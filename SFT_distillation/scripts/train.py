#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is AI generated

import re
import ast
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, pipeline
)
from peft import PeftModel
from scipy.spatial.distance import jensenshannon

# ========= User-editable config ========= #
DATA_PATH = "data/global_opinions.csv"
OUTPUT_CSV = "final_output.csv"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "best-qwen25-lora"
REPRESENTATIVE_COUNTRIES = ['United States', 'China', 'Nigeria',
                            'Brazil', 'Pakistan', 'Sweden']
QUESTIONS_PER_COUNTRY = 100
CHECKPOINT_INTERVAL = 50          # rows
# ======================================= #

# ---------- helpers ----------
num_re = re.compile(r"[-+]?\d*\.\d+|\d+")


def extract_dict(s):
    if isinstance(s, dict):
        return s
    m = re.search(r"\{.*\}", str(s))
    return ast.literal_eval(m.group(0)) if m else {}


def safe_parse_distribution(s):
    """Return list[float]; empty if none."""
    nums = [float(x) for x in num_re.findall(s)]
    if not nums:
        return []
    total = sum(nums)
    return [x / total for x in nums] if total else nums


def process_response(resp):
    think_m = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
    reasoning = think_m.group(1).strip() if think_m else ""
    dist_m = re.search(r"\{([^\}]*)\}", resp)
    distribution = safe_parse_distribution(dist_m.group(1)) if dist_m else []
    return reasoning, distribution


def jsd(p, q):
    """ Jensen-Shannon divergence; returns 0-1.  Handles length mismatch. """
    n = max(len(p), len(q))
    p = np.array(p + [0.0]*(n-len(p)), dtype=np.float64)
    q = np.array(q + [0.0]*(n-len(q)), dtype=np.float64)
    if p.sum() == 0 or q.sum() == 0:
        return 1.0
    p /= p.sum()
    q /= q.sum()
    return jensenshannon(p, q, base=2.0) ** 2  # square to get divergence


def generate_prompt(row):
    q, ctry, opts = row['question'], row['country'], row['options']
    return (f"Given the following question and answer options, estimate the distribution "
            f"of responses you would expect from people in {ctry}. "
            f"Respond with your reasoning in <think></think> tags, then output your "
            f"estimated distribution as a list of numbers that sum to one in the format {{X, Y, Z, ...}}.\n\n"
            f"Question: {q}\nOptions: {opts}\nCountry: {ctry}\n\n"
            f"<think>Think step-by-step about cultural, historical, and social factors that might influence "
            f"how people in {ctry} would answer this question.</think>\nDistribution: {{X, Y, Z, ...}}")


def build_long(df):
    rows = []
    for _, r in df.iterrows():
        for ctry, sel in extract_dict(r['selections']).items():
            rows.append(dict(question=r['question'],
                             country=ctry,
                             selections=sel,
                             options=r['options']))
    return pd.DataFrame(rows)

# ---------- main ----------


def main():
    print("🔹 Loading data")
    raw = pd.read_csv(DATA_PATH)
    df_long = build_long(raw)
    df_rep = df_long[df_long['country'].isin(REPRESENTATIVE_COUNTRIES)]

    df = (df_rep.groupby('country', group_keys=False)
          .apply(lambda x: x.sample(n=QUESTIONS_PER_COUNTRY, random_state=42)
                 if len(x) >= QUESTIONS_PER_COUNTRY else x)
          ).reset_index(drop=True)

    # ----- models -----
    print("🔹 Loading models")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16", bnb_4bit_use_double_quant=True
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_cfg,
        device_map="auto", trust_remote_code=True)

    lora_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_cfg,
        device_map="auto", trust_remote_code=True)
    lora_model = PeftModel.from_pretrained(lora_base, LORA_PATH)

    pipe_def = pipeline("text-generation", model=base_model, tokenizer=tok,
                        max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)
    pipe_lora = pipeline("text-generation", model=lora_model, tokenizer=tok,
                         max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)

    # ----- inference -----
    reason_def, dist_def, jsd_def = [], [], []
    reason_lora, dist_lora, jsd_lora = [], [], []

    print("🚀 Inference loop")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = generate_prompt(row)

        # ground-truth distribution
        true_counts = row['selections']
        true_dist = [c / sum(true_counts) for c in true_counts]

        # default
        resp_d = pipe_def(prompt)[0]["generated_text"]
        r_d, d_d = process_response(resp_d)
        reason_def.append(r_d)
        dist_def.append(d_d)
        jsd_def.append(jsd(true_dist, d_d))

        # lora
        resp_l = pipe_lora(prompt)[0]["generated_text"]
        r_l, d_l = process_response(resp_l)
        reason_lora.append(r_l)
        dist_lora.append(d_l)
        jsd_lora.append(jsd(true_dist, d_l))

        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            ck = df.iloc[:idx+1].copy()
            ck['reason_default'] = reason_def
            ck['dist_default'] = dist_def
            ck['jsd_default'] = jsd_def
            ck['reason_lora'] = reason_lora
            ck['dist_lora'] = dist_lora
            ck['jsd_lora'] = jsd_lora
            ck.to_csv(f"checkpoint_{idx+1}.csv", index=False)
            print(f"✅ checkpoint_{idx+1}.csv")

    # attach & save
    df['reason_default'] = reason_def
    df['dist_default'] = dist_def
    df['jsd_default'] = jsd_def
    df['reason_lora'] = reason_lora
    df['dist_lora'] = dist_lora
    df['jsd_lora'] = jsd_lora
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved {OUTPUT_CSV}")

    # ----- summary -----
    print("\n====== JSD summary (0 = identical, 1 = max) ======")
    print(f"Default Qwen mean JSD : {np.mean(jsd_def):.4f}")
    print(f"LoRA-tuned mean JSD  : {np.mean(jsd_lora):.4f}")


if __name__ == "__main__":
    main()
