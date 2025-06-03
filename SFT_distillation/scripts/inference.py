#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is AI generated

import re
import ast
import math
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import PeftModel
from scipy.spatial.distance import jensenshannon
import torch

import os
import sys
import time


def print_row_status(idx, prompt, raw_output, parsed_dist, jsd_val, model_type):
    os.system('clear')  # clears terminal screen (on Lambda or Unix)
    print(f"🧠 Row {idx+1} — Model: {model_type}")
    print("="*60)
    print("🔸 Prompt:")
    print(prompt)
    print("\n🔹 Model Output:")
    print(raw_output)
    print("\n🔍 Parsed Distribution:")
    print(parsed_dist)
    print(f"\n📏 JSD: {jsd_val:.4f}")
    print("="*60)
    time.sleep(0.5)  # pause just a little so it's readable


# ---------- CONFIG ---------- #
DATA_PATH = "data/not_in_training.csv"
OUTPUT_CSV = "final_output.csv"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "best-qwen25-lora"
REPR_COUNTRIES = [
    "United States", "China", "Nigeria", "Brazil", "Pakistan", "Sweden"
]
QUESTIONS_PER_COUNTRY = 100
CHECKPOINT_INTERVAL = 50
GEN_KWARGS = dict(max_new_tokens=1048, do_sample=True,
                  temperature=0.7, top_p=0.9)
# ---------------------------- #

num_re = re.compile(r"[-+]?\d*\.\d+|\d+")


def extract_dict(s):
    if isinstance(s, dict):
        return s
    m = re.search(r"\{.*\}", str(s))
    return ast.literal_eval(m.group(0)) if m else {}


def safe_parse_distribution(s):
    nums = [float(x) for x in num_re.findall(s)]
    if not nums:
        return []
    tot = sum(nums)
    return [x / tot for x in nums]


def process_response(resp):
    # Store full output in reasoning
    reasoning = resp.strip()
    # Extract distribution
    matches = re.findall(r"\{([^\}]*)\}", resp)
    dist_str = matches[-1] if matches else ""
    distribution = safe_parse_distribution(dist_str)
    return reasoning, distribution


def jsd(p, q):
    n = max(len(p), len(q))
    p = np.array(p + [0.0]*(n-len(p)))
    q = np.array(q + [0.0]*(n-len(q)))
    if p.sum() == 0 or q.sum() == 0:
        return 1.0
    p /= p.sum()
    q /= q.sum()
    return jensenshannon(p, q, base=2.0)**2


def build_long(df):
    rows = []
    for _, r in df.iterrows():
        for c, sels in extract_dict(r["selections"]).items():
            rows.append(dict(question=r["question"], country=c,
                             selections=sels, options=r["options"]))
    return pd.DataFrame(rows)


def prompt_from_row(row):
    q, c, o = row["question"], row["country"], row["options"]
    return (f"Given the following question and answer options, estimate the distribution "
            f"of responses you would expect from people in {c}. Respond with your reasoning "
            f"in <think></think> tags, then output your estimated distribution as a list "
            f"of numbers that sum to one in the format {{X, Y, Z, ...}}.\n\n"
            f"Question: {q}\nOptions: {o}\nCountry: {c}\n\n"
            f"<think>Think step-by-step about cultural, historical, and social factors that "
            f"might influence how people in {c} would answer this question.</think>\n"
            f"{{X, Y, Z, ...}}")


def generate_text(model, tokenizer, prompt):
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False
    )
    toks = tokenizer(chat, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**toks, max_new_tokens=1052, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    print("🔹 Loading and filtering data (already long-form)")
    raw = pd.read_csv(DATA_PATH)
    # parse the stringified list into a real Python list
    raw['selections'] = raw['selections'].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else s
    )
    df_long = raw
    df_rep = df_long[df_long["country"].isin(REPR_COUNTRIES)]
    df = (
        df_rep
        .groupby("country", group_keys=False)
        .apply(lambda x: x.sample(n=QUESTIONS_PER_COUNTRY, random_state=42)
               if len(x) >= QUESTIONS_PER_COUNTRY else x)
        .reset_index(drop=True)
    )

    print("🔹 Loading models (4-bit QLoRA)")
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype="bfloat16",
                                 bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_cfg,
        device_map="auto", trust_remote_code=True)

    lora_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_cfg,
        device_map="auto", trust_remote_code=True)
    lora = PeftModel.from_pretrained(lora_base, LORA_PATH)

    reason_def, dist_def, jsd_def = [], [], []
    reason_lora, dist_lora, jsd_lora = [], [], []

    print("🚀 Inference loop")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = prompt_from_row(row)

        counts = row["selections"]
        if not counts or sum(counts) == 0:
            reason_def.append("")
            dist_def.append([])
            jsd_def.append(1.0)
            reason_lora.append("")
            dist_lora.append([])
            jsd_lora.append(1.0)
            continue
        true_d = [c/sum(counts) for c in counts]

        # default (vanilla)
        resp_d = generate_text(base, tok, prompt)
        r_d, d_d = process_response(resp_d)
        reason_def.append(r_d)
        dist_def.append(d_d)
        jsd_def.append(jsd(true_d, d_d))
        print_row_status(idx, prompt, resp_d, d_d,
                         jsd_def[-1], model_type="Vanilla")

        # lora-tuned
        resp_l = generate_text(lora, tok, prompt)
        r_l, d_l = process_response(resp_l)
        jsd_l = jsd(true_d, d_l)
        print_row_status(idx, prompt, resp_l, d_l, jsd_l, model_type="LoRA")
        reason_lora.append(r_l)
        dist_lora.append(d_l)
        jsd_lora.append(jsd_l)

        if (idx+1) % CHECKPOINT_INTERVAL == 0:
            tmp = df.iloc[:idx+1].copy()
            tmp["reason_default"] = reason_def
            tmp["dist_default"] = dist_def
            tmp["jsd_default"] = jsd_def
            tmp["reason_lora"] = reason_lora
            tmp["dist_lora"] = dist_lora
            tmp["jsd_lora"] = jsd_lora
            tmp.to_csv(f"checkpoint_{idx+1}.csv", index=False)
            print(f"✅ checkpoint_{idx+1}.csv")

    df["reason_default"] = reason_def
    df["dist_default"] = dist_def
    df["jsd_default"] = jsd_def
    df["reason_lora"] = reason_lora
    df["dist_lora"] = dist_lora
    df["jsd_lora"] = jsd_lora
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {OUTPUT_CSV}")

    print("\n====== Mean Jensen–Shannon divergence ======")
    print(f"Vanilla Qwen : {np.mean(jsd_def):.4f}")
    print(f"LoRA-tuned   : {np.mean(jsd_lora):.4f}")


if __name__ == "__main__":
    main()
