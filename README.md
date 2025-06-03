## GlobalOpinionBench

This repo contains the code for my group's capstone project for DATASCI 112, an introductory data science course at Stanford.

There are three main components to this project:

1. Benchmarking. Compare the performance of different leading LLMs on recreating opinion distributions in the GlobalOpinionQA dataset compiled by Anthropic. We discovered that the best performing model was Qwen2.5-7B-Instruct, which we fine-tuned in a later step.
2. Interpretability. Analyze the performance on different countries and question types and try to find trends.
3. Fine-tuning. Use the best performing examples as synthetic training data for SFT to improve the performance of Qwen2.5-7B-Instruct on this task.

When it remembered to box in its answers, our fine-tuned model significantly outperformed every leading model on this task. However, our model only followed our precise formatting instructions about 40% of the time. The base Qwen model properly formatted its answers 99% of the time. In other words, our fine-tuning process greatly improved the model's cultural reasoning capabilities, but degraded its ability to follow instructions.

<<<<<<< HEAD
I used AI to help with techniques beyond the class (regex parsing, writing SFT script, some JSD calculation). I marked the parts that I used AI.
=======
You can view our writeup and graphics at https://docs.google.com/presentation/d/1PnIjzhjew34sjsbPokTx3Akb8-x4rZEy/edit?usp=sharing&ouid=111715931402009471082&rtpof=true&sd=true

I used AI to write the SFT script and to debug code if I got stuck. I marked the parts that I used AI.
>>>>>>> refs/remotes/origin/main
