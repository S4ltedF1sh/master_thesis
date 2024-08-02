from llama import Llama
from typing import List
import pandas as pd
import sys

def main(
    prompts: List[str],
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.5,
    top_p: float = 0.5,
    max_seq_len: int = 1000,
    max_gen_len: int = 64,
    max_batch_size: int = 8,
    out_file: str = "results/generated_question.txt"
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    results = []
    start = 0
    end = start + max_batch_size
    while end <= len(prompts):
        results += generator.text_completion(
            prompts[start : end],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        start = end
        end = min(start + max_batch_size, len(prompts))
    
    with open(out_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(result['generation'])
            f.write("\n====================\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        out_file = sys.argv[1]
    else:
        out_file = "results/generated_question.txt"
    prompts = []
    ckpt_dir = "/storage/ukp/shared/shared_model_weights/models--llama-2/llama-2-7b"
    tokenizer_path = "/storage/ukp/shared/shared_model_weights/models--llama-2/tokenizer.model"
    baseline = pd.read_csv("baseline_with_context.csv", sep="\t")
    for index, row in baseline.iterrows():
        context = row["context"]
        trigger = row["trigger"]
        prompt = f"{context}After reading the above EMR, what question do you have about \"{trigger}\"? Question:"
        prompts.append(prompt)
    main(prompts, ckpt_dir, tokenizer_path, out_file=out_file)
    
