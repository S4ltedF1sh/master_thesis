import os, torch, logging
from datasets import load_dataset, load_from_disk
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from torch.utils.data import Dataset
import datasets
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("results/llama2/7b-chat/checkpoint-435")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

device = torch.device("cuda")
model = model.to(device)
output_path = "results/llama2_finetuned_generated.txt"
results = []
baseline = pd.read_csv("baselines/baseline_with_context.csv", sep='\t')
prompts = []
sys_msg = "<<SYS>>\nYou're a doctor and you were given the following EMR by another doctor. You have some questions for the doctor who gave that EMR to you to get more details about the patient. \n<</SYS>>\n\n"


for index, row in baseline.iterrows():
    context = row["context"].strip()
    trigger = row["trigger"]
    context = context.strip().rstrip(".")
    prompt = f"<s>[INST] {sys_msg}Given the EMR: \"{context}\"\nAfter reading the above EMR, what question do you have about \"{trigger}\"?\nQuestion:\n[/INST]"
    prompts.append(prompt)

text_gen = pipeline(
    task="text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_length=500,
    device=device,
    do_sample=True,
    temperature=0.5,
    top_p=0.5,
)

outputs = text_gen(prompts, batch_size=64)

for i, out in enumerate(outputs):
    txt_file = f"results/llama2/7b-chat/generated/{i}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(out[0]['generated_text'])
