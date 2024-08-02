import os, torch, logging
from datasets import load_dataset, load_from_disk
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from transformers import LongT5ForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
from torch.utils.data import Dataset
import datasets
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True, max_length=4000, truncate=True)
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained("results/llama2/7b-trigger-5e/checkpoint-55")
device = torch.device("cuda")

text_gen = pipeline(
    task="text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_length=4000,
    device=device
)

with open("mimic-s2orc/pdf_list.txt", "r") as f:
    pdfs_list = f.readlines()
dirs_list = ["mimic-s2orc/" + txt.strip().replace(".pdf", "") for txt in pdfs_list if ".pdf" in txt]

inst = "You're a doctor and you were given the following EMR by another doctor. You need to find 5 clinical triggers in this EMR that you think are the most important and need more infomation."
#trig_txt = ",\n".join(triggers)
template = "<s>[INST] <<SYS>>\n{inst}\n<<SYS>>\n\nPlease find the clinical triggers for the following EMR.\n\nEMR: \"{auxiliary}\"\n\nClinical triggers found:\n[/INST]"

for dir_path in tqdm(dirs_list):
    out_dir = dir_path.replace("/pdf/", "/pdf_trigger/")
    os.makedirs(out_dir, exist_ok=True)
    txts = [txt for txt in os.listdir(dir_path) if ".txt" in txt]
    if len(txts) < 3:
        n = len(txts)
    elif len(txts) == 3:
        n = len(txts) - 1
    elif len(txts) >= 4 and len(txts) < 7:
        n = len(txts) - 2
    elif len(txts) >= 7 and len(txts) < 10:
        n = len(txts) - 3
    else:
        n = 7
    for txt in txts[:n]:
        with open(os.path.join(dir_path, txt), "r", encoding="utf-8") as f:
            content = f.read()
        prompt = template.format(inst=inst, auxiliary=content)
        len_prompt = len(tokenizer.tokenize(prompt))
        if len_prompt > 3500:
            continue
        out = text_gen(prompt)
        out_path = os.path.join(out_dir, txt)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out[0]['generated_text'])
