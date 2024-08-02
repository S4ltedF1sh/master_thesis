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

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True, max_length=4000, truncate=True)
model = AutoModelForCausalLM.from_pretrained("results/llama2/7b-trigger-5e/checkpoint-55")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

device = torch.device("cuda")

text_gen = pipeline(
    task="text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_length=4000,
    device=device
)

txt_list = os.listdir("trigger_dataset_3k7/data")
txt_list = [txt for txt in txt_list if "ipynb" not in txt]

txt_list_head = [txt for txt in txt_list if "p2" not in txt]
txt_list_tail = [txt for txt in txt_list if "p2" in txt]

def get_context(path, max_length, head=True):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    token_length = [len(tokenizer.tokenize(line.strip())) for line in lines]
    if head:
        i = 0
        while sum(token_length[:i+1]) <= max_length and i <= len(lines):
            i = i + 1
        print(path, sum(token_length[:i]))
        return "".join(lines[:i]), path
    else:
        i = -1
        while sum(token_length[i-1:]) <= max_length and i*(-1) <= len(lines):
            i = i - 1
        print(path, sum(token_length[i:]))
        return "".join(lines[i:]), path

prompts = []
files = []
for txt in txt_list_head:
    prompt, path = get_context("trigger_dataset_3k7/data/" + txt, 3200)
    prompts.append(prompt)
    files.append(path)
for txt in txt_list_tail:
    prompt, path = get_context("trigger_dataset_3k7/data/" + txt, 3200, head=False)
    prompts.append(prompt)
    files.append(path)

from tqdm import tqdm

inst = "You're a doctor and you were given the following EMR by another doctor. You need to find 5 clinical triggers in this EMR that you think are the most important and need more infomation."
#trig_txt = ",\n".join(triggers)
template = "<s>[INST] <<SYS>>\n{inst}\n<<SYS>>\n\nPlease find the clinical triggers for the following EMR.\n\nEMR: \"{auxiliary}\"\n\nClinical triggers found:\n[/INST]"

prompts = [template.format(inst=inst, auxiliary=prompt) for prompt in prompts]
outputs = text_gen(prompts, batch_size=2)

import os

root = 'trigger_dataset_3k7/pdf_trigger'

# list to store files name
path_list = []
file_list = []
for path, subdirs, files in os.walk(root):
    for name in files:
        file_list.append(os.path.join(path, name))
file_list = [txt for txt in file_list if "checkpoint" not in txt]

cond = ["1", "2", "3", "4", "5"]
for path in file_list:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    out_path = path.replace(".txt", "_triggers.txt")
    #txt = output[0]['generated_text']
    txt = txt.split("[/INST]")[-1]
    parts = txt.split("\n")
    triggers = []
    potential = False
    for part in parts:
        if len(part) == 0:
            continue
        if part[0] in cond:
            triggers.append(part)
            if part[-1] == ":":
                potential = True
        elif part[0] not in cond and potential:
            triggers[-1] += part
            potential = False
        else:
            continue
    #print("\n".join(triggers))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(triggers[:5]))

cond = ["1", "2", "3", "4", "5"]
for output, path in zip(outputs, files):
    out_path = path.replace("/data/", "/triggers/")
    txt = output[0]['generated_text']
    txt = txt.split("[/INST]")[-1]
    parts = txt.split("\n")
    triggers = []
    potential = False
    for part in parts:
        if len(part) == 0:
            continue
        if part[0] in cond:
            triggers.append(part)
            if part[-1] == ":":
                potential = True
        elif part[0] not in cond and potential:
            triggers[-1] += part
            potential = False
        else:
            continue
    #print("\n".join(triggers))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(triggers[:5]))
