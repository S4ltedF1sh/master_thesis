import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import torch
import os
import json
from tqdm import tqdm

llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)
llama_model = AutoModelForCausalLM.from_pretrained("results/llama2/7b-chat/checkpoint-435")
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

device = torch.device("cuda")
llama_model = llama_model.to(device)
llama_model.eval()
text_gen = pipeline(
    task="text-generation", 
    model=llama_model, 
    tokenizer=llama_tokenizer, 
    max_length=4000,
    device=device,
)

inst = "You're a doctor and you were given the following EMR by another doctor. You have some questions for the doctor who gave that EMR to you to get more details about the patient, \nYou have read an external text that contain additional information related to {trigger}: \n\"{auxiliary}\""
template = "<s>[INST] <<SYS>>\n{inst}\n<<SYS>>\n\nGiven the context in the EMR: \n\"{document}\"\n\nAfter reading the above EMR, what question do you have about \"{trigger}\"?\nQuestion:\n[/INST]"

def get_text(path):
    with open(path, "r", encoding="utf-8") as f:
        doc = f.read()
    return doc.split("[/INST]")[1].strip()

root = "results/llama2/7b-chat/generated_additional/"
files = os.listdir("trigger_dataset_3k7/triggers")
json_files = [txt for txt in files if "json" in txt]

for j in tqdm(json_files):
    json_path = "trigger_dataset_3k7/triggers/" + j
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    trigger = data["trigger"].split(":")[0].strip()
    external_path = "trigger_dataset_3k7/external_context/" + j.replace(".json", ".txt")
    ehr_path = "trigger_dataset_3k7/ehr_context/" + j.replace(".json", ".txt")
    auxiliary = get_text(external_path)
    document = get_text(ehr_path)
    instruct = inst.format(trigger=trigger, auxiliary=auxiliary)
    prompt = template.format(inst=instruct, trigger=trigger, document=document)
    
    out = text_gen(prompt)
    out_path = root + j.replace(".json", ".txt")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out[0]['generated_text'])
