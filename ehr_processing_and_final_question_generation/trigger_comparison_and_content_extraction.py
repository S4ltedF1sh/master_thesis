import os
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, BertConfig
from sentence_transformers import SentenceTransformer
import re
import nltk
import timeit
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
import json

# Trigger Comparison

model = SentenceTransformer("dmis-lab/biobert-base-cased-v1.2")

txts = []
for path, subdirs, files in os.walk("trigger_dataset_3k7/pdf_trigger"):
    for name in files:
        txts.append(os.path.join(path, name))
txts = [txt for txt in txts if "_triggers.txt" in txt]

def process_trigger_text(txt):
    with open(txt, "r", encoding="utf-8") as f:
        lines = f.readlines()
    triggers = []
    contexts = []
    for line in lines:
        parts = line.strip().split(":")
        triggers.append(parts[0][2:].strip())
        contexts.append(parts[1].strip())
    return triggers, contexts

txts_2 = os.listdir("trigger_dataset_3k7/triggers")
txts_2 = [txt for txt in txts_2 if "txt" in txt]

def cosine_sim(a, b):
    return (a @ b.T) / (norm(a)*norm(b))
    
def get_similarity_score(ehr, trig_list):
    ehr_trigs = []
    pdf_trigs = []
    with open(ehr, "r", encoding="utf-8") as f:
        lines = f.readlines()
    ehr_trigs = [line[2:].strip() for line in lines]
    for txt in trig_list:
        with open(txt, "r", encoding="utf-8") as f:
            lines = f.readlines()
        pdf_trigs += [line[2:].strip() for line in lines]
    ehr_embeds = model.encode(ehr_trigs, device="cuda")
    pdf_embeds = model.encode(pdf_trigs, device="cuda")
    avg_scores = []
    scores_list = []
    for e in ehr_embeds:
        scores = []
        for pe in pdf_embeds:
            scores.append(cosine_sim(e, pe))
        scores_list.append(scores)
        avg_scores.append(sum(scores)/len(scores))
    idx = np.argmax(avg_scores)
    pdf_idx = np.argmax(scores_list[idx])
    relate_trig = pdf_trigs[pdf_idx]
    pdf = trig_list[int(pdf_idx/5)]
    best_trig = ehr_trigs[np.argmax(avg_scores)]
    return best_trig, relate_trig, pdf

for txt2 in txts_2:
    ehr_trig = "trigger_dataset_3k7/triggers/" + txt2
    if "_" in txt2:
        kw = "/" + re.sub(r"_[a-z0-9]+.txt", "/", txt2)
    else:
        kw = "/" + txt2.replace(".txt", "/")
    trig_list = [txt for txt in txts if kw in txt]
    best_trig, relate_trig, pdf = get_similarity_score(ehr_trig, trig_list)
    trig_dict = {}
    trig_dict["trigger"] = best_trig
    trig_dict["related_trigger"] = relate_trig
    trig_dict["context_pdf"] = pdf
    path = "trigger_dataset_3k7/triggers/" + txt2.replace(".txt", ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trig_dict, f)

# Content Extraction

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import torch
import os
import json
from tqdm import tqdm

llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
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

inst = "You're a doctor and you were given the following EMR by another doctor. You think that {trigger} is important."
template = "<s>[INST] <<SYS>>\n{inst}\n<<SYS>>\n\nGiven the following EMR, please find some content related to {trigger}.\n\nEMR: \"{document}\"\n\nContent found:\n[/INST]"

files = os.listdir("trigger_dataset_3k7/triggers")
json_files = [txt for txt in files if "json" in txt]
txt_files = [txt for txt in files if "txt" in txt]

def get_context(path, max_length, head=True):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    token_length = [len(llama_tokenizer.tokenize(line.strip())) for line in lines]
    if head:
        i = 0
        while sum(token_length[:i+1]) <= max_length and i <= len(lines):
            i = i + 1
        return "".join(lines[:i]), path
    else:
        i = -1
        while sum(token_length[i-1:]) <= max_length and i*(-1) <= len(lines):
            i = i - 1
        return "".join(lines[i:]), path

for j in tqdm(json_files):
    json_path = "trigger_dataset_3k7/triggers/" + j
    txt_path = "trigger_dataset_3k7/data/" + j.replace(".json", ".txt")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    head = "p2" not in t
    doc,_ = get_context(txt_path, 3200, head=head)
    trig_1 = data["trigger"].split(":")[0].strip()
    instruct = inst.format(trigger=trig_1)
    prompt = template.format(inst=instruct, trigger=trig_1, document=doc)
    out = text_gen(prompt)
    out_path = txt_path.replace("/data/", "/ehr_context/")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out[0]['generated_text'])
