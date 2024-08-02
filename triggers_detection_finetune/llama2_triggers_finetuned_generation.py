import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import torch
import os
import json
from tqdm import tqdm

model = AutoModelForCausalLM.from_pretrained("results/llama2/7b-trigger-5e/checkpoint-55")
device = torch.device("cuda")

text_gen = pipeline(
    task="text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_length=4000,
    device=device
)

txts_list = os.listdir("trigger_dataset_3k7/data")
txts_list = ["trigger_dataset_3k7/data/" + txt for txt in txts_list if "ipynb" not in txt]
txts_list

inst = "You're a doctor and you were given the following EMR by another doctor. You need to find 5 clinical triggers in this EMR that you think are the most important and need more infomation."
#trig_txt = ",\n".join(triggers)
template = "<s>[INST] <<SYS>>\n{inst}\n<<SYS>>\n\nPlease find the clinical triggers for the following EMR.\n\nEMR: \"{auxiliary}\"\n\nClinical triggers found:\n[/INST]"

prompts = []
for txt in txts_list:
    with open(txt, "r", encoding="utf-8") as f:
        content = f.read()
    prompt = template.format(inst=inst, auxiliary=content)
    prompts.append(prompt)

outputs = text_gen(prompts, batch_size=64)

for i, out in enumerate(outputs):
    txt_file = f"trigger_dataset_3k7/generated/{i}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(out[0]['generated_text'])
