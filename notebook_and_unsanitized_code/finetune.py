import transformers, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from transformers import TrainingArguments, Trainer
from tqdm import tqdm
import sys
import pandas as pd
import json

model_name = sys.argv[1]
model = None
tokenizer = None
config_path = ""
max_length = 700

assert model_name in ["bart", "long-t5", "t0"], print("no legit model")

if model_name == "bart":
    from transformers import BartForConditionalGeneration, BartTokenizer
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    config_path = "config/bart_config.json"
elif model_name == "long-t5":
    from transformers import AutoTokenizer, LongT5Model
    model = LongT5Model.from_pretrained("google/long-t5-local-base")
    tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base")
    config_path = "config/long_t5_config.json"
else:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp")
    config_path = "config/t0_config.json"

data = pd.read_csv("new/baseline_with_context_400.csv")
data = data.loc[data["model"] == "gold"]

class EhrDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer(txt, truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

prompts = []
for index, row in data.iterrows():
  context = row["context"]
  trigger = row["trigger"]
  context = context.strip().rstrip(".")
  prompt = f"{context}. After reading the above EMR, what question do you have about \"{trigger}\"? Question:"
  prompts.append(prompt)

dataset = EhrDataset(prompts, tokenizer, max_length=max_length)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

with open('config/bart_config.json') as f:
    args = json.load(f)
training_args = TrainingArguments(**args)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                  eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                        'attention_mask': torch.stack([f[1] for f in data]),
                                                                        'labels': torch.stack([f[0] for f in data])})
train_results = trainer.train()

log_dir = f"results/{model_name}/logs"
logs_path = log_dir + "/logs.txt"
result_path = log_dir + "/result.txt"
with open(logs_path, "w") as f:
    for obj in trainer.state.log_history:
        f.write(json.dumps(obj))
        f.write("\n")

with open(result_path, "w") as f:
    f.write(json.dumps(train_results))
