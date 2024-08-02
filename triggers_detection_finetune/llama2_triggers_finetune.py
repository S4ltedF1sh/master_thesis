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
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Data Processing

def apply_prompt(txt, path, triggers):
    inst = "You're a doctor and you were given the following EMR by another doctor. You need to find some clinical triggers in this EMR that you think are important and need more infomation."
    #trig_txt = ",\n".join(triggers)
    with open(os.path.join(path, txt), "r", encoding="utf-8") as f:
        context = f.read()
    prompt = "<s>[INST] <<SYS>>\n{inst}\n<<SYS>>\n\nPlease find the clinical triggers for the following EMR.\n\nEMR: \"{auxiliary}\"\n\nClinical triggers found:\n[/INST]\n{trigs}\n</s>\n"
    return prompt.format(inst=inst, auxiliary=context, trigs=triggers)

with open("trigger_dataset_1k/dataset.json") as f:
    dataset = json.load(f)
    
new_data = {}
for k, v in dataset.items():
    triggers = ["- " + trig.strip(" .") for trig in v]
    trig_txt = "\n".join(triggers)
    new_data[k] = trig_txt

txts = []
path = "trigger_dataset_1k/data"
for k, v in new_data.items():
    txt = apply_prompt(k, path, v)
    txts.append(txt)

train_txts, eval_txts = train_test_split(txts, test_size=0.2)
train_data = pd.DataFrame.from_dict({'text': train_txts})
eval_data = pd.DataFrame.from_dict({'text': eval_txts})

train_dataset = datasets.Dataset.from_pandas(train_data)
eval_dataset = datasets.Dataset.from_pandas(eval_data)

# Training

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

model.add_adapter(peft_config)

training_args = TrainingArguments(
    output_dir=f"results/llama2/7b-trigger",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=10,
    weight_decay=1e-6,
    learning_rate=2e-4,
    logging_dir=f"results/logs",
    adam_epsilon=1e-3,
    max_grad_norm=1.0,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    disable_tqdm=False,
    gradient_accumulation_steps=16,
    #bf16=True,
)

args_max_length = 1100

trainer = SFTTrainer(
    model=model, 
    tokenizer=tokenizer,
    args=training_args, 
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, 
    dataset_text_field="text",
    max_seq_length = args_max_length,
)

train_results = trainer.train()
