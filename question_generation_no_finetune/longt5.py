from transformers import AutoTokenizer, LongT5ForConditionalGeneration
import pandas as pd
import torch

device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")
model = LongT5ForConditionalGeneration.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps").to(device)

results = []
baseline = pd.read_csv("baseline_with_context.csv", sep="\t")
for index, row in baseline.iterrows():
    context = row["context"]
    trigger = row["trigger"]
    prompt = f"{context}After reading the above EMR, what question do you have about \"{trigger}\"? Question:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    outputs = model.generate(input_ids)
    results.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

with open("results/longt5_generated_questions.txt", "w", encoding="utf-8") as f:
    for txt in results:
        f.write(txt.replace("\n", ". ").strip())
        f.write("\n")
    

