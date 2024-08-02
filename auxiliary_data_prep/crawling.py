"""
Pipeline: 
- NOTEVENT["HADM_ID"]
- DIAGNOSES_ICD["HADM_ID"] 
- DIAGNOSES_ICD["ICD9_CODE"] (First 5 ["SEQ_NUM"]) 
- D_ICD_DIAGNOSES["ICD9_CODE"]
- D_ICD_DIAGNOSES["LONG_TITLE"]
- Bio.Entrez query diagnoses (long title)
- pubmed id list 
- metapub find + download pdfs
- nougat read pdfs
"""

from Bio.Entrez import esearch
from Bio import Entrez
import pandas as pd
import os
from urllib.request import urlretrieve
import metapub

Entrez.email = ''

icd9toicd10 = {}
with open("icd9to10dictionary.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        parts = line.strip().split("|")
        icd9 = parts[0].replace(".", "")
        desc = parts[-1]
        icd9toicd10[icd9] = desc

def process_term(term, n):
    tokens = term.split()
    return " ".join(tokens[:-n])

def get_pdf(path, pmid_list, max_file):
    count = 0
    files_list = os.listdir(path)
    files_list = [f.replace(".pdf", "") for f in files_list]
    count += len(files_list)
    for query in pmid_list:
        if query in files_list:
            count += 1
            if count >= max_file:
                break
            else:
                continue
        url = None
        try:
            url = metapub.FindIt(query).url
        except Exception as e:
            continue
        if url:
            try:
                out_file = os.path.join(path, query)
                urlretrieve(url, out_file + '.pdf') 
                count += 1
                if count >= max_file:
                    break
            except Exception as ex:
                continue

from metapub import PubMedFetcher
fetch = PubMedFetcher()

pmids = ['33845964', '25857560', '28149579', '35613080', '29629687']
articles = {}
for pmid in pmids:
    articles[pmid] = fetch.article_by_pmid(pmid)

redo_files = []
redo_icd9 = []
for f in files:
    dirs = os.listdir("pdf/" + f)
    dirs = [d for d in dirs if ".txt" not in d and "ipynb" not in d]
    for d in dirs:
        path = "pdf/" + f + "/" + d
        fn = len([d for d in os.listdir(path) if "ipynb" not in d])
        if fn < 3:
            print(path, fn)
            redo_files.append(f)
            redo_icd9.append(d)
redo_files = list(set(redo_files))
redo_icd9 = [f for f in redo_icd9 if "ipynb" not in f]

for f in files:
    if f not in redo_files:
        continue
    print(f)
    path = f"pdf/{f}"
    with open(path + "/diagnosis.txt", "r") as f:
        diags = f.readlines()
    n = min(len(diags), 3)
    for diag in diags[:n]:
        parts = diag.split(":")
        icd9 = parts[0].strip()
        term = parts[1].strip()
        if icd9 not in redo_icd9 or icd9 not in icd9toicd10:
            continue
        term = process_term(term, 5)
        pdf_path = path + "/" + icd9
        if not os.path.isdir(pdf_path):
            os.mkdir(pdf_path)
        handle = Entrez.esearch(db="pubmed", term=term, retmax=200, sort='relevance')
        records = Entrez.read(handle)
        pmid_list = records['IdList']
        get_pdf(pdf_path, pmid_list, 3)
