import json
import os
import time
from tqdm import tqdm
from transformers import AutoTokenizer
import fm_index  # This is the SWIG-generated module
import multiprocessing
from multiprocessing import Manager, Lock
from queue import Empty

use_lower_corpus = True

# Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("transformers_models/LLMs/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("transformers_models/LLMs/Mistral-7B-Instruct-v0.3")
# tokenizer = AutoTokenizer.from_pretrained("transformers_models/LLMs/Qwen2-7B-Instruct")

def tokenize_document(text):
    return tokenizer.encode(text, add_special_tokens=False)

# Initialize FMIndexManager
fm_manager = fm_index.FMIndexManager()

# Read and process the document corpus
corpus_path = "Datasets/Wiki_datasets/collection/wiki_document_corpus.json"
processed_documents = []

def process_line(line):
    doc = json.loads(line)
    docid = str(doc['docid'])
    if use_lower_corpus:
        tokenized_text = tokenize_document(doc['document'].lower())
    else:
        tokenized_text = tokenize_document(doc['document'])
    return (docid, tokenized_text)

t0 = time.time()

num_processes = multiprocessing.cpu_count() // 4

print("Reading and tokenizing documents...")
with open(corpus_path, 'r') as file:
    lines = file.readlines()
with multiprocessing.Pool(processes=num_processes) as pool:
    results = list(tqdm(pool.imap_unordered(process_line, lines), total=len(lines)))
    processed_documents.extend(results)

print(f'Time Cost: {time.time() - t0}')
t0 = time.time()



print("Initializing FMIndexManager...")
fm_manager.initialize(processed_documents)
print(f'Time Cost: {time.time() - t0}')
t0 = time.time()


# Save FMIndexManager
save_path = "Projects/2024/GenRAG/data/fm_index_map_mistral_wiki_lower.bin"
print(f"Saving FMIndexManager to {save_path}...")
fm_manager.saveAll(save_path)

print(f'Time Cost: {time.time() - t0}')
t0 = time.time()



# Load FMIndexManager
print(f"Loading FMIndexManager from {save_path}...")
loaded_fm_manager = fm_index.FMIndexManager()
loaded_fm_manager.loadAll(save_path)

print(f'Time Cost: {time.time() - t0}')
t0 = time.time()

# Perform some queries
print("Performing queries...")
test_docid = '290'  # Use the first document's ID for testing
test_fm_index = loaded_fm_manager.getFMIndex(test_docid)

print(f"Occurring distinct for document {test_docid}: {list(test_fm_index.occurring_distinct)}")

print(f'Time Cost: {time.time() - t0}')
