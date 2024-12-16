import json
import os
import time
from tqdm import tqdm
from transformers import AutoTokenizer
import fm_index  # This is the SWIG-generated module
import multiprocessing
from multiprocessing import Manager, Lock
from queue import Empty

# build title or document FM-Index
mode = 'document'
use_lower_corpus = False

# Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("transformers_models/LLMs/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("transformers_models/LLMs/Mistral-7B-Instruct-v0.3")
# tokenizer = AutoTokenizer.from_pretrained("transformers_models/LLMs/Qwen2-7B-Instruct")

def tokenize(text):
    text = ' ' + text
    return tokenizer.encode(text, add_special_tokens=False)

# Read and process the document corpus
corpus_path = "Projects/2024/GenRAG/data/wiki_document_corpus.json"

processed_documents = []

def process_line(line):
    doc = json.loads(line)
    docid = str(doc['docid'])
    if mode == 'title':
        tokenized_text = tokenize(doc['title'])
    elif mode == 'document':
        if use_lower_corpus:
            tokenized_text = tokenize(doc['document'].lower())
        else:
            tokenized_text = tokenize(doc['document'])
    return (docid, tokenized_text)

t0 = time.time()
num_processes = multiprocessing.cpu_count() // 2

print("Reading and tokenizing documents...")
with open(corpus_path, 'r') as file:
    lines = file.readlines()
with multiprocessing.Pool(processes=num_processes) as pool:
    results = list(tqdm(pool.imap_unordered(process_line, lines), total=len(lines)))
    processed_documents.extend(results)

print(f'Time Cost: {time.time() - t0}')
t0 = time.time()


# Initialize FMIndex
print('Initializing FMIndex')
fmindex = fm_index.FMIndex()
fmindex.initialize_with_doc_info(processed_documents)
print(f'Time Cost: {time.time() - t0}')
t0 = time.time()


print('Saving FMIndex')
save_path = 'Projects/2024/GenRAG/data/corpus_fm_index_mistral_wiki.bin'
fmindex.save(save_path)
print(f'Time Cost: {time.time() - t0}')
t0 = time.time()



# Load the saved FMIndex
print('Loading FMIndex')
fmindex = fm_index.load_FMIndex(save_path)
print(f'Time Cost: {time.time() - t0}')
t0 = time.time()


# Test ngram_occurrence_count for a specific n-gram
ngram = [578, 40582, 2373]
print(f"\nTesting ngram_occurrence_count for n-gram {ngram}:")
range_start, range_end = fmindex.get_range(ngram)
occurrences2 = fmindex.ngram_occurrence_count(range_start, range_end)
print(f"Range: ({range_start}, {range_end})")

for doc_id, count in occurrences2.items():
    print(f"Document {doc_id}: {count}")


