import os
import time
import argparse
import csv, json
import torch
import pickle
import math, heapq
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification
from transformers import LlamaTokenizerFast, MistralForCausalLM
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List
# from scripts.trie import Trie
# from seal import FMIndex, fm_index_generate
from seal.cpp_modules_new import fm_index
from splade.models.transformer_rep import Splade
from peft import PeftModel, PeftConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

CODE_DIR='YOUR_CODE_DIR'

max_input_len=64
max_target_len=1200
eval_batch_size=128
dataset_name = 'nq'  # nq, triviaqa, hotpotqa, popqa, 2wiki
enable_constrain = True  # Set to True to enable constraints
flash_rag = True
use_subset_num = 0  # 0 for all
load_checkpoint = True

# 模型和输出路径
# ----- Mistral Series -----
model_type = 'mistral'
model_name = f"LLM_PATH"  # Mistral-7B-Instruct-v0.3
output_dir = f"CHECKPOINT_PATH"
# ----- Llama3 Series -----
# model_type = 'llama3'
# model_name = f"LLM_PATH"  # Llama-3-8B-Instruct, Llama-3.2-3B-Instruct
# output_dir = f"CHECKPOINT_PATH"
# ----- Qwen2.5 Series -----
# model_type = 'qwen'
# model_name = f"LLM_PATH"  # Qwen2.5-7B-Instruct
# output_dir = f"CHECKPOINT_PATH"

# 数据路径
eval_data_path = f"FLASHRAG_EVAL_DATA_PATH"
corpus_fm_index_path = f"CORPUS_FM_INDEX_PATH"  # corpus_fm_index_llama3, corpus_fm_index_mistral
docid_fm_index_map_path = f"FM_INDEX_MAP_PATH"  # fm_index_map_llama3, fm_index_map_mistral

apply_chat_template = True
if 'nochat' in output_dir or 'mistral' in output_dir:
    apply_chat_template = False
if load_checkpoint:
    model_name = output_dir
retriever_path = f"BGE_RERANKER_PATH"  # base, large
splade_path = f"SPLADE_V3_PATH"
runs_path = f'SPLADE_RUN_PATH'

# Load retriever
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_path)
retriever_model = AutoModelForSequenceClassification.from_pretrained(retriever_path, torch_dtype=torch.bfloat16)
# Load retriever
splade_tokenizer = AutoTokenizer.from_pretrained(splade_path)
splade_model = Splade(splade_path, agg="max")
reverse_voc = {v: k for k, v in splade_tokenizer.vocab.items()}

# Load LLM
def load_lora_model(base_model_path, lora_model_path):
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16
    )
    print('Merging LoRA weights...')
    tokenizer = AutoTokenizer.from_pretrained(lora_model_path)
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    # Load and update embeddings for new tokens
    new_embeddings_data = torch.load(f"{lora_model_path}/new_token_embeddings.pt")
    num_new_tokens = new_embeddings_data['num_new_tokens']
    model.get_input_embeddings().weight.data[-num_new_tokens:] = new_embeddings_data['input_embeddings_for_new_tokens']
    if not new_embeddings_data['is_tied']:
        model.get_output_embeddings().weight.data[-num_new_tokens:] = new_embeddings_data['output_embeddings_for_new_tokens']
    # Merge LoRA weights with base model for faster inference (optional)
    model = model.merge_and_unload()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    print('Done')
    return model, tokenizer
# Load and Merge
model, tokenizer = load_lora_model(model_name, output_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)
retriever_model.eval()
retriever_model.to(device)
splade_model.eval()
splade_model.to(device)


class HierarchicalConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        title_fm_index,
        corpus_fm_index,
        docid_fm_index_map,
        tokenizer,
        retriever_tokenizer,
        retriever_model,
        sep_symbol: str,
        clue_start: str,
        clue_end: str,
        evidence_start: str,
        evidence_end: str,
        answer_start: str,
        max_input_len: int,
        max_target_len: int,
        batch_size: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        max_occurrence: int = 1e3,
        max_window_num = 256, 
        max_window_num_per_doc = 32,
        window_size: int = 32,
        top_k: int = 10,
        top_k_wd: int = 5,
    ):
        super().__init__()
        self.title_fm_index = title_fm_index
        self.corpus_fm_index = corpus_fm_index
        self.docid_fm_index_map = docid_fm_index_map
        self.total_docs = docid_fm_index_map.getDocCount()
        self.tokenizer = tokenizer
        self.retriever_tokenizer = retriever_tokenizer
        self.retriever_model = retriever_model

        self.i = 0
        if tokenizer.bos_token_id in self.tokenizer.encode('.'):
            self.i = 1
        self.sep_symbol = self.tokenizer.encode(sep_symbol)[self.i]
        self.clue_start = self.tokenizer.encode(clue_start)[self.i]
        self.evidence_start = self.tokenizer.encode(evidence_start)[self.i]
        self.clue_end = self.tokenizer.encode(clue_end)[self.i]
        self.evidence_end = self.tokenizer.encode(evidence_end)[self.i]
        self.answer_start = self.tokenizer.encode(answer_start)[self.i:]
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.top_k = top_k  # Num of top-relevant docs for evidence constraints
        self.top_k_wd = top_k_wd  # Num of top-relevant windows for evidence constraints
        self.window_size = window_size
        self.max_occurrence = max_occurrence
        self.max_window_num = max_window_num
        self.max_window_num_per_doc = max_window_num_per_doc
        self.min_evidence_length = 2 * window_size
        self.max_evidence_length = 5 * window_size
        self.max_merged_window_length = 4 * window_size
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.batch_size = batch_size

        # Initialize variables for each batch
        self.reset_state()
        self.all_retrieved_docids = []
        self.start_idx = 0

    def reset_state(self):
        self.cur_stages = ['free'] * self.batch_size
        self.cur_start_position_idx = [0] * self.batch_size
        self.cur_clues = [[] for _ in range(self.batch_size)]
        self.retrieved_docids = [{} for _ in range(self.batch_size)]
        self.window_sequences = [[] for _ in range(self.batch_size)]  # [[(docid, allowed_tokens)]]
        self.clue_positions = [{} for _ in range(self.batch_size)]  # [{docid: {clue: [positions]}}]
        self.cur_window_id = [0 for _ in range(self.batch_size)]  # [window_id]
        self.query_texts = None
        self.expanded_clue_words = None
        self.splade_retrieved_docids = None  # splade_retrieved_docids[batch_id][k] = docid
        self.input_ids = None

    def compute_similarity_scores(self, pairs: List[List]) -> List[float]:  # Using reranker model
        """Compute similarity scores for a batch of [question, text] pairs."""
        with torch.no_grad():
            inputs = self.retriever_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=self.window_size * 4).to(device)
            logits = self.retriever_model(**inputs, return_dict=True).logits.view(-1).float()
            scores = logits.cpu().tolist()
        return scores

    def clue_fmindex_processor(self, clue_prefix: List[int]):
        prefix_allowed_tokens = set()
        if len(clue_prefix) == 0:
            distinct = list(self.corpus_fm_index.occurring_distinct)
        elif len(clue_prefix) > 6:  # max clue length
            distinct = []
        else:
            low, high = self.corpus_fm_index.get_range(clue_prefix)
            distinct = self.corpus_fm_index.get_distinct(low, high)
        prefix_allowed_tokens.update(distinct)
        if self.tokenizer.eos_token_id in prefix_allowed_tokens:
            prefix_allowed_tokens.remove(self.tokenizer.eos_token_id)
        if len(clue_prefix) > 1: 
            prefix_allowed_tokens.add(self.sep_symbol)
            prefix_allowed_tokens.add(self.clue_end)
        return list(prefix_allowed_tokens)

    def find_next_tokens(self, sentence_prefix, window_sequence):
        next_tokens_set = set()  # 初始化集合
        prefix_length = len(sentence_prefix)
        for i in range(len(window_sequence) - prefix_length + 1):
            if window_sequence[i:i + prefix_length] == sentence_prefix:  # 检查当前位置是否与prefix匹配
                if i + prefix_length < len(window_sequence):  # 如果匹配，检查是否有下一个token
                    next_tokens_set.add(window_sequence[i + prefix_length])
        return next_tokens_set

    def evidence_fmindex_processor(self, batch_id, sentence_prefix: List[int]):
        start_token_split = 5
        prefix_allowed_tokens = set()
        if len(sentence_prefix) == 0:  # evidence第一个token直接用window_sequences
            _, allowed_tokens = self.window_sequences[batch_id][self.cur_window_id[batch_id]]
            start_idx = (len(allowed_tokens) - 1) // start_token_split
            prefix_allowed_tokens.update(allowed_tokens[:start_idx])  # 只保留前半部分tokens
            # print(f'1st token prefix_allowed_tokens[{batch_id}]', allowed_tokens[:start_idx], tokenizer.decode(allowed_tokens[:start_idx]))
        else:
            docid, allowed_tokens = self.window_sequences[batch_id][self.cur_window_id[batch_id]]
            if len(sentence_prefix) < (len(allowed_tokens) - 1) // start_token_split:
                start_idx = (len(allowed_tokens) - 1) // start_token_split + len(sentence_prefix)
                distinct = self.find_next_tokens(sentence_prefix, allowed_tokens[:start_idx])  # 只保留前半部分 + len(sentence_prefix)的tokens
                # print('sentence_prefix', sentence_prefix)
                # print('distinct', distinct, tokenizer.decode(list(distinct)))
                prefix_allowed_tokens.update(distinct)
            if len(sentence_prefix) >= (len(allowed_tokens) - 1) // start_token_split or len(distinct) == 0:
                cur_index = self.docid_fm_index_map.getFMIndex(docid)
                low, high = cur_index.get_range(sentence_prefix)
                distinct = cur_index.get_distinct(low, high)
                prefix_allowed_tokens.update(set(distinct))
            # print(f'allowed_tokens[{batch_id}]', allowed_tokens, f'distinct[{batch_id}]', distinct)
            if len(sentence_prefix) > self.min_evidence_length:
                prefix_allowed_tokens.update([self.evidence_end, self.sep_symbol])
            # if len(sentence_prefix) >= len(allowed_tokens):
            #     prefix_allowed_tokens.update([self.evidence_end, self.sep_symbol])
        if self.tokenizer.eos_token_id in prefix_allowed_tokens:
            prefix_allowed_tokens.remove(self.tokenizer.eos_token_id)
        if len(prefix_allowed_tokens) == 0:
            # print(f'batch_id[{batch_id}]', 'sentence_prefix no allowed tokens', sentence_prefix)
            return list(self.tokenizer.get_vocab().values())
        return list(prefix_allowed_tokens)

    def compute_window_allowed_tokens(self, batch_id):
        self.clue_positions[batch_id] = defaultdict(lambda: defaultdict(list))
        allowed_docids = self.retrieved_docids[batch_id]
        if len(allowed_docids) == 0:
            print('No allowed_docids!!!')
            return
        # print('allowed_docids', allowed_docids)

        # tw = time.time()
        window_infos = []  # To store window information with positions
        for docid in allowed_docids:
            cur_index = self.docid_fm_index_map.getFMIndex(docid)
            for clue in self.cur_clues[batch_id]:  # clue is tokens
                low, high = cur_index.get_range(clue)
                if high - low == 0:
                    continue
                elif high - low > 64:  # 只考虑每一个clue最先出现的k次
                    low = high - 64
                max_occurrences = min(high - low, self.max_window_num)
                # Iterate from high-1 to high - max_occurrences in descending order，在文档里就是从前往后合并
                for idx in range(high, high - max_occurrences, -1):
                    if idx < low:
                        break
                    clue_position = cur_index.locate(idx)
                    start_position = max(clue_position - self.window_size, 0)
                    end_position = min(clue_position + self.window_size, cur_index.size() - 1)
                    window_infos.append({
                        'docid': docid,
                        'cur_index': cur_index,
                        'start': start_position,
                        'end': end_position,
                        'clue_positions': [clue_position],
                        'clues': [clue],
                    })
        if not window_infos:
            print('No window_sequences.')
            return
        # Sort windows by start descending (from high to low in document)
        window_infos.sort(key=lambda x: x['start'], reverse=True)
        # print('window_infos', window_infos)
        # print('Locate window', time.time() - tw)
        # tw = time.time()

        # Merge overlapping windows
        # Merge overlapping windows within each docid
        merged_windows = []
        docid_to_windows = defaultdict(list)
        for info in window_infos:
            docid_to_windows[info['docid']].append(info)
        for docid, windows in docid_to_windows.items():
            # Sort windows by start descending
            sorted_windows = sorted(windows, key=lambda x: x['start'], reverse=True)
            if not sorted_windows:
                continue
            doc_window_count = 0
            current = sorted_windows[0].copy()
            for info in sorted_windows[1:]:
                if info['start'] <= current['end']:  # Overlap, merge
                    new_start = min(current['start'], info['start'])
                    new_end = max(current['end'], info['end'])
                    if (new_end - new_start + 1) <= self.max_merged_window_length:
                        current['start'] = new_start
                        current['end'] = new_end
                        current['clue_positions'].extend(info['clue_positions'])
                        current['clues'].extend(info['clues'])
                    else:
                        merged_windows.append(current)
                        current = info.copy()
                        doc_window_count += 1
                else:
                    merged_windows.append(current)
                    current = info.copy()
                    doc_window_count += 1
                if doc_window_count >= self.max_window_num_per_doc - 1 or len(merged_windows) > self.max_window_num:
                    break
            merged_windows.append(current)
            if len(merged_windows) > self.max_window_num:
                break
        
        # print('Merge window', time.time() - tw)
        # tw = time.time()
        # Extract merged window sequences
        window_texts = []
        window_info = []
        for window in merged_windows:
            docid = window['docid']
            cur_index = window['cur_index']
            window_sequence = cur_index.extract_text(window['start'], window['end'])
            window_sequence = [token for token in list(window_sequence) if 0 <= int(token) < 1e6]
            window['window_sequence'] = window_sequence  # Store the window_sequence in window
            window_text = self.tokenizer.decode(window_sequence, skip_special_tokens=True)
            window_texts.append(window_text)
            window_info.append(window)
        # print('extract_text', time.time() - tw)
        # tw = time.time()

        # Compute similarities between query and window_texts
        question_window_pairs = [[self.query_texts[batch_id], window] for window in window_texts]
        similarities = self.compute_similarity_scores(question_window_pairs)
        # Select top_k_wd windows based on similarity
        top_k_indices = np.argsort(similarities)[::-1][:self.top_k_wd].tolist()
        # print(f'top_k_wd_indices[{batch_id}]', top_k_indices)
        # print('Rerank window', time.time() - tw)
        # tw = time.time()

        # Process top_k windows
        for idx in top_k_indices:
            info = window_info[idx]
            docid = info['docid']
            window_sequence = info['window_sequence']
            # print(f"docid {docid}", f"clue_positions {info['clue_positions']}", f"clues {info['clues']}", f"window_text[{batch_id}]", window_texts[idx])
            self.window_sequences[batch_id].append((docid, window_sequence))  # 按相似度顺序保存window_sequences

    def weighted_reciprocal_rank_fusion(self, retrieved_docids, splade_retrieved_docids, w1=1.0, w2=2.0, top_n=10):
        """
        Perform Weighted Reciprocal Rank Fusion on two ranked lists of document IDs.
        Args:
            retrieved_docids (list): Ranked list from FM-Index retrieval.
            splade_retrieved_docids (list): Ranked list from SPLADE retrieval.
            w1 (float): Weight for FM-Index retrieval.
            w2 (float): Weight for SPLADE retrieval.
            top_n (int): Number of top documents to return.
        Returns:
            list: Fused top_n document IDs.
        """
        score_dict = defaultdict(float)
        # Process FM-Index retrieved_docids
        for rank, doc_id in enumerate(retrieved_docids, start=1):
            score_dict[doc_id] += w1 / rank
        # Process SPLADE retrieved_docids
        for rank, doc_id in enumerate(splade_retrieved_docids, start=1):
            score_dict[doc_id] += w2 / rank
        # Sort documents by their total score in descending order
        sorted_docs = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
        # Extract the top_n document IDs
        top_docs = [doc_id for doc_id, score in sorted_docs[:top_n]]
        return top_docs

    def document_scoring_fn(self, batch_id):
        # 计算fm-index文档评分
        doc_scores = defaultdict(float)
        for clue in self.cur_clues[batch_id]:
            # Corpus Fm-Index
            low, high = self.corpus_fm_index.get_range(clue)
            total_occurrences = high - low  # Total occurrences of n-gram
            if total_occurrences == 0 or total_occurrences > self.max_occurrence:
                continue
            clue_occurrence_count = self.corpus_fm_index.ngram_occurrence_count(low, high)
            # Number of documents containing n-gram
            doc_frequency = len(clue_occurrence_count)
            # Compute n-gram weight
            w_i = 1 * (self.alpha * math.log(self.total_docs / total_occurrences)
                                      + self.beta * math.log(self.total_docs / doc_frequency))
            # For each document containing the n-gram, compute score
            for docid, tf in clue_occurrence_count.items():
                f_ti_d = math.log(1 + tf)
                doc_scores[docid] += w_i * f_ti_d
        # Find top-k docids
        fm_index_top_k = 5
        if len(doc_scores) > fm_index_top_k:
            fm_index_retrieved_docids = heapq.nlargest(fm_index_top_k, doc_scores.items(), key=lambda x: x[1])
        else:
            fm_index_retrieved_docids = doc_scores.items()
        fm_index_retrieved_docids = [item[0] for item in fm_index_retrieved_docids]
        # Rank fusion of FM-Index and SPLADE retrieved docids
        self.retrieved_docids[batch_id] = self.weighted_reciprocal_rank_fusion(fm_index_retrieved_docids, self.splade_retrieved_docids[batch_id], w1=1.0, w2=2.0, top_n=self.top_k)
        # Expand SPLADE clue words
        # print(f'fm_index_retrieved_docids[{batch_id}]', fm_index_retrieved_docids)
        # print(f'self.retrieved_docids[{batch_id}]', self.retrieved_docids[batch_id])
        # print(f'self.cur_clues[{batch_id}]', self.cur_clues[batch_id])
        # print(f'self.expanded_clue_words[{batch_id}]', self.expanded_clue_words[batch_id])
        expanded_clues = []
        for clue in self.cur_clues[batch_id]:
            if clue != '':
                clue_words = self.tokenizer.decode(clue, skip_special_tokens=True)
                w_tokens = self.tokenizer.encode(' ' + clue_words.lower().strip())[self.i:]
                if w_tokens not in expanded_clues:
                    expanded_clues.append(w_tokens)
        for clue_words, clue_similarity in self.expanded_clue_words[batch_id]:
            if clue_words != '':
                w_tokens = self.tokenizer.encode(' ' + clue_words.lower().strip())[self.i:]
                if w_tokens not in expanded_clues:
                    expanded_clues.append(w_tokens)
        # print('expanded_clues', expanded_clues)
        self.cur_clues[batch_id] = expanded_clues
    
    def has_unused_windows(self, batch_id):
        return self.cur_window_id[batch_id] + 1 < len(self.window_sequences[batch_id])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float('-inf'))
        input_seq_num, seq_len = input_ids.size()
        input_ids = input_ids[:, self.max_input_len:].tolist()
        if len(input_ids[0]) == 0:
            mask[:, self.clue_start] = 0.0
            return scores + mask

        # t_start = time.time()
        # t_free = 0
        # t_clue = 0
        # t_evidence = 0

        # t_doc_score, t_get_window = 0, 0

        for batch_id in range(input_seq_num):
            last_token = input_ids[batch_id][-1]

            if self.cur_stages[batch_id] == 'free':
                # t00 = time.time()
                if last_token == self.clue_start:  # 若最后一个token是clue_start，从free进入clue状态
                    self.cur_stages[batch_id] = 'clue'
                    self.cur_start_position_idx[batch_id] = len(input_ids[batch_id])
                    allowed_tokens = self.clue_fmindex_processor([])
                    mask[batch_id, allowed_tokens] = 0.0
                # elif last_token == self.answer_end:
                #     mask[batch_id, self.tokenizer.eos_token_id] = 0.0
                else:
                    if self.answer_start[-1] not in input_ids[batch_id]:
                        if last_token == self.answer_start[-2]:
                            mask[batch_id, self.answer_start[-1]] = 0.0
                        elif last_token == self.answer_start[-3]:
                            mask[batch_id, self.answer_start[-2]] = 0.0
                    else:
                        mask[batch_id] = torch.zeros_like(scores[batch_id])
                # t_free += time.time() - t00

            elif self.cur_stages[batch_id] == 'clue':
                # t01 = time.time()
                if last_token == self.sep_symbol:  # <|sep|>
                    clue_tokens = input_ids[batch_id][self.cur_start_position_idx[batch_id]:-1]
                    self.cur_start_position_idx[batch_id] = len(input_ids[batch_id])
                    # print('---\n', 'clue_text', clue_text, '\nclue_similarity', clue_similarity)
                    if clue_tokens not in self.cur_clues[batch_id]:
                        self.cur_clues[batch_id].append(clue_tokens)
                    # 计算下一个允许的tokens
                    allowed_tokens = self.clue_fmindex_processor([])
                    mask[batch_id, allowed_tokens] = 0.0
                elif last_token == self.clue_end:  # 若是 <|/clue|>，则直接进入evidence状态
                    # 计算这个clue的相似度, 保存
                    clue_tokens = input_ids[batch_id][self.cur_start_position_idx[batch_id]:-1]
                    # print('---\n', 'clue_text', clue_text, '\nclue_similarity', clue_similarity)
                    if clue_tokens not in self.cur_clues[batch_id]:
                        self.cur_clues[batch_id].append(clue_tokens)
                    # _t = time.time()
                    self.document_scoring_fn(batch_id)
                    # t_doc_score += time.time() - _t
                    # _t = time.time()
                    self.compute_window_allowed_tokens(batch_id)
                    # t_get_window += time.time() - _t
                    self.cur_clues[batch_id] = []
                    self.all_retrieved_docids[self.start_idx + batch_id].extend(list(self.retrieved_docids[batch_id]))
                    self.cur_stages[batch_id] = 'evidence'
                    mask[batch_id, self.evidence_start] = 0.0
                else:  # clue生成过程中
                    clue_prefix = input_ids[batch_id][self.cur_start_position_idx[batch_id]:]
                    allowed_tokens = self.clue_fmindex_processor(clue_prefix)
                    if self.sep_symbol in allowed_tokens and len(self.cur_clues[batch_id]) >= 2:
                        allowed_tokens.remove(self.sep_symbol)
                    mask[batch_id, allowed_tokens] = 0.0
                # t_clue += time.time() - t01

            elif self.cur_stages[batch_id] == 'evidence':
                # t02 = time.time()
                if last_token == self.evidence_start:
                    self.cur_start_position_idx[batch_id] = len(input_ids[batch_id])
                    allowed_tokens = self.evidence_fmindex_processor(batch_id, [])
                    mask[batch_id, allowed_tokens] = 0.0
                elif last_token == self.sep_symbol:
                    self.cur_window_id[batch_id] += 1
                    self.cur_start_position_idx[batch_id] = len(input_ids[batch_id])
                    allowed_tokens = self.evidence_fmindex_processor(batch_id, [])
                    mask[batch_id, allowed_tokens] = 0.0
                elif last_token == self.evidence_end:
                    self.cur_stages[batch_id] = 'free'
                    mask[batch_id, [self.answer_start[0]]] = 0.0  # For single-hop QA
                else:
                    evidence_prefix = input_ids[batch_id][self.cur_start_position_idx[batch_id]:]
                    if len(evidence_prefix) > self.max_evidence_length:
                        allowed_tokens = [self.sep_symbol, self.evidence_end]
                    else:
                        allowed_tokens = self.evidence_fmindex_processor(batch_id, evidence_prefix)
                    if self.sep_symbol in allowed_tokens and \
                        (not self.has_unused_windows(batch_id) or self.max_target_len - len(input_ids[batch_id]) - self.max_input_len < self.window_size * 2.5):  # 若所有记录的windows都用过了，则不允许生成sep_symbol
                        allowed_tokens.remove(self.sep_symbol)
                        # print('所有记录的windows都用过了，不允许生成sep_symbol.')
                    if self.max_target_len - len(input_ids[batch_id]) - self.max_input_len < 15:
                        allowed_tokens = [self.evidence_end]
                    # if self.evidence_end in allowed_tokens and self.cur_window_id[batch_id] <= 1:
                    #     allowed_tokens.remove(self.evidence_end)
                    mask[batch_id, allowed_tokens] = 0.0
                # t_evidence += time.time() - t02

        # total_time = time.time() - t_start

        # print('---')
        # print('t_free', t_free)
        # print('t_clue', t_clue, 't_doc_score', t_doc_score, 't_get_window', t_get_window)
        # print('t_evidence', t_evidence)
        # print('total_time', total_time)

        return scores + mask


class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_input_len, max_target_len):
        self.tokenizer = tokenizer
        self.data = []
        
        with open(data_path, 'r') as file:
            for line in tqdm(file):
                item = json.loads(line)
                if use_subset_num and len(self.data) >= use_subset_num:
                    break
                if flash_rag:
                    item['input'] = item['question']
                    item['answer'] = item['golden_answers']
                    item['output'] = ''
                    item['wiki_ids'] = ''
                if item['answer'] == []:
                    item['answer'] = ['']
                self.tokenizer.padding_side = 'left'
                if apply_chat_template:
                    input_text = [{"role": "user", "content": f"Question: {item['input']}"}]
                    input_text = self.tokenizer.apply_chat_template(input_text, tokenize=False, add_generation_prompt=True)
                    if self.tokenizer.bos_token is not None:
                        input_text = input_text.replace(self.tokenizer.bos_token, '')
                else:
                    input_text = f"Question: {item['input']}\nYour Response:"

                encoded_input = self.tokenizer(input_text, max_length=max_input_len, padding='max_length', truncation=True)
                encoded_target = self.tokenizer(item['output'], max_length=max_target_len, padding='max_length', truncation=True)
                self.data.append({
                    'input_ids': torch.tensor(encoded_input['input_ids']),
                    'attention_mask': torch.tensor(encoded_input['attention_mask']),
                    'labels': torch.tensor(encoded_target['input_ids']),
                    'id': item['id'],
                    'label_docids': str(item['wiki_ids']).strip("['']"),
                    'answer': str(item['answer']).strip("['']"),
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Prepare DataLoader
print('Loading dataset...')
eval_dataset = CustomDataset(eval_data_path, tokenizer, max_input_len, max_target_len)
eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=16)
# Load runs file
with open(runs_path, 'r', encoding='utf-8') as f:
    runs = [json.loads(line) for line in f]

avg_num_docids = 0
queryid_to_docids = {}
for run in runs:
    query_id = next(iter(run))
    top_passages = sorted(run[query_id].items(), key=lambda x: x[1], reverse=True)
    top_docids = []
    for pid, score in top_passages:
        docid = pid.split('-')[0]
        if docid not in top_docids:
            top_docids.append(docid)
    queryid_to_docids[query_id] = top_docids
    avg_num_docids += len(top_docids)
    
print('Total queries', len(queryid_to_docids))
print('avg_num_retrieved_docids', avg_num_docids / len(queryid_to_docids))
# Load Corpus FMIndex and FMIndexManager
if enable_constrain:
    t0 = time.time()
    print(f"Loading Corpus FMIndex from {corpus_fm_index_path}...")
    corpus_fm_index = fm_index.load_FMIndex(corpus_fm_index_path)
    print(f'Time Cost: {time.time() - t0}')
    t0 = time.time()

    print(f"Loading FMIndexManager from {docid_fm_index_map_path}...")
    docid_fm_index_map = fm_index.FMIndexManager()
    docid_fm_index_map.loadAll(docid_fm_index_map_path)
    print(f'Time Cost: {time.time() - t0}')

# Define Logits processor
logits_processor = None
if enable_constrain:
    constrained_decoding_processor = HierarchicalConstrainedLogitsProcessor(
        title_fm_index=None,
        corpus_fm_index=corpus_fm_index,
        docid_fm_index_map=docid_fm_index_map,
        max_input_len=0,  # Will be updated per batch
        max_target_len=max_target_len,  # Will be updated per batch
        batch_size=eval_batch_size,
        sep_symbol=" <|sep|>",  # Ensure proper spacing
        clue_start=" <|clue|>",
        clue_end=" <|/clue|>",
        evidence_start=" <|evidence|>",
        evidence_end=" <|/evidence|>",
        answer_start=" The answer is",
        tokenizer=tokenizer,
        retriever_tokenizer=retriever_tokenizer,
        retriever_model=retriever_model,
        window_size=32,
        max_window_num=512,
        # separate config for single-hop and multi-hop QA
        max_window_num_per_doc=12 if dataset_name in ['hotpotqa', '2wiki'] else 32,  
        top_k_wd=10 if dataset_name in ['hotpotqa', '2wiki'] else 5,
        top_k=50 if dataset_name in ['hotpotqa', '2wiki'] else 25,
    )
    logits_processor = LogitsProcessorList()
    logits_processor.append(constrained_decoding_processor)

# Generate
print('Start generating...')
start_time = time.time()
inputs, predictions, labels, ids, label_docids, answer_lists = [], [], [], [], [], []
for batch in tqdm(eval_dataloader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    label = batch['labels']
    id = batch['id']

    tt0 = time.time()

    if enable_constrain:
        # Set initial configurations
        batch_size = len(id)
        constrained_decoding_processor.max_input_len = len(input_ids[0])  # 更新max_input_len
        constrained_decoding_processor.reset_state()  # 重置状态
        constrained_decoding_processor.start_idx = len(constrained_decoding_processor.all_retrieved_docids)  # 更新start_idx
        constrained_decoding_processor.all_retrieved_docids.extend([[] for _ in range(batch_size)])  # 更新all_retrieved_docids
        constrained_decoding_processor.splade_retrieved_docids = [queryid_to_docids[queryid] for queryid in id]  # 更新splade_retrieved_docids
        # Get query text
        queries = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        queries = [text.split('Question: ')[-1].replace('\nYour response:', '') for text in queries]
        constrained_decoding_processor.query_texts = queries  # 更新query_texts
        # Get query representation with SPLADE
        with torch.no_grad():
            query_reps = splade_model(q_kwargs=splade_tokenizer(queries, padding=True, return_tensors="pt").to(device))["q_rep"].squeeze()  # (sparse) doc rep in voc space, shape (30522,)
            expanded_clue_words = []
            for query_rep in query_reps:
                col = torch.nonzero(query_rep).squeeze().cpu().tolist()
                # print("number of actual clue words", len(col))
                weights = query_rep[col].cpu().tolist()
                d = {k: v for k, v in zip(col, weights)}
                sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
                max_score = list(sorted_d.values())[0]
                query_bow_rep = []
                for k, v in sorted_d.items():
                    if '##' in reverse_voc[k]:
                        continue
                    query_bow_rep.append((reverse_voc[k], v))  # [('word', score), ...]
                expanded_clue_words.append(query_bow_rep[:8])
        constrained_decoding_processor.expanded_clue_words = expanded_clue_words  # 更新expanded_clue_words
        # print('expanded_clue_words', expanded_clue_words)

    print('splade clue expansion', time.time() - tt0)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_target_len,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=logits_processor,
    )

    inputs.extend(input_ids.tolist())
    predictions.extend(outputs.tolist())
    labels.extend(label)
    ids.extend(id)
    for docids, answers in zip(batch['label_docids'], batch['answer']):
        label_docids.append(docids)
        answer_lists.append(answers)

end_time = time.time()
query_latency = (end_time - start_time) / len(inputs) * 1000
print(f'Query latency: {query_latency:.0f} ms')

if enable_constrain:
    all_retrieved_docids = constrained_decoding_processor.all_retrieved_docids
else:
    all_retrieved_docids = [''] * len(inputs)

from collections import Counter
import re
import string

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.strip().split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def evaluate_predictions(output, answer_list, pred_docids, label_docids, label_output):
    final_metric = {"ans_in_context": 0, "clue_acc": 0, "em": 0, "f1": 0, "recall": 0}
    if 'the answer is' in output.lower():
        pred_answer = output.lower().split('the answer is')[1].strip('.')
        pred_answer = normalize_answer(pred_answer)
        for answer in answer_list:
            normalized_ground_truth = normalize_answer(answer)
            em = int(pred_answer == normalized_ground_truth)

            prediction_tokens = pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["em", "f1"]:
                final_metric[k] = max(eval(k), final_metric[k])

    for answer in answer_list:
        if answer.lower() in output.lower():
            final_metric["ans_in_context"] = 1
            
    if len(label_docids) > 0:
        for label in label_docids:
            if label.replace('\'','').replace('"','').replace(' ','') in pred_docids:
                final_metric["recall"] += 1
        final_metric["recall"] /= len(label_docids)

    return final_metric

# avg_mrr_score = 0
avg_topic_acc, avg_AIC_score, avg_em, avg_f1, avg_recall = 0, 0, 0, 0, 0
tokens_count = 0
# Save Results
t = time.localtime()
output_file = os.path.join(output_dir.split('checkpoint')[0], f'evidence.{dataset_name}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json')
with open(output_file, 'w') as f:
    for _input, _pred, _label, _id, _answer_list, _pred_docids, _label_docids in zip(inputs, predictions, labels, ids, answer_lists, all_retrieved_docids, label_docids):
        _input = tokenizer.decode(_input, skip_special_tokens=False).replace(tokenizer.pad_token, '').replace(tokenizer.eos_token, '')
        _label = tokenizer.decode(_label, skip_special_tokens=False).replace(tokenizer.pad_token, '').replace(tokenizer.eos_token, '')
        tokens_count += len(tokenizer.encode(tokenizer.decode(_pred, skip_special_tokens=True)))
        _pred_ = tokenizer.decode(_pred, skip_special_tokens=False).replace(tokenizer.pad_token, '').replace(tokenizer.eos_token, '')[len(_input):]
        _answer_list = set(_answer_list.split("', '"))
        _label_docids = set(_label_docids.split("', '"))
        final_metric = evaluate_predictions(_pred_, _answer_list, _pred_docids, _label_docids, _label)
        avg_recall += final_metric['recall']
        avg_AIC_score += final_metric['ans_in_context']
        avg_em += final_metric['em']
        avg_f1 += final_metric['f1']
        json_data = {
            'id': str(_id),
            'recall': final_metric['recall'],
            'ans_in_context': final_metric['ans_in_context'],
            'ans_em': final_metric['em'],
            'ans_f1': final_metric['f1'],
            'input': _input,
            'output': _pred_,
            'target': _label,
            'pred_docids': _pred_docids,
            'label_docids': list(_label_docids),
            'answer': list(_answer_list),
        }
        f.write(json.dumps(json_data) + "\n")
    f.write(json.dumps({
        'avg_recall':avg_recall / len(inputs),
        'avg_ans_in_context':avg_AIC_score / len(inputs), 
        'avg_ans_em':avg_em / len(inputs),
        'avg_ans_f1':avg_f1 / len(inputs),
        }) + "\n")
    f.write(json.dumps({
        'query_latency': query_latency,
        'tokens_count': tokens_count / len(inputs),
        }) + "\n")


print('tokens_count:', tokens_count / len(inputs))
print('avg_Ans_in_Context:', avg_AIC_score / len(inputs))
print('avg_EM:', avg_em / len(inputs))
print('avg_F1:', avg_f1 / len(inputs))
