import hydra
from omegaconf import DictConfig
import os
import json

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader, TextAnswerCollectionDataLoader
from .datasets.datasets import CollectionDatasetPreLoad
from .evaluate import evaluate
from .models.models_utils import get_model
from .tasks.transformer_evaluator import SparseRetrieval
from .utils.utils import get_dataset_name, get_initialize_config


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
def retrieve_evaluate(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)

    #if HF: need to udate config.
    if "hf_training" in config:
       init_dict.model_type_or_dir=os.path.join(config.checkpoint_dir,"model")
       init_dict.model_type_or_dir_q=os.path.join(config.checkpoint_dir,"model/query") if init_dict.model_type_or_dir_q else None


    model = get_model(config, init_dict)

    batch_size = 1
    # NOTE: batch_size is set to 1, currently no batched implem for retrieval (TODO)
    for data_dir in set(exp_dict["data"]["Q_COLLECTION_PATH"]):
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                        max_length=model_training_config["max_length"], batch_size=batch_size,
                                        shuffle=False, num_workers=1)
        evaluator = SparseRetrieval(config=config, model=model, dataset_name=get_dataset_name(data_dir),
                                    compute_stats=True, dim_voc=model.output_dim)
        evaluator.retrieve(q_loader, top_k=exp_dict["config"]["top_k"], threshold=exp_dict["config"]["threshold"])
    evaluate(exp_dict)


"""@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
def retrieve_evaluate(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)

    # 如果是 HF 训练，需要更新配置
    if "hf_training" in config:
        init_dict.model_type_or_dir = os.path.join(config.checkpoint_dir, "model")
        init_dict.model_type_or_dir_q = os.path.join(config.checkpoint_dir, "model/query") if init_dict.model_type_or_dir_q else None

    model = get_model(config, init_dict)

    batch_size = 1
    # NOTE: batch_size 设置为 1，目前检索没有批处理实现（待办事项）
    all_retrievals = {}
    all_answers = {}
    for data_dir in set(exp_dict["data"]["Q_COLLECTION_PATH"]):
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = TextAnswerCollectionDataLoader(
            dataset=q_collection,
            tokenizer_type=model_training_config["tokenizer_type"],
            max_length=model_training_config["max_length"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=1
        )
        evaluator = SparseRetrieval(
            config=config,
            model=model,
            dataset_name=get_dataset_name(data_dir),
            compute_stats=True,
            dim_voc=model.output_dim
        )
        # 设置 return_d=True 以获取检索结果和答案
        retrieval_result = evaluator.retrieve(
            q_loader,
            top_k=exp_dict["config"]["top_k"],
            threshold=exp_dict["config"]["threshold"],
            return_d=True
        )
        all_retrievals.update(retrieval_result["retrieval"])
        all_answers.update(retrieval_result["answers"])

    # 评估模型（假设 evaluate 函数保持不变）
    evaluate(exp_dict)

    # 计算 has_answer 指标
    has_answer_metric = compute_has_answer_metric(all_retrievals, all_answers)
    print(f"has_answer metric: {has_answer_metric:.4f}")

    # 将 has_answer 指标保存到文件
    has_answer_path = os.path.join(config["out_dir"], "has_answer_metric.json")
    with open(has_answer_path, "w") as f:
        json.dump({"has_answer": has_answer_metric}, f, indent=2)
    print(f"has_answer metric has been saved to {has_answer_path}")
"""

def compute_has_answer_metric(retrieval_results, answers):
    """
    计算 has_answer 指标。

    Args:
        retrieval_results (dict): 检索结果，格式为 {query_id: {doc_id: score, ...}, ...}
        answers (dict): 查询的答案，格式为 {query_id: answer_text, ...}

    Returns:
        float: has_answer 指标，表示包含答案的查询比例
    """
    has_answer_count = 0
    total_queries = len(retrieval_results)
    for q_id, doc_dict in retrieval_results.items():
        answer = answers.get(q_id, "").lower().strip()
        if not answer:
            continue  # 如果没有答案，跳过
        # 检查答案是否出现在任何一个检索到的文档中
        answer_found = False
        for doc_id in doc_dict.keys():
            doc_text = get_document_text(doc_id).lower()
            if answer in doc_text:
                answer_found = True
                break
        if answer_found:
            has_answer_count += 1
    has_answer_metric = has_answer_count / total_queries if total_queries > 0 else 0.0
    return has_answer_metric

def get_document_text(doc_id):
    # 实现获取文档文本的逻辑
    # 例如，文档存储在某个目录下，每个文档为一个文本文件，文件名为 {doc_id}.txt
    documents_dir = "/path/to/documents/"  # 替换为实际路径
    doc_path = os.path.join(documents_dir, f"{doc_id}.txt")
    if not os.path.exists(doc_path):
        return ""  # 或者处理文件不存在的情况
    with open(doc_path, "r", encoding="utf-8") as f:
        return f.read()
    

if __name__ == "__main__":
    retrieve_evaluate()
