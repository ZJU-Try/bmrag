"""
RAG 评估脚本
计算检索指标（Recall@K, MRR, NDCG）和生成指标（Faithfulness, Answer Relevancy）

用法:
    python evaluate.py                          # 默认评估全部 30 题
    python evaluate.py --limit 5                # 只评估前 5 题（快速测试）
    python evaluate.py --top-k 5 --rerank-k 3   # 自定义检索参数
"""

import json
import time
import math
import argparse
import os
from pathlib import Path
from typing import List

# 导入后端 RAG 管线
from backend import retrieve, rerank, generate, ask_with_details


# ==================== 检索评估指标 ====================

def recall_at_k(retrieved_doc_ids: List[int], relevant_doc_ids: List[int], k: int = 5) -> float:
    """
    Recall@K: Top-K 检索结果中包含正确文档的比例

    Args:
        retrieved_doc_ids: 检索返回的 doc_id 列表
        relevant_doc_ids: 标注的正确 doc_id 列表
        k: Top-K
    Returns:
        0.0 ~ 1.0
    """
    if not relevant_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    hits = len(set(top_k) & set(relevant_doc_ids))
    return hits / len(set(relevant_doc_ids))


def mrr(retrieved_doc_ids: List[int], relevant_doc_ids: List[int]) -> float:
    """
    MRR (Mean Reciprocal Rank): 第一个正确文档的排名倒数

    Returns:
        0.0 ~ 1.0
    """
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in relevant_doc_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_doc_ids: List[int], relevant_doc_ids: List[int], k: int = 5) -> float:
    """
    NDCG@K: 归一化折损累积增益

    Returns:
        0.0 ~ 1.0
    """
    # DCG: 相关性按位置折损
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        rel = 1.0 if doc_id in relevant_doc_ids else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 因为 log2(1)=0

    # IDCG: 理想情况（所有相关文档排在最前）
    ideal_hits = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ==================== 生成评估指标（LLM-as-Judge） ====================

def llm_judge(prompt: str) -> str:
    """调用 DeepSeek 进行评估（支持 CoT 推理）"""
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个严格的评估助手。先简要分析，最后一行只输出一个数字。"},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return response.choices[0].message.content.strip()


def parse_score(response: str, max_score: int = 10) -> float:
    """从 LLM 响应中提取分数（最后一行的数字）"""
    lines = response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip().replace('：', ':')
        if ':' in line:
            line = line.split(':')[-1].strip()
        try:
            score = float(line)
            return max(0.0, min(1.0, score / max_score))
        except ValueError:
            continue
    return -1.0


def faithfulness(answer: str, chunks: List[dict]) -> float:
    """
    忠实度：回答是否基于检索内容，而非编造
    采用 CoT（先推理后打分）+ 0-10 分制，提升区分度

    Returns:
        0.0 ~ 1.0
    """
    chunks_text = '\n'.join(c['text'] for c in chunks)
    prompt = f"""请评估以下回答对检索片段的忠实度。

【检索片段】
{chunks_text}

【回答】
{answer}

【评估步骤】
1. 找出回答中的所有事实性声明
2. 逐个判断每个声明是否能从检索片段中找到支持
3. 统计：有支持的声明数 / 总声明数
4. 检查是否存在编造的信息

【评分标准】（0-10 分）
- 10分：所有声明都有据可查，无编造
- 7-9分：大部分声明有支持，少量无法验证
- 4-6分：部分声明有支持，存在编造
- 1-3分：大部分声明无支持，严重编造
- 0分：完全编造

请先简要分析（1-2句），最后一行输出分数（只输出数字）："""

    try:
        response = llm_judge(prompt)
        return parse_score(response, max_score=10)
    except Exception:
        return -1.0


def answer_relevancy(query: str, answer: str) -> float:
    """
    回答相关性：回答与问题的相关程度
    采用 CoT（先推理后打分）+ 0-10 分制

    Returns:
        0.0 ~ 1.0
    """
    prompt = f"""请评估以下回答与问题的相关性。

【问题】
{query}

【回答】
{answer}

【评估步骤】
1. 分析问题的核心诉求是什么
2. 判断回答是否直接针对该诉求
3. 评估回答是否提供了有用且切题的信息
4. 检查是否存在答非所问或跑题的内容

【评分标准】（0-10 分）
- 10分：完全切题，信息完整且直接回答问题
- 7-9分：基本切题，但信息略有缺失
- 4-6分：部分切题，存在跑题内容
- 1-3分：大部分跑题
- 0分：完全答非所问

请先简要分析（1-2句），最后一行输出分数（只输出数字）："""

    try:
        response = llm_judge(prompt)
        return parse_score(response, max_score=10)
    except Exception:
        return -1.0


# ==================== 主评估流程 ====================

def evaluate(dataset_path: str = None,
             top_k: int = 5,
             rerank_top_k: int = 3,
             limit: int = None,
             skip_generation: bool = False) -> dict:
    """
    运行完整评估

    Args:
        dataset_path: QA 数据集路径
        top_k: 检索数量
        rerank_top_k: 重排后数量
        limit: 只评估前 N 题（None = 全部）
        skip_generation: 跳过生成评估（只评检索，节省时间）
    """
    # 1. 加载测试集（自动查找路径）
    if dataset_path is None:
        for candidate in ['./tests/qa_dataset.json', '../tests/qa_dataset.json', '/app/tests/qa_dataset.json']:
            if os.path.exists(candidate):
                dataset_path = candidate
                break
        if dataset_path is None:
            raise FileNotFoundError("找不到 qa_dataset.json，请用 --dataset 指定路径")
    # 1. 加载测试集
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]

    print(f"加载测试集: {len(dataset)} 题")
    print(f"参数: top_k={top_k}, rerank_top_k={rerank_top_k}")
    print(f"生成评估: {'跳过' if skip_generation else '启用'}")
    print(f"{'='*60}\n")

    results = []

    for i, item in enumerate(dataset):
        query = item['question']
        relevant_ids = item['relevant_doc_ids']
        print(f"[{i+1}/{len(dataset)}] {query}")

        # 2. 检索评估
        t0 = time.time()
        retrieved = retrieve(query, top_k=top_k)
        retrieved_ids = [c['doc_id'] for c in retrieved]
        reranked = rerank(query, retrieved, top_k=rerank_top_k)
        reranked_ids = [c['doc_id'] for c in reranked]
        retrieve_time = round((time.time() - t0) * 1000, 1)

        # 检索指标（用初始检索结果计算）
        recall5 = recall_at_k(retrieved_ids, relevant_ids, k=5)
        mrr_score = mrr(retrieved_ids, relevant_ids)
        ndcg5 = ndcg_at_k(retrieved_ids, relevant_ids, k=5)

        # 重排后的指标
        recall_rerank = recall_at_k(reranked_ids, relevant_ids, k=rerank_top_k)

        result = {
            'id': item['id'],
            'question': query,
            'relevant_doc_ids': relevant_ids,
            'retrieved_doc_ids': retrieved_ids,
            'reranked_doc_ids': reranked_ids,
            'metrics': {
                'recall@5': round(recall5, 3),
                'mrr': round(mrr_score, 3),
                'ndcg@5': round(ndcg5, 3),
                f'recall@{rerank_top_k}(rerank)': round(recall_rerank, 3),
            },
            'timing': {
                'retrieve_rerank_ms': retrieve_time,
            }
        }

        # 3. 生成评估
        if not skip_generation:
            t0 = time.time()
            answer = generate(query, reranked)
            gen_time = round((time.time() - t0) * 1000, 1)

            faith = faithfulness(answer, reranked)
            relevancy = answer_relevancy(query, answer)

            result['answer'] = answer
            result['metrics']['faithfulness'] = round(faith, 3)
            result['metrics']['answer_relevancy'] = round(relevancy, 3)
            result['timing']['generate_ms'] = gen_time
            result['timing']['total_ms'] = retrieve_time + gen_time

            print(f"  检索: Recall@5={recall5:.2f} MRR={mrr_score:.2f} NDCG@5={ndcg5:.2f}")
            print(f"  生成: 忠实度={faith:.2f} 相关性={relevancy:.2f}")
            print(f"  耗时: 检索={retrieve_time}ms 生成={gen_time}ms\n")
        else:
            print(f"  检索: Recall@5={recall5:.2f} MRR={mrr_score:.2f} NDCG@5={ndcg5:.2f}")
            print(f"  耗时: 检索={retrieve_time}ms\n")

        results.append(result)

    # 4. 汇总统计
    summary = summarize(results)
    return {'results': results, 'summary': summary}


def summarize(results: list) -> dict:
    """汇总所有题目的指标"""
    if not results:
        return {}

    metrics_keys = results[0]['metrics'].keys()
    summary = {}

    for key in metrics_keys:
        values = [r['metrics'][key] for r in results if r['metrics'].get(key, -1) >= 0]
        if values:
            summary[f'{key}_avg'] = round(sum(values) / len(values), 3)
            summary[f'{key}_count'] = len(values)

    # 效率统计
    retrieve_times = [r['timing']['retrieve_rerank_ms'] for r in results]
    summary['retrieve_avg_ms'] = round(sum(retrieve_times) / len(retrieve_times), 1)

    gen_times = [r['timing']['generate_ms'] for r in results if 'generate_ms' in r['timing']]
    if gen_times:
        summary['generate_avg_ms'] = round(sum(gen_times) / len(gen_times), 1)
        summary['total_avg_ms'] = round(
            (sum(retrieve_times) + sum(gen_times)) / len(gen_times), 1
        )

    return summary


# ==================== 报告生成 ====================

def generate_report(eval_result: dict, output_path: str = '../docs/evaluation_report.md'):
    """生成 Markdown 评估报告"""
    summary = eval_result['summary']
    results = eval_result['results']

    report = f"""# RAG 评估报告

> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
> 测试集大小: {len(results)} 题

## 一、汇总指标

### 检索指标

| 指标 | 平均值 | 说明 |
|------|--------|------|
| Recall@5 | {summary.get('recall@5_avg', 'N/A')} | Top-5 结果中包含正确文档的比例 |
| MRR | {summary.get('mrr_avg', 'N/A')} | 第一个正确文档的排名倒数 |
| NDCG@5 | {summary.get('ndcg@5_avg', 'N/A')} | 考虑排名位置的归一化增益 |

### 生成指标

| 指标 | 平均值 | 说明 |
|------|--------|------|
| Faithfulness | {summary.get('faithfulness_avg', 'N/A')} | 回答对检索内容的忠实度 |
| Answer Relevancy | {summary.get('answer_relevancy_avg', 'N/A')} | 回答与问题的相关性 |

### 效率指标

| 指标 | 平均值 |
|------|--------|
| 检索+重排耗时 | {summary.get('retrieve_avg_ms', 'N/A')} ms |
| 生成耗时 | {summary.get('generate_avg_ms', 'N/A')} ms |
| 端到端耗时 | {summary.get('total_avg_ms', 'N/A')} ms |

## 二、详细结果

| # | 问题 | Recall@5 | MRR | NDCG@5 | 忠实度 | 相关性 | 总耗时(ms) |
|---|------|----------|-----|--------|--------|--------|-----------|
"""

    for r in results:
        m = r['metrics']
        t = r['timing']
        report += f"| {r['id']} | {r['question'][:20]}... | {m.get('recall@5', '-')} | {m.get('mrr', '-')} | {m.get('ndcg@5', '-')} | {m.get('faithfulness', '-')} | {m.get('answer_relevancy', '-')} | {t.get('total_ms', '-')} |\n"

    report += f"""
## 三、分析

### 检索效果
- Recall@5 = {summary.get('recall@5_avg', 'N/A')}：{'优秀' if summary.get('recall@5_avg', 0) >= 0.85 else '需优化' if summary.get('recall@5_avg', 0) < 0.7 else '良好'}
- MRR = {summary.get('mrr_avg', 'N/A')}：{'优秀' if summary.get('mrr_avg', 0) >= 0.65 else '需优化' if summary.get('mrr_avg', 0) < 0.5 else '良好'}

### 生成效果
- 忠实度 = {summary.get('faithfulness_avg', 'N/A')}：{'优秀' if summary.get('faithfulness_avg', 0) >= 0.8 else '需优化' if summary.get('faithfulness_avg', 0) < 0.6 else '良好'}
- 相关性 = {summary.get('answer_relevancy_avg', 'N/A')}：{'优秀' if summary.get('answer_relevancy_avg', 0) >= 0.8 else '需优化' if summary.get('answer_relevancy_avg', 0) < 0.6 else '良好'}
"""

    # 写入文件
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已生成: {output_path}")
    return report


# ==================== 入口 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 评估工具")
    parser.add_argument('--dataset', default='./tests/qa_dataset.json', help='测试集路径')
    parser.add_argument('--limit', type=int, default=None, help='只评估前 N 题')
    parser.add_argument('--top-k', type=int, default=5, help='初始检索数量')
    parser.add_argument('--rerank-k', type=int, default=3, help='重排后数量')
    parser.add_argument('--skip-generation', action='store_true', help='跳过生成评估（只评检索）')
    parser.add_argument('--output', default='./docs/evaluation_report.md', help='报告输出路径')
    args = parser.parse_args()

    result = evaluate(
        dataset_path=args.dataset,
        top_k=args.top_k,
        rerank_top_k=args.rerank_k,
        limit=args.limit,
        skip_generation=args.skip_generation,
    )

    generate_report(result, output_path=args.output)

    # 打印汇总
    print(f"\n{'='*60}")
    print("汇总指标:")
    for k, v in result['summary'].items():
        print(f"  {k}: {v}")
