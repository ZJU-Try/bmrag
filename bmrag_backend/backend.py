from typing import List, Generator
import time
import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from rank_bm25 import BM25Okapi
import jieba
import os
from dotenv import load_dotenv
from openai import OpenAI

# ==================== 初始化配置（只执行一次） ====================

# 1. 加载环境变量
load_dotenv()

# 2. 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 3. 加载 Embedding 模型
embedding_model = SentenceTransformer('./models/text2vec-base-chinese')

# 4. 加载 Cross-Encoder 重排序模型
cross_encoder = CrossEncoder('./models/mmarco-mMiniLmv2-L12-H384-v1')

# 5. 初始化 ChromaDB
chromadb_client = chromadb.PersistentClient(path="./chroma_db")
chromadb_collection = chromadb_client.get_or_create_collection(name='default')

# 6. 初始化 BM25 索引（从 ChromaDB 加载所有文档构建）
_all_docs = chromadb_collection.get(include=['documents', 'metadatas'])
_bm25_corpus = [list(jieba.cut(doc)) for doc in _all_docs['documents']]
_bm25_doc_ids = [m.get('doc_id', -1) for m in _all_docs['metadatas']]
_bm25 = BM25Okapi(_bm25_corpus) if _bm25_corpus else None


# ==================== 查询改写 ====================

def rewrite_query(query: str) -> str:
    """
    用 LLM 将用户问题改写为适合检索的关键词形式
    提升：否定句、短查询、口语化问题的检索效果
    
    示例:
        "哪些事项不得确定为国家秘密？" → "不得确定为国家秘密 工作秘密 商业秘密 个人隐私"
        "什么是密品？" → "密品 定义 设备产品 国家秘密信息"
    """
    prompt = f"""将以下问题改写为适合向量检索的关键词形式。
要求：
1. 提取核心概念和关键词
2. 补充同义词和相关术语
3. 保留否定词（如"不得""禁止"）
4. 用空格连接，不要输出其他内容

问题：{query}

关键词："""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=100
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else query
    except Exception:
        return query  # 改写失败时回退到原始查询


# ==================== 检索功能 ====================

def embed_chunk(chunk: str) -> List[float]:
    """将文本块转换为向量"""
    embedding = embedding_model.encode(chunk)
    return embedding.tolist()


def dense_retrieve(query: str, top_k: int = 5) -> List[dict]:
    """
    稠密检索（向量检索）：语义匹配
    
    Returns:
        List[dict]，每个 dict 包含:
        - text: 文档文本
        - doc_id: 条目编号
        - distance: 向量距离（越小越相似）
        - dense_rank: 在向量检索中的排名
    """
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )

    chunks = []
    for rank, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        chunks.append({
            'text': doc,
            'doc_id': meta.get('doc_id', -1) if meta else -1,
            'distance': dist,
            'dense_rank': rank
        })

    return chunks


def bm25_search(query: str, top_k: int = 5) -> List[dict]:
    """
    稀疏检索（BM25）：关键词精确匹配
    
    Returns:
        List[dict]，每个 dict 包含:
        - text: 文档文本
        - doc_id: 条目编号
        - bm25_score: BM25 分数（越高越相关）
        - sparse_rank: 在 BM25 检索中的排名
    """
    if _bm25 is None:
        return []

    tokenized_query = list(jieba.cut(query))
    scores = _bm25.get_scores(tokenized_query)

    # 取 Top-K
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    chunks = []
    for rank, idx in enumerate(ranked_indices):
        chunks.append({
            'text': _all_docs['documents'][idx],
            'doc_id': _bm25_doc_ids[idx],
            'bm25_score': float(scores[idx]),
            'sparse_rank': rank
        })

    return chunks


def hybrid_retrieve(query: str, top_k: int = 5, rrf_k: int = 60) -> List[dict]:
    """
    混合检索：向量检索 + BM25，用 RRF（Reciprocal Rank Fusion）融合
    
    RRF 公式: score(d) = Σ 1/(k + rank_i(d))
    
    Returns:
        List[dict]，每个 dict 包含:
        - text: 文档文本
        - doc_id: 条目编号
        - rrf_score: 融合分数（越高越相关）
    """
    # 双路检索
    dense_results = dense_retrieve(query, top_k=top_k)
    sparse_results = bm25_search(query, top_k=top_k)

    # RRF 融合
    rrf_scores = {}

    for chunk in dense_results:
        doc_id = chunk['doc_id']
        rrf_scores[doc_id] = rrf_scores.get(doc_id, {'chunk': chunk, 'score': 0})
        rrf_scores[doc_id]['score'] += 1.0 / (rrf_k + chunk['dense_rank'])

    for chunk in sparse_results:
        doc_id = chunk['doc_id']
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {'chunk': chunk, 'score': 0}
        rrf_scores[doc_id]['score'] += 1.0 / (rrf_k + chunk['sparse_rank'])

    # 按融合分数排序
    merged = sorted(rrf_scores.values(), key=lambda x: x['score'], reverse=True)

    chunks = []
    for item in merged[:top_k]:
        chunk = item['chunk']
        chunk['rrf_score'] = item['score']
        chunks.append(chunk)

    return chunks


def retrieve(query: str, top_k: int = 5) -> List[dict]:
    """
    检索入口（默认使用混合检索）
    
    Returns:
        List[dict]，每个 dict 包含:
        - text: 文档文本
        - doc_id: 条目编号
        - distance: 向量距离
    """
    return hybrid_retrieve(query, top_k=top_k)


def rerank(query: str, retrieved_chunks: List[dict], top_k: int = 3) -> List[dict]:
    """
    对检索结果进行重排序

    Returns:
        List[dict]，每个 dict 包含:
        - text: 文档文本
        - doc_id: 条目编号
        - distance: 原始向量距离
        - rerank_score: 重排序得分（越高越相关）
    """
    pairs = [(query, chunk['text']) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    for chunk, score in zip(retrieved_chunks, scores):
        chunk['rerank_score'] = float(score)

    # 按重排序分数降序排列
    reranked = sorted(retrieved_chunks, key=lambda x: x['rerank_score'], reverse=True)
    return reranked[:top_k]


def generate(query: str, chunks: List[dict]) -> str:
    """使用 DeepSeek 生成回答"""
    chunks_text = '\n\n'.join(
        f"[来源: 条目{c['doc_id']}]\n{c['text']}" for c in chunks
    )
    prompt = f'''请根据用户的问题和下列片段生成准确的回答。
用户问题：{query}
相关片段：
{chunks_text}

请基于上述内容，不要编造信息。引用时标注来源条目编号。'''

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一位保密知识助手。"},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )

    return response.choices[0].message.content


def generate_stream(query: str, chunks: List[dict]) -> Generator[str, None, None]:
    """使用 DeepSeek 流式生成回答（生成器，逐块产出文本）"""
    chunks_text = '\n\n'.join(
        f"[来源: 条目{c['doc_id']}]\n{c['text']}" for c in chunks
    )
    prompt = f'''请根据用户的问题和下列片段生成准确的回答。
用户问题：{query}
相关片段：
{chunks_text}

请基于上述内容，不要编造信息。引用时标注来源条目编号。'''

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一位保密知识助手。"},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ==================== 主要接口：RAG 问答函数 ====================

def ask(query: str, top_k: int = 5, rerank_top_k: int = 3) -> str:
    """
    RAG 问答主函数

    Args:
        query: 用户问题
        top_k: 初始检索返回的文档数量（默认 5）
        rerank_top_k: 重排序后返回的文档数量（默认 3）

    Returns:
        模型生成的回答
    """
    # 1. 检索相关文档
    retrieved_chunks = retrieve(query, top_k)

    # 2. 重排序
    reranked_chunks = rerank(query, retrieved_chunks, rerank_top_k)

    # 3. 生成回答
    answer = generate(query, reranked_chunks)

    return answer


def ask_stream(query: str, top_k: int = 5, rerank_top_k: int = 3) -> Generator[str, None, None]:
    """
    RAG 流式问答主函数（生成器）

    Args:
        query: 用户问题
        top_k: 初始检索返回的文档数量（默认 5）
        rerank_top_k: 重排序后返回的文档数量（默认 3）

    Yields:
        生成的回答文本块
    """
    # 1. 检索相关文档
    retrieved_chunks = retrieve(query, top_k)

    # 2. 重排序
    reranked_chunks = rerank(query, retrieved_chunks, rerank_top_k)

    # 3. 流式生成回答
    yield from generate_stream(query, reranked_chunks)


def ask_with_details(query: str, top_k: int = 5, rerank_top_k: int = 3) -> dict:
    """
    RAG 问答主函数（带检索详情，用于评估和前端展示）

    Returns:
        dict 包含:
        - answer: 生成的回答
        - retrieved_chunks: 检索结果列表（含 doc_id, text, distance, rerank_score）
        - timing: 各阶段耗时 {retrieve_ms, rerank_ms, generate_ms, total_ms}
    """
    timing = {}

    # 1. 检索
    t0 = time.time()
    retrieved_chunks = retrieve(query, top_k)
    timing['retrieve_ms'] = round((time.time() - t0) * 1000, 1)

    # 2. 重排序
    t0 = time.time()
    reranked_chunks = rerank(query, retrieved_chunks, rerank_top_k)
    timing['rerank_ms'] = round((time.time() - t0) * 1000, 1)

    # 3. 生成
    t0 = time.time()
    answer = generate(query, reranked_chunks)
    timing['generate_ms'] = round((time.time() - t0) * 1000, 1)
    timing['total_ms'] = round(
        timing['retrieve_ms'] + timing['rerank_ms'] + timing['generate_ms'], 1
    )

    return {
        'query': query,
        'answer': answer,
        'retrieved_chunks': reranked_chunks,
        'timing': timing
    }


# ==================== 使用示例 ====================

if __name__ == "__main__":
    questions = [
        "国家秘密的密级分为哪几级？",
        "个人隐私可以确定为国家秘密吗？",
        "非密品是否需要作出秘密标识？"
    ]

    for q in questions:
        print(f"\n{'='*50}")
        print(f"问题: {q}")
        print(f"{'='*50}")
        result = ask_with_details(q)
        print(f"回答: {result['answer']}\n")
        print(f"检索详情:")
        for chunk in result['retrieved_chunks']:
            print(f"  [条目 {chunk['doc_id']}] "
                  f"distance={chunk['distance']:.4f} "
                  f"rerank={chunk['rerank_score']:.4f}")
        print(f"耗时: {result['timing']}")
