from typing import List
import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
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


# ==================== 核心功能函数 ====================

def embed_chunk(chunk: str) -> List[float]:
    """将文本块转换为向量"""
    embedding = embedding_model.encode(chunk)
    return embedding.tolist()


def retrieve(query: str, top_k: int = 5) -> List[str]:
    """从向量数据库中检索相关文档块"""
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]


def rerank(query: str, retrieved_chunks: List[str], top_k: int = 3) -> List[str]:
    """对检索结果进行重排序"""
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    chunk_with_score_list = [(chunk, score) for chunk, score in zip(retrieved_chunks, scores)]
    chunk_with_score_list.sort(key=lambda pair: pair[1], reverse=True)

    return [chunk for chunk, _ in chunk_with_score_list][:top_k]


def generate(query: str, chunks: List[str]) -> str:
    """使用 DeepSeek 生成回答"""
    chunks_text = '\n\n'.join(chunks)
    prompt = f'''请根据用户的问题和下列片段生成准确的回答。
用户问题：{query}
相关片段：
{chunks_text}

请基于上述内容，不要编造信息。'''

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一位保密知识助手。"},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )

    return response.choices[0].message.content


def generate_stream(query: str, chunks: List[str]):
    """使用 DeepSeek 流式生成回答（生成器，逐块产出文本）"""
    chunks_text = '\n\n'.join(chunks)
    prompt = f'''请根据用户的问题和下列片段生成准确的回答。
用户问题：{query}
相关片段：
{chunks_text}

请基于上述内容，不要编造信息。'''

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


def ask_stream(query: str, top_k: int = 5, rerank_top_k: int = 3):
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


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 测试查询
    # query = "可以用手机拍摄保密信息吗？"
    # answer = ask(query)
    # print(answer)
    
    #连续问答示例
    questions = [
        "国家秘密的密级分为哪几级？",
        "个人隐私可以确定为国家秘密吗？",
        "非密品是否需要作出秘密标识？"
    ]
    
    for q in questions:
        print(f"\n{'='*50}")
        print(f"问题: {q}")
        print(f"{'='*50}")
        answer = ask(q)
        print(f"回答: {answer}\n")