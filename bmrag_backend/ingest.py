"""
文档摄入管线
将 bm.md 按条目分块，携带元数据存入 ChromaDB

用法:
    python ingest.py                      # 默认摄入 bm.md
    python ingest.py --file other.md      # 摄入指定文件
    python ingest.py --rebuild            # 清空并重建索引
"""

import re
import argparse
import chromadb
from sentence_transformers import SentenceTransformer


# ==================== 分块策略 ====================

def parse_bm_md(file_path: str) -> list[dict]:
    """
    解析 bm.md，按条目号分块

    每个块包含:
        - doc_id: 条目编号 (1-105)
        - text: 条目全文（含子项）
    """
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    lines = content.split('\n')
    chunks = []
    current_id = None
    current_text = []

    # 匹配以数字开头的条目（兼容 * 前缀和无前缀两种格式）
    item_pattern = re.compile(r'^\*?\s*(\d+)\.\s*(.*)')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = item_pattern.match(line)
        if match:
            # 保存上一个块
            if current_id is not None:
                chunks.append({
                    'doc_id': current_id,
                    'text': '\n'.join(current_text).strip()
                })
            current_id = int(match.group(1))
            current_text = [match.group(2)]
        elif current_id is not None:
            current_text.append(line)

    # 保存最后一个块
    if current_id is not None:
        chunks.append({
            'doc_id': current_id,
            'text': '\n'.join(current_text).strip()
        })

    return chunks


# ==================== 摄入逻辑 ====================

def ingest(file_path: str = './bm.md',
           collection_name: str = 'default',
           rebuild: bool = True):
    """
    摄入文档到 ChromaDB

    Args:
        file_path: 文档路径
        collection_name: ChromaDB 集合名
        rebuild: 是否清空重建（False 则增量添加）
    """
    # 1. 解析文档
    chunks = parse_bm_md(file_path)
    print(f"解析到 {len(chunks)} 个文档块")
    for c in chunks[:3]:
        print(f"  [条目 {c['doc_id']}] {c['text'][:50]}...")
    print(f"  ...")

    # 2. 加载 Embedding 模型
    print("加载 Embedding 模型...")
    embedding_model = SentenceTransformer('./models/text2vec-base-chinese')

    # 3. 初始化 ChromaDB
    chromadb_client = chromadb.PersistentClient(path="./chroma_db")

    if rebuild:
        try:
            chromadb_client.delete_collection(collection_name)
            print(f"已删除旧集合: {collection_name}")
        except Exception:
            pass

    collection = chromadb_client.get_or_create_collection(name=collection_name)

    # 4. 向量化并存储
    print("开始向量化并存储...")
    for chunk in chunks:
        chunk_id = f"doc_{chunk['doc_id']}"

        # 增量模式：跳过已存在的
        if not rebuild:
            existing = collection.get(ids=[chunk_id])
            if existing['ids']:
                continue

        embedding = embedding_model.encode(chunk['text']).tolist()
        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk['text']],
            metadatas=[{
                'doc_id': chunk['doc_id'],
                'source': 'bm.md'
            }]
        )

    # 5. 验证
    count = collection.count()
    print(f"\n摄入完成！集合 '{collection_name}' 共 {count} 条记录")

    # 抽样验证
    sample = collection.peek(3)
    print("\n抽样验证:")
    for i, (doc_id, doc_text) in enumerate(zip(sample['metadatas'], sample['documents'])):
        print(f"  [{doc_id}] {doc_text[:60]}...")

    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="文档摄入工具")
    parser.add_argument('--file', default='./bm.md', help='文档路径')
    parser.add_argument('--rebuild', action='store_true', default=True, help='清空重建索引')
    args = parser.parse_args()

    ingest(file_path=args.file, rebuild=args.rebuild)
