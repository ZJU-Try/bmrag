import json
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from backend import ask, ask_stream

app = Flask(__name__)
# 允许跨域，支持前端直连后端（不经过 nginx 反代的场景）
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    """健康检查接口"""
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST", "GET"])
def ask_query():
    """
    问答接口（非流式）
    GET:  /ask?query=国家秘密分几级
    POST: {"query": "国家秘密分几级"}
    可选参数: top_k, rerank_top_k
    """
    if request.method == "GET":
        query = request.args.get("query", "")
        top_k = int(request.args.get("top_k", 5))
        rerank_top_k = int(request.args.get("rerank_top_k", 3))
    else:
        data = request.get_json(silent=True) or {}
        query = data.get("query", "")
        top_k = int(data.get("top_k", 5))
        rerank_top_k = int(data.get("rerank_top_k", 3))

    if not query:
        return jsonify({"error": "参数 query 不能为空"}), 400

    try:
        answer = ask(query, top_k=top_k, rerank_top_k=rerank_top_k)
        return jsonify({
            "query": query,
            "answer": answer
        })
    except Exception as e:
        return jsonify({
            "query": query,
            "error": str(e)
        }), 500


@app.route("/ask/stream", methods=["POST"])
def ask_stream_query():
    """
    流式问答接口（SSE）
    POST: {"query": "国家秘密分几级"}
    可选参数: top_k, rerank_top_k

    返回 text/event-stream，每条消息格式:
        data: {"content": "..."}
    结束时发送:
        data: [DONE]
    """
    data = request.get_json(silent=True) or {}
    query = data.get("query", "")
    top_k = int(data.get("top_k", 5))
    rerank_top_k = int(data.get("rerank_top_k", 3))

    if not query:
        return jsonify({"error": "参数 query 不能为空"}), 400

    def generate():
        try:
            for chunk in ask_stream(query, top_k=top_k, rerank_top_k=rerank_top_k):
                yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲，确保流式实时输出
        }
    )


if __name__ == "__main__":
    # 监听 0.0.0.0 才能在容器外访问
    app.run(host="0.0.0.0", port=5000, debug=False)
