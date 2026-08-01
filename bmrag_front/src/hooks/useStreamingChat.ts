import { useState, useCallback, useRef } from 'react';

const API_BASE = '/api';

interface StreamCallbacks {
  onChunk: (fullText: string) => void;
  onDone: (fullText: string) => void;
  onError: (error: string) => void;
}

export function useStreamingChat() {
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const send = useCallback(async (
    query: string,
    callbacks: StreamCallbacks,
    options?: { top_k?: number; rerank_top_k?: number }
  ) => {
    setIsStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;

    let fullText = '';

    try {
      const response = await fetch(`${API_BASE}/ask/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          top_k: options?.top_k ?? 5,
          rerank_top_k: options?.rerank_top_k ?? 3,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`服务器错误: ${response.status}`);
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;

          const data = trimmed.slice(6);
          if (data === '[DONE]') continue;

          try {
            const parsed = JSON.parse(data);
            if (parsed.content) {
              fullText += parsed.content;
              callbacks.onChunk(fullText);
            }
            if (parsed.error) {
              callbacks.onError(parsed.error);
              return;
            }
          } catch {
            // 忽略解析错误
          }
        }
      }

      callbacks.onDone(fullText || '(无回复)');
    } catch (err) {
      if ((err as Error).name === 'AbortError') return;
      callbacks.onError('请求失败: ' + (err as Error).message);
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
    }
  }, []);

  const abort = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { isStreaming, send, abort };
}
