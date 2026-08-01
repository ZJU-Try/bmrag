import { useEffect, useRef } from 'react';
import type { Message } from '../types';

interface MessageBubbleProps {
  message: Message;
  isStreaming?: boolean;
}

export function MessageBubble({ message, isStreaming }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const textRef = useRef<HTMLDivElement>(null);

  // 流式时自动滚动到底部
  useEffect(() => {
    if (isStreaming && textRef.current) {
      textRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [message.content, isStreaming]);

  // 等待回复时的加载动画
  const showLoading = !isUser && isStreaming && !message.content;

  return (
    <div className={`message ${message.role}`}>
      <div className="message-avatar">{isUser ? '我' : 'AI'}</div>
      <div className="message-content">
        <div className="message-role">{isUser ? '用户' : '助手'}</div>
        <div
          ref={textRef}
          className={`message-text${isStreaming ? ' typing-cursor' : ''}`}
        >
          {showLoading ? (
            <div className="loading-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          ) : (
            message.content
          )}
        </div>
      </div>
    </div>
  );
}
