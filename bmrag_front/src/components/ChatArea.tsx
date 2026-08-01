import { useEffect, useRef } from 'react';
import type { Conversation } from '../types';
import { MessageBubble } from './MessageBubble';
import { WelcomePage } from './WelcomePage';
import { InputArea } from './InputArea';

interface ChatAreaProps {
  conversation: Conversation | null;
  streamingContent: string;
  isStreaming: boolean;
  onSend: (query: string) => void;
  onSuggestionClick: (query: string) => void;
}

export function ChatArea({
  conversation,
  streamingContent,
  isStreaming,
  onSend,
  onSuggestionClick,
}: ChatAreaProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 有新消息时滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation?.messages.length, streamingContent]);

  const hasMessages = conversation && conversation.messages.length > 0;

  return (
    <main className="chat-area">
      <div className="chat-header">
        <span>{conversation?.title ?? '新对话'}</span>
      </div>

      <div className="messages">
        {!hasMessages && !isStreaming ? (
          <WelcomePage onSuggestionClick={onSuggestionClick} />
        ) : (
          <>
            {conversation!.messages.map((msg, i) => (
              <MessageBubble key={i} message={msg} />
            ))}
            {isStreaming && (
              <MessageBubble
                message={{ role: 'assistant', content: streamingContent }}
                isStreaming
              />
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <InputArea isStreaming={isStreaming} onSend={onSend} />
    </main>
  );
}
