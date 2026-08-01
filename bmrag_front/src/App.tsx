import { useState, useCallback } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { useConversations } from './hooks/useConversations';
import { useStreamingChat } from './hooks/useStreamingChat';

export default function App() {
  const {
    conversations,
    currentId,
    current,
    createConversation,
    switchConversation,
    deleteConversation,
    addMessage,
  } = useConversations();

  const { isStreaming, send } = useStreamingChat();
  const [streamingContent, setStreamingContent] = useState('');

  const handleSend = useCallback((query: string) => {
    // 确保有对话
    let convId = currentId;
    if (!convId) {
      convId = createConversation();
    }

    // 添加用户消息
    addMessage(convId, { role: 'user', content: query });
    setStreamingContent('');

    // 发起流式请求，流式内容单独渲染，结束后才存入对话
    send(query, {
      onChunk: (text) => setStreamingContent(text),
      onDone: (text) => {
        addMessage(convId!, { role: 'assistant', content: text });
        setStreamingContent('');
      },
      onError: (err) => {
        addMessage(convId!, { role: 'assistant', content: '⚠️ ' + err });
        setStreamingContent('');
      },
    });
  }, [currentId, createConversation, addMessage, send]);

  const handleSuggestionClick = useCallback((query: string) => {
    if (!isStreaming) handleSend(query);
  }, [isStreaming, handleSend]);

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        currentId={currentId}
        isStreaming={isStreaming}
        onNewChat={() => !isStreaming && createConversation()}
        onSwitch={switchConversation}
        onDelete={deleteConversation}
      />
      <ChatArea
        conversation={current}
        streamingContent={streamingContent}
        isStreaming={isStreaming}
        onSend={handleSend}
        onSuggestionClick={handleSuggestionClick}
      />
    </div>
  );
}
