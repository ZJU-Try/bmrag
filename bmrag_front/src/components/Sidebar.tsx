import type { Conversation } from '../types';

interface SidebarProps {
  conversations: Conversation[];
  currentId: string | null;
  isStreaming: boolean;
  onNewChat: () => void;
  onSwitch: (id: string) => void;
  onDelete: (id: string) => void;
}

export function Sidebar({
  conversations,
  currentId,
  isStreaming,
  onNewChat,
  onSwitch,
  onDelete,
}: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1>保密知识助手</h1>
        <button
          className="btn-new-chat"
          onClick={onNewChat}
          disabled={isStreaming}
          title="新建对话"
        >
          + 新对话
        </button>
      </div>

      <div className="conversation-list">
        {conversations.length === 0 ? (
          <div className="empty-hint">暂无历史对话</div>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              className={`conversation-item${conv.id === currentId ? ' active' : ''}`}
              onClick={() => !isStreaming && onSwitch(conv.id)}
            >
              <span className="conv-title">{conv.title}</span>
              <button
                className="delete-btn"
                title="删除"
                onClick={(e) => {
                  e.stopPropagation();
                  if (!isStreaming) onDelete(conv.id);
                }}
              >
                &times;
              </button>
            </div>
          ))
        )}
      </div>

      <div className="sidebar-footer">
        <span className="badge">RAG 问答系统</span>
      </div>
    </aside>
  );
}
