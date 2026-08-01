import { useState, useCallback } from 'react';
import type { Conversation, Message } from '../types';

const STORAGE_KEY = 'bmrag_conversations';

function load(): Conversation[] {
  try {
    const data = localStorage.getItem(STORAGE_KEY);
    return data ? (JSON.parse(data) as Conversation[]) : [];
  } catch {
    return [];
  }
}

function save(convs: Conversation[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(convs));
}

function genId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>(load);
  const [currentId, setCurrentId] = useState<string | null>(
    conversations.length > 0 ? conversations[0].id : null
  );

  const persist = useCallback((convs: Conversation[]) => {
    setConversations(convs);
    save(convs);
  }, []);

  const createConversation = useCallback((): string => {
    const conv: Conversation = {
      id: genId(),
      title: '新对话',
      messages: [],
      createdAt: Date.now(),
    };
    persist([conv, ...conversations]);
    setCurrentId(conv.id);
    return conv.id;
  }, [conversations, persist]);

  const switchConversation = useCallback((id: string) => {
    setCurrentId(id);
  }, []);

  const deleteConversation = useCallback((id: string) => {
    const next = conversations.filter((c) => c.id !== id);
    persist(next);
    if (currentId === id) {
      setCurrentId(next.length > 0 ? next[0].id : null);
    }
  }, [conversations, currentId, persist]);

  const addMessage = useCallback((id: string, msg: Message) => {
    setConversations((prev) => {
      const next = prev.map((c) =>
        c.id === id
          ? {
              ...c,
              messages: [...c.messages, msg],
              title:
                c.messages.length === 0 && msg.role === 'user'
                  ? msg.content.length > 20
                    ? msg.content.slice(0, 20) + '...'
                    : msg.content
                  : c.title,
            }
          : c
      );
      save(next);
      return next;
    });
  }, []);

  const current = conversations.find((c) => c.id === currentId) ?? null;

  return {
    conversations,
    currentId,
    current,
    createConversation,
    switchConversation,
    deleteConversation,
    addMessage,
  };
}
