// ==================== 配置 ====================
const API_BASE = '/api';
const STORAGE_KEY = 'bmrag_conversations';

// ==================== 状态管理 ====================
let conversations = loadConversations();
let currentId = null;
let isStreaming = false;

// ==================== DOM 元素 ====================
const messagesEl = document.getElementById('messages');
const welcomeEl = document.getElementById('welcome');
const inputEl = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');
const conversationListEl = document.getElementById('conversationList');
const currentTitleEl = document.getElementById('currentTitle');

// ==================== 持久化 ====================
function loadConversations() {
    try {
        const data = localStorage.getItem(STORAGE_KEY);
        return data ? JSON.parse(data) : [];
    } catch (e) {
        return [];
    }
}

function saveConversations() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
}

// ==================== 对话管理 ====================
function createConversation() {
    const conv = {
        id: Date.now().toString(36) + Math.random().toString(36).slice(2, 8),
        title: '新对话',
        messages: [],
        createdAt: Date.now()
    };
    conversations.unshift(conv);
    currentId = conv.id;
    saveConversations();
    renderConversationList();
    renderMessages();
    inputEl.focus();
}

function switchConversation(id) {
    if (isStreaming) return;
    currentId = id;
    renderConversationList();
    renderMessages();
}

function deleteConversation(id, event) {
    event.stopPropagation();
    if (isStreaming) return;
    conversations = conversations.filter(c => c.id !== id);
    if (currentId === id) {
        currentId = conversations.length > 0 ? conversations[0].id : null;
    }
    saveConversations();
    renderConversationList();
    renderMessages();
}

function getCurrentConversation() {
    return conversations.find(c => c.id === currentId);
}

// ==================== 渲染 ====================
function renderConversationList() {
    conversationListEl.innerHTML = '';
    if (conversations.length === 0) {
        conversationListEl.innerHTML = '<div style="padding:16px;text-align:center;opacity:0.5;font-size:13px;">暂无历史对话</div>';
        return;
    }
    conversations.forEach(conv => {
        const item = document.createElement('div');
        item.className = 'conversation-item' + (conv.id === currentId ? ' active' : '');
        item.innerHTML = `
            <span class="conv-title">${escapeHtml(conv.title)}</span>
            <button class="delete-btn" title="删除">&times;</button>
        `;
        item.addEventListener('click', () => switchConversation(conv.id));
        item.querySelector('.delete-btn').addEventListener('click', (e) => deleteConversation(conv.id, e));
        conversationListEl.appendChild(item);
    });
}

function renderMessages() {
    const conv = getCurrentConversation();
    if (!conv || conv.messages.length === 0) {
        welcomeEl.style.display = 'block';
        currentTitleEl.textContent = conv ? conv.title : '新对话';
        // 清空除欢迎语外的内容
        const msgs = messagesEl.querySelectorAll('.message');
        msgs.forEach(m => m.remove());
        return;
    }

    welcomeEl.style.display = 'none';
    currentTitleEl.textContent = conv.title;

    // 清空后重新渲染
    const existingMsgs = messagesEl.querySelectorAll('.message');
    existingMsgs.forEach(m => m.remove());

    conv.messages.forEach(msg => {
        appendMessage(msg.role, msg.content);
    });
    scrollToBottom();
}

function appendMessage(role, content) {
    const el = document.createElement('div');
    el.className = `message ${role}`;

    const avatar = role === 'user' ? '我' : 'AI';
    const roleLabel = role === 'user' ? '用户' : '助手';

    el.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-role">${roleLabel}</div>
            <div class="message-text">${escapeHtml(content)}</div>
        </div>
    `;

    messagesEl.appendChild(el);
    scrollToBottom();
    return el.querySelector('.message-text');
}

function updateMessageText(textEl, content, isStreaming = false) {
    textEl.innerHTML = escapeHtml(content);
    if (isStreaming) {
        textEl.classList.add('typing-cursor');
    } else {
        textEl.classList.remove('typing-cursor');
    }
    scrollToBottom();
}

function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ==================== 发送消息（流式） ====================
async function sendMessage(query) {
    if (!query.trim() || isStreaming) return;

    // 确保有一个对话
    if (!currentId) {
        createConversation();
    }
    const conv = getCurrentConversation();

    // 隐藏欢迎语
    welcomeEl.style.display = 'none';

    // 添加用户消息
    conv.messages.push({ role: 'user', content: query });
    appendMessage('user', query);

    // 如果是第一条消息，用问题作为标题
    if (conv.messages.length === 1) {
        conv.title = query.length > 20 ? query.slice(0, 20) + '...' : query;
        currentTitleEl.textContent = conv.title;
        renderConversationList();
    }

    // 清空输入框
    inputEl.value = '';
    autoResize();
    saveConversations();

    // 创建助手消息占位
    isStreaming = true;
    sendBtn.disabled = true;
    const textEl = appendMessage('assistant', '');
    textEl.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';

    let fullText = '';

    try {
        const response = await fetch(`${API_BASE}/ask/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: 5, rerank_top_k: 3 })
        });

        if (!response.ok) {
            throw new Error(`服务器错误: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // 保留最后不完整的行

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed.startsWith('data: ')) continue;

                const data = trimmed.slice(6);
                if (data === '[DONE]') {
                    continue;
                }

                try {
                    const parsed = JSON.parse(data);
                    if (parsed.content) {
                        fullText += parsed.content;
                        updateMessageText(textEl, fullText, true);
                    }
                    if (parsed.error) {
                        fullText = '⚠️ ' + parsed.error;
                        textEl.classList.add('error');
                        updateMessageText(textEl, fullText, false);
                    }
                } catch (e) {
                    // 忽略解析错误
                }
            }
        }

        // 流式结束
        updateMessageText(textEl, fullText || '(无回复)', false);
        conv.messages.push({ role: 'assistant', content: fullText });
        saveConversations();
    } catch (err) {
        textEl.classList.add('error');
        updateMessageText(textEl, '⚠️ 请求失败: ' + err.message, false);
    } finally {
        isStreaming = false;
        sendBtn.disabled = false;
        inputEl.focus();
    }
}

// ==================== 输入框自适应 ====================
function autoResize() {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + 'px';
}

// ==================== 事件绑定 ====================
sendBtn.addEventListener('click', () => {
    sendMessage(inputEl.value);
});

inputEl.addEventListener('input', autoResize);

inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage(inputEl.value);
    }
});

newChatBtn.addEventListener('click', () => {
    if (isStreaming) return;
    createConversation();
});

// 推荐问题点击
document.querySelectorAll('.suggestion-item').forEach(btn => {
    btn.addEventListener('click', () => {
        const query = btn.dataset.query;
        if (!isStreaming) {
            sendMessage(query);
        }
    });
});

// ==================== 初始化 ====================
if (conversations.length > 0) {
    currentId = conversations[0].id;
    renderConversationList();
    renderMessages();
} else {
    renderConversationList();
}
inputEl.focus();
