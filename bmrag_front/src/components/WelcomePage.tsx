interface WelcomePageProps {
  onSuggestionClick: (query: string) => void;
}

const SUGGESTIONS = [
  '国家秘密的密级分为哪几级？',
  '个人隐私可以确定为国家秘密吗？',
  '非密品是否需要作出秘密标识？',
];

export function WelcomePage({ onSuggestionClick }: WelcomePageProps) {
  return (
    <div className="welcome">
      <div className="welcome-icon">🔒</div>
      <h2>欢迎使用保密知识助手</h2>
      <p>我可以为你解答保密相关的知识问题，请直接提问。</p>
      <div className="suggestions">
        {SUGGESTIONS.map((q) => (
          <button
            key={q}
            className="suggestion-item"
            onClick={() => onSuggestionClick(q)}
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
