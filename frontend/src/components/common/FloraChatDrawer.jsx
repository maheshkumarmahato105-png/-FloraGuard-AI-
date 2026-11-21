import { useState } from 'react';
import { Bot, Send, Sparkles, X } from 'lucide-react';

const INITIAL_MESSAGES = [
  {
    id: 1,
    sender: 'bot',
    text: 'Hello! I am FloraAI, your agronomic assistant. Upload a plant leaf photo or test our preset samples to generate an official pathology diagnostic report!',
  },
];

export function FloraChatDrawer() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState(INITIAL_MESSAGES);
  const [input, setInput] = useState('');

  const handleSend = () => {
    const text = input.trim();
    if (!text) return;

    const userMsg = { id: Date.now(), sender: 'user', text };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');

    setTimeout(() => {
      let reply = 'I am FloraAI. You can upload a leaf photograph in the scanner above to run real AgriVision-CNN inference with Grad-CAM explainability!';
      const q = text.toLowerCase();
      if (q.includes('early blight') || q.includes('tomato')) {
        reply = 'For Tomato Early Blight: Prune infected lower foliage, avoid overhead watering, apply copper-based organic fungicide every 7 days, and ensure proper plant spacing.';
      } else if (q.includes('corn') || q.includes('rust')) {
        reply = 'For Corn Common Rust: Caused by Puccinia sorghi. Ensure early planting, use resistant cultivars, and apply triazole fungicides if lesions spread before silking.';
      } else if (q.includes('potato') || q.includes('late blight')) {
        reply = 'For Potato Late Blight: Phytophthora infestans spreads rapidly in cool, wet weather. Destroy cull piles, hill soil deeply, and apply preventive protective fungicides.';
      } else if (q.includes('healthy')) {
        reply = 'Healthy foliage shows uniform chlorophyll, intact vascular veining, and zero necrotic lesions. Maintain balanced N-P-K fertilization and drip irrigation.';
      } else if (q.includes('report') || q.includes('pdf')) {
        reply = 'Click the "View Official Sample Diagnostic Report" button on the dashboard to inspect or print an A4 laboratory pathology report with severity breakdown and treatment plans.';
      }

      setMessages((prev) => [...prev, { id: Date.now() + 1, sender: 'bot', text: reply }]);
    }, 600);
  };

  return (
    <>
      <button
        type="button"
        className="chat-toggle-btn"
        onClick={() => setOpen(!open)}
        aria-label={open ? 'Close FloraAI chat assistant' : 'Open FloraAI chat assistant'}
        title="FloraAI Agronomist Assistant"
      >
        <Bot size={28} />
      </button>

      <aside className={`chat-drawer${open ? ' open' : ''}`} aria-label="FloraAI Agronomist Chat">
        <div className="chat-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, color: '#fff', fontSize: '0.95rem' }}>
            <span style={{ display: 'inline-flex', padding: 5, borderRadius: 8, background: 'var(--accent-primary)', color: '#000' }}>
              <Bot size={16} />
            </span>
            <span>FloraAI Agronomist</span>
          </div>
          <button
            type="button"
            onClick={() => setOpen(false)}
            aria-label="Close chat"
            style={{ color: 'var(--text-muted)', cursor: 'pointer', display: 'inline-flex', padding: 4 }}
          >
            <X size={18} />
          </button>
        </div>

        <div className="chat-messages">
          {messages.map((m) => (
            <div key={m.id} className={`chat-msg ${m.sender}`}>
              {m.text}
            </div>
          ))}
        </div>

        <div className="chat-input-box">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Ask FloraAI about crop disease, care..."
            aria-label="Ask FloraAI a question"
          />
          <button type="button" className="btn-primary" style={{ padding: '8px 14px', borderRadius: 20 }} onClick={handleSend} aria-label="Send message">
            <Send size={14} />
          </button>
        </div>
      </aside>
    </>
  );
}

export default FloraChatDrawer;
