/* ── DocMind Frontend ── */

const API = '';  // same origin; change to 'http://localhost:8000' for dev

/* ── State ───────────────────────────────────────────────────────────────── */
let sessionId = localStorage.getItem('dm_session') || generateId();
let chatHistory = [];
let isStreaming = false;

localStorage.setItem('dm_session', sessionId);
document.getElementById('sessionBadge').textContent = sessionId.slice(0, 8);

/* ── Init ─────────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  setupDragDrop();
  document.getElementById('fileInput').addEventListener('change', handleFileSelect);
});

/* ── Markdown renderer (no deps) ─────────────────────────────────────────── */
function renderMarkdown(text) {
  return text
    // code blocks
    .replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
      `<pre><code class="lang-${lang}">${escHtml(code.trim())}</code></pre>`)
    // inline code
    .replace(/`([^`]+)`/g, (_, c) => `<code>${escHtml(c)}</code>`)
    // headings
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm,  '<h2>$1</h2>')
    .replace(/^# (.+)$/gm,   '<h1>$1</h1>')
    // bold / italic
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,     '<em>$1</em>')
    // blockquotes
    .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
    // hr
    .replace(/^---$/gm, '<hr>')
    // unordered lists
    .replace(/^[-•] (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`)
    // ordered lists
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
    // links
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
    // line breaks → paragraphs
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

/* ── Chat ─────────────────────────────────────────────────────────────────── */
function sendMessage() {
  const input = document.getElementById('msgInput');
  const text = input.value.trim();
  if (!text || isStreaming) return;
  input.value = '';
  autoResize(input);
  doSend(text);
}

function sendQuick(text) {
  doSend(text);
}

async function doSend(text) {
  if (isStreaming) return;

  // Hide welcome card on first message
  const wc = document.getElementById('welcomeCard');
  if (wc) wc.remove();

  appendUserMessage(text);
  chatHistory.push({ role: 'user', content: text });

  setStreaming(true);
  showThinking('Searching & reasoning…');

  const aiRow = appendAiMessage('');
  const bubble = aiRow.querySelector('.msg-bubble');
  let accum = '';

  try {
    const resp = await fetch(`${API}/api/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: sessionId, history: chatHistory.slice(-10) })
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    hideThinking();

    // Add streaming cursor
    bubble.innerHTML = '<span class="cursor"></span>';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const data = JSON.parse(line.slice(6));
          if (data.delta) {
            accum += data.delta;
            bubble.innerHTML = renderMarkdown(accum) + '<span class="cursor"></span>';
            scrollToBottom();
          }
          if (data.done) {
            bubble.innerHTML = renderMarkdown(data.full || accum);
            chatHistory.push({ role: 'assistant', content: data.full || accum });
            scrollToBottom();
          }
          if (data.error) {
            bubble.innerHTML = `<span style="color:var(--danger)">⚠ ${escHtml(data.error)}</span>`;
          }
        } catch {}
      }
    }

    // Fallback if streaming didn't emit done
    if (!bubble.querySelector('.cursor') === false) {
      bubble.innerHTML = renderMarkdown(accum);
    }

  } catch (err) {
    hideThinking();
    bubble.innerHTML = `<span style="color:var(--danger)">⚠ ${escHtml(err.message)}</span>`;
    toast('Connection error. Is the server running?', 'error');
  } finally {
    setStreaming(false);
  }
}

function appendUserMessage(text) {
  const chatArea = document.getElementById('chatArea');
  const row = document.createElement('div');
  row.className = 'msg-row user';
  row.innerHTML = `
    <div class="msg-avatar">U</div>
    <div class="msg-bubble">${escHtml(text)}</div>
  `;
  chatArea.appendChild(row);
  scrollToBottom();
  return row;
}

function appendAiMessage(content) {
  const chatArea = document.getElementById('chatArea');
  const row = document.createElement('div');
  row.className = 'msg-row ai';
  row.innerHTML = `
    <div class="msg-avatar">⬡</div>
    <div class="msg-bubble">${content}</div>
  `;
  chatArea.appendChild(row);
  scrollToBottom();
  return row;
}

function scrollToBottom() {
  const ca = document.getElementById('chatArea');
  ca.scrollTop = ca.scrollHeight;
}

function clearChat() {
  const ca = document.getElementById('chatArea');
  ca.innerHTML = '';
  chatHistory = [];
  sessionId = generateId();
  localStorage.setItem('dm_session', sessionId);
  document.getElementById('sessionBadge').textContent = sessionId.slice(0, 8);

  // Re-add welcome card
  const wc = document.createElement('div');
  wc.className = 'welcome-card';
  wc.id = 'welcomeCard';
  wc.innerHTML = `
    <span class="welcome-glyph">⬡</span>
    <h2>Hello, I'm DocMind</h2>
    <p>Upload your documents and ask me anything. I'll search, reason, and cite my sources.</p>
    <div class="quick-actions">
      <button class="quick-btn" onclick="sendQuick('What documents are currently uploaded?')">What's uploaded?</button>
      <button class="quick-btn" onclick="sendQuick('Search the web for latest AI news')">Latest AI news</button>
      <button class="quick-btn" onclick="sendQuick('Calculate compound interest: principal 50000, rate 8%, years 10')">Compound interest</button>
    </div>
  `;
  ca.appendChild(wc);
}

/* ── File upload ──────────────────────────────────────────────────────────── */
function handleFileSelect(e) {
  uploadFiles(Array.from(e.target.files));
  e.target.value = '';
}

function setupDragDrop() {
  const zone = document.getElementById('uploadZone');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    uploadFiles(Array.from(e.dataTransfer.files));
  });
}

async function uploadFiles(files) {
  if (!files.length) return;

  const prog = document.getElementById('uploadProgress');
  prog.innerHTML = '';
  prog.classList.remove('hidden');

  for (const file of files) {
    const item = document.createElement('div');
    item.className = 'upload-item';
    item.innerHTML = `<div class="spinner"></div><span>${escHtml(file.name)}</span>`;
    prog.appendChild(item);

    try {
      const fd = new FormData();
      fd.append('files', file);
      const resp = await fetch(`${API}/api/upload`, { method: 'POST', body: fd });
      const data = await resp.json();
      const result = data.results?.[0];
      if (result?.status === 'ok') {
        item.className = 'upload-item done';
        item.querySelector('span').textContent = `${file.name} — ${result.chunks} chunks`;
        toast(`✓ ${file.name} indexed`, 'success');
      } else {
        item.className = 'upload-item err';
        item.querySelector('span').textContent = `${file.name} — ${result?.error || 'failed'}`;
        toast(`✗ ${file.name} failed`, 'error');
      }
    } catch (err) {
      item.className = 'upload-item err';
      item.querySelector('span').textContent = `${file.name} — ${err.message}`;
    }
  }

  await loadStats();
  setTimeout(() => { prog.classList.add('hidden'); prog.innerHTML = ''; }, 4000);
}

/* ── Stats & document list ────────────────────────────────────────────────── */
async function loadStats() {
  try {
    const data = await fetch(`${API}/api/stats`).then(r => r.json());
    document.getElementById('count-docs').textContent = data.documents ?? 0;
    document.getElementById('count-chunks').textContent = data.chunks ?? 0;
    renderDocList(data.sources || []);
  } catch {
    document.getElementById('count-docs').textContent = '?';
    document.getElementById('count-chunks').textContent = '?';
  }
}

function renderDocList(sources) {
  const list = document.getElementById('docList');
  if (!sources.length) {
    list.innerHTML = '<div class="doc-empty">No documents yet</div>';
    return;
  }
  list.innerHTML = sources.map(s => {
    const name = s.name || s;
    const ext = name.split('.').pop().toUpperCase();
    const icon = ext === 'PDF' ? '📄' : ext === 'DOCX' ? '📝' : '📃';
    return `
      <div class="doc-item">
        <span class="doc-icon">${icon}</span>
        <span class="doc-name" title="${escHtml(name)}">${escHtml(name)}</span>
        <button class="doc-del" onclick="deleteDoc('${escHtml(name)}')" title="Delete">✕</button>
      </div>
    `;
  }).join('');
}

async function deleteDoc(name) {
  if (!confirm(`Delete "${name}" from the knowledge base?`)) return;
  try {
    await fetch(`${API}/api/document`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_name: name })
    });
    toast(`Deleted ${name}`, 'success');
    await loadStats();
  } catch (err) {
    toast(`Delete failed: ${err.message}`, 'error');
  }
}

/* ── UI helpers ───────────────────────────────────────────────────────────── */
function isMobile() {
  return window.innerWidth <= 700;
}

function toggleSidebar() {
  if (isMobile()) {
    const sb = document.getElementById('sidebar');
    sb.classList.contains('mobile-open') ? closeSidebar() : openSidebar();
  } else {
    document.getElementById('sidebar').classList.toggle('collapsed');
  }
}

function openSidebar() {
  document.getElementById('sidebar').classList.add('mobile-open');
  document.getElementById('sidebarBackdrop').classList.add('active');
  document.body.style.overflow = 'hidden';
}

function closeSidebar() {
  document.getElementById('sidebar').classList.remove('mobile-open');
  document.getElementById('sidebarBackdrop').classList.remove('active');
  document.body.style.overflow = '';
}

// Close on Escape
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeSidebar();
});

function setStreaming(val) {
  isStreaming = val;
  document.getElementById('sendBtn').disabled = val;
}

function showThinking(msg = 'Thinking…') {
  const bar = document.getElementById('thinkingBar');
  document.getElementById('thinkingText').textContent = msg;
  bar.classList.remove('hidden');
}

function hideThinking() {
  document.getElementById('thinkingBar').classList.add('hidden');
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

function generateId() {
  return Math.random().toString(36).slice(2, 10);
}

function toast(msg, type = 'success') {
  let container = document.querySelector('.toast-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
  }
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  container.appendChild(t);
  setTimeout(() => t.remove(), 3200);
}
