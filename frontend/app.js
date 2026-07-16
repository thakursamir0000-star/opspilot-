/**
 * OpsPilot — Frontend Application
 *
 * Session management, file upload, chat with streaming support,
 * document list, error handling, and empty state management.
 */

(() => {
    'use strict';

    // ── Configuration ────────────────────────────────────────────────────
    const API_BASE = window.location.origin;
    const SESSION_KEY = 'opspilot_session_id';

    // ── State ────────────────────────────────────────────────────────────
    let sessionId = '';
    let documents = [];
    let isUploading = false;
    let isChatting = false;
    let hasReadyDocs = false;

    // ── DOM References ───────────────────────────────────────────────────
    const $uploadZone     = document.getElementById('upload-zone');
    const $fileInput      = document.getElementById('file-input');
    const $uploadProgress = document.getElementById('upload-progress');
    const $progressLabel  = document.getElementById('upload-progress-label');
    const $progressFill   = document.getElementById('upload-progress-fill');
    const $docList        = document.getElementById('doc-list');
    const $docEmpty       = document.getElementById('doc-empty');
    const $chatMessages   = document.getElementById('chat-messages');
    const $chatEmpty      = document.getElementById('chat-empty');
    const $chatInput      = document.getElementById('chat-input');
    const $sendBtn        = document.getElementById('send-btn');
    const $inputWrapper   = document.getElementById('input-wrapper');
    const $chatStatus     = document.getElementById('chat-status');
    const $errorContainer = document.getElementById('error-banner-container');
    const $mobileToggle   = document.getElementById('mobile-toggle');
    const $sidebar        = document.getElementById('sidebar');
    const $sidebarOverlay = document.getElementById('sidebar-overlay');


    // ═════════════════════════════════════════════════════════════════════
    // SESSION MANAGEMENT
    // ═════════════════════════════════════════════════════════════════════

    function initSession() {
        let id = localStorage.getItem(SESSION_KEY);
        if (!id) {
            id = crypto.randomUUID();
            localStorage.setItem(SESSION_KEY, id);
        }
        sessionId = id;
    }


    // ═════════════════════════════════════════════════════════════════════
    // ERROR HANDLING
    // ═════════════════════════════════════════════════════════════════════

    function showError(message, duration = 6000) {
        const banner = document.createElement('div');
        banner.className = 'error-banner';
        banner.innerHTML = `
            <span>${escapeHtml(message)}</span>
            <button class="error-banner-close" aria-label="Dismiss">&times;</button>
        `;
        banner.querySelector('.error-banner-close').addEventListener('click', () => banner.remove());
        $errorContainer.appendChild(banner);

        if (duration > 0) {
            setTimeout(() => { if (banner.parentNode) banner.remove(); }, duration);
        }
    }


    // ═════════════════════════════════════════════════════════════════════
    // DOCUMENT UPLOAD
    // ═════════════════════════════════════════════════════════════════════

    function setupUpload() {
        // File input change
        $fileInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            if (files.length > 0) uploadFiles(files);
            $fileInput.value = '';  // reset so same file can be re-uploaded
        });

        // Drag & drop
        $uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            $uploadZone.classList.add('drag-over');
        });

        $uploadZone.addEventListener('dragleave', () => {
            $uploadZone.classList.remove('drag-over');
        });

        $uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            $uploadZone.classList.remove('drag-over');
            const files = Array.from(e.dataTransfer.files).filter(f => f.name.toLowerCase().endsWith('.pdf'));
            if (files.length === 0) {
                showError('Only PDF files are accepted.');
                return;
            }
            uploadFiles(files);
        });
    }

    async function uploadFiles(files) {
        if (isUploading) return;
        isUploading = true;

        // Validate: only PDFs
        const pdfFiles = files.filter(f => f.name.toLowerCase().endsWith('.pdf'));
        const rejected = files.length - pdfFiles.length;
        if (rejected > 0) {
            showError(`${rejected} non-PDF file(s) skipped. Only PDFs are accepted.`);
        }
        if (pdfFiles.length === 0) {
            isUploading = false;
            return;
        }

        // Show progress
        $uploadProgress.style.display = 'block';
        $progressLabel.textContent = `Processing ${pdfFiles.length} file(s)...`;
        $progressFill.style.width = '10%';

        const formData = new FormData();
        formData.append('session_id', sessionId);
        pdfFiles.forEach(f => formData.append('files', f));

        try {
            $progressFill.style.width = '40%';

            const response = await fetch(`${API_BASE}/documents/upload`, {
                method: 'POST',
                body: formData,
            });

            $progressFill.style.width = '80%';

            if (!response.ok) {
                const err = await response.json().catch(() => ({ detail: 'Upload failed' }));
                throw new Error(err.detail || `Upload failed (${response.status})`);
            }

            const data = await response.json();

            $progressFill.style.width = '100%';

            // Check for warnings
            const noText = data.documents.filter(d => d.status === 'no_text_extracted');
            if (noText.length > 0) {
                const names = noText.map(d => d.filename).join(', ');
                showError(`Warning: No text extracted from ${names}. These may be scanned/image PDFs.`, 8000);
            }

            const errors = data.documents.filter(d => d.status === 'error');
            if (errors.length > 0) {
                const names = errors.map(d => d.filename).join(', ');
                showError(`Error processing: ${names}`, 8000);
            }

            // Refresh document list
            await refreshDocuments();

        } catch (err) {
            showError(err.message || 'Failed to upload documents. Please try again.');
        } finally {
            isUploading = false;
            setTimeout(() => {
                $uploadProgress.style.display = 'none';
                $progressFill.style.width = '0%';
            }, 500);
        }
    }

    async function refreshDocuments() {
        try {
            const response = await fetch(`${API_BASE}/documents?session_id=${encodeURIComponent(sessionId)}`);
            if (!response.ok) throw new Error('Failed to fetch documents');
            const data = await response.json();
            documents = data.documents;
            renderDocList();
            updateChatState();
        } catch (err) {
            console.error('Failed to refresh documents:', err);
        }
    }

    function renderDocList() {
        // Remove old items (keep the empty state element)
        $docList.querySelectorAll('.doc-item, .sidebar-section-label').forEach(el => el.remove());

        if (documents.length === 0) {
            $docEmpty.style.display = 'flex';
            return;
        }

        $docEmpty.style.display = 'none';

        // Section label
        const label = document.createElement('div');
        label.className = 'sidebar-section-label';
        label.textContent = `Documents (${documents.length})`;
        $docList.prepend(label);

        documents.forEach(doc => {
            const item = document.createElement('div');
            item.className = 'doc-item';

            const icon = document.createElement('div');
            icon.className = 'doc-icon';
            icon.textContent = '📄';

            const info = document.createElement('div');
            info.className = 'doc-info';

            const name = document.createElement('div');
            name.className = 'doc-name';
            name.textContent = doc.filename;
            name.title = doc.filename;

            const meta = document.createElement('div');
            meta.className = 'doc-meta';
            meta.textContent = `${doc.num_pages} pages · ${doc.num_chunks} chunks`;

            info.appendChild(name);
            info.appendChild(meta);

            // Warning for no_text_extracted
            if (doc.status === 'no_text_extracted') {
                const warning = document.createElement('div');
                warning.className = 'doc-warning';
                warning.textContent = '⚠ No text extracted';
                info.appendChild(warning);
            }

            const status = document.createElement('div');
            status.className = `doc-status ${doc.status}`;
            status.title = doc.status === 'ready' ? 'Ready'
                : doc.status === 'no_text_extracted' ? 'No text extracted (scanned PDF?)'
                : doc.status === 'error' ? 'Processing error'
                : doc.status;

            item.appendChild(icon);
            item.appendChild(info);
            item.appendChild(status);
            $docList.appendChild(item);
        });
    }


    // ═════════════════════════════════════════════════════════════════════
    // CHAT
    // ═════════════════════════════════════════════════════════════════════

    function updateChatState() {
        hasReadyDocs = documents.some(d => d.status === 'ready');

        if (hasReadyDocs) {
            $chatInput.disabled = false;
            $chatInput.placeholder = 'Ask a question about your documents...';
            $inputWrapper.classList.remove('disabled');
            $chatStatus.textContent = `${documents.filter(d => d.status === 'ready').length} document(s) loaded`;
        } else {
            $chatInput.disabled = true;
            $chatInput.placeholder = 'Upload a document first...';
            $inputWrapper.classList.add('disabled');
            $chatStatus.textContent = 'Upload documents to start asking questions';
        }

        updateSendButton();
    }

    function updateSendButton() {
        $sendBtn.disabled = !hasReadyDocs || isChatting || !$chatInput.value.trim();
    }

    function setupChat() {
        // Auto-resize textarea
        $chatInput.addEventListener('input', () => {
            $chatInput.style.height = 'auto';
            $chatInput.style.height = Math.min($chatInput.scrollHeight, 120) + 'px';
            updateSendButton();
        });

        // Enter to send (Shift+Enter for newline)
        $chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        $sendBtn.addEventListener('click', () => sendMessage());
    }

    async function sendMessage() {
        const message = $chatInput.value.trim();
        if (!message || isChatting || !hasReadyDocs) return;

        isChatting = true;
        updateSendButton();

        // Hide empty state
        $chatEmpty.style.display = 'none';

        // Add user message
        appendMessage('user', message);

        // Clear input
        $chatInput.value = '';
        $chatInput.style.height = 'auto';

        // Show loading
        const loadingEl = appendLoading();

        try {
            // Try streaming first, fall back to sync
            let result;
            try {
                result = await sendMessageStream(message, loadingEl);
            } catch (streamErr) {
                // Streaming not available or failed — fall back to sync
                result = await sendMessageSync(message);
                // Remove loading indicator
                if (loadingEl.parentNode) loadingEl.remove();
                // Render the full response
                appendMessage('assistant', result.answer, result.citations);
            }
        } catch (err) {
            if (loadingEl.parentNode) loadingEl.remove();
            showError(err.message || 'Failed to get a response. Please try again.');
        } finally {
            isChatting = false;
            updateSendButton();
            $chatInput.focus();
        }
    }

    async function sendMessageSync(message) {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, message }),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: 'Chat request failed' }));
            throw new Error(err.detail || `Request failed (${response.status})`);
        }

        return await response.json();
    }

    async function sendMessageStream(message, loadingEl) {
        const response = await fetch(`${API_BASE}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, message }),
        });

        if (!response.ok) {
            throw new Error('Streaming not available');
        }

        // Remove loading indicator and create assistant message bubble
        if (loadingEl.parentNode) loadingEl.remove();

        const msgEl = createMessageElement('assistant', '');
        $chatMessages.appendChild(msgEl);
        const contentEl = msgEl.querySelector('.message-content');
        scrollToBottom();

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        let citations = [];

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const data = line.slice(6).trim();
                if (data === '[DONE]') continue;

                try {
                    const parsed = JSON.parse(data);
                    if (parsed.type === 'token') {
                        fullText += parsed.content;
                        contentEl.innerHTML = formatMarkdown(fullText);
                        scrollToBottom();
                    } else if (parsed.type === 'citations') {
                        citations = parsed.citations;
                    }
                } catch (e) {
                    // skip unparseable lines
                }
            }
        }

        // Append citations if any
        if (citations.length > 0) {
            const citationsEl = createCitationsElement(citations);
            msgEl.querySelector('.message-body').appendChild(citationsEl);
        }

        scrollToBottom();
        return { answer: fullText, citations };
    }


    // ═════════════════════════════════════════════════════════════════════
    // MESSAGE RENDERING
    // ═════════════════════════════════════════════════════════════════════

    function createMessageElement(role, content) {
        const msg = document.createElement('div');
        msg.className = `message ${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? 'U' : 'O';

        const body = document.createElement('div');
        body.className = 'message-body';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = formatMarkdown(content);

        body.appendChild(contentDiv);
        msg.appendChild(avatar);
        msg.appendChild(body);

        return msg;
    }

    function appendMessage(role, content, citations) {
        const msg = createMessageElement(role, content);
        $chatMessages.appendChild(msg);

        if (citations && citations.length > 0) {
            const citEl = createCitationsElement(citations);
            msg.querySelector('.message-body').appendChild(citEl);
        }

        scrollToBottom();
    }

    function appendLoading() {
        const msg = document.createElement('div');
        msg.className = 'message assistant';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = 'O';

        const body = document.createElement('div');
        body.className = 'message-body';

        const loading = document.createElement('div');
        loading.className = 'message-loading';
        loading.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';

        body.appendChild(loading);
        msg.appendChild(avatar);
        msg.appendChild(body);
        $chatMessages.appendChild(msg);

        scrollToBottom();
        return msg;
    }

    function createCitationsElement(citations) {
        const wrapper = document.createElement('div');
        wrapper.className = 'citations';

        const toggle = document.createElement('button');
        toggle.className = 'citations-toggle';
        toggle.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="9 18 15 12 9 6"/>
            </svg>
            ${citations.length} source${citations.length > 1 ? 's' : ''}
        `;

        const list = document.createElement('div');
        list.className = 'citations-list hidden';

        citations.forEach(cit => {
            const item = document.createElement('div');
            item.className = 'citation-item';

            const source = document.createElement('div');
            source.className = 'citation-source';
            source.textContent = `📄 ${cit.filename}, Page ${cit.page}`;

            const snippet = document.createElement('div');
            snippet.className = 'citation-snippet';
            snippet.textContent = cit.snippet;

            item.appendChild(source);
            item.appendChild(snippet);
            list.appendChild(item);
        });

        toggle.addEventListener('click', () => {
            list.classList.toggle('hidden');
            toggle.classList.toggle('expanded');
        });

        wrapper.appendChild(toggle);
        wrapper.appendChild(list);
        return wrapper;
    }

    function scrollToBottom() {
        requestAnimationFrame(() => {
            $chatMessages.scrollTop = $chatMessages.scrollHeight;
        });
    }


    // ═════════════════════════════════════════════════════════════════════
    // TEXT FORMATTING
    // ═════════════════════════════════════════════════════════════════════

    function formatMarkdown(text) {
        if (!text) return '';

        // Basic markdown-like formatting
        let html = escapeHtml(text);

        // Bold: **text**
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Italic: *text*
        html = html.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');

        // Inline code: `text`
        html = html.replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.06);padding:1px 4px;border-radius:3px;font-size:0.85em;">$1</code>');

        // Line breaks and paragraphs
        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');

        // Bullet lists: lines starting with "- " or "• "
        html = html.replace(/(?:^|<br>)[-•]\s+(.+?)(?=<br>|<\/p>|$)/g, '<li>$1</li>');
        html = html.replace(/(<li>.*?<\/li>(?:\s*<li>.*?<\/li>)*)/gs, '<ul>$1</ul>');

        // Numbered lists: lines starting with "1. ", "2. ", etc.
        html = html.replace(/(?:^|<br>)\d+\.\s+(.+?)(?=<br>|<\/p>|$)/g, '<li>$1</li>');

        return `<p>${html}</p>`;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }


    // ═════════════════════════════════════════════════════════════════════
    // MOBILE
    // ═════════════════════════════════════════════════════════════════════

    function setupMobile() {
        $mobileToggle.addEventListener('click', () => {
            $sidebar.classList.toggle('open');
            $sidebarOverlay.classList.toggle('active');
        });

        $sidebarOverlay.addEventListener('click', () => {
            $sidebar.classList.remove('open');
            $sidebarOverlay.classList.remove('active');
        });
    }


    // ═════════════════════════════════════════════════════════════════════
    // INIT
    // ═════════════════════════════════════════════════════════════════════

    function init() {
        initSession();
        setupUpload();
        setupChat();
        setupMobile();

        // Load existing documents for this session (in case of page refresh)
        refreshDocuments();
    }

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
