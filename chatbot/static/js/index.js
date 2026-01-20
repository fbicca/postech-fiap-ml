// static/js/index.js

const chat = document.querySelector('#chat');
const input = document.querySelector('#input');
const btnEnviar = document.querySelector('#botao-enviar');

let btnUpload = document.getElementById('btnUpload');
let iconUpload = document.getElementById('iconUpload');
let fileInput = document.getElementById('fileInput');

if (!fileInput) {
  fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.id = 'fileInput';
  fileInput.accept = 'image/*';
  fileInput.hidden = true;
  document.body.appendChild(fileInput);
}

disableUpload();
window.type_conversation = window.type_conversation || null;

// ---------- Envio de mensagens ----------
async function enviarMensagem() {
  const mensagem = (input.value || '').trim();
  if (!mensagem) return;

  appendBubble('user', mensagem);
  input.value = '';

  const placeholder = appendBubble('bot', 'Analisando ...');

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        msg: mensagem,
        type_conversation: window.type_conversation || null
      })
    });

    const data = await resp.json();
    if (data && data.type_conversation) {
      window.type_conversation = data.type_conversation;
    }

    if (data && data.ui && data.ui.enable_upload) enableUpload();
    else disableUpload();

    const msg = data?.msg || '⚠️ Resposta vazia do servidor.';
    placeholder.innerHTML = escapeHtml(msg).replace(/\n/g, '<br>');
    scrollToEnd();
  } catch (err) {
    console.error(err);
    placeholder.textContent = '❌ Erro ao comunicar com o servidor.';
    scrollToEnd();
  }
}

// ---------- Clique no "+" ----------
function handleUploadClick() {
  if (isUploadDisabled()) return;
  if (fileInput) fileInput.click();
}

if (btnUpload) btnUpload.addEventListener('click', (e) => { e.preventDefault(); handleUploadClick(); });
if (iconUpload) iconUpload.addEventListener('click', (e) => { e.preventDefault(); handleUploadClick(); });

// ---------- Preview + Upload ----------
fileInput.addEventListener('change', async (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;

  const objectUrl = URL.createObjectURL(file);
  const bubble = document.createElement('div');
  bubble.className = 'chat__bolha chat__bolha--usuario';
  bubble.innerHTML = `
    <figure style="margin:0">
      <img id="preview-img" alt="${escapeHtml(file.name)}"
           src="${objectUrl}"
           style="max-width:240px; max-height:200px; border-radius:10px; border:1px solid #e0e0e0; object-fit:cover"/>
      <figcaption style="margin-top:6px;font-size:0.95rem;line-height:1.35">
        📸 Enviando: <strong>${escapeHtml(file.name)}</strong>
      </figcaption>
    </figure>
  `;
  chat.appendChild(bubble);
  scrollToEnd();

  const img = bubble.querySelector('#preview-img');
  if (img) img.onload = () => URL.revokeObjectURL(objectUrl);

  // -------- upload para /upload --------
  const formData = new FormData();
  formData.append('file', file);

  try {
    const uploadResp = await fetch('/upload', { method: 'POST', body: formData });
    const uploadData = await uploadResp.json();

    if (uploadData?.url) {
      bubble.querySelector('figcaption').innerHTML = `✅ Upload concluído: <strong>${escapeHtml(file.name)}</strong>`;

      // informar backend para continuar o fluxo (já guarda em db_memory)
      await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          msg: '',
          type_conversation: window.type_conversation,
          uploaded_filename: uploadData.filename,
          uploaded_url: uploadData.url
        })
      }).then(r => r.json()).then(data => {
        appendBubble('bot', escapeHtml(data.msg).replace(/\n/g, '<br>'));
        if (data.type_conversation) window.type_conversation = data.type_conversation;
      });
    } else {
      bubble.querySelector('figcaption').innerHTML = `❌ Falha ao enviar ${escapeHtml(file.name)}`;
    }
  } catch (err) {
    console.error('Erro no upload:', err);
    bubble.querySelector('figcaption').innerHTML = `❌ Erro ao enviar ${escapeHtml(file.name)}`;
  }
});

// ---------- Helpers ----------
function enableUpload() {
  if (btnUpload) {
    btnUpload.disabled = false;
    btnUpload.classList.remove('is-disabled');
    btnUpload.style.pointerEvents = 'auto';
  }
  if (iconUpload) {
    iconUpload.classList.remove('is-disabled');
    iconUpload.style.pointerEvents = 'auto';
  }
}

function disableUpload() {
  if (btnUpload) {
    btnUpload.disabled = true;
    btnUpload.classList.add('is-disabled');
    btnUpload.style.pointerEvents = 'none';
  }
  if (iconUpload) {
    iconUpload.classList.add('is-disabled');
    iconUpload.style.pointerEvents = 'none';
  }
}

function isUploadDisabled() {
  if (btnUpload) return !!btnUpload.disabled;
  if (iconUpload) return iconUpload.classList.contains('is-disabled');
  return true;
}

function appendBubble(sender, text) {
  const p = document.createElement('p');
  p.className = sender === 'user' ? 'chat__bolha chat__bolha--usuario' : 'chat__bolha chat__bolha--bot';

  if (sender === 'user') {
    // Escapa HTML para segurança
    p.innerHTML = escapeHtml(text).replace(/\n/g, '<br>');
  } else {
    // Renderiza o HTML vindo do backend (mantém <br>, **, etc.)
    p.innerHTML = text;
  }

  chat.appendChild(p);
  return p;
}


function scrollToEnd() {
  chat.scrollTop = chat.scrollHeight;
}

function escapeHtml(s) {
  return String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

// Envio por clique e Enter
btnEnviar.addEventListener('click', enviarMensagem);
input.addEventListener('keyup', (ev) => {
  if (ev.key === 'Enter') {
    ev.preventDefault();
    btnEnviar.click();
  }
});
