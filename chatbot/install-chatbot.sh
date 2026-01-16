#!/usr/bin/env bash
# ============================================================
# 🤖 BotHealth Chatbot — Installer & Runner (Flask)
# ------------------------------------------------------------
# Um clique: cria venv, instala dependências e roda o servidor.
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "==========================================="
echo " 🤖 Instalação do BotHealth (Flask)        "
echo "==========================================="

# 1) Verifica Python
if ! command -v python3 &> /dev/null; then
  echo "❌ Python3 não encontrado. Instale antes de prosseguir."
  exit 1
fi
echo "✅ Python: $(python3 -V)"

# 2) Cria venv (se não existir)
if [ ! -d ".venv" ]; then
  echo "📦 Criando ambiente virtual .venv ..."
  python3 -m venv .venv
else
  echo "🔁 Reutilizando .venv existente"
fi

# 3) Ativa venv
source .venv/bin/activate

# 4) Atualiza pip e instaladores
python -m pip install --upgrade pip wheel setuptools

# 5) Instala dependências
pip install -r requirements.txt

# 6) Gera .env (se não existir)
if [ ! -f ".env" ]; then
  echo "🧩 Criando .env padrão ..."
  cat > .env <<'ENV'
# ================= .env (BotHealth) =================
FLASK_SECRET_KEY="$(python - <<'PY'
import secrets; print(secrets.token_hex(16))
PY
)"
# URL da API de predição de RISCO CARDÍACO (FastAPI)
API_PREDICT_HEART="http://127.0.0.1:8001/predict"
# URL da API de predição de PNEUMONIA (FastAPI)
API_PREDICT_PNEUMONIA="http://127.0.0.1:8002/predict"
# Porta do Flask (dev server)
PORT="5000"
# ====================================================
ENV
fi

# 7) Executa o servidor (Flask dev server)
clear
echo "✅ Instalação concluída!"
echo "🚀 Iniciando BotHealth (Flask) na porta ${PORT:-5000} ..."
echo "👉 Acesse: http://127.0.0.1:${PORT:-5000}/"
echo "👉 Endpoint do chat: POST /chat"
export FLASK_APP=app.py
flask run --host 0.0.0.0 --port ${PORT:-5000}
