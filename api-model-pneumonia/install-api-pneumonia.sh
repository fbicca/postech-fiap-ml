#!/usr/bin/env bash
# ============================================================
# 🫁 Pneumonia Detection API — Installer Script
# ------------------------------------------------------------
# Este script prepara o ambiente, instala dependências e
# executa a API FastAPI (TensorFlow CPU) automaticamente.
# ============================================================

# 🚀 Parar execução em caso de erro
set -e

# 🎯 Diretório atual do projeto
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "==========================================="
echo " 🩺 Instalação da Pneumonia Detection API  "
echo "==========================================="

# 🧰 1️⃣ Verificar Python
echo "🔍 Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 não encontrado. Instale antes de prosseguir."
    exit 1
fi
PY_VER=$(python3 -V)
echo "✅ Python detectado: $PY_VER"

# 🧱 2️⃣ Criar ambiente virtual
if [ ! -d ".venv" ]; then
    echo "📦 Criando ambiente virtual (.venv)..."
    python3 -m venv .venv
else
    echo "🔁 Ambiente virtual já existe. Usando .venv existente."
fi

# 🪄 3️⃣ Ativar ambiente virtual
echo "⚙️  Ativando ambiente virtual..."
source .venv/bin/activate

# 🧩 4️⃣ Atualizar pip
echo "⬆️  Atualizando pip..."
python -m pip install --upgrade pip

# 💾 5️⃣ Instalar dependências essenciais
echo "📦 Instalando dependências (FastAPI, TensorFlow CPU, etc.)..."
pip install --upgrade wheel setuptools
pip install fastapi uvicorn pillow numpy tensorflow-cpu
pip install python-multipart
clear
# 🧠 6️⃣ Mensagem de sucesso
echo "✅ Instalação concluída!"

# 🩻 7️⃣ Executar API
echo "🚀 Iniciando PNEUMONIA FAILURE API  (porta 8002)..."
echo "----------------------------------------------------"
echo "Acesse: http://127.0.0.1:8002/docs                  "
echo "Para interromper: CTRL+C                            "
echo "----------------------------------------------------"

# 🧭 Executar servidor Uvicorn
uvicorn api-model-pneumonia:app --host 0.0.0.0 --port 8002
