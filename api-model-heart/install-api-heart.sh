#!/usr/bin/env bash
# ============================================================
# ❤️ Heart Failure Predictor API — Installer
# ------------------------------------------------------------
# Cria .venv, instala dependências e inicia a API na porta 8001
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "==========================================="
echo " ❤️ Instalação da Heart Failure API        "
echo "==========================================="

if ! command -v python3 &> /dev/null; then
  echo "❌ Python3 não encontrado. Instale antes de prosseguir."
  exit 1
fi

echo "✅ Python: $(python3 -V)"

if [ ! -d ".venv" ]; then
  echo "📦 Criando ambiente virtual .venv ..."
  python3 -m venv .venv
else
  echo "🔁 Reutilizando .venv existente"
fi

echo "⚙️  Ativando .venv ..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "⬆️  Atualizando pip/wheel/setuptools ..."
python -m pip install --upgrade pip wheel setuptools

echo "📦 Instalando dependências do requirements.txt ..."
pip install -r requirements.txt || { echo "❌ Falha ao instalar dependências."; exit 1; }

clear
echo "✅ Instalação concluída!"
echo "🚀 Iniciando HEART FAILURE API (porta 8001)..."
echo "----------------------------------------------------"
echo "Acesse: http://127.0.0.1:8001/docs                  "
echo "Para interromper: CTRL+C                            "
echo "----------------------------------------------------"
uvicorn api-model-heart:app --host 0.0.0.0 --port 8001
