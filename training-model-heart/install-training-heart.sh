#!/usr/bin/env bash
# ============================================================
# 💖 install-training-heart.sh
# ------------------------------------------------------------
# Cria/ativa .venv, instala dependências e executa o treino.
# Requisitos: Python 3.10+ e o arquivo 'heart.csv' na raiz.
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=============================================="
echo " 💖 Setup e Treinamento — Heart Model"
echo "=============================================="

# 1) Verificar Python
if ! command -v python3 &> /dev/null; then
  echo "❌ Python3 não encontrado. Instale-o e tente novamente."
  exit 1
fi
echo "✅ Python: $(python3 -V)"

# 2) Criar .venv (se não existir) e ativar
if [ ! -d ".venv" ]; then
  echo "📦 Criando ambiente virtual (.venv) ..."
  python3 -m venv .venv
fi
source .venv/bin/activate

# 3) Atualizar instaladores e instalar dependências
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# 4) Conferir dataset
if [ ! -f "heart.csv" ]; then
  echo "❌ Arquivo 'heart.csv' não encontrado na raiz do projeto."
  echo "   Coloque o dataset nesta pasta e rode novamente."
  exit 1
fi
echo "✅ Dataset localizado: heart.csv"

# 5) Executar pipeline de treinamento/avaliação
clear
echo "=============================================="
echo " 💖 Setup e Treinamento — Heart Model"
echo "=============================================="
echo "🚀 Executando: python main.py"
python main.py

echo "----------------------------------------------"
echo "✅ Concluído! Artefatos esperados:"
echo "   - X_train.csv, X_test.csv, y_train.csv, y_test.csv"
echo "   - modelo_insuficiencia_cardiaca.pkl, scaler_dados.pkl"
echo "----------------------------------------------"
