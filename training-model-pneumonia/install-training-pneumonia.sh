#!/usr/bin/env bash
# ============================================================
# 🫁 install-training-pneumonia.sh
# ------------------------------------------------------------
# Cria/ativa .venv, instala dependências e executa o treino da CNN.
# Pré-requisitos: Python 3.10+; dataset em data/ (ver README).
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=============================================="
echo " 🫁 Setup e Treinamento — CNN Pneumonia"
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

# 4) Checagem mínima de dados
if [ ! -d "data" ]; then
  echo "❌ Pasta 'data' não encontrada."
  echo "   Estrutura esperada: data/train|val|test ou data/raw/<Classe>/"
  exit 1
fi
echo "✅ Pasta 'data' encontrada."

# 5) Executar o treino
echo "🚀 Executando: python train_cnn.py"
python train_cnn.py

echo "----------------------------------------------"
echo "✅ Concluído! Artefatos esperados em 'outputs/':"
echo "   - models/: model.h5, best_feature_extractor.keras, best_finetuned.keras"
echo "   - reports/: classification_report.csv, summary.json"
echo "   - plots/: confusion_matrix.png"
echo "----------------------------------------------"
