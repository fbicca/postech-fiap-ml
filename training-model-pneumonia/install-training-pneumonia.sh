#!/bin/bash
echo "=============================================="
echo " 🫁 Setup e Treinamento — CNN Pneumonia (v2)"
echo "=============================================="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
echo "✅ Ambiente configurado!"
echo "🚀 Para treinar e gerar evidências, execute:"
python training-model-pneumonia.py --data_dir data --img_size 224 --batch_size 32 --epochs 10 --threshold 0.35
