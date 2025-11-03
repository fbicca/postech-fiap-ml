# Postech FIAP ML

> Projeto de Machine Learning da FIAP / Pós-Tech desenvolvido por Edmilson Teixeira e Filipe Bicca

## Visão Geral

Este repositório abriga vários modelos de Machine Learning desenvolvidos como parte de um curso do programa de pós-graduação, envolvendo tarefas como previsão de doenças cardíacas e pneumonia através de um chatbot.

Ele está organizado em módulos distintos:

- `training‐model‐heart` — treinamento de modelo para doença cardíaca
- `training‐model‐pneumonia` — treinamento de modelo para pneumonia
- `api‐model‐heart` — API para expor o modelo de doença cardíaca
- `api‐model‐pneumonia` — API para expor o modelo de pneumonia
- `chatbot` — interface frontend para o usuário interagir com os modelos e api's.

## Tecnologias

- Linguagem principal: **Python** (≈ 76,6%) :contentReference[oaicite:0]{index=0}
- Também há scripts em Shell, JavaScript, CSS e HTML. :contentReference[oaicite:1]{index=1}
- Uso de frameworks de ML (por padrão: scikit-learn, TensorFlow ou PyTorch – ver dentro dos diretórios)
- API(s) provavelmente construídas com Flask/FastAPI ou similar (ver estrutura de `api-model*`)
