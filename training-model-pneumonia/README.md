Datasets público:
Chest X-Ray – Pneumonia (Kaggle): ~5.8k radiografias AP de tórax, classes Normal vs Pneumonia; excelente para um primeiro projeto de diagnóstico binário. Exige conta no Kaggle para baixar.

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download

Chest X-Ray Images (Pneumonia)
5,863 images, 2 categories


pip install tensorflow==2.* scikit-learn pandas matplotlib opencv-python


# 🫁 Treinamento de CNN — Pneumonia em Raios‑X de Tórax

Este projeto treina uma **rede neural convolucional (CNN)** usando **transfer learning (EfficientNetB0 / Keras‑TensorFlow)** para classificar imagens de tórax em **Normal** vs **Pneumonia**. O pipeline cobre: preparação/organização dos dados, criação de `tf.data` datasets, *feature extractor training*, **fine‑tuning**, e avaliação final com **relatório de métricas**, **AUC** e **matriz de confusão**. 


---
## 📦 Dataset (exemplo)
Dataset público sugerido para testes/primeiro experimento: **Chest X‑Ray Images (Pneumonia)** (Kaggle) — ~5.8k imagens AP; 2 classes. É necessário ter conta no Kaggle para baixar. 

> Link (Kaggle): *Chest X-Ray Images (Pneumonia)* — 5,863 imagens, 2 categorias. 

## Estrutura de Pastas

```
data/
  raw/                 # opcional: imagens por classe; se existir, o script cria o split
    ClasseA/
    ClasseB/
  train/
    ClasseA/
    ClasseB/
  val/
    ClasseA/
    ClasseB/
  test/
    ClasseA/
    ClasseB/
outputs/
  models/
  plots/
  reports/
train_cnn.py
```
- Se `data/train` já existir, o script **não** refaz o split. Se só houver `data/raw`, o script cria `train/`, `val/`, `test/`.

---

## Requisitos

```bash
### 1️⃣ Instalar dependências
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "tensorflow==2.*" scikit-learn pandas matplotlib opencv-python
```

> 💡 **GPU opcional:** se tiver CUDA/cuDNN instalados, o treino acelera bastante. Em CPU também funciona (mais lento).

---

## Como Executar

## 🚀 Como executar
1) **Organize os dados**  
   - **Cenário A (rápido):** já possui `data/train`, `data/val`, `data/test` → pule o split.  
   - **Cenário B (split automático):** coloque as imagens em `data/raw/<Classe>/` e **deixe** `data/train` vazio; o script fará o split. fileciteturn12file2

2) **Treinar o modelo**
```bash
python train_cnn.py --port 7002
```

Durante o treino, o script executa:
- **Feature extractor** com a base `EfficientNetB0` **congelada** e cabeça densa; *callbacks*: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`.  
- **Fine‑tuning:** descongela a base (mantendo ~200 camadas ainda congeladas), reduz a taxa de aprendizado e treina novamente. 


3) **Saídas principais**
- Modelo final: `outputs/models/model.h5`  
- Melhor checkpoint (feature extractor): `outputs/models/best_feature_extractor.keras`  
- Melhor checkpoint (fine-tuning): `outputs/models/best_finetuned.keras`  
- Relatório de classificação (precision/recall/F1): `outputs/reports/classification_report.csv`  
- Resumo com AUC e classes: `outputs/reports/summary.json`  
- Matriz de confusão: `outputs/plots/confusion_matrix.png`

---

## Visão Geral do Pipeline

## 🧠 O que o código faz (resumo)
- **Seed e diretórios** (`data/`, `outputs/models|plots|reports`).  
- **Split opcional** a partir de `data/raw` (se `data/train` não existir).  
- **Datasets `tf.data`** com `image_dataset_from_directory` (+ *augmentations* no treino, `preprocess_input` da EfficientNet, `cache`/`prefetch`).  
- **Modelo**: EfficientNetB0 `include_top=False`, GAP → Dropout(0.25) → Dense `softmax`. Otimizador Adam, *loss* `categorical_crossentropy`.  
- **Treino** com *callbacks*; depois **fine‑tuning** com LR menor.  
- **Avaliação**: `classification_report`, **AUC** (binária ou OVR multi‑classe), **matriz de confusão** e **plots**.  
- **Persistência**: salva checkpoints e `model.h5`. fileciteturn12file2



1. **Seed & Configurações**  
   Define *seed* reprodutível, diretórios (`data`, `outputs`), hiperparâmetros (tamanho de imagem, *batch size*, *epochs*, *patience*).

2. **Preparação do Split (opcional)**  
   Se `data/train` não existir, cria `train/`, `val/`, `test/` a partir de `data/raw` usando proporções **15% val** e **15% test** por classe. Copia os arquivos mantendo o balanço.

3. **Pipelines `tf.data`**  
   Carrega imagens de `train/`, `val/`, `test/` com `image_dataset_from_directory`. Aplica *augmentations* leves (flip, rotação, zoom, translação) **apenas no treino** e `preprocess_input` da EfficientNet. Usa `cache()` + `prefetch()` para desempenho.

4. **Modelo (Transfer Learning)**  
   - **Base:** `EfficientNetB0` com pesos ImageNet, `include_top=False`, *freezada* inicialmente.  
   - **Cabeçote:** GAP → Dropout(0.25) → Dense `softmax` (número de classes).  
   - **Compilação:** Adam(1e-3), *categorical crossentropy*, *accuracy*.

5. **Treinamento (Feature Extractor)**  
   Treina apenas o cabeçote com *callbacks*:  
   - `EarlyStopping` (paciente a **6** épocas, monitorando `val_accuracy`)  
   - `ReduceLROnPlateau` (reduz LR ao estagnar `val_loss`)  
   - `ModelCheckpoint` (salva melhor modelo por `val_accuracy`)

6. **Fine-Tuning**  
   Descongela a base (com as **~200 primeiras camadas ainda congeladas** para estabilidade) e recompila com LR menor (1e-5). Treina por mais algumas épocas com os mesmos *callbacks*, salvando o melhor `best_finetuned.keras`.

7. **Avaliação em Teste**  
   Gera `y_prob`, `y_pred` e calcula:
   - **classification_report** por classe (precision, recall, F1)
   - **AUC** (binária ou *ovr* multi-classe, quando aplicável)
   - **Matriz de confusão** com *plot* salvo em PNG  
   Salva o **modelo final** em `model.h5`.

---

## Detalhamento do Código (por função/bloco)

- **Configurações e Constantes**: `SEED`, diretórios, `IMG_SIZE=(224,224)`, `BATCH_SIZE=32`, `EPOCHS=30`, `VAL_SPLIT=0.15`, `TEST_SPLIT=0.15`.

- **`ensure_dirs()`**  
  Garante a existência de `outputs/models`, `outputs/plots`, `outputs/reports`.

- **`split_from_raw_if_needed()`**  
  - Pula se `data/train` já tem conteúdo.  
  - Se `data/raw` existir, cria `train/val/test` por classe com cópia dos arquivos e *shuffle* controlado pela *seed*.

- **`build_datasets()`**  
  - Cria `train_ds`, `val_ds`, `test_ds` a partir das pastas.  
  - Define *augmentations* (apenas no treino).  
  - Aplica `preprocess_input` da EfficientNet.  
  - Retorna *datasets* com `cache().prefetch()` e `class_names`.

- **`build_model(num_classes)`**  
  - Carrega `EfficientNetB0` (ImageNet), congela a base.  
  - Cabeçote: GAP → Dropout(0.25) → Dense(softmax).  
  - Compila com Adam(1e-3).

- **`unfreeze_and_finetune(model, base, lr=1e-4)`**  
  - Descongela a base para *fine-tuning* (mantém as ~200 primeiras camadas congeladas).  
  - Recompila com LR menor (1e-4 no código-base; chamada usa 1e-5).

- **`plot_confusion(cm, classes, savepath)`**  
  - Plota e salva a matriz de confusão com rótulos e contagens por célula.

- **`main()`**  
  - Cria diretórios, realiza split se necessário.  
  - Constrói *datasets* e obtém `class_names`.  
  - Treina (feature extractor) com *callbacks*.  
  - Realiza *fine-tuning*.  
  - Avalia em teste, calcula métricas, salva relatórios, plots e modelo final.

---

## Exemplo de Inferência (pós-treino)

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("outputs/models/model.h5")
img_path = "caminho/para/uma_imagem.jpg"

img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

prob = model.predict(x)[0]            # vetor de probabilidades
pred_idx = prob.argmax()
print("Classe predita:", pred_idx, "Prob:", prob[pred_idx])
```

> **Observação:** use `class_names` impressas no treinamento para mapear índices → rótulos.

---

## Boas Práticas (saúde)

- **Não substitui avaliação clínica.**  
- Valide com **dados externos** (outras instituições).  
- Atenção a **viés** (idade, sexo, aparelho, protocolo de aquisição).  
- Cheque **termos de uso** e **privacidade** dos dados. (Consentimento/ética).

---

## 🧯 Troubleshooting

- **OOM / falta de VRAM**: reduza `BATCH_SIZE`, use `mixed_precision` e/ou imagem menor.  
- **Treino estagnado**: ajuste LR (maior no início, menor no *fine-tuning*), revise augmentations.  
- **Desbalanceamento**: use `class_weight` ou técnicas de *resampling*.  
- **AUC NaN**: pode ocorrer em classes únicas no *test*; valide o split e volume de dados.

---

## Licença & Uso

Este código é fornecido para fins educacionais/experimentais. Verifique a **licença do dataset** antes de uso acadêmico/comercial.
