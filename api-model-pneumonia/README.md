# 🫁 Pneumonia Detection API (FastAPI + TensorFlow CPU)

API em **FastAPI** para **inferência automática de pneumonia em radiografias de tórax**, utilizando uma **CNN baseada em EfficientNet**.  
O projeto acompanha um **script de instalação automatizada** (`install-api-pneumonia.sh`) que cria o ambiente virtual, instala todas as dependências e executa o servidor Uvicorn.

---

## ✨ Principais recursos

- 📸 Upload de imagem via endpoint `/predict`  
- 🧠 Inferência com **CNN EfficientNet** pré-treinada em ImageNet  
- 📊 Retorna a **classe mais provável** (*Normal* ou *Pneumonia*) e o **mapa de probabilidades completo**  
- 🔍 Endpoints de verificação de saúde (`/health`) e listagem de classes (`/classes`)  
- 🌐 Suporte a **CORS** (padrão `*`, ideal para integração com chatbot web ou aplicações Flask)  
- 💻 Compatível com **TensorFlow CPU** (sem necessidade de GPU/CUDA)  

---

## 🗂️ Estrutura de diretórios

```
.
├── api-model-pneumonia.py        # Código principal da API
├── install-api-pneumonia.sh      # Script automatizado de instalação e execução
├── outputs/
│   ├── models/
│   │   ├── model.keras
│   │   ├── best_finetuned.keras
│   │   ├── best_feature_extractor.keras
│   │   └── model.h5
│   └── reports/
│       └── summary.json          # (opcional) contém nomes das classes
├── requirements.txt
└── README.md
```

> O arquivo `summary.json` pode conter:
> ```json
> { "classes": ["NORMAL", "PNEUMONIA"] }
> ```

---

## ⚙️ Requisitos

- Python **3.10+**
- TensorFlow **CPU-only**
- FastAPI, Uvicorn, Pillow, NumPy, python-multipart

---

## ⚡ Instalação automatizada (recomendada)

O script `install-api-pneumonia.sh` faz tudo automaticamente:

```bash
chmod +x install-api-pneumonia.sh
./install-api-pneumonia.sh
```

### O que ele faz:
1. Verifica se há Python 3 instalado  
2. Cria o ambiente `.venv`  
3. Ativa o ambiente  
4. Atualiza o `pip` e `setuptools`  
5. Instala dependências (`fastapi`, `uvicorn`, `tensorflow-cpu`, `pillow`, `numpy`, `python-multipart`)  
6. Executa a API na porta `8002`  

Após iniciado:
```
Acesse: http://127.0.0.1:8002/docs
Para interromper: CTRL+C
```

---

## ⚙️ Instalação manual (alternativa)

```bash
# 1️⃣ Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2️⃣ Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt

# 3️⃣ Executar API
uvicorn api-model-pneumonia:app --host 0.0.0.0 --port 8002
```

---

## 🧩 Endpoints disponíveis

| Método | Rota | Descrição |
|---------|------|------------|
| `GET` | `/health` | Retorna status da API e informações do modelo carregado |
| `GET` | `/classes` | Lista as classes disponíveis |
| `POST` | `/predict` | Recebe uma imagem e retorna a predição e as probabilidades |

---

## 📸 Exemplo de uso (`/predict`)

### Requisição
```bash
http -f POST :8002/predict file@NORMAL2-IM-1436-0001.jpeg
```

### Resposta
```json
{
  "top_class": "PNEUMONIA",
  "top_prob": 0.984,
  "probs": {
    "NORMAL": 0.016,
    "PNEUMONIA": 0.984
  }
}
```

---

## ⚗️ Funcionamento interno

1. O modelo é automaticamente carregado de `outputs/models/`.  
2. A imagem é lida e convertida para **RGB**, redimensionada para **224x224** e processada com `preprocess_input`.  
3. O modelo faz a predição (`model.predict(x)`), produzindo um vetor de probabilidades normalizadas.  
4. O resultado é retornado como JSON contendo a classe com maior probabilidade.

---

## 🧾 Tratamento de erros

| Tipo | Código HTTP | Mensagem |
|------|--------------|-----------|
| Arquivo inválido | 400 | `"Falha ao abrir a imagem. Formatos aceitos: jpg, jpeg, png."` |
| Tipo MIME incorreto | 400 | `"Envie uma imagem .jpg/.jpeg ou .png."` |
| Modelo ausente | 500 | `"Nenhum modelo encontrado em outputs/models/"` |

---

## 🧠 Modelo utilizado

- **Arquitetura:** EfficientNet (pré-treinada em ImageNet)  
- **Camada de saída:** Softmax (`num_classes`)  
- **Entrada:** Imagens RGB 224x224  
- **Formato:** `.keras` ou `.h5`  
- **Execução:** Forçada em CPU (sem GPU/CUDA)

---

## 🩻 Boas práticas

- Utilize radiografias **PA/AP** de boa qualidade.  
- Evite compressão excessiva (JPEG com qualidade < 80).  
- Centralize o pulmão na imagem.  
- Avalie o modelo em múltiplas imagens para maior confiabilidade.

---

## 👨‍💻 Autoria e licença

Projeto desenvolvido para fins **educacionais e de pesquisa aplicada à saúde**.  
Desenvolvido por **50+Dev — Edmilson Teixeira & colaboradores**.

---

## 📦 Novo `requirements.txt` atualizado

```txt
fastapi
uvicorn[standard]
tensorflow-cpu==2.*
pillow
numpy
python-multipart
```
