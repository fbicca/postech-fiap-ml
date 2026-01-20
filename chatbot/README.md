
# 🤖 BotHealth – Chatbot de Anamnese (Flask) com Integração às APIs **Coração** e **Pneumonia**

Chatbot web em **Flask** que coleta dados clínicos de forma guiada e consome **duas APIs** de predição:
1) **API Coração** (FastAPI + scikit‑learn) – predição de risco cardiovascular (12 parâmetros).  
2) **API Pneumonia** (FastAPI + TensorFlow/EfficientNet) – classificação de radiografia de tórax (*NORMAL* vs *PNEUMONIA*).

Inclui instalador `install-chatbot.sh`, endpoints de chat e upload, e exemplos de teste rápido.

---

## 📦 Sumário
- [Arquitetura e Fluxos](#arquitetura-e-fluxos)
- [Instalação (um‑clique)](#instalação-um‑clique)
- [Variáveis de Ambiente (.env)](#variáveis-de-ambiente-env)
- [Endpoints do Chatbot](#endpoints-do-chatbot)
- [Integração com a **API Coração**](#integração-com-a-api-coração)
- [Integração com a **API Pneumonia**](#integração-com-a-api-pneumonia)
- [Testes rápidos (curl/HTTPie)](#testes-rápidos-curlhttpie)
- [Solução de Problemas](#solução-de-problemas)
- [Licença](#licença)

---

## 🧭 Arquitetura e Fluxos

```
[Usuário] → Chat UI (front) → Flask /chat
                                 ├─ fluxo 1 (Cardio): coleta 11 entradas → POST {API_PREDICT_HEART}/predict
                                 │        ↳ recebe {prediction,label,probability_positive,...}
                                 │        ↳ formata e responde no chat
                                 └─ fluxo 2 (Pneumonia): /upload imagem → POST {API_PREDICT_PNEUMONIA}/predict (multipart)
                                          ↳ recebe {top_class, top_prob, probs}
                                          ↳ formata e responde no chat
```

**Pastas e arquivos do chatbot**
```
.
├─ app.py            # Rotas Flask (/chat, /upload, /uploads/*) e orquestração dos fluxos
├─ anamnese.py       # Helpers de formatação (ex.: ST_Slope, ECG, booleanos)
├─ validation.py     # Validações dos passos (idade, pressão, colesterol, etc.)
├─ requirements.txt  # Dependências do chatbot
├─ templates/        # (opcional) index.html
├─ static/           # (opcional) CSS/JS/Imagens
└─ install-chatbot.sh# Instalador: venv + pip + run
```

---

## ⚙️ Instalação (um‑clique)

```bash
chmod +x install-chatbot.sh
./install-chatbot.sh
```
O script cria `.venv`, instala dependências, gera `.env` (se ausente) e inicia o Flask na **porta 5001**.  
URLs úteis:
- Home: `http://127.0.0.1:5001/`
- Chat (POST): `/chat`
- Upload de imagem (POST): `/upload`
- Servir arquivos: `/uploads/<filename>`

> Dica: defina `PORT` no `.env` para customizar a porta.

---

## 🔧 Variáveis de Ambiente (.env)

```bash
# URLs das APIs de predição
API_PREDICT_HEART="http://127.0.0.1:8001/predict"
API_PREDICT_PNEUMONIA="http://127.0.0.1:8002/predict"

# Flask
FLASK_SECRET_KEY="uma_chave_secreta_segura"
PORT=5001
```

- **API_PREDICT_HEART** → endpoint `/predict` da **API Coração**.
- **API_PREDICT_PNEUMONIA** → endpoint `/predict` da **API Pneumonia** (multipart `file`).

---

## 🌐 Endpoints do Chatbot

### `POST /chat`
- **Entrada**: `{"msg": "<texto_do_usuário>", "type_conversation": "<estado>"}`
- **Estados típicos**:
  - `await_service` → menu inicial (1 = Coração, 2 = Pneumonia)
  - `await_*` → etapas do fluxo cardio (idade, sexo, ECG, etc.)
  - `await_pneumonia_confirm` → confirmação de envio após upload
- **Saída**: JSON com a resposta que será renderizada no front (mensagem, prompt seguinte etc.).

### `POST /upload`
- **Entrada**: `multipart/form-data` com campo `file` (`.jpg/.jpeg/.png`).
- **Efeito**: salva em `uploads/`, retorna nome/URL; o chat entra no estado de confirmação para enviar à API de Pneumonia.

### `GET /uploads/<filename>`
- Serve a imagem previamente enviada.

---

## ❤️ Integração com a **API Coração**

A API de risco cardiovascular (FastAPI) expõe, entre outros, **`/predict`**, **`/health`**, **`/predict-batch`** e **`/debug-vector`**. O chatbot consome **`/predict`**.

- **URL**: `${API_PREDICT_HEART}` (ex.: `http://127.0.0.1:8001/predict`)
- **Método**: `POST`
- **Content-Type**: `application/json`
- **Payload** (*campos coletados no fluxo do chat*):
  ```json
  {
    "Age": 52,
    "Sex": "M",
    "ChestPainType": "ASY",
    "RestingBP": 110,
    "Cholesterol": 130,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 78,
    "Exang": "não",
    "Oldpeak": 0.0,
    "ST_Slope": "Flat"
  }
  ```
  - A API aceita **sinônimos/PT‑BR/EN** e normaliza os campos (`Sex`, `ChestPainType`, `RestingECG`, `ST_Slope`, `Exang/ExerciseAngina`, `Oldpeak` com vírgula etc.).
- **Resposta esperada** (exemplo):
  ```json
  {
    "prediction": 1,
    "label": "ALTO_RISCO",
    "probability_positive": 0.91,
    "modelDetails": {
      "features_expected": ["Age", "Sex", "..."],
      "model_class": "LogisticRegression"
    },
    "warnings": [],
    "explanation": "O modelo identificou um risco elevado de doença cardiovascular (91% de probabilidade) baseado nos dados fornecidos. Os principais fatores que contribuíram para este resultado incluem: idade avançada (68 anos), presença de angina induzida por exercício, hipertrofia ventricular esquerda no ECG, e depressão significativa do segmento ST. Recomenda-se consulta médica urgente para avaliação completa e definição do tratamento adequado."
  }
  ```
- **Como o chatbot usa**:
  1. Conduz o usuário pelas 12 entradas com **validações** e **formatação** amigável.
  2. Exibe um **resumo** e pede confirmação ("SIM" para enviar).
  3. Envia o JSON à API e formata o retorno (rótulo + probabilidade) em linguagem natural.
  4. **Exibe a explicação da OpenAI** (campo `explanation`) se disponível, ou usa explicação local como fallback.

### 🤖 Explicações com OpenAI

O chatbot agora exibe automaticamente as explicações em linguagem natural geradas pela OpenAI quando disponíveis:

- ✅ **Prioridade**: Se a API retornar o campo `explanation` (gerado pela OpenAI), ele é exibido ao usuário
- ✅ **Fallback**: Se não houver explicação da OpenAI, o chatbot usa a explicação local baseada em regras
- ✅ **Formatação**: As explicações são exibidas com o título "🤖 *Explicação do Diagnóstico*" para destacar que são geradas por IA

**Nota**: Para que as explicações da OpenAI sejam geradas, é necessário configurar a variável de ambiente `OPENAI_API_KEY` no servidor da API (`api-model-heart`).

> Endpoints auxiliares: `/health` (status do modelo), `/predict-batch` (lote), `/debug-vector` (vetor alinhado/escalado).

---

## 🫁 Integração com a **API Pneumonia**

A API de Pneumonia (FastAPI + TensorFlow) recebe imagem e retorna a classe mais provável e o mapa completo de probabilidades.

- **URL**: `${API_PREDICT_PNEUMONIA}` (ex.: `http://127.0.0.1:8002/predict`)
- **Método**: `POST`
- **Content-Type**: `multipart/form-data`
- **Campo**: `file=@<sua_imagem.jpg>`
- **Resposta esperada** (exemplo):
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
- **Como o chatbot usa**:
  1. Usuário envia a imagem via `/upload`.
  2. Chatbot confirma se deve analisar a imagem (“sim”/“não”).
  3. Faz `POST multipart` à API; normaliza e exibe **classe** & **probabilidade**.
- **Observações**:
  - Suporta tipos `image/jpeg`, `image/jpg`, `image/png`.
  - A API disponibiliza ainda `/health` (modelo, nº de classes) e `/classes` (nomes de classes).

---

## 🧪 Testes rápidos (curl/HTTPie)

### 1) Iniciar o chatbot (Flask)
```bash
./install-chatbot.sh
# ou manualmente:
# python3 -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt
# export PORT=5001 && flask run --host 0.0.0.0 --port $PORT
```

### 2) Cardio – enviar opções pelo chat
```bash
curl -s -X POST http://127.0.0.1:5001/chat \
  -H "Content-Type: application/json" \
  -d '{"msg":"1","type_conversation":"await_service"}' | python3 -m json.tool
```

### 3) Pneumonia – upload + confirmar envio
```bash
curl -F "file=@NORMAL2-IM-1436-0001.jpeg" http://127.0.0.1:5001/upload

curl -s -X POST http://127.0.0.1:5001/chat \
  -H "Content-Type: application/json" \
  -d '{"msg":"sim","type_conversation":"await_pneumonia_confirm"}' | python3 -m json.tool
```

> **Portas sugeridas**: API Coração em **8001**, API Pneumonia em **8002**, Chatbot Flask em **5001**.

---

## 🛠️ Solução de Problemas

- **“Falha ao chamar a API do coração/pneumonia”**  
  Verifique se as variáveis `${API_PREDICT_HEART}` e `${API_PREDICT_PNEUMONIA}` apontam para os **/predict** corretos e se os serviços FastAPI estão ativos (Uvicorn em 8001/8002).

- **Upload não aparece**  
  Confirme permissões da pasta `uploads/` (criada automaticamente). Acesse via `/uploads/<arquivo>` para depurar.

- **Front exibindo `<br>` como texto**  
  O backend sanitiza entradas HTML acidentais. Ajuste o front para não enviar tags como texto literal quando quiser quebra de linha.

- **CORS**  
  As APIs expõem CORS liberado (`*`) por padrão. Em produção, restrinja a `allow_origins` para o domínio do seu chatbot.

---

## 📝 Licença

Uso acadêmico e educacional. Adapte livremente conforme a necessidade do seu projeto.
