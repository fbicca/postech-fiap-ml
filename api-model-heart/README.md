# 💖 API – Preditor de Insuficiência Cardíaca (FastAPI)

API desenvolvida em **FastAPI** para prever o risco de **insuficiência cardíaca** com base em 11 parâmetros clínicos.
O modelo foi treinado com **scikit-learn (Regressão Logística)** e utiliza **StandardScaler** para normalização.

---

## 🚀 Endpoints principais

| Endpoint         | Método | Descrição                                                     |
|------------------|--------|---------------------------------------------------------------|
| `/health`        | GET    | Verifica se o modelo e o scaler foram carregados corretamente |
| `/predict`       | POST   | Realiza predição individual de risco cardíaco com explicação em linguagem natural (OpenAI) |
| `/predict-batch` | POST   | Permite predição em lote                                      |
| `/debug-vector`  | POST   | Retorna o vetor processado e colunas utilizadas               |

## 🤖 Integração com OpenAI

A API agora inclui integração com a OpenAI para gerar explicações em linguagem natural dos diagnósticos. 

**Funcionalidades:**
- ✅ Análise automática dos dados do paciente e resultado da predição
- ✅ Explicações claras e compreensíveis em português brasileiro
- ✅ Destaque dos principais fatores de risco identificados
- ✅ Uso do modelo `gpt-4o-mini` (econômico e eficiente)
- ✅ Fallback gracioso: se a API não estiver configurada, a predição funciona normalmente sem explicação

**Como funciona:**
1. A rota `/predict` realiza a predição normalmente
2. Se `OPENAI_API_KEY` estiver configurada, uma explicação é gerada automaticamente
3. A explicação é incluída no campo `explanation` da resposta JSON
4. Se houver erro na geração da explicação, a API continua funcionando normalmente

---

## ⚙️ Instalação e execução

### 1️⃣ Instalar dependências
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2️⃣ Estrutura esperada
```
.
├── api.py
├── modelo_insuficiencia_cardiaca.pkl
├── scaler_dados.pkl
├── exemplos.txt
└── requirements_api.txt
```

### 3️⃣ Executar a API
```bash
uvicorn api-model-heart:app --host 0.0.0.0 --port 8001
```

---

## 📬 Exemplo de uso (HTTPie ou curl)

### `/predict`
```bash
http POST :8001/predict   Age:=52 Sex=M ChestPainType=ASY RestingBP:=110 Cholesterol:=130   FastingBS:=0 RestingECG=Normal MaxHR:=78 Exang=não Oldpeak:=0.0 ST_Slope=Flat 
```

**Resposta:**
```json
{
  "prediction": 1,
  "label": "ALTO_RISCO",
  "probability_positive": 0.91,
  "warnings": [],
  "modelDetails": {
    "features_expected": ["Age", "Sex", "ChestPainType", "..."],
    "model_class": "LogisticRegression"
  },
  "explanation": "O modelo identificou um risco elevado de doença cardiovascular (91% de probabilidade) baseado nos dados fornecidos. Os principais fatores que contribuíram para este resultado incluem: idade avançada (68 anos), presença de angina induzida por exercício, hipertrofia ventricular esquerda no ECG, e depressão significativa do segmento ST. Recomenda-se consulta médica urgente para avaliação completa e definição do tratamento adequado."
}
```

---

## 🧩 Parâmetros aceitos

| Campo           | Tipo     | Descrição                                     |
|-----------------|----------|-----------------------------------------------|
| `Age`           | int      | Idade (1–120)                                 |
| `Sex`           | str      | `M` ou `F`                                    |
| `ChestPainType` | str      | `TA`, `ATA`, `NAP`, `ASY`                     |
| `RestingBP`     | int      | Pressão arterial de repouso (70–250 mmHg)     |
| `Cholesterol`   | int      | Colesterol total (100–600 mg/dL)              |
| `FastingBS`     | int/bool | 0 = normal, 1 = glicemia alterada             |
| `RestingECG`    | str      | `Normal`, `ST`, `LVH`                         |
| `MaxHR`         | int      | Frequência cardíaca máxima (40–250 bpm)       |
| `Exang`         | str      | “sim” / “não”                                 |
| `Oldpeak`       | float    | Depressão ST (0.0–10.0)                       |
| `ST_Slope`      | str      | `Up`, `Flat`, `Down`                          |

---

## 🧪 Casos de teste

### 🟢 Baixo Risco
```bash
http POST :8001/predict   Age:=50 Sex=M ChestPainType=NAP RestingBP:=125 Cholesterol:=190   FastingBS:=0 RestingECG=Normal MaxHR:=165 Exang=não Oldpeak:=0.2 ST_Slope=Up 
```

### 🔴 Alto Risco
```bash
http POST :8001/predict   Age:=68 Sex=M ChestPainType=ASY RestingBP:=160 Cholesterol:=290   FastingBS:=1 RestingECG=LVH MaxHR:=82 Exang=sim Oldpeak:=3.1 ST_Slope=Flat 
```

---

## 🧠 Modelo

- **Tipo:** LogisticRegression
- **Scaler:** StandardScaler
- **Features:** 11–12 parâmetros clínicos
- **Métrica:** Recall (reduz falsos negativos)

---

## 🩺 Health Check
```bash
curl -s http://localhost:8001/health | python3 -m json.tool
```

**Resposta:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "scaler_loaded": true,
  "feature_columns_source": "model.feature_names_in_"
}
```

---

## 📜 Licença
Uso acadêmico e educacional.  
Desenvolvido por **Filipe Bicca e Edmilson Teixeira (50+Dev)**.

### 🔧 Variáveis de Ambiente (.env)

Crie um arquivo `.env` na raiz com:

```
# ------------------------------------------------------------
# ❤️ Configurações do modelo de insuficiência cardíaca
# ------------------------------------------------------------
# Caminho do modelo treinado (arquivo .pkl gerado no treino)
MODEL_PATH=modelo_insuficiencia_cardiaca.pkl

# Caminho do objeto StandardScaler (para normalizar novas entradas)
SCALER_PATH=scaler_dados.pkl

# Caminho do CSV com as colunas originais do treino (usado como fallback)
FEATURE_COLUMNS_PATH=X_train.csv

# ------------------------------------------------------------
# 🤖 Configurações da OpenAI (para explicações em linguagem natural)
# ------------------------------------------------------------
# Chave de API da OpenAI (obtenha em https://platform.openai.com/api-keys)
OPENAI_API_KEY=sua_chave_api_aqui

# Modelo da OpenAI a ser usado (padrão: gpt-4o-mini - barato e eficiente)
OPENAI_MODEL=gpt-4o-mini
```

Essas variáveis são lidas automaticamente no `api-model-heart.py` e usadas para carregar o modelo,
o scaler e a referência de colunas do treino.

**Nota sobre explicações com OpenAI:**
- Se `OPENAI_API_KEY` não estiver configurada, a API funcionará normalmente, mas o campo `explanation` será `null`
- O modelo padrão `gpt-4o-mini` é uma opção econômica que oferece boas respostas
- As explicações são geradas automaticamente na rota `/predict` analisando os dados do paciente e o resultado da predição
