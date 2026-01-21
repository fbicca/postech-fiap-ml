# 💖 Treinamento do Modelo – Preditor de Insuficiência Cardíaca

Este repositório contém o **algoritmo de criação e teste do modelo** utilizado pela API. O pipeline treina um modelo de **classificação binária** (doença cardíaca: 0/1) a partir do dataset `heart.csv`, gera métricas e persiste **modelo** e **scaler** para uso em produção.

---

## 🗂️ Estrutura

```
.
├── main.py                       # Script de treino/avaliação do modelo
├── heart.csv                     # Dataset de entrada (features + HeartDisease)
├── X_train.csv  X_test.csv       # Features escalonadas (geradas pelo pipeline)
├── y_train.csv  y_test.csv       # Targets correspondentes
├── modelo_insuficiencia_cardiaca.pkl  # Modelo treinado (joblib)
├── scaler_dados.pkl                   # Scaler treinado (joblib)
├── requirements.txt              # Dependências para treino/avaliação
├── evidencias/                   # Diretório de saída dos experimentos
│   ├── logs/                     # Logs e histórico do algoritmo genético
│   ├── comparacao_experimentos_ag_*.json  # Relatório comparativo
│   └── [arquivos de evidências por experimento]
├── ag_*.py                       # Módulos do algoritmo genético
│   ├── ag_algoritmo.py           # Classe principal do AG
│   ├── ag_experimentos.py        # Execução de múltiplos experimentos
│   ├── ag_comparacao.py          # Comparação de modelos
│   ├── ag_criar_individuo.py     # Criação de indivíduos
│   ├── ag_criar_fitness.py       # Função de fitness
│   ├── ag_selecao.py             # Métodos de seleção
│   ├── ag_cruzamento.py          # Métodos de cruzamento
│   ├── ag_mutacao.py             # Métodos de mutação
│   ├── ag_logging.py             # Sistema de logging
│   └── ag_visualizacao.py        # Geração de gráficos
└── *.md                          # Documentação técnica
```

---
# Desativar o ambiente virtual, se estiver ativo
```bash
deactivate

# Remover toda a pasta do ambiente virtual
rm -rf .venv


## ⚙️ Ambiente

Crie e ative um ambiente virtual e instale as dependências do arquivo `requirements_model.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Se preferir versões mínimas (não fixas), você pode usar um requirements com **constraints `>=`** (ver seção “Alternativa com versões mínimas”).

---

## 🧠 Pipeline de Treinamento

O script `main.py` executa as seguintes etapas principais:

1. **Carregamento do dataset** `heart.csv` e checagem de nulos.  
2. **Codificação One‑Hot** das variáveis categóricas com `pd.get_dummies(drop_first=True)`.  
3. **Split treino/teste** estratificado (70/30) com `train_test_split`.  
4. **Escalonamento** das features com `StandardScaler` (fit no treino, transform em treino e teste).  
5. **Persistência** dos conjuntos escalonados (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`).  
6. **Treinamento** de uma **Regressão Logística** (`solver='liblinear'`, `random_state=42`).  
7. **Avaliação** com acurácia e `classification_report` (precision, recall, f1).  
8. **Exportação** dos artefatos: `modelo_insuficiencia_cardiaca.pkl` e `scaler_dados.pkl` (via `joblib`).

> O **alvo** (variável dependente) é a coluna `HeartDisease` (0/1).  
> Para aplicações clínicas, recomenda‑se acompanhar **Recall/Sensibilidade** (minimizar falsos negativos).

### Execução Básica (Treinamento Simples)

```bash
python main.py --csv heart.csv --target HeartDisease --outdir evidencias --modo simples
```

Após a execução, você deverá ver no console as métricas do modelo e os arquivos `.pkl`/`.csv` serão gerados na raiz do projeto.

---

## 🧬 Experimentos com Algoritmo Genético

O projeto inclui um sistema de **otimização de hiperparâmetros usando Algoritmo Genético (AG)** que permite comparar diferentes configurações e encontrar os melhores parâmetros para o modelo de Regressão Logística.

### 📊 Modos de Execução

O script `main.py` suporta três modos de execução:

1. **`simples`**: Executa um único algoritmo genético (sem comparação explícita)
2. **`experimentos`**: Executa múltiplos experimentos com diferentes configurações do AG
3. **`comparar`**: Executa um único AG e compara detalhadamente com o modelo original

### 🚀 Executar os 4 Experimentos

Para executar os **4 experimentos** com diferentes configurações do algoritmo genético:

```bash
python3 main.py --csv heart.csv --target HeartDisease --outdir evidencias --modo experimentos --num_experimentos 4
```

**O que este comando faz:**
- Executa 4 experimentos diferentes com configurações variadas do AG
- Cada experimento otimiza hiperparâmetros (C, max_iter, class_weight) da Regressão Logística
- Compara cada modelo otimizado com o modelo original (baseline)
- Gera relatórios comparativos, gráficos e evidências em `evidencias/`

**Configurações dos 4 Experimentos:**

1. **Experimento 1: Configuração Conservadora**
   - População: 20, Gerações: 15
   - Taxa Cruzamento: 0.7, Taxa Mutação: 0.1
   - Método: Torneio + Cruzamento Uniforme + Mutação Uniforme

2. **Experimento 2: População Maior + Mais Gerações**
   - População: 40, Gerações: 25
   - Taxa Cruzamento: 0.7, Taxa Mutação: 0.1
   - Método: Torneio + Cruzamento Uniforme + Mutação Uniforme

3. **Experimento 3: Alta Mutação + Cruzamento Aritmético**
   - População: 30, Gerações: 20
   - Taxa Cruzamento: 0.8, Taxa Mutação: 0.2
   - Método: Roleta + Cruzamento Aritmético + Mutação Gaussiana

4. **Experimento 4: Foco em RECALL (Sensibilidade)**
   - População: 30, Gerações: 25
   - Taxa Cruzamento: 0.8, Taxa Mutação: 0.15
   - Método: Torneio + Cruzamento Aritmético + Mutação Não-Uniforme
   - **Métrica Fitness: RECALL** (prioriza minimizar falsos negativos)

### 📈 Executar Comparação Detalhada

Para executar um único experimento com comparação detalhada entre modelo original e otimizado:

```bash
python3 main.py --csv heart.csv --target HeartDisease --outdir evidencias --modo comparar
```

**O que este comando faz:**
- Executa um algoritmo genético com configuração padrão
- Treina modelo original (baseline) e modelo otimizado
- Exibe tabela comparativa detalhada com métricas e melhorias percentuais
- Gera gráficos de comparação (matriz de confusão, curva ROC)

### 📁 Arquivos Gerados

Após executar os experimentos, os seguintes arquivos serão gerados em `evidencias/`:

**Relatórios e Comparações:**
- `comparacao_experimentos_ag_TIMESTAMP.json` - Relatório comparativo completo (JSON)
- `COMPARACAO_4_EXPERIMENTOS_TIMESTAMP.md` - Documento Markdown com análise detalhada

**Evidências por Experimento:**
- `classification_report_EXPERIMENTO_TIMESTAMP.txt` - Relatório de classificação
- `matriz_confusao_EXPERIMENTO_TIMESTAMP.png` - Matriz de confusão
- `roc_curve_EXPERIMENTO_TIMESTAMP.png` - Curva ROC
- `coeficientes_EXPERIMENTO_TIMESTAMP.csv` - Coeficientes do modelo
- `evidencias_treinamento_EXPERIMENTO_TIMESTAMP.json` - Métricas de treinamento
- `resumo_evidencias_EXPERIMENTO_TIMESTAMP.pdf` - Resumo em PDF

**Logs e Histórico:**
- `logs/ag_log_TIMESTAMP.log` - Log estruturado da execução
- `logs/ag_config_TIMESTAMP.json` - Configuração do AG
- `logs/ag_historico_EXPERIMENTO_TIMESTAMP.json` - Histórico de evolução
- `logs/ag_evolucao_fitness_EXPERIMENTO_TIMESTAMP.png` - Gráfico de evolução do fitness
- `logs/ag_convergencia_EXPERIMENTO_TIMESTAMP.png` - Gráfico de convergência

### 🔍 Visualizar Resultados

Após executar os 4 experimentos, você pode:

1. **Ver o relatório comparativo:**
   ```bash
   cat evidencias/comparacao_experimentos_ag_*.json | python3 -m json.tool
   ```

2. **Ler a análise detalhada:**
   ```bash
   cat evidencias/COMPARACAO_4_EXPERIMENTOS_*.md
   ```

3. **Visualizar gráficos:**
   - Abra os arquivos PNG em `evidencias/` para ver matrizes de confusão e curvas ROC
   - Abra os arquivos PNG em `evidencias/logs/` para ver evolução do algoritmo genético

### ⏱️ Tempo de Execução

- **4 Experimentos**: ~15-30 minutos (dependendo do hardware)
- **Modo Comparar**: ~5-10 minutos
- **Modo Simples**: ~3-5 minutos

> **Nota**: Os tempos podem variar significativamente dependendo da configuração do hardware e dos parâmetros do algoritmo genético (população, gerações, etc.).

---

## 🔬 Métricas e Relatórios

O script imprime no console:  
- **Acurácia** no conjunto de teste;  
- **Classification Report**: *precision*, *recall*, *f1‑score* por classe;  
- *Observação*: ajuste de limiar pode ser considerado conforme a necessidade (ex.: priorizar recall).

---

## 🔁 Reprodutibilidade

- `random_state=42` no split e no modelo;  
- `StandardScaler` treinado apenas no treino (evita *data leakage*);  
- As colunas finais usadas pelo modelo ficam registradas na propriedade `model.feature_names_in_` (útil para alinhar produção).

---

## 🧩 Handoff para Produção (API)

Na etapa de inferência (API), é **obrigatório alinhar** o vetor de entrada às **mesmas colunas** do treino:

- Usar `model.feature_names_in_` para reordenar/“completar” dummies;  
- Aplicar **o mesmo `scaler_dados.pkl`** (fit no treino) ao vetor antes de `predict`/`predict_proba`;  
- Em caso de divergência de colunas, usar `X_train.csv` como **fonte da verdade** para o conjunto de features.

---

## 🧪 Exemplo de uso dos artefatos (inferência local)

```python
import joblib
import pandas as pd

# 1) Carrega artefatos
model = joblib.load('modelo_insuficiencia_cardiaca.pkl')
scaler = joblib.load('scaler_dados.pkl')

# 2) Novo paciente (exemplo)
novo = pd.DataFrame([{
    "Age": 50, "Sex": "M", "ChestPainType": "NAP", "RestingBP": 125,
    "Cholesterol": 190, "FastingBS": 0, "RestingECG": "Normal",
    "MaxHR": 165, "ExerciseAngina": "N", "Oldpeak": 0.2, "ST_Slope": "Up"
}])

# 3) One‑Hot e alinhamento
X_cols = model.feature_names_in_
novo_d = pd.get_dummies(novo, drop_first=True)
for c in set(X_cols) - set(novo_d.columns):
    novo_d[c] = 0
novo_alinhado = novo_d[X_cols]

# 4) Escalonar e prever
X_scaled = scaler.transform(novo_alinhado)
pred = model.predict(X_scaled)[0]
prob = model.predict_proba(X_scaled)[0, 1]
print("Classe:", pred, "Prob.:", f"{prob:.2%}")
```

---

---

## 🎓 Guia Rápido para Avaliação

### Passo 1: Configurar Ambiente

```bash
# Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

# Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt
```

### Passo 2: Executar os 4 Experimentos

```bash
python3 main.py --csv heart.csv --target HeartDisease --outdir evidencias --modo experimentos --num_experimentos 4
```

**Tempo estimado:** 15-30 minutos

### Passo 3: Verificar Resultados

Após a execução, verifique:

1. **Relatório Comparativo (JSON):**
   ```bash
   ls -lh evidencias/comparacao_experimentos_ag_*.json
   ```

2. **Análise Detalhada (Markdown):**
   ```bash
   ls -lh evidencias/COMPARACAO_4_EXPERIMENTOS_*.md
   cat evidencias/COMPARACAO_4_EXPERIMENTOS_*.md
   ```

3. **Gráficos e Evidências:**
   ```bash
   ls evidencias/*.png  # Matrizes de confusão e curvas ROC
   ls evidencias/logs/*.png  # Gráficos de evolução do AG
   ```

### Passo 4: Executar Comparação Detalhada (Opcional)

Para uma comparação mais detalhada de um único experimento:

```bash
python3 main.py --csv heart.csv --target HeartDisease --outdir evidencias --modo comparar
```

**Tempo estimado:** 5-10 minutos

### 📋 Checklist de Verificação

- [ ] Ambiente virtual criado e ativado
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Arquivo `heart.csv` presente no diretório
- [ ] Comando dos 4 experimentos executado com sucesso
- [ ] Arquivos gerados em `evidencias/`
- [ ] Relatório comparativo JSON gerado
- [ ] Documento Markdown de análise gerado
- [ ] Gráficos (PNG) gerados para cada experimento

### 🔧 Troubleshooting

**Erro: "FileNotFoundError: heart.csv"**
- Verifique se o arquivo `heart.csv` está no diretório atual
- Use `--csv caminho/completo/para/heart.csv` se necessário

**Erro: "ModuleNotFoundError"**
- Certifique-se de que o ambiente virtual está ativado
- Execute `pip install -r requirements.txt` novamente

**Execução muito lenta**
- Reduza `--num_experimentos` para testar (ex: `--num_experimentos 2`)
- Os experimentos são computacionalmente intensivos

---

## 🪪 Licença
Uso acadêmico e educacional. Ajuste conforme sua necessidade.

---

## 📦 Alternativa com versões mínimas (requirements "aberto")

Se preferir dependências com `>=`:

```
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
numpy>=1.25.0
# opcionais
matplotlib>=3.8.0
jupyter>=1.0.0
```
