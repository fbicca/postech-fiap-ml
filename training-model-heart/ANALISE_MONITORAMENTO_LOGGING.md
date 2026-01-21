# Análise de Monitoramento e Logging - Algoritmo Genético

## Status Atual: O que JÁ EXISTE ✅

### 1. Histórico de Evolução (In-Memory)
**Localização:** `ag_algoritmo.py` linhas 64, 198-203

- **Implementação:** Classe `AlgoritmoGenetico` armazena histórico de fitness por geração
- **Dados capturados:**
  ```python
  {
      'geracao': int,
      'fitness_medio': float,
      'fitness_max': float,
      'fitness_min': float
  }
  ```
- **Acesso:** Método `get_historico()` (linha 237-244)
- **Limitação:** Histórico fica apenas em memória, não é persistido em arquivo

### 2. Logging Básico via Print (Verbose)
**Localização:** Vários arquivos

#### `ag_algoritmo.py`:
- Linhas 177-178: Início do AG (população, gerações)
- Linhas 206-207: Progresso por geração (fitness max, médio, min)
- Linhas 232-233: Conclusão (melhor fitness)

#### `ag_experimentos.py`:
- Linhas 77-89: Configuração do experimento
- Linha 111: Status de treinamento
- Linhas 368-377: Resumo comparativo de experimentos

#### `ag_comparacao.py`:
- Linhas 125-168: Tabela comparativa formatada

#### `main.py`:
- Linhas 156-157, 184-185, 223-224: Modo de execução
- Linhas 356-365: Lista de arquivos gerados

### 3. Salvamento de Resultados Finais
**Localização:** `main.py` linhas 309-349

#### JSON de Métricas (`evidencias_treinamento_TIMESTAMP.json`):
```json
{
    "timestamp": "20260116-101150",
    "metrics": {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1": 0.85,
        "auc": 0.92
    },
    "algoritmo_genetico": {
        "hiperparametros_otimizados": {
            "C": 1.5234,
            "penalty": "l2",
            "class_weight": "balanced",
            "max_iter": 1000
        }
    }
}
```

### 4. Log de Pré-processamento
**Localização:** `main.py` linhas 294-303

- **Arquivo:** `evidencias_preprocess_TIMESTAMP.txt`
- **Conteúdo:**
  - Checagem de nulos
  - Balanceamento de classes
  - Shapes dos datasets
  - Informações de split treino/teste

### 5. Relatório Comparativo de Experimentos
**Localização:** `ag_experimentos.py` linhas 282-379

- **Arquivo:** `comparacao_experimentos_ag_TIMESTAMP.json`
- **Conteúdo:**
  - Resultados de cada experimento
  - Comparação com modelo original
  - Ranking por métrica
  - Melhores experimentos por métrica
  - **Inclui histórico do AG** (via `exp.get_resultado_dict()`)

---

## O que FALTA para Logging/Monitoramento Adequado ❌

### ✅ **TODAS AS FUNCIONALIDADES PRINCIPAIS FORAM IMPLEMENTADAS!**

As seguintes funcionalidades foram implementadas conforme documentado na seção "Implementação de Geração de Evidências para Logging e Tracking":

- ✅ Módulo Logging Estruturado do Python - **IMPLEMENTADO** (`ag_logging.py`)
- ✅ Persistência do Histórico de Evolução - **IMPLEMENTADO** (`ag_logging.salvar_historico_evolucao`)
- ✅ Gráficos de Evolução do Fitness - **IMPLEMENTADO** (`ag_visualizacao.py`)
- ✅ Tracking de Tempo de Execução - **IMPLEMENTADO** (tempo por geração e total)
- ✅ Logging de Erros e Warnings - **IMPLEMENTADO** (níveis WARNING e ERROR)
- ✅ Logging de Configuração do AG - **IMPLEMENTADO** (`ag_logging.salvar_configuracao_ag`)

### Funcionalidades OPCIONAIS (não críticas)

### 6. Logging Detalhado de Cada Etapa do AG
**Status:** Parcialmente implementado

**O que existe:**
- Logging de fitness por geração (max, médio, min)
- Logging de tempo por geração
- Logging de configuração completa

**O que poderia ser adicionado (opcional):**
- Número de indivíduos avaliados por geração
- Taxa de cruzamento efetiva
- Taxa de mutação efetiva
- Diversidade da população (desvio padrão do fitness)
- Melhorias por geração (quantos indivíduos melhoraram)

**Prioridade:** BAIXA - Não crítico para trabalho acadêmico

---

## Funcionalidades PARCIALMENTE Implementadas ⚠️

### 1. Histórico do AG nos Experimentos
**Status:** Implementado, mas não é facilmente acessível

**Localização:** `ag_experimentos.py` linha 107
```python
self.historico_ag = ag.get_historico()
```

**Salvamento:** Sim, via `get_resultado_dict()` que inclui `'historico_ag'` no JSON de experimentos

**Problema:** Histórico fica enterrado no JSON de comparação de experimentos, não há forma fácil de acessar individualmente

### 2. Comparação de Modelos
**Status:** Implementado e funcional

**Localização:** `ag_comparacao.py`

**Problema:** Comparação é exibida, mas não há log estruturado dos resultados comparativos

---

## Resumo: Nível Atual de Monitoramento

### ✅ Implementado (Básico):
- [x] Histórico de fitness por geração (em memória)
- [x] Prints de progresso (verbose)
- [x] Salvamento de resultados finais em JSON
- [x] Log de pré-processamento
- [x] Relatório comparativo de experimentos
- [x] Exibição de comparação modelo original vs otimizado

### ✅ Totalmente Implementado:
- [x] Logging estruturado com níveis (DEBUG, INFO, WARNING, ERROR)
- [x] Persistência individual do histórico de evolução (JSON por experimento)
- [x] Gráficos de evolução do fitness (3 tipos: evolução, convergência, distribuição)
- [x] Tracking de tempo de execução (por geração e total)
- [x] Logging de erros em arquivo (níveis WARNING e ERROR)
- [x] Configuração completa salva de forma estruturada (JSON)
- [x] Gráficos de desempenho por experimento (matriz de confusão, ROC)
- [x] PDF resumo consolidado com todos os gráficos
- [x] Relatório comparativo estruturado (JSON)

### ⚠️ Parcialmente Implementado:
- [x] Logging detalhado de operadores (tem logging básico, poderia ter mais detalhes)
- [ ] Métricas de diversidade populacional (não implementado, opcional)

---

## Recomendações de Melhorias Futuras (Opcionais)

### ✅ **Todas as funcionalidades de prioridade ALTA e MÉDIA foram implementadas!**

### Melhorias Opcionais (Prioridade BAIXA):
1. **Métricas de diversidade populacional**
   - Desvio padrão do fitness por geração
   - Entropia da população
   - Número de genótipos únicos

2. **Logging detalhado de cada operação genética**
   - Taxa efetiva de cruzamento (quantos pares foram cruzados)
   - Taxa efetiva de mutação (quantos genes foram mutados)
   - Tipos de mutações aplicadas

3. **Dashboard interativo** (opcional)
   - Interface web para visualização dos resultados
   - Comparação interativa entre experimentos
   - Visualização dinâmica da evolução

4. **Exportação para formatos adicionais**
   - CSV para histórico de evolução
   - Excel com múltiplas abas
   - HTML interativo com gráficos

---

## Implementação de Geração de Evidências para Logging e Tracking ⭐

### 1. Sistema de Logging Estruturado

#### Arquivo: `ag_logging.py`

**Função Principal:** `configurar_logger()`
- **Localização:** `ag_logging.py` linhas 15-69
- **Funcionalidade:** Configura logger estruturado do Python com múltiplos handlers

**Como é gerado:**
```python
# Configuração do logger
logger = configurar_logger(
    nome_logger='AG.Main',
    diretorio_logs=str(outdir / 'logs'),
    nivel=logging.INFO,
    salvar_arquivo=True,
    formato_detalhado=True
)
```

**Evidências geradas:**
- **Arquivo de Log:** `evidencias/logs/ag_log_TIMESTAMP.log`
- **Formato:** Timestamp - Logger - Nível - [Arquivo:Linha] - Mensagem
- **Conteúdo:** Todos os logs estruturados da execução (INFO, WARNING, ERROR)

**Exemplo de log gerado:**
```
2026-01-19 21:57:37 - AG.Main - INFO - [main.py:190] - #AG Iniciando pipeline de treinamento - Modo: experimentos
2026-01-19 21:57:37 - AG.Main - INFO - [main.py:191] - #AG Dataset: heart.csv, Target: HeartDisease
2026-01-19 21:57:38 - AG.algoritmo - INFO - [ag_algoritmo.py:207] - #AG Geração 1/15 - Fitness: max=0.8840, médio=0.8605, min=0.8209 (0.67s)
```

### 2. Salvamento de Configuração do AG

#### Arquivo: `ag_logging.py`

**Função:** `salvar_configuracao_ag()`
- **Localização:** `ag_logging.py` linhas 85-114
- **Chamada:** Automaticamente pelo `AlgoritmoGenetico` após evolução

**Evidências geradas:**
- **Arquivo:** `evidencias/logs/ag_config_TIMESTAMP.json`
- **Formato:** JSON estruturado
- **Conteúdo:**
  ```json
  {
    "timestamp": "20260119_215737",
    "configuracao": {
      "tamanho_populacao": 20,
      "n_geracoes": 15,
      "taxa_cruzamento": 0.7,
      "taxa_mutacao": 0.1,
      "n_elites": 2,
      "metodo_selecao": "torneio",
      "metodo_cruzamento": "uniforme",
      "metodo_mutacao": "uniforme",
      "metric": "composite",
      "cv_folds": 5
    },
    "data_hora": "2026-01-19T21:57:37.123456"
  }
  ```

**Uso:** Rastreabilidade completa da configuração usada em cada execução

### 3. Histórico de Evolução do AG

#### Arquivo: `ag_logging.py`

**Função:** `salvar_historico_evolucao()`
- **Localização:** `ag_logging.py` linhas 117-161
- **Chamada:** Automaticamente pelo `AlgoritmoGenetico` após evolução

**Evidências geradas:**
- **Arquivo (com experimento):** `evidencias/logs/ag_historico_experimento_1_configuracao_conservadora_TIMESTAMP.json`
- **Arquivo (sem experimento):** `evidencias/logs/ag_historico_TIMESTAMP.json`
- **Formato:** JSON estruturado com histórico completo

**Conteúdo:**
```json
{
  "timestamp": "20260119_215737",
  "nome_experimento": "Experimento 1: Configuração Conservadora",
  "data_hora": "2026-01-19T21:57:37.123456",
  "historico": [
    {
      "geracao": 0,
      "fitness_medio": 0.8605,
      "fitness_max": 0.8840,
      "fitness_min": 0.8209,
      "tempo_segundos": 0.6726
    },
    ...
  ],
  "resumo": {
    "numero_geracoes": 15,
    "fitness_final_max": 0.8888,
    "fitness_final_medio": 0.8879,
    "melhor_fitness_geral": 0.8888
  }
}
```

**Uso:** Análise posterior da evolução do algoritmo, geração de gráficos, comparação entre experimentos

### 4. Gráficos de Evolução do Fitness

#### Arquivo: `ag_visualizacao.py`

**Funções:**
- `gerar_grafico_evolucao_fitness()` - linha 15
- `gerar_grafico_convergencia()` - linha 76
- `gerar_histograma_fitness()` - linha 130

**Chamada:** Automaticamente pelo `AlgoritmoGenetico` após evolução (via `ag_logging`)

**Evidências geradas:**
- **Evolução:** `evidencias/logs/ag_evolucao_fitness_experimento_1_configuracao_conservadora_TIMESTAMP.png`
- **Convergência:** `evidencias/logs/ag_convergencia_experimento_1_configuracao_conservadora_TIMESTAMP.png`
- **Distribuição:** `evidencias/logs/ag_distribuicao_fitness_experimento_1_configuracao_conservadora_TIMESTAMP.png`

**Conteúdo dos gráficos:**
1. **Evolução do Fitness:**
   - Linha verde: Fitness Máximo por geração
   - Linha azul: Fitness Médio por geração
   - Linha vermelha: Fitness Mínimo por geração
   - Área preenchida entre min e max
   - Marcador da melhor geração

2. **Convergência:**
   - Melhor fitness acumulado (nunca decresce)
   - Mostra quando o algoritmo convergiu

3. **Distribuição:**
   - Histograma do fitness na última geração
   - Mostra diversidade da população

**Qualidade:** 150 DPI, formato PNG, alta qualidade para apresentações

### 5. Gráficos de Desempenho por Experimento

#### Arquivo: `ag_experimentos.py`

**Função:** `_gerar_graficos_experimento()`
- **Localização:** `ag_experimentos.py` linhas 168-210
- **Chamada:** Automaticamente após comparação de modelos em cada experimento

**Evidências geradas (por experimento):**
- **Matriz de Confusão:** `evidencias/matriz_confusao_experimento_1_configuracao_conservadora_TIMESTAMP.png`
- **Curva ROC:** `evidencias/roc_curve_experimento_1_configuracao_conservadora_TIMESTAMP.png`

**Processo de geração:**
```python
# 1. Gera predições do modelo otimizado
y_pred = self.modelo_otimizado.predict(self.X_test)
y_prob = self.modelo_otimizado.predict_proba(self.X_test)[:, 1]

# 2. Matriz de Confusão
cm = confusion_matrix(self.y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Matriz de Confusão - {self.nome_experimento}")
plt.savefig(self.cm_png_path, dpi=140)

# 3. Curva ROC
RocCurveDisplay.from_predictions(self.y_test, y_prob)
plt.title(f"Curva ROC - {self.nome_experimento}")
plt.savefig(self.roc_png_path, dpi=140)
```

**Armazenamento:** Caminhos salvos em `exp.cm_png_path` e `exp.roc_png_path` para uso no PDF

**Qualidade:** 140 DPI, formato PNG

### 6. PDF Resumo com Todos os Gráficos

#### Arquivo: `main.py`

**Função:** `make_pdf_resumo()`
- **Localização:** `main.py` linhas 51-146
- **Chamada:** Ao final do pipeline principal

**Evidências geradas:**
- **Arquivo:** `evidencias/resumo_evidencias_TIMESTAMP.pdf`

**Estrutura do PDF:**

**Página 1 - Métricas Gerais:**
- Título e informações do dataset
- Métricas do modelo final (Accuracy, Precision, Recall, F1, AUC)
- Matriz de Confusão e Curva ROC do modelo final (lado a lado)

**Páginas Seguintes (Modo Experimentos):**
- **Uma página por experimento** contendo:
  - Título do experimento
  - Métricas do modelo otimizado (Accuracy, Recall, F1, AUC)
  - Matriz de Confusão (esquerda)
  - Curva ROC (direita)

**Exemplo de conteúdo:**
```
Página 1:
├─ Título: "Resumo de Evidências – Risco Cardíaco (Logistic Regression)"
├─ Dataset: heart.csv | Target: HeartDisease | Data: 20260119-215737
├─ Métricas:
│  ├─ ACCURACY: 0.8841
│  ├─ PRECISION: 0.8758
│  ├─ RECALL: 0.9216
│  ├─ F1: 0.8981
│  └─ AUC: 0.9325
├─ [Matriz de Confusão]  [Curva ROC]

Página 2+ (Modo Experimentos):
├─ Experimento 1: Configuração Conservadora
├─ Accuracy: 0.8551 | Recall: 0.9281 | F1: 0.8765 | AUC: 0.9318
├─ [Matriz CM]  [Curva ROC]
└─ ...

Página 3:
├─ Experimento 2: População Maior + Mais Gerações
├─ Accuracy: 0.8370 | Recall: 0.9412 | F1: 0.8649 | AUC: 0.9312
├─ [Matriz CM]  [Curva ROC]
└─ ...
```

**Tecnologia:** ReportLab (`reportlab.pdfgen.canvas`)

### 7. Relatório Comparativo de Experimentos

#### Arquivo: `ag_experimentos.py`

**Função:** `gerar_relatorio_comparativo()`
- **Localização:** `ag_experimentos.py` linhas 355-454
- **Chamada:** Automaticamente após execução de múltiplos experimentos

**Evidências geradas:**
- **Arquivo:** `evidencias/comparacao_experimentos_ag_TIMESTAMP.json`

**Conteúdo:**
```json
{
  "timestamp": "20260119-215737",
  "numero_experimentos": 4,
  "experimentos": [
    {
      "nome_experimento": "Experimento 1: Configuração Conservadora",
      "configuracao_ag": { ... },
      "hiperparametros_otimizados": { ... },
      "comparacao": {
        "modelo_original": { ... },
        "modelo_otimizado": { ... },
        "melhorias": { ... }
      },
      "historico_ag": [ ... ]
    },
    ...
  ],
  "melhores_por_metrica": { ... },
  "experimentos_ordenados_por_metricas": { ... }
}
```

**Uso:** Análise comparativa completa, geração de relatórios Markdown, apresentações

### 8. Arquivos de Evidências do Pipeline Principal

#### Arquivo: `main.py`

**Evidências geradas automaticamente:**
- `evidencias/evidencias_treinamento_TIMESTAMP.json` - Métricas e hiperparâmetros finais
- `evidencias/classification_report_TIMESTAMP.txt` - Relatório de classificação
- `evidencias/matriz_confusao_TIMESTAMP.png` - Matriz do modelo final
- `evidencias/roc_curve_TIMESTAMP.png` - ROC do modelo final
- `evidencias/coeficientes_TIMESTAMP.csv` - Coeficientes do modelo
- `evidencias/modelo_logreg_TIMESTAMP.joblib` - Modelo treinado (persistência)
- `evidencias/scaler_standard_TIMESTAMP.joblib` - Scaler usado (persistência)
- `evidencias/features_TIMESTAMP.json` - Lista de features
- `evidencias/evidencias_preprocess_TIMESTAMP.txt` - Log de pré-processamento

### 9. Fluxo Completo de Geração de Evidências

**Execução:** `python3 main.py --csv heart.csv --target HeartDisease --outdir evidencias --modo experimentos --num_experimentos 4`

**Fluxo:**
```
1. Inicialização
   └─> Configura logger principal (ag_logging.configurar_logger)
       └─> Gera: evidencias/logs/ag_log_TIMESTAMP.log

2. Para cada experimento:
   ├─> Executa AG (ag_algoritmo.AlgoritmoGenetico.evoluir)
   │   ├─> Para cada geração:
   │   │   ├─> Registra no histórico (em memória)
   │   │   └─> Log estruturado: "#AG Geração X/Y - Fitness: ..."
   │   │
   │   └─> Após evolução:
   │       ├─> Salva configuração (ag_logging.salvar_configuracao_ag)
   │       │   └─> Gera: evidencias/logs/ag_config_TIMESTAMP.json
   │       │
   │       ├─> Salva histórico (ag_logging.salvar_historico_evolucao)
   │       │   └─> Gera: evidencias/logs/ag_historico_EXPERIMENTO_TIMESTAMP.json
   │       │
   │       └─> Gera gráficos de evolução (ag_visualizacao.*)
   │           ├─> Gera: evidencias/logs/ag_evolucao_fitness_EXPERIMENTO_TIMESTAMP.png
   │           ├─> Gera: evidencias/logs/ag_convergencia_EXPERIMENTO_TIMESTAMP.png
   │           └─> Gera: evidencias/logs/ag_distribuicao_fitness_EXPERIMENTO_TIMESTAMP.png
   │
   ├─> Treina modelos (original e otimizado)
   │
   ├─> Compara modelos (ag_comparacao.comparar_modelos)
   │
   └─> Gera gráficos de desempenho (_gerar_graficos_experimento)
       ├─> Gera: evidencias/matriz_confusao_EXPERIMENTO_TIMESTAMP.png
       └─> Gera: evidencias/roc_curve_EXPERIMENTO_TIMESTAMP.png

3. Após todos os experimentos:
   └─> Gera relatório comparativo (ag_experimentos.gerar_relatorio_comparativo)
       └─> Gera: evidencias/comparacao_experimentos_ag_TIMESTAMP.json

4. Pipeline principal:
   ├─> Gera evidências do modelo final (main.py)
   │   ├─> Gera: evidencias/matriz_confusao_TIMESTAMP.png
   │   ├─> Gera: evidencias/roc_curve_TIMESTAMP.png
   │   ├─> Gera: evidencias/coeficientes_TIMESTAMP.csv
   │   ├─> Gera: evidencias/modelo_logreg_TIMESTAMP.joblib
   │   └─> Gera: evidencias/evidencias_treinamento_TIMESTAMP.json
   │
   └─> Gera PDF resumo (main.make_pdf_resumo)
       └─> Gera: evidencias/resumo_evidencias_TIMESTAMP.pdf
           └─> Inclui TODOS os gráficos de todos os experimentos
```

### 10. Estrutura de Diretórios de Evidências

**Estrutura típica após execução:**
```
evidencias/
├── logs/                                    # #AG Logs e configurações do AG
│   ├── ag_log_20260119_215737.log          # Log estruturado completo
│   ├── ag_config_experimento_1_*.json      # Configuração do AG (Exp 1)
│   ├── ag_config_experimento_2_*.json      # Configuração do AG (Exp 2)
│   ├── ag_config_experimento_3_*.json      # Configuração do AG (Exp 3)
│   ├── ag_config_experimento_4_*.json      # Configuração do AG (Exp 4)
│   ├── ag_historico_experimento_1_*.json   # Histórico completo (Exp 1)
│   ├── ag_historico_experimento_2_*.json   # Histórico completo (Exp 2)
│   ├── ag_historico_experimento_3_*.json   # Histórico completo (Exp 3)
│   ├── ag_historico_experimento_4_*.json   # Histórico completo (Exp 4)
│   ├── ag_evolucao_fitness_experimento_1_*.png  # Gráfico evolução (Exp 1)
│   ├── ag_convergencia_experimento_1_*.png      # Gráfico convergência (Exp 1)
│   ├── ag_distribuicao_fitness_experimento_1_*.png  # Gráfico distribuição (Exp 1)
│   └── ... (gráficos dos outros experimentos)
│
├── matriz_confusao_20260119-215737.png              # Matriz modelo final
├── roc_curve_20260119-215737.png                    # ROC modelo final
├── matriz_confusao_experimento_1_*.png              # Matriz Exp 1
├── roc_curve_experimento_1_*.png                    # ROC Exp 1
├── matriz_confusao_experimento_2_*.png              # Matriz Exp 2
├── roc_curve_experimento_2_*.png                    # ROC Exp 2
├── matriz_confusao_experimento_3_*.png              # Matriz Exp 3
├── roc_curve_experimento_3_*.png                    # ROC Exp 3
├── matriz_confusao_experimento_4_*.png              # Matriz Exp 4
├── roc_curve_experimento_4_*.png                    # ROC Exp 4
│
├── comparacao_experimentos_ag_20260119-215737.json  # Relatório comparativo
├── evidencias_treinamento_20260119-215737.json      # Métricas finais
├── resumo_evidencias_20260119-215737.pdf            # PDF com TODOS os gráficos ⭐
├── classification_report_20260119-215737.txt        # Relatório de classificação
├── coeficientes_20260119-215737.csv                 # Coeficientes do modelo
├── modelo_logreg_20260119-215737.joblib             # Modelo persistido
├── scaler_standard_20260119-215737.joblib           # Scaler persistido
├── features_20260119-215737.json                    # Lista de features
└── evidencias_preprocess_20260119-215737.txt        # Log de pré-processamento
```

### 11. Características de Rastreabilidade

**Timestamps consistentes:**
- Todos os arquivos de uma execução compartilham o mesmo timestamp base
- Formato: `YYYYMMDD-HHMMSS` (dados principais) ou `YYYYMMDD_HHMMSS` (logs)
- Facilita associação entre arquivos da mesma execução

**Nomes descritivos:**
- Incluem tipo de arquivo (config, historico, evolucao, etc.)
- Incluem nome do experimento (quando aplicável)
- Facilita identificação sem abrir arquivo

**Metadados completos:**
- JSONs incluem timestamp, data/hora ISO, resumo estatístico
- Logs incluem timestamp, nível, arquivo:linha, mensagem
- Rastreabilidade completa de cada execução

---

## Conclusão

**Nível atual:** ⭐⭐⭐⭐⭐ (5/5) - Monitoramento Profissional Completo

O sistema possui **monitoramento profissional completo** com:
- ✅ Logging estruturado com níveis (DEBUG, INFO, WARNING, ERROR)
- ✅ Persistência automática de configuração do AG
- ✅ Persistência automática de histórico de evolução
- ✅ Geração automática de gráficos de evolução (3 tipos)
- ✅ Geração automática de gráficos de desempenho por experimento
- ✅ PDF resumo consolidado com TODOS os gráficos
- ✅ Relatório comparativo estruturado em JSON
- ✅ Tracking de tempo por geração e total
- ✅ Rastreabilidade completa com timestamps consistentes
- ✅ Estrutura de diretórios organizada

**Arquivos gerados para logging e tracking:**
- 📝 Logs estruturados (.log)
- 📊 Históricos de evolução (.json)
- ⚙️ Configurações do AG (.json)
- 📈 Gráficos de evolução do fitness (.png)
- 📉 Gráficos de convergência (.png)
- 📊 Histogramas de distribuição (.png)
- 🎯 Matrizes de confusão por experimento (.png)
- 📈 Curvas ROC por experimento (.png)
- 📄 PDF resumo consolidado (.pdf)
- 📋 Relatório comparativo (.json)

**Recomendação:** Sistema de monitoramento adequado e completo para trabalho acadêmico e profissional, atendendo todos os requisitos de rastreabilidade e análise posterior.

