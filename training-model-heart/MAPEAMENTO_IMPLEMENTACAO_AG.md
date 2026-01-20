# Mapeamento da Implementação do Algoritmo Genético

Este documento indica **exatamente onde** cada componente do algoritmo genético está implementado no código.

---

## 1. Codificação Adequada (Representação de Genes)

**Arquivo:** `ag_criar_individuo.py`

### Localização da implementação:

#### Função principal: `criar_individuo()`

- **Linhas:** 11-54
- **Função:** Cria um indivíduo (solução candidata) representando hiperparâmetros do LogisticRegression

#### Estrutura dos genes (hiperparâmetros):

```python
individuo = {
    'C': float,              # Linha 26: intervalo [0.001, 100.0]
    'penalty': str,          # Linha 29: 'l1' ou 'l2'
    'class_weight': dict/str,# Linhas 32-41: None, 'balanced', ou {0:w0, 1:w1}
    'solver': str,           # Linha 50: 'liblinear' (fixo)
    'max_iter': int         # Linha 44: intervalo [100, 5000]
}
```

#### Funções auxiliares:

- `copiar_individuo()` - **Linha 57**: Cria cópia profunda de um indivíduo
- `validar_individuo()` - **Linha 71**: Valida estrutura e valores do indivíduo
- `corrigir_individuo()` - **Linha 95**: Corrige valores fora dos limites

---

## 2. Operadores de Seleção

**Arquivo:** `ag_selecao.py`

### Implementações:

#### 2.1. Seleção por Torneio

- **Função:** `selecao_torneio()`
- **Linhas:** 12-34
- **Descrição:** Seleciona k indivíduos aleatórios e escolhe o melhor (maior fitness)
- **Parâmetros:** `populacao_fitness`, `tamanho_torneio=3`

#### 2.2. Seleção por Roleta

- **Função:** `selecao_roleta()`
- **Linhas:** 37-61
- **Descrição:** Seleção proporcional ao fitness (maior fitness = maior probabilidade)
- **Parâmetros:** `populacao_fitness`

#### 2.3. Seleção por Elitismo

- **Função:** `selecao_elitismo()`
- **Linhas:** 64-75
- **Descrição:** Retorna os n melhores indivíduos da população
- **Parâmetros:** `populacao_fitness`, `n_elites=2`

#### 2.4. Seleção Mista

- **Função:** `selecao_mista()`
- **Linhas:** 78-107
- **Descrição:** Combina elitismo com outro método (torneio ou roleta)
- **Parâmetros:** `populacao_fitness`, `n_elites`, `tamanho_torneio`, `metodo`

#### 2.5. Seleção por Rank

- **Função:** `selecao_rank()`
- **Linhas:** 110-130
- **Descrição:** Seleciona baseado na posição (rank) do fitness, não no valor absoluto
- **Parâmetros:** `populacao_fitness`

---

## 3. Operadores de Cruzamento (Crossover)

**Arquivo:** `ag_cruzamento.py`

### Implementações:

#### 3.1. Cruzamento de Ponto Único

- **Função:** `cruzamento_ponto_unico()`
- **Linhas:** 12-59
- **Descrição:** Divide genes em duas metades e troca após o ponto de corte
- **Parâmetros:** `pai1`, `pai2`, `taxa_cruzamento=0.7`

#### 3.2. Cruzamento Uniforme

- **Função:** `cruzamento_uniforme()`
- **Linhas:** 62-99
- **Descrição:** Para cada gene, escolhe aleatoriamente de qual pai virá (50% cada)
- **Parâmetros:** `pai1`, `pai2`, `taxa_cruzamento=0.7`
- **Usado por padrão no AG principal**

#### 3.3. Cruzamento Aritmético

- **Função:** `cruzamento_aritmetico()`
- **Linhas:** 102-138
- **Descrição:** Para genes contínuos (C, max_iter), faz média ponderada dos pais
- **Parâmetros:** `pai1`, `pai2`, `taxa_cruzamento=0.7`, `alpha=0.5`

#### 3.4. Cruzamento BLX-α (Blend)

- **Função:** `cruzamento_blend()`
- **Linhas:** 141-204
- **Descrição:** Para genes contínuos, escolhe valor aleatório no intervalo expandido
- **Parâmetros:** `pai1`, `pai2`, `taxa_cruzamento=0.7`, `alpha=0.5`

---

## 4. Operadores de Mutação

**Arquivo:** `ag_mutacao.py`

### Implementações:

#### 4.1. Mutação Uniforme

- **Função:** `mutacao_uniforme()`
- **Linhas:** 12-66
- **Descrição:** Cada gene tem probabilidade independente de sofrer mutação
  - C: novo valor aleatório ou perturbação gaussiana (linha 27-35)
  - penalty: escolhe entre 'l1' ou 'l2' (linha 37-39)
  - class_weight: escolhe entre None, 'balanced' ou custom (linha 41-51)
  - max_iter: novo valor aleatório ou perturbação (linha 53-61)
- **Parâmetros:** `individuo`, `taxa_mutacao=0.1`, `forca_mutacao=0.1`
- **Usado por padrão no AG principal**

#### 4.2. Mutação Gaussiana

- **Função:** `mutacao_gaussiana()`
- **Linhas:** 69-115
- **Descrição:** Aplica perturbação gaussiana aos genes contínuos
- **Parâmetros:** `individuo`, `taxa_mutacao=0.1`, `sigma_fracao=0.1`

#### 4.3. Mutação Não-Uniforme

- **Função:** `mutacao_nao_uniforme()`
- **Linhas:** 118-177
- **Descrição:** Reduz intensidade da mutação ao longo das gerações
- **Parâmetros:** `individuo`, `geracao`, `max_geracoes`, `taxa_mutacao=0.1`, `b=2.0`

#### 4.4. Mutação Adaptativa

- **Função:** `mutacao_adaptativa()`
- **Linhas:** 180-194
- **Descrição:** Ajusta taxa de mutação baseado no fitness relativo
- **Parâmetros:** `individuo`, `fitness_relativo`, `taxa_mutacao_base=0.1`

---

## 5. Função Fitness

**Arquivo:** `ag_criar_fitness.py`

### Implementações:

#### 5.1. Função Fitness Principal

- **Função:** `fitness()`
- **Linhas:** 20-111
- **Descrição:** Calcula fitness de um indivíduo usando validação cruzada

#### 5.2. Métrica Composta (Padrão)

- **Localização:** Linhas 62-88
- **Fórmula:**
  ```
  Fitness = 0.20 × Accuracy + 0.30 × Recall + 0.25 × F1-Score + 0.25 × AUC
  ```
- **Implementação:**
  - Linha 64: `cross_val_score` para Accuracy
  - Linha 65: `cross_val_score` para Recall
  - Linha 66: `cross_val_score` para F1-Score
  - Linhas 69-79: Cálculo manual de AUC (usa `predict_proba`)
  - Linhas 83-87: Combinação ponderada das métricas

#### 5.3. Métrica AUC

- **Localização:** Linhas 90-103
- **Descrição:** Calcula apenas AUC via validação cruzada

#### 5.4. Métrica F1-Score

- **Localização:** Linha 105
- **Descrição:** Calcula apenas F1-Score via validação cruzada

#### 5.5. Métrica Recall

- **Localização:** Linha 108
- **Descrição:** Calcula apenas Recall via validação cruzada

#### 5.6. Função Fitness Detalhado

- **Função:** `fitness_detalhado()`
- **Linhas:** 114-197
- **Descrição:** Calcula todas as métricas individualmente para análise

---

## 6. Orquestração do Algoritmo Genético

**Arquivo:** `ag_algoritmo.py`

### Classe Principal: `AlgoritmoGenetico`

#### 6.1. Inicialização

- **Método:** `__init__()`
- **Linhas:** 21-66
- **Descrição:** Configura todos os parâmetros do AG

#### 6.2. Inicialização da População

- **Método:** `inicializar_populacao()`
- **Linhas:** 68-75
- **Descrição:** Gera população inicial chamando `criar_individuo()` N vezes

#### 6.3. Avaliação da População

- **Método:** `avaliar_populacao()`
- **Linhas:** 77-95
- **Descrição:** Calcula fitness de todos os indivíduos usando `fitness()`

#### 6.4. Seleção de Pais

- **Método:** `selecionar_pais()`
- **Linhas:** 97-113
- **Descrição:** Usa `selecao_torneio()` ou `selecao_roleta()`

#### 6.5. Aplicação de Cruzamento

- **Método:** `aplicar_cruzamento()`
- **Linhas:** 115-127
- **Descrição:** Usa `cruzamento_uniforme()`, `cruzamento_aritmetico()` ou `cruzamento_blend()`

#### 6.6. Aplicação de Mutação

- **Método:** `aplicar_mutacao()`
- **Linhas:** 129-141
- **Descrição:** Usa `mutacao_uniforme()`, `mutacao_gaussiana()` ou `mutacao_nao_uniforme()`

#### 6.7. Evolução Completa

- **Método:** `evoluir()`
- **Linhas:** 143-233
- **Descrição:** Orquestra todo o processo evolutivo:
  1. Inicializa população (linha 149)
  2. Loop por gerações (linha 151)
  3. Avalia população (linha 153)
  4. Ordena por fitness (linha 156)
  5. Aplica elitismo (linha 170)
  6. Gera nova população (linhas 172-185):
     - Seleção de pais (linha 175)
     - Cruzamento (linha 178)
     - Mutação (linhas 181-182)
  7. Retorna melhor indivíduo (linha 187)

---

## 7. Comparação com Modelo Original (Sem Otimização de Hiperparâmetros)

**Arquivo:** `ag_comparacao.py`

### 7.1. Treinamento do Modelo Original

#### Função: `treinar_modelo_original()`

- **Localização:** Linhas 20-43 em `ag_comparacao.py`
- **Descrição:** Treina um modelo LogisticRegression usando hiperparâmetros padrão (sem otimização)

#### Hiperparâmetros Padrão do Modelo Original:

```python
modelo_original = LogisticRegression(
    solver='liblinear',
    C=1.0,              # Valor padrão do scikit-learn
    penalty='l2',       # Penalidade padrão
    class_weight=None,  # Sem balanceamento de classes
    max_iter=1000,      # Iterações padrão
    random_state=42
)
```

**Importante:** Este modelo representa a **baseline** (linha de base) da Fase 1 do projeto, antes da otimização pelo algoritmo genético.

### 7.2. Avaliação de Modelos

#### Função: `avaliar_modelo()`

- **Localização:** Linhas 46-71 em `ag_comparacao.py`
- **Descrição:** Calcula todas as métricas de desempenho de um modelo

#### Métricas Calculadas:

- **Accuracy** (linha 62): Proporção de predições corretas
- **Precision** (linha 63): Precisão das predições positivas
- **Recall** (linha 64): Sensibilidade (proporção de positivos detectados)
- **F1-Score** (linha 65): Média harmônica de Precision e Recall
- **AUC** (linha 66): Area Under ROC Curve
- **Confusion Matrix** (linha 67): Matriz de confusão
- **Classification Report** (linha 68): Relatório detalhado

### 7.3. Função de Comparação

#### Função: `comparar_modelos()`

- **Localização:** Linhas 74-114 em `ag_comparacao.py`
- **Parâmetros:**
  - `modelo_original`: Modelo treinado com hiperparâmetros padrão
  - `modelo_otimizado`: Modelo treinado com hiperparâmetros encontrados pelo AG
  - `X_test`: Features de teste
  - `y_test`: Labels de teste

#### Processo de Comparação:

1. **Avalia modelo original** (linha 88): Calcula todas as métricas
2. **Avalia modelo otimizado** (linha 89): Calcula todas as métricas
3. **Calcula diferenças** (linhas 93-100): Para cada métrica:
   - Diferença absoluta: `diff = otimizado - original`
   - Melhoria percentual: `pct = (diff / original) × 100`
   - Identifica qual modelo é melhor
4. **Gera resumo** (linhas 106-111): Identifica qual modelo é melhor em cada métrica

#### Retorno da Função:

```python
{
    'modelo_original': {
        'accuracy': float,
        'precision': float,
        'recall': float,
        'f1': float,
        'auc': float,
        'confusion_matrix': list,
        'classification_report': dict
    },
    'modelo_otimizado': {
        # Mesmas chaves do modelo_original
    },
    'melhorias': {
        'accuracy': {'diferenca': float, 'percentual': float, 'melhor': str},
        'precision': {...},
        'recall': {...},
        'f1': {...},
        'auc': {...}
    },
    'resumo': {
        'melhor_accuracy': 'otimizado' ou 'original',
        'melhor_recall': 'otimizado' ou 'original',
        'melhor_f1': 'otimizado' ou 'original',
        'melhor_auc': 'otimizado' ou 'original'
    }
}
```

### 7.4. Exibição da Comparação

#### Função: `imprimir_comparacao()`

- **Localização:** Linhas 117-168 em `ag_comparacao.py`
- **Descrição:** Imprime tabela formatada comparando ambos os modelos

#### Saída Gerada:

```
================================================================================
#AG COMPARAÇÃO: MODELO ORIGINAL vs MODELO OTIMIZADO (ALGORITMO GENÉTICO)
================================================================================

#AG Hiperparâmetros Otimizados pelo AG:
#AG   C: 92.0762
#AG   penalty: l2
#AG   class_weight: {0: 0.55, 1: 1.53}
#AG   max_iter: 964

#AG Métricas de Desempenho:
--------------------------------------------------------------------------------
Métrica         Original     Otimizado    Diferença    Melhoria %
--------------------------------------------------------------------------------
Accuracy        0.8500       0.8800       +0.0300      +3.53%
Recall          0.8200       0.8600       +0.0400      +4.88%
F1-Score        0.8300       0.8700       +0.0400      +4.82%
AUC             0.9000       0.9200       +0.0200      +2.22%
--------------------------------------------------------------------------------

#AG Resumo:
#AG   Melhor Accuracy: otimizado
#AG   Melhor Recall: otimizado
#AG   Melhor F1-Score: otimizado
#AG   Melhor AUC: otimizado
================================================================================
```

### 7.5. Como Usar a Comparação no Código

#### Modo Comparação no `main.py`:

- **Localização:** Linhas 182-221 em `main.py`
- **Fluxo:**
  1. Executa o algoritmo genético para encontrar melhores hiperparâmetros (linhas 188-201)
  2. Treina modelo original com hiperparâmetros padrão (linha 204)
  3. Treina modelo otimizado com hiperparâmetros do AG (linhas 207-215)
  4. Compara ambos os modelos (linha 218)
  5. Exibe tabela comparativa (linha 219)

#### Exemplo de Uso:

```python
from ag_comparacao import treinar_modelo_original, comparar_modelos, imprimir_comparacao

# 1. Treina modelo original (sem otimização)
modelo_original = treinar_modelo_original(X_train, y_train)

# 2. Treina modelo otimizado (com hiperparâmetros do AG)
modelo_otimizado = LogisticRegression(
    C=melhor_individuo['C'],
    penalty=melhor_individuo['penalty'],
    class_weight=melhor_individuo['class_weight'],
    max_iter=melhor_individuo['max_iter']
)
modelo_otimizado.fit(X_train, y_train)

# 3. Compara modelos
comparacao = comparar_modelos(modelo_original, modelo_otimizado, X_test, y_test)

# 4. Exibe resultados
imprimir_comparacao(comparacao, melhor_individuo)
```

### 7.6. Diferenças-Chave entre Modelos

#### Modelo Original (Baseline):

- **C**: 1.0 (valor padrão)
- **penalty**: 'l2' (padrão)
- **class_weight**: None (sem balanceamento)
- **max_iter**: 1000 (padrão)
- **Não passa por otimização**

#### Modelo Otimizado pelo AG:

- **C**: Valor otimizado (ex: 0.5 a 100.0)
- **penalty**: 'l1' ou 'l2' (escolhido pelo AG)
- **class_weight**: None, 'balanced' ou pesos customizados (otimizado)
- **max_iter**: Valor otimizado (100 a 5000)
- **Passa por evolução genética (seleção, cruzamento, mutação)**

### 7.7. Interpretação dos Resultados

#### Melhoria Positiva (`+`):

- Indica que o modelo otimizado tem melhor desempenho que o original
- Exemplo: `+4.88%` em Recall significa que o modelo otimizado detecta 4.88% mais casos positivos

#### Melhoria Negativa (`-`):

- Indica que o modelo otimizado tem pior desempenho que o original
- Pode ocorrer em algumas métricas, mas o objetivo é melhorar no geral

#### Modelo Melhor:

- Identificado no campo `'melhor'` de cada métrica
- O resumo final mostra qual modelo vence em cada métrica
- Objetivo: modelo otimizado deve vencer na maioria das métricas, especialmente Recall (crítico em diagnóstico médico)

### 7.8. Integração nos Experimentos

#### Localização: `ag_experimentos.py`

- **Classe:** `ExperimentoAG`
- **Método:** `executar()` - Linha 69
- **Fluxo:**
  1. Executa algoritmo genético (linha 125)
  2. Treina modelo original (linha 132)
  3. Treina modelo otimizado (linhas 135-144)
  4. Compara modelos (linhas 147-152)
  5. Cada experimento gera sua própria comparação

**Nota:** Todos os experimentos realizam automaticamente a comparação com o modelo original, permitindo avaliar o impacto de diferentes configurações do AG.

---

## 8. Integração no Pipeline Principal

**Arquivo:** `main.py`

### Localização da execução do AG:

#### Modo Experimentos (Padrão)

- **Linhas:** 154-175
- **Função:** `executar_multiplos_experimentos()` de `ag_experimentos.py`
- **Comparação:** Cada experimento compara automaticamente com modelo original

#### Modo Comparação

- **Linhas:** 182-221
- **Função:** `executar_ag()` de `ag_algoritmo.py` + `comparar_modelos()`
- **Comparação:** Exibe tabela detalhada comparando modelo original vs otimizado

#### Modo Simples

- **Linhas:** 223-246
- **Função:** `executar_ag()` de `ag_algoritmo.py`
- **Comparação:** Não faz comparação explícita (apenas otimiza)

#### Uso do melhor indivíduo para treinamento

- **Linhas:** 290-297
- **Descrição:** Treina modelo final com hiperparâmetros do melhor indivíduo encontrado

---

## 9. Resumo Visual

```
MAIN.PY
  ├─> ag_experimentos.py (múltiplos experimentos + comparação)
  │     └─> ag_algoritmo.py (classe AlgoritmoGenetico)
  │           ├─> ag_criar_individuo.py (criar_individuo)
  │           ├─> ag_criar_fitness.py (fitness)
  │           ├─> ag_selecao.py (seleção de pais)
  │           ├─> ag_cruzamento.py (cruzamento)
  │           └─> ag_mutacao.py (mutação)
  │
  └─> ag_comparacao.py (comparação com modelo original)
        ├─> treinar_modelo_original()  # Modelo baseline (Fase 1)
        ├─> avaliar_modelo()           # Calcula métricas
        ├─> comparar_modelos()         # Compara original vs otimizado
        └─> imprimir_comparacao()      # Exibe tabela comparativa
```

**Fluxo Completo de Comparação:**

```
1. Treina modelo original (C=1.0, penalty='l2', class_weight=None)
   ↓
2. Executa AG para encontrar melhores hiperparâmetros
   ↓
3. Treina modelo otimizado (com hiperparâmetros do AG)
   ↓
4. Avalia ambos os modelos no conjunto de teste
   ↓
5. Calcula diferenças e melhorias percentuais
   ↓
6. Exibe tabela comparativa formatada
```

---

## 10. Parâmetros Configuráveis (Valores Padrão)

Definidos em `ag_algoritmo.py` (linhas 23-33):

- `tamanho_populacao = 20`
- `n_geracoes = 15`
- `taxa_cruzamento = 0.7`
- `taxa_mutacao = 0.1`
- `n_elites = 2`
- `metodo_selecao = 'torneio'`
- `metodo_cruzamento = 'uniforme'`
- `metodo_mutacao = 'uniforme'`
- `metric = 'composite'` (20% accuracy + 30% recall + 25% f1 + 25% AUC)
- `cv_folds = 5`

---

## 11. Como Verificar cada Componente

### Verificar codificação (genes):

```bash
python -c "from ag_criar_individuo import criar_individuo; print(criar_individuo())"
```

### Verificar seleção:

```bash
# Ver código em ag_selecao.py linha 12
```

### Verificar cruzamento:

```bash
# Ver código em ag_cruzamento.py linha 62
```

### Verificar mutação:

```bash
# Ver código em ag_mutacao.py linha 12
```

### Verificar fitness:

```bash
# Ver código em ag_criar_fitness.py linha 20, especialmente linhas 83-87
```

### Verificar comparação com modelo original:

```python
# Exemplo de código para testar comparação
from ag_comparacao import treinar_modelo_original, comparar_modelos, imprimir_comparacao
from sklearn.linear_model import LogisticRegression
import numpy as np

# Dados fictícios (substituir por dados reais)
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)
X_test = np.random.rand(30, 10)
y_test = np.random.randint(0, 2, 30)

# Modelo original
modelo_original = treinar_modelo_original(X_train, y_train)

# Modelo otimizado (com hiperparâmetros fictícios)
modelo_otimizado = LogisticRegression(
    solver='liblinear',
    C=0.5,
    penalty='l1',
    class_weight='balanced',
    max_iter=2000,
    random_state=42
)
modelo_otimizado.fit(X_train, y_train)

# Compara
comparacao = comparar_modelos(modelo_original, modelo_otimizado, X_test, y_test)

# Exibe
imprimir_comparacao(comparacao, {'C': 0.5, 'penalty': 'l1', 'class_weight': 'balanced', 'max_iter': 2000})
```

### Executar comparação via main.py:

```bash
# Modo que executa AG e compara automaticamente com modelo original
python main.py --csv heart.csv --target HeartDisease --outdir evidencias --modo comparar
```

---

**Todas as funções e classes estão marcadas com comentário `#AG` para fácil identificação no código.**
