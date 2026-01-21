# Guia Técnico Completo: Algoritmo Genético para Otimização de Hiperparâmetros

## 📚 Índice

1. [Fundamentos do Algoritmo Genético](#1-fundamentos-do-algoritmo-genético)
2. [Aplicação ao Problema de Otimização de Hiperparâmetros](#2-aplicação-ao-problema-de-otimização-de-hiperparâmetros)
3. [Codificação: Representação de Genes](#3-codificação-representação-de-genes)
4. [Função Fitness: Avaliação de Qualidade](#4-função-fitness-avaliação-de-qualidade)
5. [Operadores Genéticos: Seleção](#5-operadores-genéticos-seleção)
6. [Operadores Genéticos: Cruzamento](#6-operadores-genéticos-cruzamento)
7. [Operadores Genéticos: Mutação](#7-operadores-genéticos-mutação)
8. [Ciclo Evolutivo Completo](#8-ciclo-evolutivo-completo)
9. [Parâmetros e Configuração](#9-parâmetros-e-configuração)
10. [Exemplos Práticos](#10-exemplos-práticos)

---

## 1. Fundamentos do Algoritmo Genético

### 1.1. O que é um Algoritmo Genético?

Um **Algoritmo Genético (AG)** é uma técnica de otimização baseada nos princípios da evolução natural de Charles Darwin. Ele simula o processo de seleção natural onde:

- **Indivíduos** (soluções candidatas) competem pela sobrevivência
- Os **melhores** (com maior fitness) têm maior chance de se reproduzir
- A **reprodução** cria novas soluções combinando características dos pais
- **Mutações** introduzem diversidade genética
- Ao longo das **gerações**, a população evolui para soluções melhores

### 1.2. Analogia com a Evolução Natural

| Evolução Natural | Algoritmo Genético |
|------------------|-------------------|
| Indivíduo (ser vivo) | Solução candidata (conjunto de hiperparâmetros) |
| Genótipo (DNA) | Codificação (dicionário de hiperparâmetros) |
| Fenótipo (características físicas) | Modelo treinado com os hiperparâmetros |
| Aptidão (sobrevivência e reprodução) | Fitness (desempenho do modelo) |
| Seleção natural | Operador de seleção |
| Reprodução sexual | Operador de cruzamento |
| Mutação genética | Operador de mutação |
| Gerações | Iterações do algoritmo |

### 1.3. Por que usar AG para Otimização de Hiperparâmetros?

**Vantagens:**
- ✅ Não requer gradientes (funciona com problemas não-diferenciáveis)
- ✅ Explora múltiplas regiões do espaço de busca simultaneamente (população)
- ✅ Não fica preso em mínimos locais (diversidade genética)
- ✅ Funciona com espaços de busca grandes e complexos
- ✅ Pode otimizar múltiplos objetivos simultaneamente

**Desvantagens:**
- ⚠️ Pode ser computacionalmente custoso (muitas avaliações)
- ⚠️ Não garante solução ótima global
- ⚠️ Requer ajuste de parâmetros do próprio AG

---

## 2. Aplicação ao Problema de Otimização de Hiperparâmetros

### 2.1. O Problema

Queremos encontrar os **melhores hiperparâmetros** para o modelo `LogisticRegression` que maximizem o desempenho em diagnóstico cardíaco.

**Hiperparâmetros a otimizar:**
- `C`: Parâmetro de regularização (inverso da força)
- `penalty`: Tipo de penalidade ('l1' ou 'l2')
- `class_weight`: Balanceamento de classes
- `max_iter`: Número máximo de iterações

### 2.2. Espaço de Busca

O **espaço de busca** é o conjunto de todas as combinações possíveis:

```
C: [0.001, 100.0] (contínuo)
penalty: {'l1', 'l2'} (categórico)
class_weight: {None, 'balanced', {0: w0, 1: w1}} (complexo)
max_iter: [100, 5000] (inteiro)
```

**Tamanho do espaço:** Praticamente infinito devido ao contínuo!

**Solução:** O AG não testa todas as combinações, mas **explora** o espaço de forma inteligente.

### 2.3. Fluxo Geral do AG

```
┌─────────────────┐
│ 1. Inicializar  │ → Gera população inicial (20 indivíduos aleatórios)
│   População     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Avaliar      │ → Calcula fitness de cada indivíduo
│   População     │    (treina modelo e mede desempenho)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Selecionar   │ → Escolhe pais para reprodução
│   Pais          │    (favorável aos melhores)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Cruzar       │ → Combina características dos pais
│   (Crossover)   │    (gera filhos)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Mutar        │ → Introduz variação aleatória
│   (Mutation)    │    (diversidade)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. Elitismo     │ → Preserva melhores indivíduos
│                 │    (garante convergência)
└────────┬────────┘
         │
         ▼
    ┌────┴────┐
    │ Próxima │
    │ Geração │
    └────┬────┘
         │
         ▼
    [Repete 15 vezes]
         │
         ▼
┌─────────────────┐
│ Retorna Melhor  │ → Melhor indivíduo encontrado
│   Indivíduo     │
└─────────────────┘
```

---

## 3. Codificação: Representação de Genes

### 3.1. O que é Codificação?

A **codificação** é como representamos uma solução candidata (hiperparâmetros) em formato que o AG possa manipular. É como o **DNA** do indivíduo.

### 3.2. Nossa Codificação: Dicionário Python

Escolhemos representar um **indivíduo** como um dicionário Python:

```python
individuo = {
    'C': 1.5234,                    # Gene 1: Regularização (float)
    'penalty': 'l2',                # Gene 2: Tipo de penalidade (string)
    'class_weight': 'balanced',     # Gene 3: Balanceamento (complexo)
    'solver': 'liblinear',          # Gene 4: Fixo (não otimizado)
    'max_iter': 1000                # Gene 5: Iterações (inteiro)
}
```

### 3.3. Por que Esta Codificação?

**Vantagens:**
1. **Intuitiva:** Mapeamento direto para parâmetros do scikit-learn
2. **Flexível:** Suporta diferentes tipos (float, string, dict)
3. **Extensível:** Fácil adicionar novos hiperparâmetros
4. **Eficiente:** Operações de cópia e modificação são diretas

**Alternativas consideradas:**
- **Array NumPy:** Seria mais eficiente, mas menos legível
- **Tupla:** Menos flexível para valores complexos
- **Classe:** Mais verboso, sem ganho significativo

### 3.4. Inicialização Aleatória

A função `criar_individuo()` gera indivíduos aleatórios:

```python
def criar_individuo():
    # C: valor aleatório no intervalo [0.001, 100.0]
    C = np.random.uniform(0.001, 100.0)
    
    # penalty: escolhe aleatoriamente entre 'l1' ou 'l2'
    penalty = np.random.choice(['l1', 'l2'])
    
    # class_weight: escolhe entre None, 'balanced', ou pesos customizados
    class_weight_option = np.random.choice(['None', 'balanced', 'custom'])
    if class_weight_option == 'None':
        class_weight = None
    elif class_weight_option == 'balanced':
        class_weight = 'balanced'
    else:
        # Pesos customizados entre 0.5 e 2.0
        w1 = np.random.uniform(0.5, 2.0)
        w0 = np.random.uniform(0.5, 2.0)
        class_weight = {0: w0, 1: w1}
    
    # max_iter: inteiro aleatório entre 100 e 5000
    max_iter = int(np.random.uniform(100, 5000))
    
    return {
        'C': C,
        'penalty': penalty,
        'class_weight': class_weight,
        'solver': 'liblinear',
        'max_iter': max_iter
    }
```

**Por que uniforme?**
- Para `C`: Explora todo o intervalo igualmente no início
- Para `penalty`: 50% chance de cada opção
- Para `max_iter`: Distribuição uniforme no intervalo

---

## 4. Função Fitness: Avaliação de Qualidade

### 4.1. O que é Fitness?

O **fitness** é uma medida numérica que indica **quão boa é uma solução**. Quanto maior o fitness, melhor a solução.

**No nosso caso:** Fitness = desempenho do modelo com esses hiperparâmetros.

### 4.2. Como Calculamos o Fitness?

**Processo:**
1. Criamos um modelo `LogisticRegression` com os hiperparâmetros do indivíduo
2. Treinamos usando **validação cruzada estratificada** (5 folds)
3. Calculamos métricas em cada fold
4. Retornamos a média das métricas

**Por que validação cruzada?**
- ✅ Evita overfitting (não usa todos os dados para treinar)
- ✅ Dá estimativa mais robusta do desempenho
- ✅ Usa dados de treino de forma eficiente

### 4.3. Função Fitness Composta

Não queremos otimizar apenas uma métrica, mas uma **combinação**:

```python
Fitness = 0.20 × Accuracy + 0.30 × Recall + 0.25 × F1-Score + 0.25 × AUC
```

**Pesos escolhidos:**
- **Accuracy (20%):** Mede acurácia geral
- **Recall (30%):** **PRIORIDADE** - Em diagnóstico médico, falsos negativos são críticos!
- **F1-Score (25%):** Balanceia precision e recall
- **AUC (25%):** Mede capacidade de distinguir classes

**Por que Recall tem peso maior?**
No diagnóstico cardíaco, **não detectar doença quando ela existe** (falso negativo) é mais grave que detectar doença quando não existe (falso positivo). Por isso, priorizamos Recall.

### 4.4. Implementação da Função Fitness

```python
def fitness(individuo, X, y, cv_folds=5, metric='composite'):
    # 1. Cria modelo com hiperparâmetros do indivíduo
    model = LogisticRegression(
        C=individuo['C'],
        penalty=individuo['penalty'],
        class_weight=individuo['class_weight'],
        solver=individuo['solver'],
        max_iter=individuo['max_iter']
    )
    
    # 2. Validação cruzada estratificada
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True)
    
    # 3. Calcula métricas
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    
    # AUC precisa de probabilidades, então calcula manualmente
    auc_scores = []
    for train_idx, val_idx in skf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_pred_proba = model.predict_proba(X[val_idx])[:, 1]
        auc = roc_auc_score(y[val_idx], y_pred_proba)
        auc_scores.append(auc)
    
    # 4. Combinação ponderada
    fitness_value = (
        0.20 * np.mean(accuracy_scores) +
        0.30 * np.mean(recall_scores) +
        0.25 * np.mean(f1_scores) +
        0.25 * np.mean(auc_scores)
    )
    
    return fitness_value
```

**Complexidade:** O(n × m × k), onde:
- n = número de amostras
- m = número de features
- k = número de folds (5)

**Tempo estimado:** ~2-5 segundos por indivíduo (depende do tamanho do dataset)

---

## 5. Operadores Genéticos: Seleção

### 5.1. Objetivo da Seleção

A **seleção** escolhe quais indivíduos da população atual serão **pais** para gerar a próxima geração. Deve favorecer indivíduos com **maior fitness**, mas também permitir que indivíduos medianos participem (diversidade).

### 5.2. Seleção por Torneio (Implementada)

**Como funciona:**
1. Escolhe aleatoriamente `k` indivíduos da população (k = tamanho do torneio)
2. Seleciona o **melhor** entre esses k (maior fitness)

**Exemplo:**
```
População (ordenada por fitness):
[Indivíduo A: 0.90, Indivíduo B: 0.85, Indivíduo C: 0.80, ..., Indivíduo Z: 0.50]

Torneio (k=3):
- Escolhe aleatoriamente: C, M, P
- Fitness: C=0.80, M=0.65, P=0.70
- Vencedor: C (maior fitness)
```

**Vantagens:**
- ✅ Simples e eficiente
- ✅ Controla pressão seletiva (maior k = mais seletivo)
- ✅ Não precisa normalizar fitness
- ✅ Permite que indivíduos medianos vençam (se não competirem com os melhores)

**Pressão Seletiva:**
- `k=1`: Sem seleção (aleatório)
- `k=2`: Leve seleção
- `k=3`: Moderada (usamos este)
- `k=tamanho_populacao`: Seleção total (sempre o melhor)

**Código:**
```python
def selecao_torneio(populacao_fitness, tamanho_torneio=3):
    # Escolhe k competidores aleatórios
    competidores = np.random.choice(len(populacao_fitness), tamanho_torneio, replace=False)
    
    # Encontra o melhor entre os competidores
    melhor_idx = competidores[0]
    melhor_fitness = populacao_fitness[melhor_idx][1]
    
    for idx in competidores[1:]:
        if populacao_fitness[idx][1] > melhor_fitness:
            melhor_fitness = populacao_fitness[idx][1]
            melhor_idx = idx
    
    return copiar_individuo(populacao_fitness[melhor_idx][0])
```

### 5.3. Seleção por Roleta (Alternativa Implementada)

**Como funciona:**
1. Converte fitness em probabilidades (proporcionais)
2. Seleciona indivíduo baseado nessas probabilidades

**Exemplo:**
```
População:
A: fitness=0.90 → probabilidade = 0.90 / soma_total
B: fitness=0.85 → probabilidade = 0.85 / soma_total
C: fitness=0.80 → probabilidade = 0.80 / soma_total
```

**Vantagens:**
- ✅ Representa qualidade relativa naturalmente
- ✅ Permite que indivíduos com fitness intermediário também sejam selecionados

**Desvantagens:**
- ⚠️ Requer fitness positivo
- ⚠️ Indivíduos muito melhores dominam a seleção

**Quando usar:**
- Roleta: Quando diferenças de fitness são importantes
- Torneio: Quando queremos mais controle sobre a pressão seletiva

### 5.4. Elitismo

**O que é:**
Preserva os **n melhores** indivíduos da população atual na próxima geração, sem modificação.

**Por que usar:**
- ✅ Garante que o melhor fitness nunca piora
- ✅ Acelera convergência
- ✅ Evita perder boas soluções por "azar" na seleção

**Implementação:**
```python
def selecao_elitismo(populacao_fitness, n_elites=2):
    # Ordena por fitness (decrescente)
    populacao_ordenada = sorted(populacao_fitness, key=lambda x: x[1], reverse=True)
    
    # Retorna os n melhores
    elites = [copiar_individuo(pf[0]) for pf in populacao_ordenada[:n_elites]]
    return elites
```

**Quantos elites usar?**
- Poucos (1-2): Preserva diversidade
- Muitos (>20%): Reduz diversidade (pode estagnar)

**Usamos:** `n_elites=2` (10% da população de 20)

---

## 6. Operadores Genéticos: Cruzamento

### 6.1. Objetivo do Cruzamento

O **cruzamento (crossover)** combina características de dois pais para gerar **filhos** que herdam características dos dois. Isso permite **explorar combinações novas** que não existiam nos pais.

### 6.2. Cruzamento Uniforme (Implementado)

**Como funciona:**
Para cada gene (hiperparâmetro), escolhe aleatoriamente de qual pai virá (50% de chance de cada pai).

**Exemplo:**
```
Pai 1: {C: 1.0, penalty: 'l1', class_weight: 'balanced', max_iter: 1000}
Pai 2: {C: 2.0, penalty: 'l2', class_weight: None, max_iter: 2000}

Cruzamento uniforme:
- C: Escolhe Pai 2 → C = 2.0
- penalty: Escolhe Pai 1 → penalty = 'l1'
- class_weight: Escolhe Pai 1 → class_weight = 'balanced'
- max_iter: Escolhe Pai 2 → max_iter = 2000

Filho: {C: 2.0, penalty: 'l1', class_weight: 'balanced', max_iter: 2000}
```

**Vantagens:**
- ✅ Simples e eficiente
- ✅ Explora espaço de busca de forma ampla
- ✅ Funciona bem com genes de tipos diferentes

**Taxa de Cruzamento:**
- `taxa_cruzamento = 0.7` significa 70% de chance de cruzamento ocorrer
- 30% dos pares geram filhos idênticos aos pais (exploitation)
- 70% geram filhos novos (exploration)

**Implementação:**
```python
def cruzamento_uniforme(pai1, pai2, taxa_cruzamento=0.7):
    if np.random.rand() > taxa_cruzamento:
        return copiar_individuo(pai1), copiar_individuo(pai2)
    
    filho1 = copiar_individuo(pai1)
    filho2 = copiar_individuo(pai2)
    
    # Para cada gene, decide aleatoriamente se troca
    if np.random.rand() < 0.5:
        filho1['C'], filho2['C'] = filho2['C'], filho1['C']
    
    # ... mesmo processo para outros genes
    
    return filho1, filho2
```

### 6.3. Cruzamento Aritmético (Alternativa)

**Como funciona:**
Para genes contínuos (C, max_iter), faz **média ponderada** dos valores dos pais.

**Exemplo:**
```
Pai 1: C = 1.0
Pai 2: C = 3.0
α = 0.5 (média simples)

Filho 1 = α × Pai1 + (1-α) × Pai2 = 0.5×1.0 + 0.5×3.0 = 2.0
Filho 2 = (1-α) × Pai1 + α × Pai2 = 0.5×1.0 + 0.5×3.0 = 2.0
```

**Vantagens:**
- ✅ Explora região entre os pais
- ✅ Útil para genes contínuos
- ✅ Pode gerar valores intermediários melhores

**Quando usar:**
- Genes contínuos: Aritmético
- Genes categóricos: Uniforme

### 6.4. Cruzamento BLX-α (Blend)

**Como funciona:**
Expande o intervalo entre os valores dos pais e escolhe um valor aleatório dentro desse intervalo expandido.

**Exemplo:**
```
Pai 1: C = 1.0
Pai 2: C = 3.0
α = 0.5

Intervalo original: [1.0, 3.0]
Tamanho: 2.0
Expansão: α × tamanho = 0.5 × 2.0 = 1.0

Intervalo expandido: [1.0 - 1.0, 3.0 + 1.0] = [0.0, 4.0]
Escolhe valor aleatório neste intervalo
```

**Vantagens:**
- ✅ Explora além do intervalo dos pais
- ✅ Boa para problemas onde o ótimo está entre pais

---

## 7. Operadores Genéticos: Mutação

### 7.1. Objetivo da Mutação

A **mutação** introduz **variação aleatória** nos indivíduos, garantindo:
- **Diversidade genética** na população
- **Exploração** de novas regiões do espaço de busca
- Evita **estagnação** (população muito similar)

### 7.2. Mutação Uniforme (Implementada)

**Como funciona:**
Cada gene tem uma probabilidade independente (`taxa_mutacao`) de sofrer mutação.

**Para genes contínuos (C, max_iter):**
- Opção 1: Perturbação gaussiana (pequena alteração)
- Opção 2: Valor completamente novo (grande alteração)

**Para genes categóricos (penalty, class_weight):**
- Escolhe novo valor aleatório do conjunto válido

**Exemplo:**
```
Indivíduo original:
{C: 1.5, penalty: 'l1', class_weight: 'balanced', max_iter: 1000}

Mutação (taxa_mutacao = 0.1):
- C: 10% chance → Mutou! Novo valor: 2.3 (perturbação gaussiana)
- penalty: 10% chance → Não mutou (mantém 'l1')
- class_weight: 10% chance → Mutou! Novo valor: None
- max_iter: 10% chance → Não mutou (mantém 1000)

Indivíduo mutado:
{C: 2.3, penalty: 'l1', class_weight: None, max_iter: 1000}
```

**Taxa de Mutação:**
- `taxa_mutacao = 0.1` (10%) significa cada gene tem 10% de chance de mutar
- **Baixa taxa (0.05-0.1):** Mutação sutil (refinamento)
- **Alta taxa (0.2-0.3):** Mutação forte (exploração agressiva)

**Por que baixa taxa?**
Mutação muito alta pode destruir boas soluções. Preferimos:
- Cruzamento para criar novas soluções (mais controlado)
- Mutação para pequenos ajustes (baixa taxa)

**Implementação:**
```python
def mutacao_uniforme(individuo, taxa_mutacao=0.1, forca_mutacao=0.1):
    individuo_mutado = copiar_individuo(individuo)
    
    # Mutação em C
    if np.random.rand() < taxa_mutacao:
        if np.random.rand() < 0.5:
            # Perturbação gaussiana
            delta = np.random.normal(0, forca_mutacao * individuo_mutado['C'])
            individuo_mutado['C'] = individuo_mutado['C'] + delta
        else:
            # Valor completamente novo
            individuo_mutado['C'] = np.random.uniform(0.001, 100.0)
    
    # ... mesmo processo para outros genes
    
    return individuo_mutado
```

### 7.3. Mutação Gaussiana (Alternativa)

**Como funciona:**
Aplica perturbação gaussiana aos genes contínuos. O desvio padrão é uma fração do valor atual.

**Exemplo:**
```
C atual: 1.5
σ_fração = 0.1
σ = 0.1 × 1.5 = 0.15

Perturbação: N(0, 0.15) → delta = +0.23 (exemplo)
C novo: 1.5 + 0.23 = 1.73
```

**Vantagens:**
- ✅ Mutação proporcional ao valor atual
- ✅ Pequenos valores sofrem mutação pequena
- ✅ Grandes valores podem sofrer mutação maior

### 7.4. Mutação Não-Uniforme

**Como funciona:**
Reduz a intensidade da mutação ao longo das gerações.

**Início (geração 0):**
- Mutação grande → Exploração ampla

**Fim (geração final):**
- Mutação pequena → Refinamento fino

**Vantagens:**
- ✅ Balanceia exploração (início) e exploitation (fim)
- ✅ Converge suavemente para soluções refinadas

**Fórmula:**
```
fator_reducao = (1 - r^((1 - g/G)^b))
onde:
- r: aleatório [0, 1]
- g: geração atual
- G: total de gerações
- b: parâmetro de formato (ex: 2.0)
```

---

## 8. Ciclo Evolutivo Completo

### 8.1. Algoritmo Completo (Passo a Passo)

```python
# 1. INICIALIZAÇÃO
populacao = [criar_individuo() for _ in range(20)]  # 20 indivíduos aleatórios

# 2. LOOP POR GERAÇÕES (15 gerações)
for geracao in range(15):
    
    # 2.1. AVALIAÇÃO
    populacao_fitness = []
    for individuo in populacao:
        fitness_value = fitness(individuo, X_train, y_train)  # Treina e avalia
        populacao_fitness.append((individuo, fitness_value))
    
    # 2.2. ORDENAÇÃO
    populacao_fitness.sort(key=lambda x: x[1], reverse=True)  # Melhor primeiro
    
    # 2.3. REGISTRA HISTÓRICO
    fitness_max = populacao_fitness[0][1]
    fitness_medio = np.mean([pf[1] for pf in populacao_fitness])
    fitness_min = populacao_fitness[-1][1]
    historico.append({'geracao': geracao, 'fitness_max': fitness_max, ...})
    
    # 2.4. ELITISMO
    nova_populacao = selecao_elitismo(populacao_fitness, n_elites=2)  # Preserva 2 melhores
    
    # 2.5. GERA NOVA POPULAÇÃO
    while len(nova_populacao) < 20:
        # Seleção de pais
        pai1 = selecao_torneio(populacao_fitness, k=3)
        pai2 = selecao_torneio(populacao_fitness, k=3)
        
        # Cruzamento
        filho1, filho2 = cruzamento_uniforme(pai1, pai2, taxa=0.7)
        
        # Mutação
        filho1 = mutacao_uniforme(filho1, taxa=0.1)
        filho2 = mutacao_uniforme(filho2, taxa=0.1)
        
        # Adiciona à nova população
        nova_populacao.append(filho1)
        if len(nova_populacao) < 20:
            nova_populacao.append(filho2)
    
    # 2.6. SUBSTITUI POPULAÇÃO
    populacao = nova_populacao

# 3. RETORNA MELHOR INDIVÍDUO
melhor_individuo = populacao_fitness[0][0]  # Melhor da última geração
return melhor_individuo
```

### 8.2. Visualização do Processo

**Geração 0 (Inicial):**
```
População: [Ind1: 0.65, Ind2: 0.72, Ind3: 0.68, ..., Ind20: 0.61]
Melhor: 0.72
Médio: 0.67
```

**Geração 1:**
```
População: [Ind1: 0.75, Ind2: 0.72 (elite), Ind3: 0.71, ..., Ind20: 0.64]
Melhor: 0.75  ← Melhorou!
Médio: 0.69   ← Melhorou!
```

**Geração 5:**
```
População: [Ind1: 0.82, Ind2: 0.81 (elite), Ind3: 0.80, ..., Ind20: 0.72]
Melhor: 0.82  ← Melhorou!
Médio: 0.77   ← Melhorou!
```

**Geração 15 (Final):**
```
População: [Ind1: 0.89, Ind2: 0.88 (elite), Ind3: 0.87, ..., Ind20: 0.82]
Melhor: 0.89  ← MELHOR ENCONTRADO
Médio: 0.85
```

### 8.3. Convergência

O algoritmo **converge** quando:
- ✅ O melhor fitness para de melhorar (estagnação)
- ✅ A diversidade da população diminui (indivíduos muito similares)
- ✅ O número de gerações é atingido

**Critérios de Parada (nossa implementação):**
- Número fixo de gerações (15)
- Poderia adicionar: estagnação por N gerações

---

## 9. Parâmetros e Configuração

### 9.1. Parâmetros do AG

| Parâmetro | Valor Padrão | Impacto | Como Ajustar |
|-----------|--------------|---------|--------------|
| `tamanho_populacao` | 20 | Mais indivíduos = mais exploração, mais custo | 20-50 típico |
| `n_geracoes` | 15 | Mais gerações = mais evolução, mais custo | 10-30 típico |
| `taxa_cruzamento` | 0.7 | Mais cruzamento = mais exploração | 0.6-0.9 típico |
| `taxa_mutacao` | 0.1 | Mais mutação = mais diversidade | 0.05-0.2 típico |
| `n_elites` | 2 | Mais elites = menos diversidade | 1-5 típico |
| `tamanho_torneio` | 3 | Maior = mais seletivo | 2-5 típico |

### 9.2. Trade-offs

**População Grande (40) vs Pequena (10):**
- ✅ Grande: Mais diversidade, exploração melhor
- ❌ Grande: Mais custo computacional

**Muitas Gerações (30) vs Poucas (10):**
- ✅ Muitas: Mais tempo para convergir
- ❌ Muitas: Pode estagnar antes do fim

**Alta Taxa de Cruzamento (0.9) vs Baixa (0.5):**
- ✅ Alta: Mais exploração
- ❌ Alta: Menos preservação de boas características

**Alta Taxa de Mutação (0.2) vs Baixa (0.05):**
- ✅ Alta: Mais diversidade
- ❌ Alta: Pode destruir boas soluções

### 9.3. Como Escolher Parâmetros?

**Estratégia:**
1. **Comece com valores padrão** (já são razoáveis)
2. **Execute experimentos** com diferentes configurações
3. **Compare resultados** (use nosso sistema de experimentos!)
4. **Escolha o melhor** para seu problema específico

**Nosso sistema:**
- Implementa múltiplos experimentos automaticamente
- Compara configurações diferentes
- Gera relatórios comparativos

---

## 10. Exemplos Práticos

### 10.1. Exemplo 1: Evolução Simples

**Cenário:** Executar AG com configuração padrão

```python
melhor_individuo = executar_ag(
    X_train, y_train,
    tamanho_populacao=20,
    n_geracoes=15,
    taxa_cruzamento=0.7,
    taxa_mutacao=0.1,
    n_elites=2
)

# Resultado:
# melhor_individuo = {
#     'C': 1.5234,
#     'penalty': 'l2',
#     'class_weight': 'balanced',
#     'max_iter': 964
# }
```

**O que aconteceu:**
1. 20 indivíduos aleatórios foram criados
2. Cada um foi avaliado (treinamento + validação cruzada)
3. Por 15 gerações: seleção → cruzamento → mutação
4. Melhor indivíduo encontrado retornado

**Tempo:** ~5-10 minutos (depende do hardware)

### 10.2. Exemplo 2: Comparação com Modelo Original

**Cenário:** Comparar modelo otimizado vs original

```python
# Modelo original (sem otimização)
modelo_original = treinar_modelo_original(X_train, y_train)

# Modelo otimizado (com AG)
melhor_individuo = executar_ag(X_train, y_train)
modelo_otimizado = LogisticRegression(
    C=melhor_individuo['C'],
    penalty=melhor_individuo['penalty'],
    class_weight=melhor_individuo['class_weight'],
    max_iter=melhor_individuo['max_iter']
)
modelo_otimizado.fit(X_train, y_train)

# Compara
comparacao = comparar_modelos(modelo_original, modelo_otimizado, X_test, y_test)

# Resultado esperado:
# {
#     'modelo_original': {'accuracy': 0.85, 'recall': 0.82, ...},
#     'modelo_otimizado': {'accuracy': 0.88, 'recall': 0.86, ...},
#     'melhorias': {
#         'accuracy': {'diferenca': 0.03, 'percentual': 3.53%},
#         'recall': {'diferenca': 0.04, 'percentual': 4.88%},
#         ...
#     }
# }
```

**Interpretação:**
- Modelo otimizado tem **+3.53%** em accuracy
- Modelo otimizado tem **+4.88%** em recall (crítico!)
- AG melhorou o desempenho significativamente

### 10.3. Exemplo 3: Múltiplos Experimentos

**Cenário:** Testar diferentes configurações do AG

```python
experimentos_executados = executar_multiplos_experimentos(
    X_train, y_train, X_test, y_test,
    experimentos_config=[
        {
            'nome': 'Experimento 1: Conservador',
            'tamanho_populacao': 20,
            'n_geracoes': 15,
            'taxa_mutacao': 0.1
        },
        {
            'nome': 'Experimento 2: Agressivo',
            'tamanho_populacao': 40,
            'n_geracoes': 25,
            'taxa_mutacao': 0.2
        }
    ]
)

# Gera relatório comparativo
relatorio_path = gerar_relatorio_comparativo(experimentos_executados)
```

**Resultado:**
- Cada experimento executa AG completo
- Resultados são comparados
- Melhor experimento identificado
- Relatório JSON gerado

### 10.4. Interpretação de Resultados

**Histórico de Evolução:**
```json
{
  "historico": [
    {"geracao": 0, "fitness_max": 0.72, "fitness_medio": 0.67},
    {"geracao": 1, "fitness_max": 0.75, "fitness_medio": 0.69},
    {"geracao": 5, "fitness_max": 0.82, "fitness_medio": 0.77},
    {"geracao": 15, "fitness_max": 0.89, "fitness_medio": 0.85}
  ]
}
```

**Interpretação:**
- ✅ Fitness melhorou ao longo das gerações
- ✅ Convergência suave (sem grandes saltos)
- ✅ Melhoria de 0.72 → 0.89 (+23.6%)
- ✅ Fitness médio também melhorou (população como um todo evoluiu)

**Gráfico de Evolução:**
- **Linha verde (max):** Mostra melhor fitness por geração
- **Linha azul (médio):** Mostra fitness médio (evolução da população)
- **Linha vermelha (min):** Mostra pior fitness (diversidade)

**Gráfico de Convergência:**
- **Subplot superior:** Evolução do melhor fitness vs melhor acumulado
- **Subplot inferior:** Taxa de melhoria por geração
  - **Barras verdes:** Melhorou
  - **Barras vermelhas:** Piorou (normal em AG!)

---

## 11. Conceitos Avançados

### 11.1. Exploração vs Exploitation

**Exploração (Exploration):**
- Buscar em novas regiões do espaço de busca
- Aumenta com: alta mutação, cruzamento diverso, população grande

**Exploitation (Exploração Local):**
- Refinar soluções existentes
- Aumenta com: elitismo, baixa mutação, seleção forte

**Balanceamento:**
O AG precisa equilibrar:
- **Início:** Mais exploração (descobrir boas regiões)
- **Fim:** Mais exploitation (refinar melhor solução)

**Nossa implementação:**
- Elitismo garante exploitation
- Cruzamento e mutação garantem exploração
- Taxas balanceadas (70% cruzamento, 10% mutação)

### 11.2. Diversidade Genética

**Problema:** População muito similar → estagnação

**Soluções:**
- Mutação (introduz variação)
- Cruzamento diverso (combina características diferentes)
- Tamanho de população adequado (mais indivíduos = mais diversidade)

**Medição de Diversidade:**
- Variação do fitness (fitness_max - fitness_min)
- Distância entre indivíduos (mais complexo)

### 11.3. Convergência Prematura

**Problema:** AG converge para solução sub-ótima muito rápido

**Sinais:**
- Fitness para de melhorar após poucas gerações
- População muito similar
- Fitness médio ≈ fitness máximo

**Soluções:**
- Aumentar taxa de mutação
- Aumentar população
- Usar mutação não-uniforme
- Reduzir elitismo

---

## 12. Perguntas Frequentes (FAQ)

### Q1: Por que não usar Grid Search?

**Grid Search:**
- Testa todas as combinações de parâmetros
- Funciona bem com poucos hiperparâmetros
- Custo exponencial: O(n^k) onde k = número de hiperparâmetros

**Algoritmo Genético:**
- Testa subconjunto inteligente
- Funciona bem com muitos hiperparâmetros
- Custo controlado: O(p × g) onde p = população, g = gerações

**Conclusão:** AG é melhor para muitos hiperparâmetros ou espaço de busca grande.

### Q2: Quanto tempo leva para executar?

**Tempo estimado:**
- Avaliação de fitness: ~2-5s por indivíduo
- População 20: ~40-100s por geração
- 15 gerações: ~10-25 minutos

**Fatores:**
- Tamanho do dataset
- Número de features
- Configuração do AG (população, gerações)

### Q3: Como saber se o AG convergiu?

**Sinais de convergência:**
1. Fitness máximo não melhora por N gerações
2. Fitness médio ≈ fitness máximo (população homogênea)
3. Gráfico de convergência mostra platô

**Nossa implementação:**
- Número fixo de gerações (15)
- Pode adicionar critério de estagnação

### Q4: Os hiperparâmetros encontrados são ótimos?

**Resposta curta:** Provavelmente não são ótimos globais, mas são **muito bons**.

**Por quê:**
- AG não garante ótimo global
- Depende da inicialização e sorte
- Mas geralmente encontra soluções muito melhores que valores padrão

**Estratégia:**
- Execute múltiplos experimentos
- Use o melhor resultado

---

## 13. Referências Teóricas

### 13.1. Fundamentos
- **Holland, J. H. (1975)** - "Adaptation in Natural and Artificial Systems"
- **Goldberg, D. E. (1989)** - "Genetic Algorithms in Search, Optimization and Machine Learning"

### 13.2. Aplicações em ML
- **Bergstra, J. & Bengio, Y. (2012)** - "Random Search for Hyper-Parameter Optimization"
- **Larrañaga, P. & Lozano, J. A. (2002)** - "Estimation of Distribution Algorithms"

### 13.3. Operadores Genéticos
- **Deb, K. & Beyer, H. G. (2001)** - "Self-Adaptive Genetic Algorithms"
- **Eiben, A. E. & Smith, J. E. (2015)** - "Introduction to Evolutionary Computing"

---

## 14. Conclusão

Este guia técnico explicou:

✅ **Fundamentos:** O que é e por que usar AG
✅ **Codificação:** Como representar soluções
✅ **Fitness:** Como avaliar qualidade
✅ **Operadores:** Seleção, cruzamento, mutação
✅ **Ciclo completo:** Como tudo funciona junto
✅ **Prática:** Exemplos e interpretação

**Próximos Passos:**
1. Execute o código com diferentes configurações
2. Analise os gráficos de evolução
3. Compare resultados dos experimentos
4. Documente suas descobertas no trabalho

**Recursos:**
- Código completo em: `ag_*.py`
- Documentação de implementação: `MAPEAMENTO_IMPLEMENTACAO_AG.md`
- Logs e histórico: `evidencias/logs/`

---

**Boa sorte com seu trabalho acadêmico! 🎓**

