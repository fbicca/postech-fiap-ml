#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#AG Módulo principal do algoritmo genético para otimização de hiperparâmetros.
Orquestra a evolução da população: inicialização, avaliação, seleção, cruzamento e mutação.
"""

import numpy as np
from ag_criar_individuo import criar_individuo, copiar_individuo
from ag_criar_fitness import fitness, fitness_detalhado
from ag_selecao import selecao_mista, selecao_elitismo
from ag_cruzamento import cruzamento_uniforme, cruzamento_aritmetico, cruzamento_blend
from ag_mutacao import mutacao_uniforme, mutacao_gaussiana, mutacao_nao_uniforme


class AlgoritmoGenetico:
    """
    #AG Classe principal do algoritmo genético para otimização de hiperparâmetros.
    """
    
    def __init__(
        self,
        tamanho_populacao=20,
        n_geracoes=15,
        taxa_cruzamento=0.7,
        taxa_mutacao=0.1,
        n_elites=2,
        metodo_selecao='torneio',
        metodo_cruzamento='uniforme',
        metodo_mutacao='uniforme',
        metric='composite',
        cv_folds=5,
        verbose=True
    ):
        """
        #AG Inicializa o algoritmo genético com parâmetros configuráveis.
        
        Args:
            tamanho_populacao (int): Tamanho da população
            n_geracoes (int): Número de gerações
            taxa_cruzamento (float): Probabilidade de cruzamento [0, 1]
            taxa_mutacao (float): Probabilidade de mutação [0, 1]
            n_elites (int): Número de elites a manter
            metodo_selecao (str): 'torneio' ou 'roleta'
            metodo_cruzamento (str): 'uniforme', 'aritmetico', 'blend'
            metodo_mutacao (str): 'uniforme', 'gaussiana', 'nao_uniforme'
            metric (str): Métrica para fitness ('composite', 'auc', 'f1', 'recall')
            cv_folds (int): Número de folds para validação cruzada
            verbose (bool): Se True, imprime progresso
        """
        self.tamanho_populacao = tamanho_populacao
        self.n_geracoes = n_geracoes
        self.taxa_cruzamento = taxa_cruzamento
        self.taxa_mutacao = taxa_mutacao
        self.n_elites = n_elites
        self.metodo_selecao = metodo_selecao
        self.metodo_cruzamento = metodo_cruzamento
        self.metodo_mutacao = metodo_mutacao
        self.metric = metric
        self.cv_folds = cv_folds
        self.verbose = verbose
        
        # Histórico da evolução
        self.historico_fitness = []
        self.melhor_individuo = None
        self.melhor_fitness = -np.inf
    
    def inicializar_populacao(self):
        """
        #AG Inicializa a população com indivíduos aleatórios.
        
        Returns:
            list: Lista de indivíduos
        """
        return [criar_individuo() for _ in range(self.tamanho_populacao)]
    
    def avaliar_populacao(self, populacao, X, y):
        """
        #AG Avalia todos os indivíduos da população calculando seu fitness.
        
        Args:
            populacao (list): Lista de indivíduos
            X (np.array): Features de treinamento
            y (np.array): Labels de treinamento
            
        Returns:
            list: Lista de tuplas (individuo, fitness)
        """
        populacao_fitness = []
        
        for individuo in populacao:
            fitness_value = fitness(individuo, X, y, self.cv_folds, self.metric)
            populacao_fitness.append((individuo, fitness_value))
        
        return populacao_fitness
    
    def selecionar_pais(self, populacao_fitness):
        """
        #AG Seleciona pais para reprodução usando o método configurado.
        
        Args:
            populacao_fitness (list): Lista de tuplas (individuo, fitness)
            
        Returns:
            tuple: (pai1, pai2) para cruzamento
        """
        if self.metodo_selecao == 'torneio':
            from ag_selecao import selecao_torneio
            pai1 = selecao_torneio(populacao_fitness, tamanho_torneio=3)
            pai2 = selecao_torneio(populacao_fitness, tamanho_torneio=3)
        elif self.metodo_selecao == 'roleta':
            from ag_selecao import selecao_roleta
            pai1 = selecao_roleta(populacao_fitness)
            pai2 = selecao_roleta(populacao_fitness)
        else:
            raise ValueError(f"Método de seleção '{self.metodo_selecao}' não reconhecido")
        
        return pai1, pai2
    
    def aplicar_cruzamento(self, pai1, pai2):
        """
        #AG Aplica o operador de cruzamento configurado.
        
        Args:
            pai1 (dict): Primeiro pai
            pai2 (dict): Segundo pai
            
        Returns:
            tuple: (filho1, filho2)
        """
        if self.metodo_cruzamento == 'uniforme':
            return cruzamento_uniforme(pai1, pai2, self.taxa_cruzamento)
        elif self.metodo_cruzamento == 'aritmetico':
            return cruzamento_aritmetico(pai1, pai2, self.taxa_cruzamento)
        elif self.metodo_cruzamento == 'blend':
            return cruzamento_blend(pai1, pai2, self.taxa_cruzamento)
        else:
            raise ValueError(f"Método de cruzamento '{self.metodo_cruzamento}' não reconhecido")
    
    def aplicar_mutacao(self, individuo, geracao=None):
        """
        #AG Aplica o operador de mutação configurado.
        
        Args:
            individuo (dict): Indivíduo a mutar
            geracao (int): Geração atual (para mutação não-uniforme)
            
        Returns:
            dict: Indivíduo mutado
        """
        if self.metodo_mutacao == 'uniforme':
            return mutacao_uniforme(individuo, self.taxa_mutacao)
        elif self.metodo_mutacao == 'gaussiana':
            return mutacao_gaussiana(individuo, self.taxa_mutacao)
        elif self.metodo_mutacao == 'nao_uniforme':
            if geracao is None:
                geracao = 0
            return mutacao_nao_uniforme(individuo, geracao, self.n_geracoes, self.taxa_mutacao)
        else:
            raise ValueError(f"Método de mutação '{self.metodo_mutacao}' não reconhecido")
    
    def evoluir(self, X, y):
        """
        #AG Executa a evolução completa do algoritmo genético.
        
        Args:
            X (np.array): Features de treinamento
            y (np.array): Labels de treinamento
            
        Returns:
            dict: Melhor indivíduo encontrado
        """
        # Inicializa população
        populacao = self.inicializar_populacao()
        
        if self.verbose:
            print("#AG Iniciando algoritmo genético...")
            print(f"#AG População: {self.tamanho_populacao}, Gerações: {self.n_geracoes}")
        
        # Evolução por gerações
        for geracao in range(self.n_geracoes):
            # Avalia população
            populacao_fitness = self.avaliar_populacao(populacao, X, y)
            
            # Ordena por fitness (decrescente)
            populacao_fitness.sort(key=lambda x: x[1], reverse=True)
            
            # Atualiza melhor indivíduo global
            melhor_fitness_geracao = populacao_fitness[0][1]
            if melhor_fitness_geracao > self.melhor_fitness:
                self.melhor_fitness = melhor_fitness_geracao
                self.melhor_individuo = copiar_individuo(populacao_fitness[0][0])
            
            # Registra histórico
            fitness_medio = np.mean([pf[1] for pf in populacao_fitness])
            fitness_max = populacao_fitness[0][1]
            fitness_min = populacao_fitness[-1][1]
            self.historico_fitness.append({
                'geracao': geracao,
                'fitness_medio': fitness_medio,
                'fitness_max': fitness_max,
                'fitness_min': fitness_min
            })
            
            if self.verbose:
                print(f"#AG Geração {geracao+1}/{self.n_geracoes} - "
                      f"Fitness: max={fitness_max:.4f}, médio={fitness_medio:.4f}, min={fitness_min:.4f}")
            
            # Elitismo: mantém os melhores
            nova_populacao = selecao_elitismo(populacao_fitness, self.n_elites)
            
            # Gera nova população
            while len(nova_populacao) < self.tamanho_populacao:
                # Seleção de pais
                pai1, pai2 = self.selecionar_pais(populacao_fitness)
                
                # Cruzamento
                filho1, filho2 = self.aplicar_cruzamento(pai1, pai2)
                
                # Mutação
                filho1 = self.aplicar_mutacao(filho1, geracao)
                filho2 = self.aplicar_mutacao(filho2, geracao)
                
                # Adiciona filhos à nova população
                nova_populacao.append(filho1)
                if len(nova_populacao) < self.tamanho_populacao:
                    nova_populacao.append(filho2)
            
            populacao = nova_populacao
        
        if self.verbose:
            print(f"#AG Algoritmo genético concluído!")
            print(f"#AG Melhor fitness: {self.melhor_fitness:.4f}")
        
        return self.melhor_individuo
    
    def get_historico(self):
        """
        #AG Retorna o histórico da evolução.
        
        Returns:
            list: Lista de dicionários com histórico por geração
        """
        return self.historico_fitness


def executar_ag(
    X, y,
    tamanho_populacao=20,
    n_geracoes=15,
    taxa_cruzamento=0.7,
    taxa_mutacao=0.1,
    n_elites=2,
    metodo_selecao='torneio',
    metodo_cruzamento='uniforme',
    metodo_mutacao='uniforme',
    metric='composite',
    cv_folds=5,
    verbose=True
):
    """
    #AG Função de conveniência para executar o algoritmo genético.
    
    Args:
        X (np.array): Features de treinamento
        y (np.array): Labels de treinamento
        tamanho_populacao (int): Tamanho da população
        n_geracoes (int): Número de gerações
        taxa_cruzamento (float): Probabilidade de cruzamento
        taxa_mutacao (float): Probabilidade de mutação
        n_elites (int): Número de elites
        metodo_selecao (str): Método de seleção
        metodo_cruzamento (str): Método de cruzamento
        metodo_mutacao (str): Método de mutação
        metric (str): Métrica para fitness
        cv_folds (int): Número de folds
        verbose (bool): Se True, imprime progresso
        
    Returns:
        dict: Melhor indivíduo encontrado
    """
    ag = AlgoritmoGenetico(
        tamanho_populacao=tamanho_populacao,
        n_geracoes=n_geracoes,
        taxa_cruzamento=taxa_cruzamento,
        taxa_mutacao=taxa_mutacao,
        n_elites=n_elites,
        metodo_selecao=metodo_selecao,
        metodo_cruzamento=metodo_cruzamento,
        metodo_mutacao=metodo_mutacao,
        metric=metric,
        cv_folds=cv_folds,
        verbose=verbose
    )
    
    melhor_individuo = ag.evoluir(X, y)
    
    return melhor_individuo

