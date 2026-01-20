#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#AG Módulo com operadores de seleção para o algoritmo genético.
Implementa diferentes estratégias de seleção: torneio, roleta, elitismo.
"""

import numpy as np
from ag_criar_individuo import copiar_individuo


def selecao_torneio(populacao_fitness, tamanho_torneio=3):
    """
    #AG Seleção por torneio: escolhe o melhor indivíduo entre k aleatórios.
    
    Args:
        populacao_fitness (list): Lista de tuplas (individuo, fitness)
        tamanho_torneio (int): Número de indivíduos competindo no torneio
        
    Returns:
        dict: Indivíduo selecionado (cópia)
    """
    competidores = np.random.choice(len(populacao_fitness), tamanho_torneio, replace=False)
    
    # Encontra o melhor entre os competidores (maior fitness)
    melhor_idx = competidores[0]
    melhor_fitness = populacao_fitness[melhor_idx][1]
    
    for idx in competidores[1:]:
        if populacao_fitness[idx][1] > melhor_fitness:
            melhor_fitness = populacao_fitness[idx][1]
            melhor_idx = idx
    
    return copiar_individuo(populacao_fitness[melhor_idx][0])


def selecao_roleta(populacao_fitness):
    """
    #AG Seleção por roleta (proporcional ao fitness).
    Converte fitness para probabilidades e seleciona proporcionalmente.
    
    Args:
        populacao_fitness (list): Lista de tuplas (individuo, fitness)
        
    Returns:
        dict: Indivíduo selecionado (cópia)
    """
    # Extrai apenas os fitness
    fitness_values = np.array([pf[1] for pf in populacao_fitness])
    
    # Normaliza para garantir valores positivos (shift mínimo)
    min_fitness = np.min(fitness_values)
    if min_fitness < 0:
        fitness_values = fitness_values - min_fitness + 1e-10
    
    # Calcula probabilidades proporcionais
    probabilidades = fitness_values / np.sum(fitness_values)
    
    # Seleciona baseado nas probabilidades
    idx_selecionado = np.random.choice(len(populacao_fitness), p=probabilidades)
    
    return copiar_individuo(populacao_fitness[idx_selecionado][0])


def selecao_elitismo(populacao_fitness, n_elites=2):
    """
    #AG Seleção por elitismo: retorna os n melhores indivíduos.
    
    Args:
        populacao_fitness (list): Lista de tuplas (individuo, fitness)
        n_elites (int): Número de elites a retornar
        
    Returns:
        list: Lista dos n melhores indivíduos (cópias)
    """
    # Ordena por fitness (decrescente)
    populacao_ordenada = sorted(populacao_fitness, key=lambda x: x[1], reverse=True)
    
    # Retorna os n melhores
    elites = [copiar_individuo(pf[0]) for pf in populacao_ordenada[:n_elites]]
    
    return elites


def selecao_mista(populacao_fitness, n_elites=2, tamanho_torneio=3, metodo='torneio'):
    """
    #AG Seleção mista: combina elitismo com outro método (torneio ou roleta).
    
    Args:
        populacao_fitness (list): Lista de tuplas (individuo, fitness)
        n_elites (int): Número de elites a manter
        tamanho_torneio (int): Tamanho do torneio (se método for 'torneio')
        metodo (str): 'torneio' ou 'roleta' para seleção dos demais
        
    Returns:
        list: Lista com elites + indivíduos selecionados
    """
    selecionados = []
    
    # Adiciona elites
    elites = selecao_elitismo(populacao_fitness, n_elites)
    selecionados.extend(elites)
    
    # Seleciona o restante
    n_restante = len(populacao_fitness) - n_elites
    
    if metodo == 'torneio':
        for _ in range(n_restante):
            selecionados.append(selecao_torneio(populacao_fitness, tamanho_torneio))
    elif metodo == 'roleta':
        for _ in range(n_restante):
            selecionados.append(selecao_roleta(populacao_fitness))
    else:
        raise ValueError(f"Método '{metodo}' não reconhecido. Use 'torneio' ou 'roleta'")
    
    return selecionados


def selecao_rank(populacao_fitness):
    """
    #AG Seleção por rank: seleciona baseado na posição (rank) do fitness, não no valor absoluto.
    Útil quando há grande variação nos valores de fitness.
    
    Args:
        populacao_fitness (list): Lista de tuplas (individuo, fitness)
        
    Returns:
        dict: Indivíduo selecionado (cópia)
    """
    # Ordena por fitness (decrescente)
    populacao_ordenada = sorted(populacao_fitness, key=lambda x: x[1], reverse=True)
    
    # Atribui probabilidades baseadas no rank (melhor rank = maior probabilidade)
    n = len(populacao_ordenada)
    # Usa distribuição linear decrescente
    probabilidades = [(n - i) / (n * (n + 1) / 2) for i in range(n)]
    
    # Seleciona baseado nas probabilidades de rank
    idx_selecionado = np.random.choice(n, p=probabilidades)
    
    return copiar_individuo(populacao_ordenada[idx_selecionado][0])

