#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#AG Módulo com operadores de cruzamento (crossover) para o algoritmo genético.
Implementa diferentes estratégias: ponto único, uniforme, aritmético.
"""

import numpy as np
from ag_criar_individuo import copiar_individuo, corrigir_individuo


def cruzamento_ponto_unico(pai1, pai2, taxa_cruzamento=0.7):
    """
    #AG Cruzamento de ponto único: escolhe um ponto de corte e troca genes após esse ponto.
    Para hiperparâmetros, divide os genes em duas metades.
    
    Args:
        pai1 (dict): Primeiro pai
        pai2 (dict): Segundo pai
        taxa_cruzamento (float): Probabilidade de ocorrer cruzamento [0, 1]
        
    Returns:
        tuple: (filho1, filho2) ou (pai1, pai2) se não houver cruzamento
    """
    if np.random.rand() > taxa_cruzamento:
        return copiar_individuo(pai1), copiar_individuo(pai2)
    
    filho1 = copiar_individuo(pai1)
    filho2 = copiar_individuo(pai2)
    
    # Ponto de corte aleatório (divide genes em duas metades)
    # C é um gene contínuo, penalty é categórico, class_weight é complexo, max_iter é inteiro
    if np.random.rand() < 0.5:
        # Troca C
        filho1['C'], filho2['C'] = filho2['C'], filho1['C']
    
    if np.random.rand() < 0.5:
        # Troca penalty
        filho1['penalty'], filho2['penalty'] = filho2['penalty'], filho1['penalty']
    
    if np.random.rand() < 0.5:
        # Troca class_weight
        filho1['class_weight'], filho2['class_weight'] = filho2['class_weight'], filho1['class_weight']
    
    if np.random.rand() < 0.5:
        # Troca max_iter
        filho1['max_iter'], filho2['max_iter'] = filho2['max_iter'], filho1['max_iter']
    
    # Corrige valores inválidos
    corrigir_individuo(filho1)
    corrigir_individuo(filho2)
    
    return filho1, filho2


def cruzamento_uniforme(pai1, pai2, taxa_cruzamento=0.7):
    """
    #AG Cruzamento uniforme: para cada gene, escolhe aleatoriamente de qual pai virá.
    
    Args:
        pai1 (dict): Primeiro pai
        pai2 (dict): Segundo pai
        taxa_cruzamento (float): Probabilidade de ocorrer cruzamento [0, 1]
        
    Returns:
        tuple: (filho1, filho2) ou (pai1, pai2) se não houver cruzamento
    """
    if np.random.rand() > taxa_cruzamento:
        return copiar_individuo(pai1), copiar_individuo(pai2)
    
    filho1 = copiar_individuo(pai1)
    filho2 = copiar_individuo(pai2)
    
    # Para cada gene, decide aleatoriamente se troca ou não
    if np.random.rand() < 0.5:
        filho1['C'] = pai2['C']
        filho2['C'] = pai1['C']
    
    if np.random.rand() < 0.5:
        filho1['penalty'] = pai2['penalty']
        filho2['penalty'] = pai1['penalty']
    
    if np.random.rand() < 0.5:
        filho1['class_weight'] = pai2['class_weight']
        filho2['class_weight'] = pai1['class_weight']
    
    if np.random.rand() < 0.5:
        filho1['max_iter'] = pai2['max_iter']
        filho2['max_iter'] = pai1['max_iter']
    
    # Corrige valores inválidos
    corrigir_individuo(filho1)
    corrigir_individuo(filho2)
    
    return filho1, filho2


def cruzamento_aritmetico(pai1, pai2, taxa_cruzamento=0.7, alpha=0.5):
    """
    #AG Cruzamento aritmético: para genes contínuos (C, max_iter), faz média ponderada.
    Para genes categóricos, usa cruzamento uniforme.
    
    Args:
        pai1 (dict): Primeiro pai
        pai2 (dict): Segundo pai
        taxa_cruzamento (float): Probabilidade de ocorrer cruzamento [0, 1]
        alpha (float): Peso da média (0.5 = média simples) [0, 1]
        
    Returns:
        tuple: (filho1, filho2) ou (pai1, pai2) se não houver cruzamento
    """
    if np.random.rand() > taxa_cruzamento:
        return copiar_individuo(pai1), copiar_individuo(pai2)
    
    filho1 = copiar_individuo(pai1)
    filho2 = copiar_individuo(pai2)
    
    # Genes contínuos: média ponderada
    filho1['C'] = alpha * pai1['C'] + (1 - alpha) * pai2['C']
    filho2['C'] = (1 - alpha) * pai1['C'] + alpha * pai2['C']
    
    filho1['max_iter'] = int(alpha * pai1['max_iter'] + (1 - alpha) * pai2['max_iter'])
    filho2['max_iter'] = int((1 - alpha) * pai1['max_iter'] + alpha * pai2['max_iter'])
    
    # Genes categóricos: cruzamento uniforme
    if np.random.rand() < 0.5:
        filho1['penalty'] = pai2['penalty']
        filho2['penalty'] = pai1['penalty']
    
    if np.random.rand() < 0.5:
        filho1['class_weight'] = pai2['class_weight']
        filho2['class_weight'] = pai1['class_weight']
    
    # Corrige valores inválidos
    corrigir_individuo(filho1)
    corrigir_individuo(filho2)
    
    return filho1, filho2


def cruzamento_blend(pai1, pai2, taxa_cruzamento=0.7, alpha=0.5):
    """
    #AG Cruzamento BLX-alpha: para genes contínuos, escolhe valor aleatório no intervalo expandido.
    Para genes categóricos, usa cruzamento uniforme.
    
    Args:
        pai1 (dict): Primeiro pai
        pai2 (dict): Segundo pai
        taxa_cruzamento (float): Probabilidade de ocorrer cruzamento [0, 1]
        alpha (float): Fator de expansão do intervalo [0, 1]
        
    Returns:
        tuple: (filho1, filho2) ou (pai1, pai2) se não houver cruzamento
    """
    if np.random.rand() > taxa_cruzamento:
        return copiar_individuo(pai1), copiar_individuo(pai2)
    
    filho1 = copiar_individuo(pai1)
    filho2 = copiar_individuo(pai2)
    
    # Para C: intervalo expandido
    c_min = min(pai1['C'], pai2['C'])
    c_max = max(pai1['C'], pai2['C'])
    intervalo_c = c_max - c_min
    
    filho1['C'] = np.random.uniform(
        max(0.001, c_min - alpha * intervalo_c),
        min(100.0, c_max + alpha * intervalo_c)
    )
    filho2['C'] = np.random.uniform(
        max(0.001, c_min - alpha * intervalo_c),
        min(100.0, c_max + alpha * intervalo_c)
    )
    
    # Para max_iter: intervalo expandido
    iter_min = min(pai1['max_iter'], pai2['max_iter'])
    iter_max = max(pai1['max_iter'], pai2['max_iter'])
    intervalo_iter = iter_max - iter_min
    
    filho1['max_iter'] = int(np.random.uniform(
        max(100, iter_min - alpha * intervalo_iter),
        min(5000, iter_max + alpha * intervalo_iter)
    ))
    filho2['max_iter'] = int(np.random.uniform(
        max(100, iter_min - alpha * intervalo_iter),
        min(5000, iter_max + alpha * intervalo_iter)
    ))
    
    # Genes categóricos: cruzamento uniforme
    if np.random.rand() < 0.5:
        filho1['penalty'] = pai2['penalty']
        filho2['penalty'] = pai1['penalty']
    
    if np.random.rand() < 0.5:
        filho1['class_weight'] = pai2['class_weight']
        filho2['class_weight'] = pai1['class_weight']
    
    # Corrige valores inválidos
    corrigir_individuo(filho1)
    corrigir_individuo(filho2)
    
    return filho1, filho2

