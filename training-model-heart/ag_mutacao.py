#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#AG Módulo com operadores de mutação para o algoritmo genético.
Implementa diferentes estratégias: uniforme, gaussiana, não-uniforme.
"""

import numpy as np
from ag_criar_individuo import copiar_individuo, corrigir_individuo, criar_individuo


def mutacao_uniforme(individuo, taxa_mutacao=0.1, forca_mutacao=0.1):
    """
    #AG Mutação uniforme: altera genes aleatoriamente dentro dos limites permitidos.
    
    Args:
        individuo (dict): Indivíduo a mutar (modificado in-place)
        taxa_mutacao (float): Probabilidade de cada gene sofrer mutação [0, 1]
        forca_mutacao (float): Força da mutação (para genes contínuos)
        
    Returns:
        dict: Indivíduo mutado
    """
    individuo_mutado = copiar_individuo(individuo)
    
    # Mutação em C (gene contínuo)
    if np.random.rand() < taxa_mutacao:
        # Perturbação gaussiana ou valor completamente novo
        if np.random.rand() < 0.5:
            # Perturbação gaussiana
            delta = np.random.normal(0, forca_mutacao * individuo_mutado['C'])
            individuo_mutado['C'] = individuo_mutado['C'] + delta
        else:
            # Valor completamente novo
            individuo_mutado['C'] = np.random.uniform(0.001, 100.0)
    
    # Mutação em penalty (gene categórico)
    if np.random.rand() < taxa_mutacao:
        individuo_mutado['penalty'] = np.random.choice(['l1', 'l2'])
    
    # Mutação em class_weight (gene complexo)
    if np.random.rand() < taxa_mutacao:
        class_weight_option = np.random.choice(['None', 'balanced', 'custom'])
        if class_weight_option == 'None':
            individuo_mutado['class_weight'] = None
        elif class_weight_option == 'balanced':
            individuo_mutado['class_weight'] = 'balanced'
        else:
            w1 = np.random.uniform(0.5, 2.0)
            w0 = np.random.uniform(0.5, 2.0)
            individuo_mutado['class_weight'] = {0: w0, 1: w1}
    
    # Mutação em max_iter (gene inteiro)
    if np.random.rand() < taxa_mutacao:
        if np.random.rand() < 0.5:
            # Perturbação
            delta = int(np.random.normal(0, forca_mutacao * individuo_mutado['max_iter']))
            individuo_mutado['max_iter'] = individuo_mutado['max_iter'] + delta
        else:
            # Valor completamente novo
            individuo_mutado['max_iter'] = int(np.random.uniform(100, 5000))
    
    # Corrige valores inválidos
    corrigir_individuo(individuo_mutado)
    
    return individuo_mutado


def mutacao_gaussiana(individuo, taxa_mutacao=0.1, sigma_fracao=0.1):
    """
    #AG Mutação gaussiana: aplica perturbação gaussiana aos genes contínuos.
    Genes categóricos usam mutação uniforme.
    
    Args:
        individuo (dict): Indivíduo a mutar
        taxa_mutacao (float): Probabilidade de cada gene sofrer mutação [0, 1]
        sigma_fracao (float): Fração do valor atual usado como desvio padrão
        
    Returns:
        dict: Indivíduo mutado
    """
    individuo_mutado = copiar_individuo(individuo)
    
    # Mutação gaussiana em C
    if np.random.rand() < taxa_mutacao:
        sigma = sigma_fracao * individuo_mutado['C']
        delta = np.random.normal(0, sigma)
        individuo_mutado['C'] = individuo_mutado['C'] + delta
    
    # Mutação em penalty (categórico - uniforme)
    if np.random.rand() < taxa_mutacao:
        individuo_mutado['penalty'] = np.random.choice(['l1', 'l2'])
    
    # Mutação em class_weight (complexo)
    if np.random.rand() < taxa_mutacao:
        class_weight_option = np.random.choice(['None', 'balanced', 'custom'])
        if class_weight_option == 'None':
            individuo_mutado['class_weight'] = None
        elif class_weight_option == 'balanced':
            individuo_mutado['class_weight'] = 'balanced'
        else:
            w1 = np.random.uniform(0.5, 2.0)
            w0 = np.random.uniform(0.5, 2.0)
            individuo_mutado['class_weight'] = {0: w0, 1: w1}
    
    # Mutação gaussiana em max_iter (discreto, mas tratado como contínuo para perturbação)
    if np.random.rand() < taxa_mutacao:
        sigma = sigma_fracao * individuo_mutado['max_iter']
        delta = int(np.random.normal(0, sigma))
        individuo_mutado['max_iter'] = individuo_mutado['max_iter'] + delta
    
    # Corrige valores inválidos
    corrigir_individuo(individuo_mutado)
    
    return individuo_mutado


def mutacao_nao_uniforme(individuo, geracao, max_geracoes, taxa_mutacao=0.1, b=2.0):
    """
    #AG Mutação não-uniforme: reduz a intensidade da mutação ao longo das gerações.
    Útil para convergência mais suave no final da evolução.
    
    Args:
        individuo (dict): Indivíduo a mutar
        geracao (int): Geração atual
        max_geracoes (int): Número máximo de gerações
        taxa_mutacao (float): Probabilidade de cada gene sofrer mutação [0, 1]
        b (float): Parâmetro que controla o formato da redução
        
    Returns:
        dict: Indivíduo mutado
    """
    # Calcula o fator de redução baseado na geração
    r = np.random.rand()
    fator_reducao = (1 - r ** ((1 - geracao / max_geracoes) ** b))
    
    # Aplica mutação com intensidade reduzida
    if np.random.rand() < taxa_mutacao:
        individuo_mutado = copiar_individuo(individuo)
        
        # Mutação em C com intensidade reduzida
        range_c = 100.0 - 0.001
        delta = fator_reducao * range_c * (2 * np.random.rand() - 1)
        individuo_mutado['C'] = individuo_mutado['C'] + delta
        
        # Mutação em penalty (sem redução - categórico)
        if np.random.rand() < 0.5:
            individuo_mutado['penalty'] = np.random.choice(['l1', 'l2'])
        
        # Mutação em class_weight (sem redução - categórico)
        if np.random.rand() < 0.5:
            class_weight_option = np.random.choice(['None', 'balanced', 'custom'])
            if class_weight_option == 'None':
                individuo_mutado['class_weight'] = None
            elif class_weight_option == 'balanced':
                individuo_mutado['class_weight'] = 'balanced'
            else:
                w1 = np.random.uniform(0.5, 2.0)
                w0 = np.random.uniform(0.5, 2.0)
                individuo_mutado['class_weight'] = {0: w0, 1: w1}
        
        # Mutação em max_iter com intensidade reduzida
        range_iter = 5000 - 100
        delta_iter = int(fator_reducao * range_iter * (2 * np.random.rand() - 1))
        individuo_mutado['max_iter'] = individuo_mutado['max_iter'] + delta_iter
        
        # Corrige valores inválidos
        corrigir_individuo(individuo_mutado)
        
        return individuo_mutado
    
    return copiar_individuo(individuo)


def mutacao_adaptativa(individuo, fitness_relativo, taxa_mutacao_base=0.1):
    """
    #AG Mutação adaptativa: ajusta a taxa de mutação baseado no fitness relativo.
    Indivíduos com fitness baixo têm maior probabilidade de mutação.
    
    Args:
        individuo (dict): Indivíduo a mutar
        fitness_relativo (float): Fitness relativo (0=pior, 1=melhor)
        taxa_mutacao_base (float): Taxa de mutação base
        
    Returns:
        dict: Indivíduo mutado
    """
    # Taxa de mutação inversamente proporcional ao fitness
    # Indivíduos piores (fitness_relativo baixo) mutam mais
    taxa_mutacao = taxa_mutacao_base * (1 - fitness_relativo) + 0.01
    
    return mutacao_uniforme(individuo, taxa_mutacao=taxa_mutacao)

