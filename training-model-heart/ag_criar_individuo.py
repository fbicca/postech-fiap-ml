#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#AG Módulo para criação de indivíduos (representação de genes/hiperparâmetros)
Cada indivíduo representa um conjunto de hiperparâmetros do modelo LogisticRegression.
"""

import numpy as np
import copy

def criar_individuo():
    """
    #AG Cria um indivíduo aleatório representando hiperparâmetros do LogisticRegression.
    
    Representação (genes):
    - C: parâmetro de regularização (float) - intervalo [0.001, 100]
    - penalty: tipo de penalidade (str) - 'l1' ou 'l2'
    - class_weight: balanceamento de classes (dict/str) - 'balanced', None, ou dict
    - solver: algoritmo de otimização (str) - 'liblinear' (fixo para compatibilidade)
    - max_iter: número máximo de iterações (int) - intervalo [100, 5000]
    
    Returns:
        dict: Dicionário representando um indivíduo com hiperparâmetros
    """
    # C: valor de regularização (log scale para melhor exploração)
    C = np.random.uniform(0.001, 100.0)
    
    # penalty: l1 ou l2
    penalty = np.random.choice(['l1', 'l2'])
    
    # class_weight: None, 'balanced', ou dict com pesos customizados
    class_weight_option = np.random.choice(['None', 'balanced', 'custom'])
    if class_weight_option == 'None':
        class_weight = None
    elif class_weight_option == 'balanced':
        class_weight = 'balanced'
    else:
        # pesos customizados levemente desbalanceados
        w1 = np.random.uniform(0.5, 2.0)
        w0 = np.random.uniform(0.5, 2.0)
        class_weight = {0: w0, 1: w1}
    
    # max_iter: número máximo de iterações
    max_iter = int(np.random.uniform(100, 5000))
    
    individuo = {
        'C': C,
        'penalty': penalty,
        'class_weight': class_weight,
        'solver': 'liblinear',  # fixo para compatibilidade com l1/l2
        'max_iter': max_iter
    }
    
    return individuo


def copiar_individuo(individuo):
    """
    #AG Cria uma cópia profunda de um indivíduo.
    
    Args:
        individuo (dict): Indivíduo a ser copiado
        
    Returns:
        dict: Cópia do indivíduo
    """
    return copy.deepcopy(individuo)


def validar_individuo(individuo):
    """
    #AG Valida se um indivíduo tem estrutura e valores válidos.
    
    Args:
        individuo (dict): Indivíduo a validar
        
    Returns:
        bool: True se válido, False caso contrário
    """
    required_keys = ['C', 'penalty', 'class_weight', 'solver', 'max_iter']
    
    # Verifica se todas as chaves necessárias estão presentes
    if not all(key in individuo for key in required_keys):
        return False
    
    # Valida C
    if not (0.001 <= individuo['C'] <= 100.0):
        return False
    
    # Valida penalty
    if individuo['penalty'] not in ['l1', 'l2']:
        return False
    
    # Valida solver
    if individuo['solver'] != 'liblinear':
        return False
    
    # Valida max_iter
    if not (100 <= individuo['max_iter'] <= 5000):
        return False
    
    # Valida class_weight
    if individuo['class_weight'] not in [None, 'balanced']:
        if isinstance(individuo['class_weight'], dict):
            if not all(k in [0, 1] for k in individuo['class_weight'].keys()):
                return False
        else:
            return False
    
    return True


def corrigir_individuo(individuo):
    """
    #AG Corrige valores inválidos de um indivíduo, garantindo que esteja dentro dos limites.
    
    Args:
        individuo (dict): Indivíduo a corrigir (modificado in-place)
        
    Returns:
        dict: Indivíduo corrigido
    """
    # Garante C dentro do intervalo
    individuo['C'] = np.clip(individuo['C'], 0.001, 100.0)
    
    # Garante penalty válido
    if individuo['penalty'] not in ['l1', 'l2']:
        individuo['penalty'] = np.random.choice(['l1', 'l2'])
    
    # Garante solver válido
    if individuo['solver'] != 'liblinear':
        individuo['solver'] = 'liblinear'
    
    # Garante max_iter dentro do intervalo
    individuo['max_iter'] = int(np.clip(individuo['max_iter'], 100, 5000))
    
    # Corrige class_weight se necessário
    if individuo['class_weight'] not in [None, 'balanced']:
        if isinstance(individuo['class_weight'], dict):
            if not all(k in [0, 1] for k in individuo['class_weight'].keys()):
                individuo['class_weight'] = np.random.choice([None, 'balanced'])
        else:
            individuo['class_weight'] = np.random.choice([None, 'balanced'])
    
    return individuo

