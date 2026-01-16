#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#AG Módulo para cálculo da função fitness baseada em métricas de desempenho.
A função fitness combina accuracy, recall, F1-score e AUC usando validação cruzada.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    make_scorer
)


def fitness(individuo, X, y, cv_folds=5, metric='composite'):
    """
    #AG Calcula a função fitness de um indivíduo baseada em métricas de desempenho.
    
    A função fitness pode ser calculada de diferentes formas:
    - 'composite': média ponderada de accuracy, recall, f1 e AUC
    - 'auc': apenas AUC (melhor para problemas desbalanceados)
    - 'f1': apenas F1-score
    - 'recall': apenas recall (importante para minimizar falsos negativos)
    
    Args:
        individuo (dict): Indivíduo contendo hiperparâmetros
        X (np.array): Features de treinamento
        y (np.array): Labels de treinamento
        cv_folds (int): Número de folds para validação cruzada
        metric (str): Métrica a usar ('composite', 'auc', 'f1', 'recall')
        
    Returns:
        float: Valor da função fitness (maior é melhor)
    """
    try:
        # Cria modelo com os hiperparâmetros do indivíduo
        model = LogisticRegression(
            C=individuo['C'],
            penalty=individuo['penalty'],
            class_weight=individuo['class_weight'],
            solver=individuo['solver'],
            max_iter=individuo['max_iter'],
            random_state=42
        )
        
        # Validação cruzada estratificada
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        if metric == 'composite':
            # Calcula múltiplas métricas e combina
            accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall')
            f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
            
            # Para AUC, precisamos usar predict_proba, então calculamos manualmente
            auc_scores = []
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                model.fit(X_train_fold, y_train_fold)
                y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                try:
                    auc = roc_auc_score(y_val_fold, y_pred_proba)
                    auc_scores.append(auc)
                except ValueError:
                    auc_scores.append(0.5)  # AUC neutro se houver problema
            
            # Média ponderada: 20% accuracy, 30% recall, 25% f1, 25% AUC
            # Priorizamos recall para minimizar falsos negativos em diagnóstico cardíaco
            fitness_value = (
                0.20 * np.mean(accuracy_scores) +
                0.30 * np.mean(recall_scores) +
                0.25 * np.mean(f1_scores) +
                0.25 * np.mean(auc_scores)
            )
            
        elif metric == 'auc':
            # Calcula apenas AUC via validação cruzada customizada
            auc_scores = []
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                model.fit(X_train_fold, y_train_fold)
                y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                try:
                    auc = roc_auc_score(y_val_fold, y_pred_proba)
                    auc_scores.append(auc)
                except ValueError:
                    auc_scores.append(0.5)
            fitness_value = np.mean(auc_scores)
            
        elif metric == 'f1':
            fitness_value = np.mean(cross_val_score(model, X, y, cv=skf, scoring='f1'))
            
        elif metric == 'recall':
            fitness_value = np.mean(cross_val_score(model, X, y, cv=skf, scoring='recall'))
            
        else:
            raise ValueError(f"Métrica '{metric}' não reconhecida. Use 'composite', 'auc', 'f1' ou 'recall'")
        
        return float(fitness_value)
        
    except Exception as e:
        # Em caso de erro (ex: não convergência), retorna fitness baixo
        print(f"#AG Aviso: Erro ao calcular fitness: {e}")
        return 0.0


def fitness_detalhado(individuo, X, y, cv_folds=5):
    """
    #AG Calcula todas as métricas individualmente para análise detalhada.
    
    Args:
        individuo (dict): Indivíduo contendo hiperparâmetros
        X (np.array): Features de treinamento
        y (np.array): Labels de treinamento
        cv_folds (int): Número de folds para validação cruzada
        
    Returns:
        dict: Dicionário com todas as métricas calculadas
    """
    try:
        model = LogisticRegression(
            C=individuo['C'],
            penalty=individuo['penalty'],
            class_weight=individuo['class_weight'],
            solver=individuo['solver'],
            max_iter=individuo['max_iter'],
            random_state=42
        )
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        metrics = {
            'accuracy': np.mean(cross_val_score(model, X, y, cv=skf, scoring='accuracy')),
            'recall': np.mean(cross_val_score(model, X, y, cv=skf, scoring='recall')),
            'precision': np.mean(cross_val_score(model, X, y, cv=skf, scoring='precision')),
            'f1': np.mean(cross_val_score(model, X, y, cv=skf, scoring='f1'))
        }
        
        # AUC calculado manualmente
        auc_scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            model.fit(X_train_fold, y_train_fold)
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            try:
                auc = roc_auc_score(y_val_fold, y_pred_proba)
                auc_scores.append(auc)
            except ValueError:
                auc_scores.append(0.5)
        
        metrics['auc'] = np.mean(auc_scores)
        
        # Fitness composto
        metrics['fitness'] = (
            0.20 * metrics['accuracy'] +
            0.30 * metrics['recall'] +
            0.25 * metrics['f1'] +
            0.25 * metrics['auc']
        )
        
        return metrics
        
    except Exception as e:
        print(f"#AG Aviso: Erro ao calcular fitness detalhado: {e}")
        return {
            'accuracy': 0.0,
            'recall': 0.0,
            'precision': 0.0,
            'f1': 0.0,
            'auc': 0.0,
            'fitness': 0.0
        }

