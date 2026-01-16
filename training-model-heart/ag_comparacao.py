#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#AG Módulo para comparação de desempenho entre modelos otimizados (com AG) e originais (sem AG).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


def treinar_modelo_original(X_train, y_train, random_state=42):
    """
    #AG Treina um modelo LogisticRegression com hiperparâmetros padrão (sem otimização).
    
    Args:
        X_train (np.array): Features de treinamento
        y_train (np.array): Labels de treinamento
        random_state (int): Seed para reprodutibilidade
        
    Returns:
        sklearn.linear_model.LogisticRegression: Modelo treinado
    """
    # #AG Modelo original com hiperparâmetros padrão (sem otimização)
    model = LogisticRegression(
        solver='liblinear',
        C=1.0,  # valor padrão
        penalty='l2',  # padrão
        class_weight=None,  # padrão
        max_iter=1000,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    return model


def avaliar_modelo(model, X_test, y_test):
    """
    #AG Avalia um modelo calculando todas as métricas relevantes.
    
    Args:
        model: Modelo treinado
        X_test (np.array): Features de teste
        y_test (np.array): Labels de teste
        
    Returns:
        dict: Dicionário com todas as métricas
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='binary', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='binary', zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, average='binary', zero_division=0)),
        'auc': float(roc_auc_score(y_test, y_prob)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return metrics


def comparar_modelos(modelo_original, modelo_otimizado, X_test, y_test):
    """
    #AG Compara o desempenho entre modelo original e modelo otimizado.
    
    Args:
        modelo_original: Modelo treinado com hiperparâmetros padrão
        modelo_otimizado: Modelo treinado com hiperparâmetros otimizados pelo AG
        X_test (np.array): Features de teste
        y_test (np.array): Labels de teste
        
    Returns:
        dict: Dicionário com comparação detalhada
    """
    # Avalia ambos os modelos
    metrics_original = avaliar_modelo(modelo_original, X_test, y_test)
    metrics_otimizado = avaliar_modelo(modelo_otimizado, X_test, y_test)
    
    # Calcula melhorias/diferenças
    melhorias = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        diff = metrics_otimizado[metric] - metrics_original[metric]
        pct_improvement = (diff / metrics_original[metric] * 100) if metrics_original[metric] > 0 else 0.0
        melhorias[metric] = {
            'diferenca': float(diff),
            'percentual': float(pct_improvement),
            'melhor': 'otimizado' if diff > 0 else 'original'
        }
    
    comparacao = {
        'modelo_original': metrics_original,
        'modelo_otimizado': metrics_otimizado,
        'melhorias': melhorias,
        'resumo': {
            'melhor_accuracy': 'otimizado' if melhorias['accuracy']['diferenca'] > 0 else 'original',
            'melhor_recall': 'otimizado' if melhorias['recall']['diferenca'] > 0 else 'original',
            'melhor_f1': 'otimizado' if melhorias['f1']['diferenca'] > 0 else 'original',
            'melhor_auc': 'otimizado' if melhorias['auc']['diferenca'] > 0 else 'original',
        }
    }
    
    return comparacao


def imprimir_comparacao(comparacao, hiperparametros_otimizados=None):
    """
    #AG Imprime uma tabela comparativa formatada dos resultados.
    
    Args:
        comparacao (dict): Resultado da função comparar_modelos
        hiperparametros_otimizados (dict): Hiperparâmetros otimizados pelo AG
    """
    print("\n" + "="*80)
    print("#AG COMPARAÇÃO: MODELO ORIGINAL vs MODELO OTIMIZADO (ALGORITMO GENÉTICO)")
    print("="*80)
    
    if hiperparametros_otimizados:
        print("\n#AG Hiperparâmetros Otimizados pelo AG:")
        print(f"#AG   C: {hiperparametros_otimizados['C']:.4f}")
        print(f"#AG   penalty: {hiperparametros_otimizados['penalty']}")
        print(f"#AG   class_weight: {hiperparametros_otimizados['class_weight']}")
        print(f"#AG   max_iter: {hiperparametros_otimizados['max_iter']}")
    
    print("\n#AG Métricas de Desempenho:")
    print("-" * 80)
    print(f"{'Métrica':<15} {'Original':<12} {'Otimizado':<12} {'Diferença':<12} {'Melhoria %':<12}")
    print("-" * 80)
    
    metrics_names = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score',
        'auc': 'AUC'
    }
    
    for metric, nome in metrics_names.items():
        orig = comparacao['modelo_original'][metric]
        otim = comparacao['modelo_otimizado'][metric]
        diff = comparacao['melhorias'][metric]['diferenca']
        pct = comparacao['melhorias'][metric]['percentual']
        
        sinal = "+" if diff >= 0 else ""
        print(f"{nome:<15} {orig:<12.4f} {otim:<12.4f} {sinal+str(diff):<12.4f} {sinal+str(pct):<11.2f}%")
    
    print("-" * 80)
    print("\n#AG Resumo:")
    resumo = comparacao['resumo']
    print(f"#AG   Melhor Accuracy: {resumo['melhor_accuracy']}")
    print(f"#AG   Melhor Recall: {resumo['melhor_recall']}")
    print(f"#AG   Melhor F1-Score: {resumo['melhor_f1']}")
    print(f"#AG   Melhor AUC: {resumo['melhor_auc']}")
    print("="*80 + "\n")

