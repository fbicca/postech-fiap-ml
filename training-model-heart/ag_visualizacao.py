#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#AG Módulo para visualização e gráficos de monitoramento do algoritmo genético.
Gera gráficos de evolução do fitness, convergência, etc.
"""

import matplotlib
matplotlib.use('Agg')  # Para evitar problemas de display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def gerar_grafico_evolucao_fitness(historico, caminho_saida=None, titulo="Evolução do Fitness - Algoritmo Genético"):
    """
    #AG Gera gráfico da evolução do fitness ao longo das gerações.
    
    Args:
        historico (list): Lista de dicionários com histórico por geração
        caminho_saida (Path ou str): Caminho para salvar o gráfico. Se None, retorna figura
        titulo (str): Título do gráfico
        
    Returns:
        matplotlib.figure.Figure: Figura criada (se caminho_saida=None)
    """
    if not historico:
        raise ValueError("Histórico vazio. Não é possível gerar gráfico.")
    
    # Extrai dados
    geracoes = [h['geracao'] for h in historico]
    fitness_max = [h['fitness_max'] for h in historico]
    fitness_medio = [h['fitness_medio'] for h in historico]
    fitness_min = [h['fitness_min'] for h in historico]
    
    # Cria figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plota linhas
    ax.plot(geracoes, fitness_max, label='Fitness Máximo', linewidth=2, color='#2ecc71', marker='o', markersize=4)
    ax.plot(geracoes, fitness_medio, label='Fitness Médio', linewidth=2, color='#3498db', marker='s', markersize=4)
    ax.plot(geracoes, fitness_min, label='Fitness Mínimo', linewidth=1.5, color='#e74c3c', marker='^', markersize=3, alpha=0.7)
    
    # Preenche área entre max e min (opcional)
    ax.fill_between(geracoes, fitness_min, fitness_max, alpha=0.1, color='#3498db')
    
    # Configurações
    ax.set_xlabel('Geração', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fitness', fontsize=12, fontweight='bold')
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, len(geracoes) - 1)
    
    # Adiciona informações estatísticas
    melhor_geracao = np.argmax(fitness_max)
    ax.axvline(x=melhor_geracao, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(melhor_geracao, fitness_max[melhor_geracao], 
           f'Melhor: {fitness_max[melhor_geracao]:.4f}\nGeração: {melhor_geracao}', 
           fontsize=9, ha='right', va='bottom',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Salva ou retorna
    if caminho_saida:
        caminho_saida = Path(caminho_saida)
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(caminho_saida, dpi=150, bbox_inches='tight')
        plt.close()
        return caminho_saida
    else:
        return fig


def gerar_grafico_convergencia(historico, caminho_saida=None, titulo="Convergência do Algoritmo Genético"):
    """
    #AG Gera gráfico de convergência (melhoria do fitness ao longo das gerações).
    
    Args:
        historico (list): Lista de dicionários com histórico por geração
        caminho_saida (Path ou str): Caminho para salvar o gráfico
        titulo (str): Título do gráfico
        
    Returns:
        Path: Caminho do arquivo salvo
    """
    if not historico:
        raise ValueError("Histórico vazio. Não é possível gerar gráfico.")
    
    fitness_max = [h['fitness_max'] for h in historico]
    
    # Calcula melhor fitness acumulado (não decresce)
    melhor_acumulado = []
    melhor_atual = -np.inf
    for f in fitness_max:
        if f > melhor_atual:
            melhor_atual = f
        melhor_acumulado.append(melhor_atual)
    
    # Calcula taxa de melhoria (diferença entre gerações)
    taxa_melhoria = [0]
    for i in range(1, len(fitness_max)):
        taxa_melhoria.append(fitness_max[i] - fitness_max[i-1])
    
    geracoes = list(range(len(historico)))
    
    # Cria figura com subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Subplot 1: Evolução do melhor fitness
    ax1.plot(geracoes, fitness_max, label='Fitness Máximo por Geração', 
            linewidth=2, color='#3498db', marker='o', markersize=3)
    ax1.plot(geracoes, melhor_acumulado, label='Melhor Fitness Acumulado', 
            linewidth=2.5, color='#2ecc71', linestyle='--', marker='s', markersize=3)
    ax1.set_ylabel('Fitness', fontsize=11, fontweight='bold')
    ax1.set_title('Evolução do Melhor Fitness', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Subplot 2: Taxa de melhoria
    cores_taxa = ['green' if t > 0 else 'red' if t < 0 else 'gray' for t in taxa_melhoria]
    ax2.bar(geracoes, taxa_melhoria, color=cores_taxa, alpha=0.6, width=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Geração', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Taxa de Melhoria', fontsize=11, fontweight='bold')
    ax2.set_title('Taxa de Melhoria por Geração', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.suptitle(titulo, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Salva
    if caminho_saida:
        caminho_saida = Path(caminho_saida)
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(caminho_saida, dpi=150, bbox_inches='tight')
        plt.close()
        return caminho_saida
    else:
        return fig


def gerar_grafico_distribuicao_fitness(historico_geracao_atual, caminho_saida=None, geracao=0):
    """
    #AG Gera gráfico de distribuição do fitness na geração atual.
    
    Args:
        historico_geracao_atual (list): Lista de fitness de todos os indivíduos
        caminho_saida (Path ou str): Caminho para salvar o gráfico
        geracao (int): Número da geração
        
    Returns:
        Path: Caminho do arquivo salvo
    """
    if not historico_geracao_atual:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histograma
    ax.hist(historico_geracao_atual, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    
    # Linhas estatísticas
    media = np.mean(historico_geracao_atual)
    mediana = np.median(historico_geracao_atual)
    desvio = np.std(historico_geracao_atual)
    
    ax.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Média: {media:.4f}')
    ax.axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.4f}')
    
    ax.set_xlabel('Fitness', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequência', fontsize=11, fontweight='bold')
    ax.set_title(f'Distribuição do Fitness - Geração {geracao}', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Adiciona estatísticas no gráfico
    texto_stats = f'μ = {media:.4f}\nσ = {desvio:.4f}\nMin = {min(historico_geracao_atual):.4f}\nMax = {max(historico_geracao_atual):.4f}'
    ax.text(0.98, 0.98, texto_stats, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if caminho_saida:
        caminho_saida = Path(caminho_saida)
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(caminho_saida, dpi=150, bbox_inches='tight')
        plt.close()
        return caminho_saida
    else:
        return fig

