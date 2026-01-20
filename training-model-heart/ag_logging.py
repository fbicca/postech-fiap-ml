#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#AG Módulo de logging estruturado para monitoramento e tracking de desempenho do algoritmo genético.
Implementa logging profissional com níveis, formatação e salvamento em arquivo.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json


def configurar_logger(nome_logger='AG', diretorio_logs='logs', nivel=logging.INFO, 
                     salvar_arquivo=True, formato_detalhado=True):
    """
    #AG Configura um logger estruturado para o algoritmo genético.
    
    Args:
        nome_logger (str): Nome do logger
        diretorio_logs (str): Diretório onde salvar logs
        nivel (int): Nível de logging (DEBUG, INFO, WARNING, ERROR)
        salvar_arquivo (bool): Se True, salva logs em arquivo
        formato_detalhado (bool): Se True, usa formato detalhado com timestamp
        
    Returns:
        logging.Logger: Logger configurado
    """
    logger = logging.getLogger(nome_logger)
    logger.setLevel(nivel)
    
    # Remove handlers existentes para evitar duplicação
    logger.handlers.clear()
    
    # Formato detalhado
    if formato_detalhado:
        formato = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formato = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(nivel)
    console_handler.setFormatter(formato)
    logger.addHandler(console_handler)
    
    # Handler para arquivo (se solicitado)
    if salvar_arquivo:
        diretorio_logs = Path(diretorio_logs)
        diretorio_logs.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arquivo_log = diretorio_logs / f"ag_log_{timestamp}.log"
        
        file_handler = logging.FileHandler(arquivo_log, encoding='utf-8')
        file_handler.setLevel(nivel)
        file_handler.setFormatter(formato)
        logger.addHandler(file_handler)
        
        logger.info(f"#AG Logging iniciado. Logs serão salvos em: {arquivo_log}")
    
    return logger


def criar_logger_simples(nome='AG'):
    """
    #AG Cria um logger simples para uso rápido.
    
    Args:
        nome (str): Nome do logger
        
    Returns:
        logging.Logger: Logger configurado
    """
    return configurar_logger(nome_logger=nome, salvar_arquivo=False, formato_detalhado=False)


def salvar_configuracao_ag(config, diretorio_logs='logs', timestamp=None):
    """
    #AG Salva configuração do algoritmo genético em JSON para rastreabilidade.
    
    Args:
        config (dict): Configuração do AG
        diretorio_logs (str): Diretório onde salvar
        timestamp (str): Timestamp para nomeação. Se None, gera automaticamente
        
    Returns:
        Path: Caminho do arquivo JSON criado
    """
    diretorio_logs = Path(diretorio_logs)
    diretorio_logs.mkdir(parents=True, exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    arquivo_config = diretorio_logs / f"ag_config_{timestamp}.json"
    
    config_completa = {
        'timestamp': timestamp,
        'configuracao': config,
        'data_hora': datetime.now().isoformat()
    }
    
    with open(arquivo_config, 'w', encoding='utf-8') as f:
        json.dump(config_completa, f, indent=2, ensure_ascii=False)
    
    return arquivo_config


def salvar_historico_evolucao(historico, diretorio_logs='logs', nome_experimento=None, timestamp=None):
    """
    #AG Salva histórico de evolução do algoritmo genético em JSON.
    
    Args:
        historico (list): Lista de dicionários com histórico por geração
        diretorio_logs (str): Diretório onde salvar
        nome_experimento (str): Nome do experimento (opcional)
        timestamp (str): Timestamp para nomeação. Se None, gera automaticamente
        
    Returns:
        Path: Caminho do arquivo JSON criado
    """
    diretorio_logs = Path(diretorio_logs)
    diretorio_logs.mkdir(parents=True, exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Nome do arquivo
    if nome_experimento:
        # Remove caracteres especiais do nome
        nome_limpo = nome_experimento.replace(' ', '_').replace(':', '').replace('/', '_')
        arquivo_historico = diretorio_logs / f"ag_historico_{nome_limpo}_{timestamp}.json"
    else:
        arquivo_historico = diretorio_logs / f"ag_historico_{timestamp}.json"
    
    # Prepara dados para salvar
    dados_historico = {
        'timestamp': timestamp,
        'nome_experimento': nome_experimento,
        'data_hora': datetime.now().isoformat(),
        'historico': historico,
        'resumo': {
            'numero_geracoes': len(historico),
            'fitness_final_max': historico[-1]['fitness_max'] if historico else None,
            'fitness_final_medio': historico[-1]['fitness_medio'] if historico else None,
            'melhor_fitness_geral': max([h['fitness_max'] for h in historico]) if historico else None
        }
    }
    
    with open(arquivo_historico, 'w', encoding='utf-8') as f:
        json.dump(dados_historico, f, indent=2, ensure_ascii=False)
    
    return arquivo_historico


def criar_logger_por_modulo(nome_modulo):
    """
    #AG Cria logger específico para um módulo.
    
    Args:
        nome_modulo (str): Nome do módulo (ex: 'ag_algoritmo', 'ag_fitness')
        
    Returns:
        logging.Logger: Logger configurado para o módulo
    """
    logger = logging.getLogger(f'AG.{nome_modulo}')
    if not logger.handlers:
        # Se não tem handlers, herda do logger principal
        logger_principal = logging.getLogger('AG')
        if logger_principal.handlers:
            # Adiciona handlers do logger principal
            for handler in logger_principal.handlers:
                logger.addHandler(handler)
        else:
            # Se não existe logger principal, configura um básico
            configurar_logger('AG')
            logger_principal = logging.getLogger('AG')
            for handler in logger_principal.handlers:
                logger.addHandler(handler)
    
    logger.setLevel(logging.INFO)
    return logger

