#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#AG Módulo principal do algoritmo genético para otimização de hiperparâmetros.
Orquestra a evolução da população: inicialização, avaliação, seleção, cruzamento e mutação.
"""

import numpy as np
import time
from datetime import datetime
from pathlib import Path
from ag_criar_individuo import criar_individuo, copiar_individuo
from ag_criar_fitness import fitness, fitness_detalhado
from ag_selecao import selecao_mista, selecao_elitismo
from ag_cruzamento import cruzamento_uniforme, cruzamento_aritmetico, cruzamento_blend
from ag_mutacao import mutacao_uniforme, mutacao_gaussiana, mutacao_nao_uniforme
from ag_logging import criar_logger_por_modulo
from ag_visualizacao import gerar_grafico_evolucao_fitness, gerar_grafico_convergencia


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
        verbose=True,
        salvar_historico=True,
        diretorio_logs='logs',
        gerar_graficos=True
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
            salvar_historico (bool): Se True, salva histórico em arquivo
            diretorio_logs (str): Diretório para salvar logs e histórico
            gerar_graficos (bool): Se True, gera gráficos de evolução
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
        self.salvar_historico = salvar_historico
        self.diretorio_logs = Path(diretorio_logs)
        self.gerar_graficos = gerar_graficos
        
        # #AG Logger estruturado para monitoramento
        self.logger = criar_logger_por_modulo('ag_algoritmo')
        
        # Histórico da evolução
        self.historico_fitness = []
        self.melhor_individuo = None
        self.melhor_fitness = -np.inf
        
        # Tracking de tempo
        self.tempo_execucao = {
            'inicio': None,
            'fim': None,
            'duracao_total': None,
            'tempo_por_geracao': []
        }
        
        # Configuração salva para rastreabilidade
        self.configuracao = {
            'tamanho_populacao': tamanho_populacao,
            'n_geracoes': n_geracoes,
            'taxa_cruzamento': taxa_cruzamento,
            'taxa_mutacao': taxa_mutacao,
            'n_elites': n_elites,
            'metodo_selecao': metodo_selecao,
            'metodo_cruzamento': metodo_cruzamento,
            'metodo_mutacao': metodo_mutacao,
            'metric': metric,
            'cv_folds': cv_folds
        }
    
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
    
    def evoluir(self, X, y, nome_experimento=None):
        """
        #AG Executa a evolução completa do algoritmo genético.
        
        Args:
            X (np.array): Features de treinamento
            y (np.array): Labels de treinamento
            nome_experimento (str): Nome do experimento (para salvamento)
            
        Returns:
            dict: Melhor indivíduo encontrado
        """
        # #AG Inicia tracking de tempo
        self.tempo_execucao['inicio'] = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # #AG Log de início
        self.logger.info("#AG Iniciando algoritmo genético...")
        self.logger.info(f"#AG Configuração: População={self.tamanho_populacao}, Gerações={self.n_geracoes}, "
                        f"Taxa Cruzamento={self.taxa_cruzamento}, Taxa Mutação={self.taxa_mutacao}")
        
        if self.verbose:
            print("#AG Iniciando algoritmo genético...")
            print(f"#AG População: {self.tamanho_populacao}, Gerações: {self.n_geracoes}")
        
        # #AG Salva configuração
        if self.salvar_historico:
            from ag_logging import salvar_configuracao_ag
            config_path = salvar_configuracao_ag(self.configuracao, self.diretorio_logs, timestamp)
            self.logger.info(f"#AG Configuração salva em: {config_path}")
        
        # Inicializa população
        populacao = self.inicializar_populacao()
        self.logger.info(f"#AG População inicial criada com {len(populacao)} indivíduos")
        
        # Evolução por gerações
        for geracao in range(self.n_geracoes):
            tempo_geracao_inicio = time.time()
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
            tempo_geracao = time.time() - tempo_geracao_inicio
            
            self.historico_fitness.append({
                'geracao': geracao,
                'fitness_medio': fitness_medio,
                'fitness_max': fitness_max,
                'fitness_min': fitness_min,
                'tempo_segundos': tempo_geracao
            })
            self.tempo_execucao['tempo_por_geracao'].append(tempo_geracao)
            
            # #AG Log estruturado da geração
            self.logger.info(f"#AG Geração {geracao+1}/{self.n_geracoes} - "
                           f"Fitness: max={fitness_max:.4f}, médio={fitness_medio:.4f}, min={fitness_min:.4f}, "
                           f"Tempo: {tempo_geracao:.2f}s")
            
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
        
        # #AG Finaliza tracking de tempo
        self.tempo_execucao['fim'] = time.time()
        self.tempo_execucao['duracao_total'] = self.tempo_execucao['fim'] - self.tempo_execucao['inicio']
        
        # #AG Log de conclusão
        self.logger.info(f"#AG Algoritmo genético concluído!")
        self.logger.info(f"#AG Melhor fitness: {self.melhor_fitness:.4f}")
        self.logger.info(f"#AG Tempo total de execução: {self.tempo_execucao['duracao_total']:.2f} segundos "
                        f"({self.tempo_execucao['duracao_total']/60:.2f} minutos)")
        
        if self.verbose:
            print(f"#AG Algoritmo genético concluído!")
            print(f"#AG Melhor fitness: {self.melhor_fitness:.4f}")
            print(f"#AG Tempo total: {self.tempo_execucao['duracao_total']:.2f}s")
        
        # #AG Salva histórico em arquivo
        if self.salvar_historico:
            from ag_logging import salvar_historico_evolucao
            historico_path = salvar_historico_evolucao(
                self.historico_fitness, 
                self.diretorio_logs, 
                nome_experimento=nome_experimento,
                timestamp=timestamp
            )
            self.logger.info(f"#AG Histórico salvo em: {historico_path}")
        
        # #AG Gera gráficos de evolução
        if self.gerar_graficos and self.historico_fitness:
            try:
                if nome_experimento:
                    nome_limpo = nome_experimento.replace(' ', '_').replace(':', '').replace('/', '_')
                    grafico_evolucao_path = self.diretorio_logs / f"ag_evolucao_fitness_{nome_limpo}_{timestamp}.png"
                    grafico_convergencia_path = self.diretorio_logs / f"ag_convergencia_{nome_limpo}_{timestamp}.png"
                else:
                    grafico_evolucao_path = self.diretorio_logs / f"ag_evolucao_fitness_{timestamp}.png"
                    grafico_convergencia_path = self.diretorio_logs / f"ag_convergencia_{timestamp}.png"
                
                gerar_grafico_evolucao_fitness(self.historico_fitness, grafico_evolucao_path)
                gerar_grafico_convergencia(self.historico_fitness, grafico_convergencia_path)
                
                self.logger.info(f"#AG Gráficos gerados:")
                self.logger.info(f"#AG   - Evolução: {grafico_evolucao_path}")
                self.logger.info(f"#AG   - Convergência: {grafico_convergencia_path}")
            except Exception as e:
                self.logger.warning(f"#AG Erro ao gerar gráficos: {e}")
        
        return self.melhor_individuo
    
    def get_historico(self):
        """
        #AG Retorna o histórico da evolução.
        
        Returns:
            list: Lista de dicionários com histórico por geração
        """
        return self.historico_fitness
    
    def get_tempo_execucao(self):
        """
        #AG Retorna informações de tempo de execução.
        
        Returns:
            dict: Dicionário com informações de tempo
        """
        return self.tempo_execucao.copy()


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

