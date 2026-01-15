# Mestrado-Giovani

Este repositório contém as ferramentas computacionais desenvolvidas para o design e validação de interferômetros de guias de onda (chips fotônicos). O objetivo principal é a identificação de parâmetros físicos que permitam a geração de Bases Mutuamente Não Enviesadas (MUB).

O projeto está dividido entre a busca numérica de hardware (Julia) e a simulação de protocolos de informação quântica (Python).

src/ (Julia)
Scripts focados em performance para a busca de parâmetros físicos ($L, \delta, \kappa, \theta$):
chip_2d.jl & chip_3d.jl: Algoritmos de busca por força bruta utilizando avaliação de curto-circuito para encontrar matrizes de transferência ideais.
NBS_busca_continua_CMAES.jl: Implementação da estratégia evolutiva CMA-ES para encontrar divisores de feixe simétricos (NBS) de alta dimensão.
CMA_MUB.jl: Otimizador para encontrar a matriz $U_2$ (MUB) a partir de uma matriz $U_1$ de referência carregada de arquivo CSV.

src/protocolos/ (Python)
Validação das matrizes encontradas em cenários de uso real:
bb84_generalizado.py: Simulação Monte Carlo do protocolo de distribuição de chaves quânticas em dimensões $d > 2$.
qrac_generalizado.py: Teste de códigos de acesso aleatório quântico (Quantum Random Access Codes).Utiliza a biblioteca QuTiP para modelagem de estados e operadores.

Requisitos
Julia 1.9+: LinearAlgebra, Symbolics, CMAEvolutionStrategy, Polynomials, CSV, DataFrames.
Python 3.8+: numpy, qutip, pandas.
