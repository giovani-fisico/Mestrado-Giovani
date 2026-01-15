import numpy as np
import qutip as qu
from math import sqrt
from qutip import Qobj

def medir_em_base(estado, base_autoestados):
    """
    Mede um estado em uma base específica, com normalização robusta.
    """
    d = len(base_autoestados)
    probabilidades = [np.abs(estado.overlap(autoestado))**2 for autoestado in base_autoestados]
    prob_array = np.array(probabilidades, dtype=float)
    prob_array /= np.sum(prob_array)
    indice = np.random.choice(d, p=prob_array)
    return indice, base_autoestados[indice]

def run_single_simulation(matriz_base1, matriz_base2, d, tentativas):
    """
    Executa UMA simulação completa do QRAC e retorna a probabilidade de sucesso.
    Esta versão não imprime resultados, apenas retorna o valor final.
    """
    autoestados_base1 = [Qobj(matriz_base1[:, i]) for i in range(d)]
    autoestados_base2 = [Qobj(matriz_base2[:, i]) for i in range(d)]

    mensagem = np.empty((d, d), dtype=object)
    for m in range(d):
        for n in range(d):
            psi_m = autoestados_base1[m]
            psi_n = autoestados_base2[n]
            overlap = psi_m.overlap(psi_n)
            fase_correcao = np.conj(overlap) / np.abs(overlap) if np.abs(overlap) > 1e-9 else 1.0
            estado_soma = psi_m + fase_correcao * psi_n
            mensagem[m, n] = estado_soma.unit()

    acertos = 0
    for _ in range(tentativas):
        m_aleatorio = np.random.randint(d)
        n_aleatorio = np.random.randint(d)
        qudit_enviado = mensagem[m_aleatorio, n_aleatorio]
        escolha_de_bob = np.random.randint(2)

        if escolha_de_bob == 0:
            palpite, _ = medir_em_base(qudit_enviado, autoestados_base1)
            if m_aleatorio == palpite:
                acertos += 1
        else:
            palpite, _ = medir_em_base(qudit_enviado, autoestados_base2)
            if n_aleatorio == palpite:
                acertos += 1

    return acertos / tentativas

# --- SCRIPT PRINCIPAL PARA ANÁLISE ESTATÍSTICA ---

if __name__ == "__main__":
    # 1. Defina suas matrizes
    
    #da forma como está, estão sendo escolhidas bases mutuamente não enviesadas de dimensão 5
    U1 = np.array([[0.0099761+0.443416j, 0.282795+0.349522j, -0.0148809-0.444292j, -0.241719+0.371378j, 0.267151-0.368522j], [0.282795+0.349522j, 0.0668591+0.436427j, -0.411391+0.179198j, -0.297668-0.334562j, -0.234102+0.382395j], [-0.0148809-0.444292j, -0.411391+0.179198j, -0.416428+0.166938j, -0.421442+0.162008j, -0.0228067-0.442005j], [-0.241719+0.371378j, -0.297668-0.334562j, -0.421442+0.162008j, 0.0443289+0.4494j, 0.279279+0.342536j], [0.267151-0.368522j, -0.234102+0.382395j, -0.0228067-0.442005j, 0.279279+0.342536j, 0.00708932+0.447801j]])
    U2 = np.array([[0.0099761+0.443416j, 0.282795+0.349522j, -0.0148809-0.444292j, -0.241719+0.371378j, 0.267151-0.368522j], [-0.0297718-0.448611j, 0.196793-0.395236j, 0.439495+0.0905489j, 0.0505473+0.444953j, 0.411656-0.177678j], [0.416502-0.15538j, -0.300708-0.333061j, -0.290685-0.341735j, -0.287606-0.348056j, 0.411813-0.162167j], [-0.243453+0.370244j, -0.2961-0.33595j, -0.422195+0.160035j, 0.0422271+0.449602j, 0.277674+0.343838j], [0.432756+0.14107j, -0.435809-0.105358j, 0.413642-0.157443j, -0.240218+0.370975j, -0.423985+0.144265j]])
    
    # 2. Defina os parâmetros da análise
    d = 5
    num_tentativas_por_simulacao = 100000  # "M" - tentativas dentro de CADA simulação
    num_execucoes = 30                     # "N" - quantas vezes vamos rodar a simulação completa

    print(f"Iniciando análise estatística com {num_execucoes} execuções...")

    # 3. Rode o "loop externo" para coletar os dados
    lista_de_resultados = []
    for i in range(num_execucoes):
        print(f"  Executando rodada {i+1}/{num_execucoes}...")
        probabilidade = run_single_simulation(U1, U2, d, num_tentativas_por_simulacao)
        lista_de_resultados.append(probabilidade)

    print("Análise concluída.")
    
    # 4. Calcule e exiba as estatísticas
    resultados_np = np.array(lista_de_resultados)
    media = np.mean(resultados_np)
    desvio_padrao = np.std(resultados_np)
    erro_padrao = desvio_padrao / np.sqrt(num_execucoes)
    
    probabilidade_esperada = 0.5 * (1 + (1 / sqrt(d)))

    print("\n--- Resultados da Análise Estatística ---")
    print(f"Probabilidade Teórica Esperada: {probabilidade_esperada:.4f}")
    print("-" * 40)
    print(f"Média das probabilidades:         {media:.4f}")
    print(f"Desvio Padrão (σ):                {desvio_padrao:.4f}")
    print(f"Erro Padrão da Média (SEM):       {erro_padrao:.4f}")
    print("-" * 40)
    print(f"Resultado Final (Média ± Erro Padrão): {media:.4f} ± {erro_padrao:.4f}")