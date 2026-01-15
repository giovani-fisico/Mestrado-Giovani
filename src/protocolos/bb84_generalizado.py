import numpy as np
import qutip as qu
from math import cos, sin, pi, sqrt, log
from qutip import Qobj, basis, fidelity
from collections import Counter

# --- Funções auxiliares (sem modificações) ---


def ordenar_autoestados(operador):
    """
    Calcula, fixa a fase, ordena universalmente pela fase do autovalor e
    limpa numericamente os autoestados de um operador, eliminando valores muito próximos a zero.
    Esta é a versão robusta que funciona para qualquer d e k.

    Args:
        operador (Qobj): O operador quântico a ser diagonalizado.

    Returns:
        (list, list): Uma tupla contendo:
                      1. A lista de autovalores ordenados pela fase.
                      2. A lista de autoestados correspondentes ordenados.
    """

    d = operador.shape[0]   
    autovalores_originais, autoestados_originais = operador.eigenstates()

    #A seguir, multiplicarei os autoestados por uma fase global de tal forma que a primeira entrada de cada um deles seja um número real
    autoestados_fixados = []
    for vet in autoestados_originais:
        primeiro_elemento = vet.full()[0, 0] #extraindo o primeiro elemento do autoestado
        if np.abs(primeiro_elemento) < 1e-10: #para o caso de o primeiro elemento for 0
            fase = 1.0
            for elem in vet.full().flatten():
                if np.abs(elem) > 1e-10:
                    fase = elem / np.abs(elem)
                    break
        else:
            fase = primeiro_elemento / np.abs(primeiro_elemento) #extraindo a fase do primeiro elemento
        fase_global = np.conj(fase) 
        vet_fixado = fase_global * vet #multiplicando o autoestado pelo conjugado da fase do primeiro elemento
        autoestados_fixados.append(vet_fixado)

    # --- Reordenando autoestados de tal forma que a ordem de seus autovalores seja 1, w, w^2, ..., w^(d-1) ---

    ordem_potencias = list(range(d)) 
    w = complex(cos(2 * pi / d), sin(2 * pi / d))
    autovalores_alvo = [w**k for k in ordem_potencias] #ordem desejada de autovalores
    autoestados_reordenados = [None] * d #lista que conterá os autoestados ordenados da maneira desejada

    # Dicionário para mapear os autovalores calculados aos seus autoestados fixados.
    mapa_val_vet = dict(zip(autovalores_originais, autoestados_fixados))


    tolerance = 1e-9
    for i, alvo_autoval in enumerate(autovalores_alvo):
        for calc_val, calc_vec in mapa_val_vet.items():
            if abs(alvo_autoval - calc_val) < tolerance:
                autoestados_reordenados[i] = calc_vec
                break
    autoestados_limpos = []
    tolerancia = 1e-10

    for vet in autoestados_reordenados:
        if vet is None:
            autoestados_limpos.append(None)
            continue
        
        # Converte o Qobj para um array NumPy para manipulação
        dados_np = vet.full()
        
        # Itera sobre cada número complexo no array
        for i, x in np.ndenumerate(dados_np):
            real_part = x.real
            imag_part = x.imag

            # Se a parte real ou imaginária for muito pequena, zera ela
            if abs(real_part) < tolerancia:
                real_part = 0.0
            if abs(imag_part) < tolerancia:
                imag_part = 0.0
            
            # Atualiza o array com o valor limpo
            dados_np[i] = complex(real_part, imag_part)
            
        # Cria um novo Qobj com os dados limpos, mantendo as dimensões originais
        vetor_limpo = Qobj(dados_np, dims=vet.dims)
        autoestados_limpos.append(vetor_limpo)

    # 4. Retorna os resultados finais
    return autovalores_alvo, autoestados_limpos

def calcular_entropia(casos_Alice, casos_Bob, d):
    if len(casos_Alice) == 0:
        return 0.0
    casos_Alice_arr = np.array(casos_Alice, dtype=int)
    contagem_Alice = np.bincount(casos_Alice_arr, minlength=d)[:, np.newaxis]
    matriz_contagem = np.zeros((d, d), dtype=int)
    for bit_alice, bit_bob in zip(casos_Alice, casos_Bob):
        matriz_contagem[bit_alice, bit_bob] += 1
    
    matriz_prob = np.divide(matriz_contagem.astype(float), contagem_Alice, 
                           out=np.zeros_like(matriz_contagem, dtype=float), 
                           where=contagem_Alice!=0)
    
    prob_positivas = matriz_prob[matriz_prob > 0]
    if d > 1:
        entropia = -np.sum(prob_positivas * (np.log(prob_positivas) / np.log(d)))
    else: # Evita log(1) que é 0, levando a divisão por zero se houver probs
        entropia = 0.0
    return entropia

def medir_em_base(estado, base_autoestados):
    d = len(base_autoestados)
    probabilidades = [np.abs(estado.overlap(autoestado))**2 for autoestado in base_autoestados]
    prob_array = np.array(probabilidades)
    prob_array /= np.sum(prob_array)
    indice = np.random.choice(d, p=prob_array)
    return indice, base_autoestados[indice]

# --- Função de simulação "silenciosa" ---

def simular_protocolo_qkd_silencioso(matriz_base1, matriz_base2, d, m):
    """
    Executa UMA simulação completa do QKD e retorna a Figura de Mérito.
    Esta versão não imprime resultados, apenas o valor final.
    """
    autoestados_base1 = [Qobj(matriz_base1[:, i]) for i in range(d)]
    autoestados_base2 = [Qobj(matriz_base2[:, i]) for i in range(d)]
    
    msg_Alice = np.random.randint(d, size=m)
    bases_Alice = np.random.randint(2, size=m)

    qudits = [autoestados_base1[msg_Alice[i]] if bases_Alice[i] == 0 else autoestados_base2[msg_Alice[i]] for i in range(m)]

    bases_Bob = np.random.randint(2, size=m)
    msg_Bob = []
    for i in range(m):
        base_de_medicao = autoestados_base1 if bases_Bob[i] == 0 else autoestados_base2
        medida, _ = medir_em_base(qudits[i], base_de_medicao)
        msg_Bob.append(medida)

    alice_coincide, alice_incoincide = [], []
    bob_coincide, bob_incoincide = [], []

    for b_alice, b_bob, val_alice, val_bob in zip(bases_Alice, bases_Bob, msg_Alice, msg_Bob):
        if b_alice == b_bob:
            alice_coincide.append(val_alice)
            bob_coincide.append(val_bob)
        else:
            alice_incoincide.append(val_alice)
            bob_incoincide.append(val_bob)

    Sc = calcular_entropia(alice_coincide, bob_coincide, d)
    Si = calcular_entropia(alice_incoincide, bob_incoincide, d)
    
    figura_de_merito = (Si - Sc) / d if d != 0 else 0.0
        
    return figura_de_merito

d=5
# --- SCRIPT PRINCIPAL PARA ANÁLISE ESTATÍSTICA ---
Zd_np = np.zeros((d, d), dtype=np.complex128)
w = complex(cos(2 * pi / d), sin(2 * pi / d))
for i in range(d):
    Zd_np[i, i] = w**i
Zd = Qobj(Zd_np)


# --- Encontrando autoestados e autovalores de XZ e X ---
Xd_np = np.zeros((d, d), dtype=np.complex128)
Xd_np[0, d - 1] = 1
for i in range(d - 1):
    Xd_np[i + 1, i] = 1
Xd = Qobj(Xd_np)
XZ = Xd * Zd

autovalores_XZ, autoestados_XZ = ordenar_autoestados(XZ)

autovalores_X, autoestados_X = ordenar_autoestados(Xd)


if __name__ == "__main__":
    # 1. Defina os parâmetros da análise
    d = 5  # <--- Dimensão a ser testada
    m_por_simulacao = 20000  # Tamanho da mensagem em CADA simulação
    num_execucoes = 30      # Quantas vezes vamos rodar a simulação completa

    # 2. Seleção das matrizes com base na dimensão 'd'
    U1, U2 = None, None
    if d == 2:
        U1 = np.array([[-0.402824+0.581148j, 0.416647-0.571319j], [0.416647-0.571319j, 0.430231-0.561161j]])
        U2 = np.array([[-0.402824+0.581148j, 0.416647-0.571319j], [0.571319+0.416647j, 0.561161+0.430231j]])
    elif d == 3:
        U1 = np.array([[0.487394+0.309485j, 0.575538+0.0457148j, -0.511718+0.267353j], [0.575538+0.0457148j, -0.0670419+0.573445j, 0.575538+0.0457148j], [-0.511718+0.267353j, 0.575538+0.0457148j, 0.487394+0.309485j]])
        U2 = np.array([[0.487394+0.309485j, 0.575538+0.0457148j, -0.511718+0.267353j], [-0.327359+0.475573j, -0.463097-0.344782j, -0.327359+0.475573j], [0.0243246-0.576838j, -0.327359+0.475573j, -0.511718+0.267353j]])
    elif d == 4:
        U1 = np.array([[-0.460108+0.195706j, 0.48477-0.122466j, -0.429896+0.255323j, 0.389412-0.313622j], [0.48477-0.122466j, -0.461535+0.192316j, -0.387748+0.315676j, 0.433089-0.249867j], [-0.429896+0.255323j, -0.387748+0.315676j, -0.281915+0.412945j, -0.344105+0.362756j], [0.389412-0.313622j, 0.433089-0.249867j, -0.344105+0.362756j, -0.289273+0.407825j]])
        U2 = np.array([[-0.460108+0.195706j, 0.48477-0.122466j, -0.429896+0.255323j, 0.389412-0.313622j], [-0.48477+0.122467j, 0.461535-0.192317j, 0.387748-0.315676j, -0.433089+0.249868j], [0.429896-0.255323j, 0.387748-0.315676j, 0.281915-0.412946j, 0.344105-0.362756j], [-0.389411+0.313622j, -0.433089+0.249868j, 0.344105-0.362756j, 0.289273-0.407825j]])
    elif d == 5:
        U1 = np.array([[0.0099761+0.443416j, 0.282795+0.349522j, -0.0148809-0.444292j, -0.241719+0.371378j, 0.267151-0.368522j], [0.282795+0.349522j, 0.0668591+0.436427j, -0.411391+0.179198j, -0.297668-0.334562j, -0.234102+0.382395j], [-0.0148809-0.444292j, -0.411391+0.179198j, -0.416428+0.166938j, -0.421442+0.162008j, -0.0228067-0.442005j], [-0.241719+0.371378j, -0.297668-0.334562j, -0.421442+0.162008j, 0.0443289+0.4494j, 0.279279+0.342536j], [0.267151-0.368522j, -0.234102+0.382395j, -0.0228067-0.442005j, 0.279279+0.342536j, 0.00708932+0.447801j]])
        U2 = np.array([[0.0099761+0.443416j, 0.282795+0.349522j, -0.0148809-0.444292j, -0.241719+0.371378j, 0.267151-0.368522j], [-0.0297718-0.448611j, 0.196793-0.395236j, 0.439495+0.0905489j, 0.0505473+0.444953j, 0.411656-0.177678j], [0.416502-0.15538j, -0.300708-0.333061j, -0.290685-0.341735j, -0.287606-0.348056j, 0.411813-0.162167j], [-0.243453+0.370244j, -0.2961-0.33595j, -0.422195+0.160035j, 0.0422271+0.449602j, 0.277674+0.343838j], [0.432756+0.14107j, -0.435809-0.105358j, 0.413642-0.157443j, -0.240218+0.370975j, -0.423985+0.144265j]])

    if U1 is None or U2 is None:
        print(f"Erro: Dimensão d={d} não suportada. Escolha um valor entre 2 e 5.")
    else:
        print(f"Iniciando análise estatística para d={d} com {num_execucoes} execuções...")
        
        # 3. Rode o "loop externo" para coletar os dados
        lista_de_resultados = []
        for i in range(num_execucoes):
            print(f"  Executando rodada {i+1}/{num_execucoes}...")
            resultado = simular_protocolo_qkd_silencioso(autoestados_XZ, autoestados_X, d, m_por_simulacao)
            lista_de_resultados.append(resultado)

        print("Análise concluída.")

        # 4. Calcule e exiba as estatísticas
        resultados_np = np.array(lista_de_resultados)
        media = np.mean(resultados_np)
        desvio_padrao = np.std(resultados_np)
        erro_padrao = desvio_padrao / np.sqrt(num_execucoes)
        
        print("\n--- Resultados da Análise Estatística ---")
        print(f"Figura de Mérito Ideal Esperada: 1.0000")
        print("-" * 45)
        print(f"Média da Figura de Mérito:          {media:.5f}")
        print(f"Desvio Padrão (σ):                  {desvio_padrao:.5f}")
        print(f"Erro Padrão da Média (SEM):         {erro_padrao:.5f}")
        print("-" * 45)
        print(f"Resultado Final (Média ± Erro Padrão): {media:.5f} ± {erro_padrao:.5f}")