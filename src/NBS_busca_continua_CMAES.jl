# ===================================================================
# PARTE 0: PACOTES E A FUNÇÃO BASE
# ===================================================================
using LinearAlgebra
using Statistics
using CMAEvolutionStrategy
using Polynomials
using CSV

# Função auxiliar para gerar Uₙ(x)
function get_chebyshev_u(n::Int)
    n < 0 && return Polynomial([0.0])
    n == 0 && return Polynomial([1.0])
    x = variable(Polynomial{Float64})
    U0 = Polynomial([1.0]); U1 = 2*x
    n == 1 && return U1
    for _ in 2:n
        Unext = 2*x*U1 - U0; U0 = U1; U1 = Unext
    end
    return U1
end

function calculate_nbs_numerically_FAST(n_val::Int, delta_val::Float64, kappa_val::Float64, L_val::Float64)
    if abs(kappa_val) < 1e-9
        return nothing
    end

    x = variable(Polynomial{Float64})
    Un = get_chebyshev_u(n_val)
    Un_minus_2 = get_chebyshev_u(n_val - 2)
    Un_minus_3 = get_chebyshev_u(n_val - 3)

    poly_numerical = (kappa_val^2 * Un
                     + (delta_val^2 - 4 * kappa_val * delta_val * x) * Un_minus_2
                     + 2 * kappa_val * delta_val * Un_minus_3)
    
    solutions_for_x = roots(poly_numerical)

    if isempty(solutions_for_x); return nothing; end

    function CalcEigenvector_Numerical(N_size, x_k, d_val, k_val)
        v = zeros(ComplexF64, N_size)
        factor = -d_val / k_val + 2 * x_k
        for i in 1:N_size
            if i == 1; v[i] = abs(factor) < 1e-9 ? 1.0 : factor
            elseif i < N_size
                u_i_minus_1 = get_chebyshev_u(i - 1)(x_k)
                u_i_minus_2 = get_chebyshev_u(i - 2)(x_k)
                term = u_i_minus_1 - (d_val / k_val) * u_i_minus_2
                v[i] = factor * term
            else
                if abs(v[i-1]) < 1e-9; v[i] = -v[1]; else
                    u_n_minus_2 = get_chebyshev_u(N_size - 2)(x_k)
                    u_n_minus_3 = get_chebyshev_u(N_size - 3)(x_k)
                    term = u_n_minus_2 - (d_val / k_val) * u_n_minus_3
                    v[i] = term
                end
            end
        end; return v
    end

    NormalizeEigenvector(v) = (norm_v = norm(v); norm_v > 1e-9 ? v / norm_v : v)
    
    eigenvectors = Vector{ComplexF64}[]
    diagonal_elements = ComplexF64[]
    for x_k in solutions_for_x
        if abs(imag(x_k)) > 1e-9; continue; end
        x_k_real = real(x_k)
        lambda_k = -delta_val / 2 + 2 * kappa_val * x_k_real
        v_k = CalcEigenvector_Numerical(n_val, x_k_real, delta_val, kappa_val)
        push!(eigenvectors, NormalizeEigenvector(v_k)); push!(diagonal_elements, exp(im * L_val * lambda_k))
    end
    
    if isempty(eigenvectors); @warn "Nenhuma solução real encontrada."; return nothing; end

    if length(eigenvectors) != n_val
        return nothing
    end

    try  # O try/catch para SingularException (matriz não invertível)
        MatrixEigenvectors = hcat(eigenvectors...)
        M_kk = diagm(diagonal_elements)
        
        NBS_julia = MatrixEigenvectors * M_kk * MatrixEigenvectors'
        
        prod = NBS_julia * transpose(NBS_julia) 
        
        return NBS_julia, prod
    catch e
        if e isa SingularException
            return nothing
        else
            rethrow(e)
        end
    end
end

# Função auxiliar para salvar os parâmetros em um CSV
function save_params_to_csv(filepath::String, std_val::Float64, params::AbstractVector{Float64}, N_DIM::Int)
    L, delta, kappa = params[1], params[2], params[3]
    thetas = params[4:end]

    header_exists = isfile(filepath) && filesize(filepath) > 0

    # Abre o arquivo em modo append ("a")
    open(filepath, "a") do io
        # Escreve o cabeçalho se o arquivo for novo ou estiver vazio
        if !header_exists
            header = "STD,L,delta,kappa," * join(["theta$i" for i in 1:N_DIM], ",") * "\n"
            write(io, header)
        end
        # Escreve a linha de dados
        data_row = "$std_val,$L,$delta,$kappa," * join(thetas, ",") * "\n"
        write(io, data_row)
    end
end

# ===================================================================
# PARTE 1: O MODELO FÍSICO (COM ASSINATURA CORRIGIDA)
# ===================================================================
# MUDANÇA AQUI: Vector{Float64} -> AbstractVector{Float64}
function mach_zender_nd(params::AbstractVector{Float64}, n::Int)
    L, δ, kappa = params[1], params[2], params[3]
    θ = params[4:end]
    if length(θ) != n
        error("O número de ângulos (thetas) deve ser igual à dimensão n.")
    end
    resultado_bs = calculate_nbs_numerically_FAST(n, δ, kappa, L)
    isnothing(resultado_bs) && return nothing
    bs_matrix, _ = resultado_bs #matriz do divisor de feixes
    pm_matrix = diagm(exp.(im .* θ)) #matriz de fase
    return bs_matrix * pm_matrix * bs_matrix #matriz do chip
end



# ===================================================================
# PARTE 2: O PROCESSO DE OTIMIZAÇÃO E ANÁLISE FINAL
# ===================================================================

# --- Função de Custo Robusta (Soma Ponderada) ---
function fitness_function(params::AbstractVector{Float64}, n::Int, weight::Float64)
    mz_matrix = mach_zender_nd(params, n)
    isnothing(mz_matrix) && return Inf
    
    magnitudes = abs.(mz_matrix)
    std_val = std(magnitudes)
    mean_val = mean(magnitudes)
    target_mean = 1.0 / sqrt(n)
    mean_error = abs(mean_val - target_mean)
    
    return std_val + weight * mean_error
end


# --- Função Principal em MODO DE DESCOBERTA CONTÍNUA ---
function main()
    N_DIM = 2 # Dimensão para a otimização
    
    # --- HIPERPARÂMETROS DA BUSCA ---
    FAIXA_DE_PESOS = 10.0:1.0:100.0
    POP_SIZE = 400
    MAX_ITER = 10000 

    # --- NOMES DOS ARQUIVOS DE SAÍDA ---
    PARAMS_FILE = "campeoes_std_menor_001_parametros_teste.csv"
    STD_THRESHOLD = 0.01 # Limiar de 1% para o desvio padrão

    println("Iniciando MODO DE DESCOBERTA CONTÍNUA para um sistema de $N_DIM dimensões.")
    println("Pressione Ctrl+C a qualquer momento para parar e ver o melhor resultado.")

    # --- Variáveis para guardar o "Campeão dos Campeões" ---
    overall_best_std = Inf
    overall_best_params = nothing
    overall_best_weight = 0.0
    run_count = 0

    try
        while true #código permanecerá procurando pelos parâmetros e ao se apertar ctrl + c, será mostrado o melhor obtido.
            run_count += 1
            println("\n" * "*"^50)
            println("--- INICIANDO TENTATIVA DE BUSCA #$run_count ---")
            
            peso_sorteado = rand(FAIXA_DE_PESOS)
            
            L_bounds     = (20.0, 30.0); delta_bounds = (0.0, 0.5); kappa_bounds = (0.0, 1.0); theta_bounds = (0.0, 2 * pi)
            lower_bounds = [L_bounds[1], delta_bounds[1], kappa_bounds[1], fill(theta_bounds[1], N_DIM)...]
            upper_bounds = [L_bounds[2], delta_bounds[2], kappa_bounds[2], fill(theta_bounds[2], N_DIM)...]
            x0_aleatorio = lower_bounds .+ rand(length(lower_bounds)) .* (upper_bounds .- lower_bounds)
            sigma0 = 0.3
            
            println("Configuração: Peso = $(round(peso_sorteado, digits=2)), Ponto de Partida Aleatório")
            println("*"^50 * "\n")

            f_to_optimize(p) = fitness_function(p, N_DIM, peso_sorteado)

            result = minimize(f_to_optimize, x0_aleatorio, sigma0; 
                           lower=lower_bounds, upper=upper_bounds,
                           popsize = POP_SIZE, maxiter = MAX_ITER, verbosity=0, multi_threading=true)

            xbest_list = result.logger.xbest
            best_real_std_desta_rodada = Inf
            best_params_desta_rodada = nothing

            for i in 1:length(xbest_list)
                current_params = xbest_list[i]
                unnormalized_matrix = mach_zender_nd(current_params, N_DIM)
                if !isnothing(unnormalized_matrix) && (unnormalized_matrix * unnormalized_matrix') ≈ I(N_DIM)
                    real_std = std(abs.(unnormalized_matrix))
                    if real_std < best_real_std_desta_rodada
                        best_real_std_desta_rodada = real_std
                        best_params_desta_rodada = current_params
                    end
                end
            end
            
            if best_real_std_desta_rodada < overall_best_std
                println("\n" * "!"^50)
                println(">>> NOVO CAMPEÃO GERAL ENCONTRADO NA TENTATIVA #$run_count <<<")
                println("Novo Melhor Desvio Padrão Real: $best_real_std_desta_rodada")
                println("Obtido com peso: $peso_sorteado")
                println("!"^50 * "\n")
                
                overall_best_std = best_real_std_desta_rodada
                overall_best_params = best_params_desta_rodada
                overall_best_weight = peso_sorteado
            else
                println("\n--- Tentativa #$run_count concluída. Nenhum recorde quebrado. Melhor std da rodada: $best_real_std_desta_rodada ---")
            end
            println("\n--- Fim da Tentativa #$run_count. Melhor STD geral até agora: $overall_best_std ---")

            if best_real_std_desta_rodada < STD_THRESHOLD && !isnothing(best_params_desta_rodada)
                println("-> Salvando resultado da Tentativa #$run_count (STD=$(best_real_std_desta_rodada)) nos arquivos...")
                
                # Salva os parâmetros no CSV
                save_params_to_csv(PARAMS_FILE, best_real_std_desta_rodada, best_params_desta_rodada, N_DIM)
                
            end
        end
    catch e
        # Verifica se a exceção é uma Interrupção Direta
        if e isa InterruptException
            println("\n\nBusca interrompida pelo usuário.")
        # Verifica se a exceção é uma falha de tarefa causada por uma Interrupção
        elseif e isa TaskFailedException
            # Extrai a exceção aninhada da TaskFailedException
            nested_exception = e.task.exception
            if nested_exception isa InterruptException
                println("\n\nBusca interrompida pelo usuário (tarefas paralelas foram interrompidas).")
            else
                # Se a TaskFailedException foi por outro motivo, rethrow
                println("\n\nERRO INESPERADO EM TAREFA PARALELA: ", nested_exception)
                rethrow(e)
            end
        else
            # Qualquer outro tipo de erro
            println("\n\nERRO INESPERADO: ", e)
            rethrow(e) 
        end
    end
    # --- LÓGICA DE SALVAMENTO DO CAMPEÃO FINAL (FORA DO LAÇO) ---
    # Esta parte é importante para garantir que o *melhor* resultado absoluto
    # seja salvo, mesmo que ele não tenha sido salvo anteriormente (ex: por
    # ultrapassar o limite de 1% só no final, ou ser o único a atingir).
    if !isnothing(overall_best_params) && overall_best_std < STD_THRESHOLD
        println("\n--- Salvando o Campeão FINAL (STD=$(overall_best_std)) nos arquivos... ---")
        save_params_to_csv(PARAMS_FILE, overall_best_std, overall_best_params, N_DIM)
        final_matrix_to_save = mach_zender_nd(overall_best_params, N_DIM)
    end
    
    println("\nExibindo o melhor resultado encontrado após $run_count tentativas...")
    return overall_best_params, overall_best_std, overall_best_weight, N_DIM
end

# --- Executa o programa e processa o resultado final ---
println("Iniciando a busca... Os resultados finais serão salvos e exibidos.")
melhores_parametros, melhor_std, melhor_peso, N_DIM = main()

if isnothing(melhores_parametros)
    println("\nBusca concluída sem resultados válidos.")
else
    println("\n" * "="^50)
    println("--- RESULTADO FINAL VENCEDOR ---")
    println("Melhor Peso Encontrado: ", melhor_peso)
    println("Melhor Desvio Padrão Real Geral: ", melhor_std)
    println("\nMelhores parâmetros correspondentes:")
    println("  - L     = ", melhores_parametros[1])
    println("  - delta = ", melhores_parametros[2])
    println("  - kappa = ", melhores_parametros[3])
    println("  - thetas: ")
    for (i, th) in enumerate(melhores_parametros[4:end])
        println("    - θ[$i] = ", th)
    end
    println("="^50 * "\n")

    # Geramos a matriz final uma última vez para exibir e verificar
    final_matrix = mach_zender_nd(melhores_parametros, N_DIM)
    
    if !isnothing(final_matrix)
        final_matrix_normalized = final_matrix .* sqrt(N_DIM)
        println("Melhor matriz:")
        display(abs.(final_matrix))
        println("Matriz ajustada (normalizada por sqrt(N)):")
        display(final_matrix_normalized)
        println("\nMagnitude dos elementos da matriz ajustada (normalizada):")
        display(abs.(final_matrix_normalized))
        
        # A verificação final de unitariedade
        println("\n--- VERIFICANDO SE A MATRIZ FINAL É UNITÁRIA ---")
        is_unitary = (final_matrix * final_matrix') ≈ I(N_DIM)
        println("O resultado de `(U * U') ≈ I` é: ", is_unitary)
        if is_unitary
            println("Resultado: A matriz final selecionada é UNITÁRIA.")
        else
            println("Resultado: A matriz final selecionada NÃO é unitária.")
        end
    end
end