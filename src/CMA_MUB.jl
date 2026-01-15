# ===================================================================
# SCRIPT PARA RECONSTRUIR U1 E ENCONTRAR U2 (MUB) VIA CMA-ES
# ===================================================================
using LinearAlgebra
using Statistics
using Polynomials
using CSV
using DataFrames
using CMAEvolutionStrategy # Adicionado para o CMA-ES

# ===================================================================
# PARTE 1: CÓPIAS DAS FUNÇÕES DE SIMULAÇÃO NECESSÁRIAS
# (Copiadas do script principal para que este funcione de forma independente)
# ===================================================================
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
    if abs(kappa_val) < 1e-9; return nothing; end
    x = variable(Polynomial{Float64})
    Un = get_chebyshev_u(n_val)
    Un_minus_2 = get_chebyshev_u(n_val - 2)
    Un_minus_3 = get_chebyshev_u(n_val - 3)
    poly_numerical = (kappa_val^2 * Un + (delta_val^2 - 4 * kappa_val * delta_val * x) * Un_minus_2 + 2 * kappa_val * delta_val * Un_minus_3)
    solutions_for_x = roots(poly_numerical)
    if isempty(solutions_for_x); return nothing; end
    function CalcEigenvector_Numerical(N_size, x_k, d_val, k_val)
        v = zeros(ComplexF64, N_size); factor = -d_val / k_val + 2 * x_k
        for i in 1:N_size
            if i == 1; v[i] = abs(factor) < 1e-9 ? 1.0 : factor
            elseif i < N_size
                u_i_minus_1 = get_chebyshev_u(i - 1)(x_k); u_i_minus_2 = get_chebyshev_u(i - 2)(x_k)
                term = u_i_minus_1 - (d_val / k_val) * u_i_minus_2; v[i] = factor * term
            else
                if abs(v[i-1]) < 1e-9; v[i] = -v[1]; else
                    u_n_minus_2 = get_chebyshev_u(N_size - 2)(x_k); u_n_minus_3 = get_chebyshev_u(N_size - 3)(x_k)
                    term = u_n_minus_2 - (d_val / k_val) * u_n_minus_3; v[i] = term
                end
            end
        end; return v
    end
    NormalizeEigenvector(v) = (norm_v = norm(v); norm_v > 1e-9 ? v / norm_v : v)
    eigenvectors = Vector{ComplexF64}[]; diagonal_elements = ComplexF64[]
    for x_k in solutions_for_x
        if abs(imag(x_k)) > 1e-9; continue; end
        x_k_real = real(x_k); lambda_k = -delta_val / 2 + 2 * kappa_val * x_k_real
        v_k = CalcEigenvector_Numerical(n_val, x_k_real, delta_val, kappa_val)
        push!(eigenvectors, NormalizeEigenvector(v_k)); push!(diagonal_elements, exp(im * L_val * lambda_k))
    end
    if isempty(eigenvectors) || length(eigenvectors) != n_val; return nothing; end
    try
        MatrixEigenvectors = hcat(eigenvectors...); M_kk = diagm(diagonal_elements)
        NBS_julia = MatrixEigenvectors * M_kk * MatrixEigenvectors'
        prod = NBS_julia * transpose(NBS_julia); return NBS_julia, prod
    catch e; return nothing; end
end

function mach_zender_nd(params::AbstractVector{Float64}, n::Int)
    L, δ, kappa = params[1], params[2], params[3]
    thetas = params[4:end] # Pega todos os thetas fornecidos
    
    resultado_bs = calculate_nbs_numerically_FAST(n, δ, kappa, L)
    isnothing(resultado_bs) && return nothing
    bs_matrix, _ = resultado_bs
    
    pm_matrix = diagm(exp.(im .* thetas))
    return bs_matrix * pm_matrix * bs_matrix
end

# ===================================================================
# PARTE 2: LÓGICA DE RECONSTRUÇÃO DA MELHOR U1 (DO CSV)
# ===================================================================
function load_u1_from_csv(filepath::String, N_DIM::Int)
    df = CSV.read(filepath, DataFrames.DataFrame)
    
    # Filtra pela dimensão correta se necessário
    if "N_DIM" in names(df)
        df = filter(row -> row.N_DIM == N_DIM, df)
    end
    
    if isempty(df); error("Nenhuma solução para N_DIM=$N_DIM encontrada."); end

    # Pega o melhor resultado (menor STD)
    sort!(df, :STD)
    best = df[1, :]

    # Reconstroi o vetor de parâmetros completo [L, delta, kappa, theta1...thetaN]
    # Aqui lemos EXPLICITAMENTE de 1 até N_DIM
    thetas = [Float64(best[Symbol("theta$i")]) for i in 1:N_DIM]
    params_U1 = [best.L, best.delta, best.kappa, thetas...] 

    # Gera a matriz U1 usando todos os parâmetros
    U1_matrix = mach_zender_nd(params_U1, N_DIM) 

    return U1_matrix, params_U1
end

# ===================================================================
# PARTES 3 e 4 (fitness_mub, find_mub_U2, main_mub_search)
# Essas funções permanecem as mesmas, mas find_mub_U2 vai precisar 
# lidar com N_DIM=1 separadamente caso seja necessário no futuro
# ===================================================================

# Fitness Function: Queremos MINIMIZAR isso
function fitness_mub(phi_vector::AbstractVector{Float64}, U1::Matrix{ComplexF64}, N_DIM::Int)
    
    # Constrói o vetor de fases completo para P2, fixando phi1=0 para remover redundância
    # (N-1) thetas para otimizar
    if N_DIM == 1
        phi_for_P2 = [0.0]
    else
        if length(phi_vector) != (N_DIM - 1)
            error("Erro interno: phi_vector deve ter N_DIM-1 elementos. Recebido: $(length(phi_vector)) para N_DIM=$N_DIM")
        end
        phi_for_P2 = [0.0; phi_vector] # phi1 = 0.0, seguido dos otimizados
    end

    P2 = diagm(exp.(im .* phi_for_P2))
    
    U2 = P2 * U1
    
    W = U1' * U2 # U1' é a adjunta (transposta conjugada) de U1. Cada entrada tera o produto interno entre os vetores
    
    # O objetivo é que abs.(W)^2 seja 1/N para todos os elementos
    target_magnitude_squared = 1.0 / N_DIM
    
    # Métrica de aptidão: desvio médio absoluto das magnitudes ao quadrado em relação ao target
    # Queremos MINIMIZAR este valor.
    deviation = sum(abs.(abs.(W).^2 .- target_magnitude_squared)) / (N_DIM^2)
    
    return deviation 
end

function main_mub_search_continuous()
    N_DIM = 4 # Dimensão para a busca de MUB
    
    # --- Carrega a U1 de referência ---
    U1_file_path = "$(N_DIM)D_melhor_STD_(antigo).csv"
    U1_matrix, U1_params = load_u1_from_csv(U1_file_path, N_DIM)
    if isnothing(U1_matrix)
        error("Não foi possível carregar U1, abortando busca de MUB U2.")
    end

    # --- HIPERPARÂMETROS DA BUSCA DE U2 ---
    POP_SIZE_MUB = 400
    MAX_ITER_MUB = 10000 
    SIGMA0_MUB = 0.5 

    # --- NOMES DOS ARQUIVOS DE SAÍDA PARA U2 ---
    MUB_PARAMS_FILE = "$(N_DIM)D_best_MUB_U2_params.csv"
    MUB_THRESHOLD = 0.01 # Limiar para o desvio MUB

    println("\n" * "#"^80)
    println("Iniciando MODO DE DESCOBERTA CONTÍNUA para MUB U2 em relação a U1 para N_DIM = $N_DIM.")
    println("Pressione Ctrl+C a qualquer momento para parar e ver o melhor resultado geral.")
    println("#"^80 * "\n")

    # --- Variáveis para guardar o "Campeão dos Campeões" de U2 ---
    overall_best_mub_deviation = Inf
    overall_best_phi_vector = nothing
    overall_best_U2_matrix = nothing # Para armazenar a U2 correspondente
    run_count = 0

    try
        while true # O loop infinito de busca
            run_count += 1
            println("\n" * "*"^50)
            println("--- INICIANDO TENTATIVA DE BUSCA DE U2 #$run_count ---")
            
            N_PARAMS_PHI = N_DIM - 1 

            lower_bounds_phi = fill(0.0, N_PARAMS_PHI)
            upper_bounds_phi = fill(2 * pi, N_PARAMS_PHI)
            x0_phi_aleatorio = lower_bounds_phi .+ rand(N_PARAMS_PHI) .* (upper_bounds_phi .- lower_bounds_phi)
            
            println("Configuração: Ponto de Partida Aleatório para ângulos phi.")
            println("*"^50 * "\n")

            f_mub_to_optimize(p_phi) = fitness_mub(p_phi, U1_matrix, N_DIM)

            result_mub = minimize(f_mub_to_optimize, x0_phi_aleatorio, SIGMA0_MUB; 
                                 lower=lower_bounds_phi, upper=upper_bounds_phi,
                                 popsize = POP_SIZE_MUB, maxiter = MAX_ITER_MUB, verbosity=0,
                                 multi_threading=true)

            # --- PÓS-PROCESSAMENTO ROBUSTO DO LOGGER (como discutimos) ---
            best_mub_deviation_this_run = Inf
            best_phi_vector_this_run = nothing
            best_U2_matrix_this_run = nothing

            xbest_history = result_mub.logger.xbest
            fbest_history = result_mub.logger.fbest # Podemos usar para comparação, mas confiamos no recalculado

            println("Iniciando pós-processamento do histórico do CMA-ES para U2...")

            for i in 1:length(xbest_history)
                current_phi_vector = xbest_history[i]
                
                phi_full_current = [0.0; current_phi_vector]
                P2_current = diagm(exp.(im .* phi_full_current))
                U2_current = P2_current * U1_matrix

                if (U2_current * U2_current') ≈ I(N_DIM) # Verifica unitariedade de U2
                    deviation_recalculated = fitness_mub(current_phi_vector, U1_matrix, N_DIM)
                    
                    if deviation_recalculated < best_mub_deviation_this_run
                        best_mub_deviation_this_run = deviation_recalculated
                        best_phi_vector_this_run = current_phi_vector
                        best_U2_matrix_this_run = U2_current # Guarda a U2 correspondente
                    end
                end
            end

            if isnothing(best_phi_vector_this_run)
                println("AVISO: Nenhuma solução unitária para U2 foi encontrada no histórico desta rodada.")
                # Fallback: Usar o último xbest do logger como último recurso
                best_phi_vector_this_run = result_mub.logger.xbest[end]
                best_mub_deviation_this_run = fitness_mub(best_phi_vector_this_run, U1_matrix, N_DIM)
                phi_full_fallback = [0.0; best_phi_vector_this_run]
                P2_fallback = diagm(exp.(im .* phi_full_fallback))
                best_U2_matrix_this_run = P2_fallback * U1_matrix
            end
            
            # --- Comparação com o Campeão Geral ---
            if best_mub_deviation_this_run < overall_best_mub_deviation
                overall_best_mub_deviation = best_mub_deviation_this_run
            end
            println("\n--- Fim da Tentativa #$run_count. Melhor Desvio MUB geral até agora: $(overall_best_mub_deviation) ---")

            # --- Salvamento Condicional ---
            if best_mub_deviation_this_run < MUB_THRESHOLD && !isnothing(best_phi_vector_this_run)
                #println("-> Salvando resultado de U2 da Tentativa #$run_count (Desvio MUB=$(best_mub_deviation_this_run)) nos arquivos...")
                
                # Para o CSV, podemos salvar os phi's.
                # Criar uma função save_mub_params_to_csv similar a save_params_to_csv
                save_mub_params_to_csv(MUB_PARAMS_FILE, best_mub_deviation_this_run, best_phi_vector_this_run, N_DIM, U1_params)           
            end
        end # Fim do while true
    catch e
        if e isa InterruptException
            println("\n\nBusca por MUB U2 interrompida pelo usuário.")
        elseif e isa TaskFailedException
            nested_exception = e.task.exception
            if nested_exception isa InterruptException
                println("\n\nBusca por MUB U2 interrompida pelo usuário (tarefas paralelas foram interrompidas).")
            else
                println("\n\nERRO INESPERADO EM TAREFA PARALELA (Busca MUB U2): ", nested_exception)
                rethrow(e)
            end
        else
            println("\n\nERRO INESPERADO (Busca MUB U2): ", e)
            rethrow(e) 
        end
    end
    # --- LÓGICA DE SALVAMENTO DO CAMPEÃO FINAL DE U2 (FORA DO LAÇO) ---
    if !isnothing(overall_best_phi_vector) && overall_best_mub_deviation < MUB_THRESHOLD
        println("\n--- Salvando o Campeão FINAL de U2 (Desvio MUB=$(overall_best_mub_deviation)) no arquivo CSV... ---")
        # === Passe U1_params ===
        save_mub_params_to_csv(MUB_PARAMS_FILE, overall_best_mub_deviation, overall_best_phi_vector, N_DIM, U1_params)
    end
    
    println("\nExibindo o melhor resultado de U2 encontrado após $run_count tentativas...")
    return overall_best_phi_vector, overall_best_mub_deviation, overall_best_U2_matrix, U1_matrix, N_DIM
end

# --- FUNÇÕES DE SALVAMENTO PARA MUB (Novas ou Adaptadas) ---
function save_mub_params_to_csv(filepath::String, mub_deviation::Float64, phi_vector::AbstractVector{Float64}, N_DIM::Int, u1_params::AbstractVector{Float64})
    header_exists = isfile(filepath) && filesize(filepath) > 0
    open(filepath, "a") do io
        if !header_exists
            # === CONSTRUÇÃO DO CABEÇALHO PARA U1 ===
            u1_header_parts = String[]
            push!(u1_header_parts, "U1_L", "U1_delta", "U1_kappa")
            for i in 1:N_DIM
                push!(u1_header_parts, "U1_theta$i")
            end

            # === CONSTRUÇÃO DO CABEÇALHO PARA U2 ===
            u2_header_parts = String[]
            push!(u2_header_parts, "MUB_Deviation")
            # Para phi_vector de U2, ele já contém phi2...phiN, então o índice deve ser i+1 para o nome da coluna.
            for i in 1:length(phi_vector) # length(phi_vector) é N_DIM-1
                push!(u2_header_parts, "U2_phi$(i+1)") # Começa com U2_phi2, U2_phi3, etc.
            end
            push!(u2_header_parts, "N_DIM")

            # === COMBINAÇÃO DOS CABEÇALHOS ===
            header = join([u1_header_parts..., u2_header_parts...], ",") * "\n"
            write(io, header)
        end
        
        # === CONSTRUÇÃO DOS DADOS PARA U1 ===
        u1_data_parts = String[]
        push!(u1_data_parts, string(u1_params[1])) # L
        push!(u1_data_parts, string(u1_params[2])) # delta
        push!(u1_data_parts, string(u1_params[3])) # kappa
        for i in 4:length(u1_params) # theta1, theta2, ..., thetaN
            push!(u1_data_parts, string(u1_params[i]))
        end

        # === CONSTRUÇÃO DOS DADOS PARA U2 ===
        u2_data_parts = String[]
        push!(u2_data_parts, string(mub_deviation)) # MUB_Deviation
        for phi_val in phi_vector # phi_vector já é phi2...phiN
            push!(u2_data_parts, string(phi_val))
        end
        push!(u2_data_parts, string(N_DIM))

        # === COMBINAÇÃO DOS DADOS DA LINHA ===
        data_row = join([u1_data_parts..., u2_data_parts...], ",") * "\n"
        write(io, data_row)
    end
end

# --- Executa o programa para busca contínua de U2 ---
println("Iniciando a busca contínua por MUB U2...")
melhor_phi, melhor_desvio_mub, melhor_U2_matrix, U1_matrix_global, N_DIM_global = main_mub_search_continuous()

if !isnothing(melhor_phi)
    println("\nBusca por MUB U2 finalizada. Os melhores resultados foram salvos no arquivo: ")
    println("  - $(N_DIM_global)D_best_MUB_U2_params.csv")
else
    println("\nBusca por MUB U2 finalizada. Nenhum resultado válido foi salvo.")
end
