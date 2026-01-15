using LinearAlgebra, DelimitedFiles, Base.Threads, Symbolics

let
    # Variáveis simbólicas (todas reais)
    @syms θ ϕ ρ τ

    # Definição das matrizes (ρ e τ são reais)
    BS2 = [ρ      im*τ
           im*τ    conj(ρ)] 

    Fase_ϕ = [1   0
              0   exp(im*ϕ)]

    Fase_θ = [1   0
              0   exp(im*θ)]

    global Ut = Fase_θ * BS2 * Fase_ϕ * BS2

    # Obter elementos simbólicos da matriz
    Ut_simbolico = [Ut[i, j] for i in 1:2, j in 1:2]

    # Iterar sobre cada elemento para criar funções Ut_ij_func , de cada entrada 
    for i in 1:2, j in 1:2
        func_name = Symbol("Ut_$(i-1)$(j-1)_func")
        @eval global $func_name = $(eval(build_function(Ut_simbolico[i, j], θ, ϕ, ρ, τ)))
    end

end

# Função para garantir unitariedade de BS2 (única condição nesse caso é |ρ|²+τ² = 1)
function modulo_1(valores_ρ)
    combinacoes = []
    for ρ_val in valores_ρ
        τ_squared = 1 - abs2(ρ_val)
        if τ_squared >= 0
            τ = sqrt(τ_squared)
            push!(combinacoes, (ρ_val, τ))
            push!(combinacoes, (ρ_val, -τ))
        end
    end
    return combinacoes
end

# Função para verificar proximidade
function proximo_complexo(valor, alvo_real, alvo_imag, tolerancia=1e-2)
    dx = real(valor) - alvo_real
    dy = imag(valor) - alvo_imag
    return (dx^2 + dy^2) < tolerancia^2
end

# GERAÇÃO DE VALORES
passo = 0.001
passo_angulo = 2π/8
angulos_ϕ = collect(Float64, 0:passo_angulo:(2π - passo_angulo))
angulos_θ = collect(Float64, 0:passo_angulo:(2π - passo_angulo)) 
intervalos_ρ_real = -0.4:passo:0.4
intervalos_ρ_im = 0.2:passo:0.6

valores_ρ = [a + im*b for a in intervalos_ρ_real, b in intervalos_ρ_im]
combinacoes_ρ_τ = modulo_1(valores_ρ)
combinacoes_ρ_τ = [(ComplexF64(ρ), ComplexF64(τ)) for (ρ, τ) in combinacoes_ρ_τ] #garantindo o tipo das variáveis

# Matriz desejada
matriz_desejada = (1/sqrt(2)) * ComplexF64[1      1;
                                           1    -1
]

# Tolerância e "alvos"
alvos_reais = real.(matriz_desejada)
alvos_imaginarios = imag.(matriz_desejada)
pi1 = π/1

# Processamento
function processar(angulos_θ, angulos_ϕ, combinacoes_ρ_τ, alvos_reais, alvos_imaginarios, tolerancia)
    resultados = []
    @threads for ϕ_val in [pi1] #angulos_ϕ
        resultados_thread = []  # Resultados locais para cada thread
        ϕ_val = Float64(ϕ_val)
        for θ_val in [0] #angulos_θ
            θ_val = Float64(θ_val)
            for (ρ_val, τ_val) in combinacoes_ρ_τ
                ρ_val = ComplexF64(ρ_val)
                τ_val = Float64(τ_val)
                Ut_00 = Ut_00_func(θ_val, ϕ_val, ρ_val, τ_val)
                if proximo_complexo(Ut_00, alvos_reais[1, 1], alvos_imaginarios[1, 1], tolerancia)
                    Ut_01 = Ut_01_func(θ_val, ϕ_val, ρ_val, τ_val)
                    if proximo_complexo(Ut_01, alvos_reais[1, 2], alvos_imaginarios[1, 2], tolerancia)
                        Ut_10 = Ut_10_func(θ_val, ϕ_val, ρ_val, τ_val)
                        if proximo_complexo(Ut_10, alvos_reais[2, 1], alvos_imaginarios[2, 1], tolerancia)
                            Ut_11 = Ut_11_func(θ_val, ϕ_val, ρ_val, τ_val)
                            if proximo_complexo(Ut_11, alvos_reais[2, 2], alvos_imaginarios[2, 2], tolerancia)
                                push!(resultados_thread, (θ_val, ϕ_val, ρ_val, τ_val, Ut_00, Ut_01, Ut_10, Ut_11))
                                push!(resultados_thread, (θ_val, ϕ_val, -ρ_val, -τ_val, Ut_00, Ut_01, Ut_10, Ut_11))
                            end
                        end
                    end
                end
            end
        end
        # Adiciona os resultados da thread ao vetor principal
        append!(resultados, resultados_thread)
    end
    return resultados
end

# Executa
start_time = time()
resultados = processar(angulos_θ, angulos_ϕ, combinacoes_ρ_τ, alvos_reais, alvos_imaginarios, 0.25)
end_time = time()

# Salva
open("resultados_teste_01_07_2025.txt", "w") do arquivo
    write(arquivo, "θ, ϕ, ρ, τ, Ut_00, Ut_01, Ut_10, Ut_11\n")
    for resultado in resultados
        write(arquivo, join(resultado, ", ") * "\n")
    end
end

println("Tempo de execução: ", end_time - start_time, " segundos")