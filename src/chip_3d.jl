using LinearAlgebra, DelimitedFiles, Base.Threads, Symbolics


let 
    # Variáveis simbólicas
    @syms θ1 θ2 ϕ1 ϕ2 ρ::Complex τ δ
    BS3 = (1/2) * [
        ρ + exp(im * δ)       im * √2 * τ    -(exp(im * δ) - ρ);
        im * √2 * τ  2 * conj(ρ)   im * √2 * τ;
        -(exp(im * δ) - ρ)    im * √2 * τ     ρ + exp(im * δ)
    ]
    Fase_ϕ = [1   0   0
              0   exp(im*ϕ1)  0
              0   0   exp(im*ϕ2)
    ]
    Fase_θ = [1   0   0
              0   exp(im*θ1)  0
              0   0   exp(im*θ2)
    ]
    Ut = Fase_θ * BS3 * Fase_ϕ * BS3

    # Extrair entradas simbólicas
    Ut_00_simbolico = Ut[1,1]
    Ut_01_simbolico = Ut[1,2]
    Ut_02_simbolico = Ut[1,3]
    Ut_10_simbolico = Ut[2,1]
    Ut_11_simbolico = Ut[2,2]
    Ut_12_simbolico = Ut[2,3]
    Ut_20_simbolico = Ut[3,1]
    Ut_21_simbolico = Ut[3,2]
    Ut_22_simbolico = Ut[3,3]

    # Obter elementos simbólicos da matriz
    Ut_simbolico = [Ut[i, j] for i in 1:3, j in 1:3]

    # Iterar sobre cada elemento para criar funções Ut_ij_func , de cada entrada 
    for i in 1:3, j in 1:3
        func_name = Symbol("Ut_$(i-1)$(j-1)_func")
        @eval global $func_name = $(eval(build_function(Ut_simbolico[i, j], θ1, θ2, ϕ1, ϕ2, ρ, τ, δ)))
    end

    # Gerar funções numéricas
    global Ut_00_func = eval(
        build_function(
            Ut_00_simbolico,
            θ1, θ2, ϕ1, ϕ2, ρ, τ, δ
        )
    )
    global Ut_01_func = eval(
        build_function(
            Ut_01_simbolico,
            θ1, θ2, ϕ1, ϕ2, ρ, τ, δ
        )
    )
    global Ut_02_func = eval(
        build_function(
            Ut_02_simbolico,
            θ1, θ2, ϕ1, ϕ2, ρ, τ, δ
        )
    )
    global Ut_10_func = eval(
        build_function(
            Ut_10_simbolico,
            θ1, θ2, ϕ1, ϕ2, ρ, τ, δ
        )
    )
    global Ut_11_func = eval(
        build_function(
            Ut_11_simbolico,
            θ1, θ2, ϕ1, ϕ2, ρ, τ, δ
        )
    )
    global Ut_12_func = eval(
        build_function(
            Ut_12_simbolico,
            θ1, θ2, ϕ1, ϕ2, ρ, τ, δ
        )
    )
    global Ut_20_func = eval(
        build_function(
            Ut_20_simbolico,
            θ1, θ2, ϕ1, ϕ2, ρ, τ, δ
        )
    )
    global Ut_21_func = eval(
        build_function(
            Ut_21_simbolico,
            θ1, θ2, ϕ1, ϕ2, ρ, τ, δ
        )
    )
    global Ut_22_func = eval(
        build_function(
            Ut_22_simbolico,
            θ1, θ2, ϕ1, ϕ2, ρ, τ, δ
        )
    )
end


# Função para garantir unitariedade de BS2 (única condição nesse caso é |ρ|²+τ² = 1)
function modulo_1(valores_ρ)
    combinacoes = []
    for ρ in valores_ρ
        τ_squared = 1 - abs2(ρ)
        if τ_squared >= 0
            τ = sqrt(τ_squared)
            push!(combinacoes, (ρ, τ))
            push!(combinacoes, (ρ, -τ))
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
passo_angulo = 2π/64              
angulos_ϕ1 = 0:passo_angulo:(2π - passo_angulo)
angulos_ϕ2 = 0:passo_angulo:(2π - passo_angulo)
angulos_θ1 = 0:passo_angulo:(2π - passo_angulo)
angulos_θ2 = 0:passo_angulo:(2π - passo_angulo)
angulos_δ = 0:(2π/128):(2π - (2π/128))
intervalos_ρ_real = 0.290:passo:0.294
intervalos_ρ_im = 0.080:passo:0.084

valores_ρ = [a + im*b for a in intervalos_ρ_real, b in intervalos_ρ_im]
combinacoes_ρ_τ = modulo_1(valores_ρ)
combinacoes_ρ_τ = [(ComplexF64(ρ), Float64(τ)) for (ρ, τ) in combinacoes_ρ_τ] #garantindo o tipo das variáveis

# Matriz desejada
ω = exp(im * 2π / 3)
matriz_desejada = (1/√3) * ComplexF64[1   1       1;
                                      1   ω^2       ω;
                                      1   ω     ω^2]


# Alvos
alvos_reais = real.(matriz_desejada)
alvos_imaginarios = imag.(matriz_desejada)

# Processamento
# Processamento paralelizado
function processar(angulos_θ1, angulos_θ2, angulos_ϕ1, angulos_ϕ2, combinacoes_ρ_τ, angulos_δ, alvos_reais, alvos_imaginarios, tolerancia)
    resultados = []  # Vetor global para armazenar todos os resultados
    @threads for ϕ1_val in [3.9269908169872414] #angulos_ϕ1
        resultados_thread = []  # Resultados locais para cada thread
        ϕ1_val = Float64(ϕ1_val)
        for ϕ2_val in [3.9269908169872414] #angulos_ϕ2
            ϕ2_val = Float64(ϕ2_val)
            for θ1_val in angulos_θ1
                θ1_val = Float64(θ1_val)
                for θ2_val in angulos_θ2
                    θ2_val = Float64(θ2_val)
                    for (ρ_val, τ_val) in combinacoes_ρ_τ
                        ρ_val = ComplexF64(ρ_val)
                        τ_val = Float64(τ_val)
                        for δ_val in [5.18722339] #angulos_δ
                            δ_val = Float64(δ_val)
                            Ut_00 = Ut_00_func(θ1_val, θ2_val, ϕ1_val, ϕ2_val, ρ_val, τ_val, δ_val)
                            if proximo_complexo(Ut_00, alvos_reais[1, 1], alvos_imaginarios[1, 1], tolerancia) #veririca se entradas em 00 batem
                                Ut_01 = Ut_01_func(θ1_val, θ2_val, ϕ1_val, ϕ2_val, ρ_val, τ_val, δ_val)
                                if proximo_complexo(Ut_01, alvos_reais[1, 2], alvos_imaginarios[1, 2], tolerancia) #veririca se entradas em 01 batem
                                    #println("chegou1")
                                    Ut_02 = Ut_02_func(θ1_val, θ2_val, ϕ1_val, ϕ2_val, ρ_val, τ_val, δ_val)
                                    if proximo_complexo(Ut_02, alvos_reais[1, 3], alvos_imaginarios[1, 3], tolerancia) #veririca se entradas em 02 batem
                                        #println("chegou")
                                        Ut_10 = Ut_10_func(θ1_val, θ2_val, ϕ1_val, ϕ2_val, ρ_val, τ_val, δ_val)
                                        if proximo_complexo(Ut_10, alvos_reais[2, 1], alvos_imaginarios[2, 1], tolerancia) #veririca se entradas em 10 batem
                                            #println("chegou")
                                            #push!(resultados_thread, (θ1_val, θ2_val, ϕ1_val, ϕ2_val, ρ_val, τ_val, δ_val, Ut_00, Ut_01, Ut_02, Ut_10))
                                            #push!(resultados_thread, (θ1_val, θ2_val, ϕ1_val, ϕ2_val, -ρ_val, -τ_val, δ_val, Ut_00, Ut_01, Ut_02, Ut_10))
                                            Ut_11 = Ut_11_func(θ1_val, θ2_val, ϕ1_val, ϕ2_val, ρ_val, τ_val, δ_val)
                                            if proximo_complexo(Ut_11, alvos_reais[2, 2], alvos_imaginarios[2, 2], tolerancia) #veririca se entradas em 11 batem
                                                println("chegou")
                                                Ut_12 = Ut_12_func(θ1_val, θ2_val, ϕ1_val, ϕ2_val, ρ_val, τ_val, δ_val)
                                                if proximo_complexo(Ut_12, alvos_reais[2, 3], alvos_imaginarios[2, 3], tolerancia) #veririca se entradas em 12 batem
                                                    Ut_20 = Ut_20_func(θ1_val, θ2_val, ϕ1_val, ϕ2_val, ρ_val, τ_val, δ_val)
                                                    if proximo_complexo(Ut_20, alvos_reais[3, 1], alvos_imaginarios[3, 1], tolerancia) #veririca se entradas em 20 batem
                                                        Ut_21 = Ut_21_func(θ1_val, θ2_val, ϕ1_val, ϕ2_val, ρ_val, τ_val, δ_val)
                                                        if proximo_complexo(Ut_21, alvos_reais[3, 2], alvos_imaginarios[3, 2], tolerancia) #veririca se entradas em 21 batem
                                                            Ut_22 = Ut_22_func(θ1_val, θ2_val, ϕ1_val, ϕ2_val, ρ_val, τ_val, δ_val)
                                                            if proximo_complexo(Ut_22, alvos_reais[3, 3], alvos_imaginarios[3, 3], tolerancia) #veririca se entradas em 22 batem
                                                                push!(resultados_thread, (θ1_val, θ2_val, ϕ1_val, ϕ2_val, ρ_val, τ_val, δ_val, Ut_00, Ut_01, Ut_02, Ut_10, Ut_11, Ut_12, Ut_20, Ut_21, Ut_22))
                                                                push!(resultados_thread, (θ1_val, θ2_val, ϕ1_val, ϕ2_val, -ρ_val, -τ_val, δ_val, Ut_00, Ut_01, Ut_02, Ut_10, Ut_11, Ut_12, Ut_20, Ut_21, Ut_22))
                                                            end
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
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
resultados = processar(angulos_θ1, angulos_θ2, angulos_ϕ1, angulos_ϕ2, combinacoes_ρ_τ, angulos_δ, alvos_reais, alvos_imaginarios, 30e-2)
end_time = time()


# Salva
open("resultados_julia_3d_final.txt", "w") do arquivo
    write(arquivo, "θ1, θ2, ϕ1, ϕ2, ρ, τ, δ, Ut_00, Ut_01, Ut_02, Ut_10, Ut_11, Ut_12, Ut_20, Ut_21, Ut_22\n")
    for resultado in resultados
        write(arquivo, join(resultado, ", ") * "\n")
    end
end

println("Tempo de execução: ", end_time - start_time, " segundos")
