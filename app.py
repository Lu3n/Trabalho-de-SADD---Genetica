import pandas as pd
import random
import matplotlib.pyplot as plt

# ============================================
# 1 - CARREGAR OS DADOS REAIS DO ENEM
# ============================================


caminho_arquivo = r"C:\\Users\\pichau\\Downloads\\microdados_enem_2023\\DADOS\\MICRODADOS_ENEM_2023.csv"


# Colunas necessárias

colunas = [
    "NU_INSCRICAO",
    "NU_NOTA_MT", "NU_NOTA_CN", "NU_NOTA_LC", "NU_NOTA_CH", "NU_NOTA_REDACAO",
    "Q006", "Q002", "TP_ESCOLA", "TP_COR_RACA", "SG_UF_PROVA"
]



print(" Carregando dados... (isso pode levar alguns segundos)")
df = pd.read_csv(caminho_arquivo, sep=";", usecols=colunas, encoding="latin1", nrows=10000)

# Limpeza básica
df = df.dropna(subset=["NU_NOTA_MT", "NU_NOTA_CN", "NU_NOTA_LC", "NU_NOTA_CH", "NU_NOTA_REDACAO"])

# Criar média geral
df["media_geral"] = df[["NU_NOTA_MT","NU_NOTA_CN","NU_NOTA_LC","NU_NOTA_CH","NU_NOTA_REDACAO"]].mean(axis=1)

print(" Dados carregados com sucesso:", df.shape)
print(df.head())

# ============================================
# 2 - PARÂMETROS DO ALGORITMO GENÉTICO
# ============================================

TAMANHO_GRUPO = 100          # cada grupo terá 100 alunos
TAMANHO_POPULACAO = 20       # número de grupos em cada geração
NUM_GERACOES = 100            # número de gerações
TAXA_MUTACAO = 0.1            # chance de mutação

# ============================================
# 3 - FUNÇÃO DE APTIDÃO (fitness)
# ============================================

def fitness(grupo):
    media_notas = grupo["media_geral"].mean()
    diversidade = (
        grupo["Q006"].nunique() + grupo["TP_COR_RACA"].nunique() + grupo["TP_ESCOLA"].nunique()
    ) / 10  # normalização simples
    cobertura = grupo["SG_UF_PROVA"].nunique() / 27  # 27 estados

    # ponderação dos critérios
    return (0.5 * media_notas) + (0.3 * diversidade * 100) + (0.2 * cobertura * 100)

# ============================================
# 4 - GERAR POPULAÇÃO INICIAL
# ============================================

def gerar_populacao():
    populacao = []
    for _ in range(TAMANHO_POPULACAO):
        grupo_ids = random.sample(list(df.index), TAMANHO_GRUPO)
        populacao.append(grupo_ids)
    return populacao

# ============================================
# 5 - SELEÇÃO (torneio simples)
# ============================================

def selecao(populacao):
    melhores = []
    for _ in range(2):
        g1 = random.choice(populacao)
        g2 = random.choice(populacao)
        f1 = fitness(df.loc[g1])
        f2 = fitness(df.loc[g2])
        melhores.append(g1 if f1 > f2 else g2)
    return melhores

# ============================================
# 6 - CROSSOVER (mistura entre dois grupos)
# ============================================

def crossover(pai1, pai2):
    corte = random.randint(1, TAMANHO_GRUPO - 1)
    filho = pai1[:corte] + [x for x in pai2 if x not in pai1[:corte]]
    return filho[:TAMANHO_GRUPO]

# ============================================
# 7 - MUTAÇÃO (troca aleatória de candidatos)
# ============================================

def mutacao(grupo):
    if random.random() < TAXA_MUTACAO:
        pos = random.randint(0, TAMANHO_GRUPO - 1)
        novo_id = random.choice(list(df.index))
        grupo[pos] = novo_id
    return grupo

# ============================================
# 8 - EXECUÇÃO DO ALGORITMO GENÉTICO
# ============================================

print("\n Iniciando algoritmo genético...\n")
populacao = gerar_populacao()

for geracao in range(NUM_GERACOES):
    nova_pop = []
    for _ in range(TAMANHO_POPULACAO):
        pais = selecao(populacao)
        filho = crossover(pais[0], pais[1])
        filho = mutacao(filho)
        nova_pop.append(filho)
    populacao = nova_pop

    # Mostrar progresso a cada 10 gerações
    if geracao % 10 == 0 or geracao == NUM_GERACOES - 1:
        melhor_temp = max(populacao, key=lambda g: fitness(df.loc[g]))
        print(f"Geração {geracao + 1}: melhor aptidão = {fitness(df.loc[melhor_temp]):.2f}")

# ============================================
# 9 - RESULTADO FINAL
# ============================================

melhor_grupo = max(populacao, key=lambda g: fitness(df.loc[g]))
melhor_fitness = fitness(df.loc[melhor_grupo])
grupo_final = df.loc[melhor_grupo]

print("\n Algoritmo finalizado!")
print(f"Melhor valor de aptidão: {melhor_fitness:.2f}")
print(f"Média de notas do grupo ideal: {grupo_final['media_geral'].mean():.2f}")
print(f"Diversidade (renda única): {grupo_final['Q006'].nunique()}")
print(f"Cobertura geográfica: {grupo_final['SG_UF_PROVA'].nunique()} estados")


# ----------------------------------------------------------
# Exibir a combinação final dos 100 candidatos selecionados
# ----------------------------------------------------------

print("\nLista final dos 100 candidatos selecionados:\n")
print(grupo_final[[
    "NU_INSCRICAO",
    "media_geral",
    "TP_COR_RACA",
    "Q006",
    "Q002",
    "TP_ESCOLA",
    "SG_UF_PROVA"
]].sort_values(by="media_geral", ascending=False))

# Salvar resultado em CSV
grupo_final.to_csv("bolsistas_selecionados.csv", index=False, sep=";")
print("\nArquivo 'bolsistas_selecionados.csv' salvo com sucesso!")


# ============================================
# 10 - ANÁLISE E GRÁFICOS
# ============================================

plt.figure(figsize=(8,5))
plt.hist(grupo_final["media_geral"], bins=10)
plt.title("Distribuição das médias do grupo ideal")
plt.xlabel("Média geral das notas")
plt.ylabel("Número de alunos")
plt.show()

plt.figure(figsize=(8,5))
grupo_final["SG_UF_PROVA"].value_counts().plot(kind='bar')
plt.title("Distribuição dos bolsistas por estado")
plt.xlabel("Estado")
plt.ylabel("Quantidade de alunos")
plt.show()
