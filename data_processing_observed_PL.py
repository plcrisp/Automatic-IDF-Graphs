import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from processing_datasets_PL import *  # Certifique-se de que estas funções estão documentadas e funcionando




# ----------------- ETAPA 1: LEITURA E VERIFICAÇÃO DE GAPS ----------------- #
# Função auxiliar para leitura e verificação de séries temporais
def load_and_verify(file_name, frequency):
    """
    Lê um arquivo CSV e realiza a verificação de gaps.

    Args:
        file_name (str): Nome do arquivo sem extensão.
        frequency (str): Frequência esperada ('daily', 'monthly', etc.).

    Returns:
        pd.DataFrame: DataFrame com os dados carregados e verificados.
    """
    df = read_csv(file_name, frequency)  # Lê o arquivo usando a função específica definida no módulo
    verification(df)  # Verifica gaps ou inconsistências na série
    return df

# Leitura e verificação dos dados de cada estação
INMET_aut = load_and_verify('inmet_aut', 'daily')  # Estação automática INMET
INMET_conv = load_and_verify('inmet_conv', 'daily')  # Estação convencional INMET
agua_vermelha = load_and_verify('cemaden_agua', 'daily')  # Cemaden (Água Vermelha)
jd_sp = load_and_verify('cemaden_jardim', 'daily')  # Cemaden (Jardim SP)
maplu = load_and_verify('maplu', 'daily')  # MAPLU

# Exibição inicial de amostra dos dados para verificar leitura
print(INMET_conv.head())  # Exibe as primeiras linhas da estação convencional INMET





# ----------------- ETAPA 2: PREENCHIMENTO DE GAPS ----------------- #
# Preenche gaps (datas ausentes) para todas as séries temporais
INMET_aut = complete_date_series('inmet_aut', 'daily')  # Preenche para INMET automática
INMET_conv = complete_date_series('inmet_conv', 'daily')  # Preenche para INMET convencional
agua_vermelha = complete_date_series('cemaden_agua', 'daily')  # Preenche para Cemaden (Água Vermelha)
jd_sp = complete_date_series('cemaden_jardim', 'daily')  # Preenche para Cemaden (Jardim SP)
maplu = complete_date_series('maplu', 'daily')  # Preenche para MAPLU







# ----------------- ETAPA 3: UNIÃO E PROCESSAMENTO ----------------- #
# Junta todas as séries em um único DataFrame
df = left_join_precipitation(agua_vermelha, jd_sp, INMET_aut, INMET_conv, maplu)


df.columns = ['Date', 'P_av', 'P_jdsp', 'P_inmet_aut', 'P_inmet_conv', 'P_maplu_esc'] # Renomeia as colunas para facilitar a compreensão

df = df.dropna() # Remove linhas com valores NaN (dados ausentes)


df['P_average'] = df[['P_av', 'P_jdsp', 'P_inmet_aut', 'P_inmet_conv', 'P_maplu_esc']].mean(axis=1) # Calcula a média das colunas de precipitação para cada dia


for col in ['P_av', 'P_jdsp', 'P_inmet_aut', 'P_inmet_conv', 'P_maplu_esc', 'P_average']: # Calcula as somas acumuladas para cada estação e para a média
    df[f'Pacum_{col}'] = df[col].fillna(0).cumsum()  # Soma acumulada, preenchendo NaN com 0


df.to_csv('Results/dupla_massa_new.csv', index=False) # Salva o DataFrame processado em um arquivo CSV para análises futuras





# ----------------- ETAPA 4: PLOTAGEM ----------------- #
# Configurações de estilo do Seaborn
sns.set_context("talk", font_scale=0.8)  # Define um estilo apropriado para apresentações
fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=True)  # Cria figura com 5 subplots lado a lado

# Lista das colunas de precipitação acumulada para as estações
stations = ['P_av', 'P_jdsp', 'P_inmet_aut', 'P_inmet_conv', 'P_maplu_esc']
titles = ['Água Vermelha', 'JDSP', 'INMET Automática', 'INMET Convencional', 'MAPLU ESC']

# Loop para criar gráficos de dispersão para cada estação
for ax, station, title in zip(axes, stations, titles):
    sns.scatterplot(
        x="Pacum_P_average",  # Eixo X: soma acumulada da precipitação média
        y=f"Pacum_{station}",  # Eixo Y: soma acumulada da estação específica
        data=df,  # DataFrame com os dados
        ax=ax  # Define o subplot atual
    )
    ax.set_xlabel("Pacum Average (mm)")  # Define o rótulo do eixo X
    ax.set_ylabel(f"Pacum {title} (mm)")  # Define o rótulo do eixo Y
    ax.set_title(f"Scatterplot of {title}")  # Define o título do subplot

# Ajusta o layout para evitar sobreposição dos subplots
plt.tight_layout()

# Exibe o gráfico
plt.show()







# ----------------- ETAPA 5: ANÁLISES ESTATÍSTICAS ----------------- #
# Calcula o P90 (percentil 90) para as séries do INMET
print('P90 INMET / Automatic: ', calculate_p90(INMET_aut))  # P90 da estação automática
print('P90 INMET / Conventional: ', calculate_p90(INMET_conv))  # P90 da estação convencional

# Calcula a precipitação máxima anual para cada estação
max_annual_precipitation(INMET_conv, 'inmet_conv')  # Estação convencional
max_annual_precipitation(INMET_aut, 'inmet_aut')  # Estação automática

# Define o nível de significância para testes de tendência
alpha_value = 0.1

# Testa tendências anuais na precipitação
print('Annual precipitation')
group = 'INMET'
sites_list = ['inmet_aut', 'inmet_conv']
get_trend('Year', sites_list, alpha_value, group)  # Testa tendência na precipitação anual

# Testa tendências nos valores máximos diários
print('Daily maximum')
get_trend('Max_daily', sites_list, alpha_value, group, plot_graphs=True)  # Testa tendência nos máximos diários com gráficos
