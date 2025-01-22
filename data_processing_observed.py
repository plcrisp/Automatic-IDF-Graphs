import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from processing_datasets import *  # Certifique-se de que estas funções estão documentadas e funcionando


INMET_aut = read_csv('inmet_aut', 'daily')  # Estação automática INMET
INMET_conv = read_csv('inmet_conv', 'daily')  # Estação convencional INMET


def process_precipitation_series(file_names, frequency):
    """
    Processa séries temporais de precipitação, realiza leitura, preenchimento de gaps,
    união, cálculo de médias acumuladas e cria gráficos de dupla massa.

    Args:
        file_names (list): Lista com os nomes dos arquivos (sem extensão).
        frequency (str): Frequência das séries ('daily', 'monthly', etc.).
        output_csv (str): Caminho do arquivo de saída CSV para salvar os dados processados.

    Returns:
        None
    """

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

    # ----------------- ETAPA 1: LEITURA E VERIFICAÇÃO DE GAPS ----------------- #
    dataframes = {name: load_and_verify(name, frequency) for name in file_names}

    # ----------------- ETAPA 2: PREENCHIMENTO DE GAPS ----------------- #
    dataframes = {name: fill_missing_data(name, frequency) for name in file_names}

    # ----------------- ETAPA 3: UNIÃO E PROCESSAMENTO ----------------- #
    df = left_join_precipitation(*dataframes.values())
    df.columns = ['Date'] + [f'P_{name}' for name in file_names]

    df = df.dropna()  # Remove linhas com valores NaN (dados ausentes)
    df['P_average'] = df.iloc[:, 1:].mean(axis=1)  # Calcula a média das colunas de precipitação para cada dia

    for col in df.columns[1:]:  # Calcula as somas acumuladas para cada estação e para a média
        df[f'Pacum_{col}'] = df[col].fillna(0).cumsum()

    # ----------------- ETAPA 4: PLOTAGEM ----------------- #
    sns.set_context("talk", font_scale=0.8)  # Define um estilo apropriado para apresentações
    fig, axes = plt.subplots(1, len(file_names), figsize=(20, 6), sharey=True)  # Cria figura com subplots lado a lado

    for ax, name in zip(axes, file_names):
        sns.scatterplot(
            x="Pacum_P_average",  # Eixo X: soma acumulada da precipitação média
            y=f"Pacum_P_{name}",  # Eixo Y: soma acumulada da estação específica
            data=df,  # DataFrame com os dados
            ax=ax  # Define o subplot atual
        )
        ax.set_xlabel("Pacum Average (mm)")  # Define o rótulo do eixo X
        ax.set_ylabel(f"Pacum {name} (mm)")  # Define o rótulo do eixo Y
        ax.set_title(f"Scatterplot of {name}")  # Define o título do subplot

    plt.tight_layout()  # Ajusta o layout para evitar sobreposição dos subplots
    plt.show()



process_precipitation_series(['inmet_aut', 'inmet_conv', 'cemaden_agua', 'cemaden_jardim', 'maplu'], 'daily', 'all_stations_daily.csv')





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
