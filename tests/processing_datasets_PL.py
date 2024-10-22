import pandas as pd
import os

VALID_SOURCES = ['CEMADEN', 'INMET', 'INMET_DAILY', 'MAPLU', 'MAPLU_USP']

def convert_to_numeric(df, columns):
    """
    Converte colunas especificadas de um DataFrame para tipo numérico.

    Parâmetros:
        df (DataFrame): O DataFrame a ser processado.
        columns (list): Lista com os nomes das colunas a serem convertidas.
    
    Retorna:
        DataFrame: O DataFrame com as colunas convertidas.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df


def process_data(source, year_start=None, year_end=None):
    """
    Processa dados meteorológicos de diferentes fontes.

    Parâmetros:
        source (str): A fonte de dados, deve ser uma das seguintes: 
                      'CEMADEN', 'INMET', 'INMET_DAILY', 'MAPLU', 'MAPLU_USP'.
        mode (str, opcional): Modo de processamento, se aplicável.
        year_start (int, opcional): Ano inicial para filtragem, se aplicável.
        year_end (int, opcional): Ano final para filtragem, se aplicável.

    Retornos:
        - Se source for 'CEMADEN': Retorna três DataFrames correspondentes aos sites:
          (DataFrame Jd_Sao_Paulo, DataFrame Cidade_Jardim, DataFrame Agua_Vermelha).
        - Se source for 'INMET' ou 'INMET_DAILY': Retorna dois DataFrames
          (DataFrame aut, DataFrame conv).
        - Se source for 'MAPLU': Retorna dois DataFrames
          (DataFrame Escola, DataFrame Posto).
        - Se source for 'MAPLU_USP': Retorna um DataFrame (DataFrame USP).

    Exemplo de uso:
        df1, df2 = process_data('INMET')
    """

    if source not in VALID_SOURCES:
        raise ValueError(f"Fonte '{source}' inválida. As fontes válidas são: {', '.join(VALID_SOURCES)}")

    if source == 'CEMADEN':
        print("Processando dados do CEMADEN...")

        # Lê e concatena os arquivos CSV em um único DataFrame
        CEMADEN_df = pd.concat(
            [pd.read_csv(f'datasets/CEMADEN/data ({i}).csv', sep=';') for i in range(62)],
            ignore_index=True,
            sort=False
        )

        # Renomeia as colunas e seleciona as relevantes
        CEMADEN_df.columns = ['1', '2', '3', '4', '5', '6', '7', '8']
        CEMADEN_df = CEMADEN_df[['5', '6', '7']]
        CEMADEN_df.columns = ['Site', 'Date', 'Precipitation']

        # Substitui vírgulas por pontos nas precipitações e renomeia locais
        CEMADEN_df['Precipitation'] = CEMADEN_df['Precipitation'].str.replace(',', '.')
        site_replacements = {
            '-22,031': 'Jd_Sao_Paulo',
            '-21,997': 'Cidade_Jardim',
            '-21,898': 'Agua_Vermelha'
        }
        for old, new in site_replacements.items():
            CEMADEN_df['Site'] = CEMADEN_df['Site'].str.replace(old, new)

        # Divide a coluna Date em Year, Month, Day, Hour
        CEMADEN_df[['Year', 'Month', 'Day_hour']] = CEMADEN_df.Date.str.split("-", expand=True)
        CEMADEN_df[['Day', 'Hour_min']] = CEMADEN_df.Day_hour.str.split(" ", expand=True)
        CEMADEN_df[['Hour', 'Min', 'Seg']] = CEMADEN_df.Hour_min.str.split(":", expand=True)

        # Seleciona as colunas relevantes para o DataFrame final
        CEMADEN_df = CEMADEN_df[['Site', 'Year', 'Month', 'Day', 'Hour', 'Precipitation']]

        # Converte as colunas especificadas para numérico
        CEMADEN_df['Precipitation'] = pd.to_numeric(CEMADEN_df['Precipitation'])
        CEMADEN_df = convert_to_numeric(CEMADEN_df, ['Year', 'Month', 'Day', 'Hour'])

        # Filtra os DataFrames por site
        jd_sp = CEMADEN_df[CEMADEN_df['Site'] == 'Jd_Sao_Paulo']
        cidade_jardim = CEMADEN_df[CEMADEN_df['Site'] == 'Cidade_Jardim']
        agua_vermelha = CEMADEN_df[CEMADEN_df['Site'] == 'Agua_Vermelha']

        return jd_sp, cidade_jardim, agua_vermelha

    elif source == 'INMET':
        print("Processando dados do INMET...")

        def process_inmet_data(file_path):
            df = pd.read_csv(file_path, sep=';')
            df.columns = ['Date', 'Hour', 'Precipitation', 'Null']
            df = df[['Date', 'Hour', 'Precipitation']]
            df[['Year', 'Month', 'Day']] = df.Date.str.split("-", expand=True)
            df['Hour'] = (df['Hour'] / 100)  # Converte Hora para formato decimal
            df = df[['Year', 'Month', 'Day', 'Hour', 'Precipitation']]
            
            # Converte colunas para tipo numérico
            return convert_to_numeric(df, ['Hour', 'Year', 'Month', 'Day'])

        # Processa os arquivos
        INMET_aut_df = process_inmet_data('datasets/INMET/data_aut_8h.csv')
        INMET_conv_df = process_inmet_data('datasets/INMET/data_conv_8h.csv')

        return INMET_aut_df, INMET_conv_df
    
    elif source == 'INMET_DAILY':
        print("Processando dados diários do INMET...")

        def process_inmet_daily_data(file_path):
            df = pd.read_csv(file_path, sep=';') 
            df.columns = ['Date', 'Precipitation', 'Null']
            df = df[['Date', 'Precipitation']]
            df[['Year', 'Month', 'Day']] = df.Date.str.split("-", expand=True)

            # Converte colunas para tipo numérico
            return convert_to_numeric(df, ['Year', 'Month', 'Day'])

        # Processa os arquivos
        INMET_aut_df = process_inmet_daily_data('datasets/INMET/data_aut_daily.csv')
        INMET_conv_df = process_inmet_daily_data('datasets/INMET/data_conv_daily.csv')

        return INMET_aut_df, INMET_conv_df

    elif source == 'MAPLU':
        print("Processando dados do MAPLU...")

        def process_maplu_data(file_path, site_name):
            df = pd.read_csv(file_path)
            df['Site'] = df['Site'].str.replace(file_path.split('/')[-1].split('.')[0], site_name)  # Substitui o nome do site
            df.columns = ['Site', 'Date', 'Precipitation']
            df[['Year', 'Month', 'Day_hour']] = df.Date.str.split("-", expand=True)
            df[['Day', 'Hour_min']] = df.Day_hour.str.split(" ", expand=True)
            df[['Hour', 'Min']] = df.Hour_min.str.split(":", expand=True)
            df = df[['Site', 'Year', 'Month', 'Day', 'Hour', 'Min', 'Precipitation']]
            df['Precipitation'] = pd.to_numeric(df['Precipitation'])
            
            # Converte colunas para tipo numérico
            return convert_to_numeric(df, ['Year', 'Month', 'Day', 'Hour', 'Min'])

        # Processa os dados da Escola
        MAPLU_esc_df = pd.concat(
            [process_maplu_data(f'datasets/MAPLU/escola{i}.csv', 'Escola Sao Bento') for i in range(year_start, year_end + 1)],
            ignore_index=True,
            sort=False
        )

        # Processa os dados do Posto de Saúde
        MAPLU_post_df = pd.concat(
            [process_maplu_data(f'datasets/MAPLU/postosaude{i}.csv', 'Posto Santa Felicia') for i in range(year_start, year_end + 1)],
            ignore_index=True,
            sort=False
        )

        return MAPLU_esc_df, MAPLU_post_df

    elif source == 'MAPLU_USP':
        print("Processando dados do MAPLU_USP...")

        # Lendo os dados da USP
        MAPLU_usp_df = pd.read_csv('datasets/MAPLU/USP2.csv')

        # Extraindo Hora e Minuto da coluna Time
        MAPLU_usp_df[['Hour', 'Min']] = MAPLU_usp_df.Time.str.split(":", expand=True)

        # Selecionando e renomeando as colunas
        MAPLU_usp_df = MAPLU_usp_df[['Year', 'Month', 'Day', 'Hour', 'Min', 'Precipitation']]

        # Convertendo colunas para tipo numérico
        MAPLU_usp_df['Precipitation'] = pd.to_numeric(MAPLU_usp_df['Precipitation'])
        MAPLU_usp_df = convert_to_numeric(CEMADEN_df, ['Year', 'Month', 'Day', 'Hour', 'Min'])

        return MAPLU_usp_df




# Função para agregação flexível
def aggregate(df, vars):
    """
    Agrega os dados de precipitação com base nas colunas fornecidas.
    
    Parâmetros:
    df (DataFrame): O DataFrame com os dados.
    vars (list): Lista de variáveis para agrupar (ex: ['Year'], ['Year', 'Month']).
    
    Retorna:
    DataFrame: DataFrame com os dados agregados.
    """
    return df.groupby(vars).Precipitation.sum().reset_index()

# Função para salvar os arquivos CSV
def save_to_csv(df, name, var, directory):
    """
    Salva um DataFrame em formato CSV no diretório especificado.
    
    Parâmetros:
    df (DataFrame): O DataFrame a ser salvo.
    name (str): Nome base do arquivo.
    var (str): Nome da variável de agregação (ex: 'yearly', 'monthly').
    directory (str): Caminho do diretório onde salvar.
    """
    # Garante que o diretório existe
    os.makedirs(directory, exist_ok=True)
    
    # Define o caminho completo e salva o arquivo CSV
    file_path = os.path.join(directory, f'{name}_{var}.csv')
    df.to_csv(file_path, index=False)

# Função agregada e mais flexível para salvar diferentes agregações
def aggregate_to_csv(df, name, directory='Results/tests'):
    """
    Agrega os dados e salva em arquivos CSV anuais, mensais e diários.
    
    Parâmetros:
    df (DataFrame): O DataFrame com os dados.
    name (str): Nome base para os arquivos.
    directory (str): Caminho do diretório onde salvar os resultados.
    """
    # Agrega por ano, mês, dia, hora
    df_yearly = aggregate(df, ['Year'])
    save_to_csv(df_yearly, name, 'yearly', directory)

    df_monthly = aggregate(df, ['Year', 'Month'])
    save_to_csv(df_monthly, name, 'monthly', directory)

    df_daily = aggregate(df, ['Year', 'Month', 'Day'])
    save_to_csv(df_daily, name, 'daily', directory)
    
    df_hourly = aggregate(df, ['Year', 'Month', 'Day', 'Hour'])
    save_to_csv(df_hourly, name, 'hourly', directory)

    # Salva o DataFrame original completo (não agregado)
    save_to_csv(df, name, 'min', directory)

# Função para ler CSV
def read_csv(name, var, directory='Results/tests'):
    """
    Lê um arquivo CSV gerado pela agregação.
    
    Parâmetros:
    name (str): Nome base do arquivo.
    var (str): Variável de agregação (ex: 'yearly', 'monthly').
    directory (str): Diretório onde os arquivos estão salvos.
    
    Retorna:
    DataFrame: O DataFrame lido do arquivo CSV.
    """
    file_path = os.path.join(directory, f'{name}_{var}.csv')
    return pd.read_csv(file_path)

    
    
# Teste a função
jd_sp, cidade_jardim, agua_vermelha = process_data('CEMADEN')
INMET_aut_df, INMET_conv_df = process_data('INMET')
INMET_DAILY_aut_df, INMET_DAILY_conv_df = process_data('INMET_DAILY')
MAPLU_esc_df, MAPLU_post_df = process_data('MAPLU', year_start=2015, year_end=2018)


# Exibir os primeiros resultados de cada DataFrame
"""
print("Jardim São Paulo:\n", jd_sp.head(), "\n")
print("Cidade Jardim:\n", cidade_jardim.head(), "\n")
print("Água Vermelha:\n", agua_vermelha.head(), "\n")
print("Inmet Aut:\n", INMET_aut_df.head(), "\n")
print("Inmet conv:\n", INMET_conv_df.head(), "\n")
print("Inmet Daily Aut:\n", INMET_DAILY_aut_df.head(), "\n")
print("Inmet Daily conv:\n", INMET_DAILY_conv_df.head(), "\n")
print("MAPLU Escola:\n", MAPLU_esc_df.head(), "\n")
print("MAPLU Posto Saude:\n", MAPLU_post_df.head(), "\n")
"""

# Supondo que 'df' seja o DataFrame carregado e processado
aggregate_to_csv(INMET_aut_df, 'dados_precipitacao')

# Para ler um arquivo CSV específico
df_yearly = read_csv('dados_precipitacao', 'yearly')
df_monthly = read_csv('dados_precipitacao', 'monthly')