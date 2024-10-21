import pandas as pd
import os

VALID_SOURCES = ['CEMADEN', 'INMET', 'MAPLU', 'MAPLU_USP']

def process_data(source, mode=None, year_start=None, year_end=None):
    
    """
    Processa dados meteorológicos de diferentes fontes.

    Parâmetros:
        source (str): A fonte de dados, deve ser uma das seguintes: 'CEMADEN', 'INMET', 'MAPLU', 'MAPLU_USP'.
        mode (str, opcional): Modo de processamento, se aplicável.
        year_start (int, opcional): Ano inicial para filtragem, se aplicável.
        year_end (int, opcional): Ano final para filtragem, se aplicável.
        
    Retorna:
        DataFrame: Um ou mais DataFrames processados conforme a fonte escolhida.
        
    Exemplo de uso:
        df1, df2 = process_data('INMET')
    """


    if source not in VALID_SOURCES:
        raise ValueError(f"Fonte '{source}' inválida. As fontes válidas são: {', '.join(VALID_SOURCES)}")
    
    if source == 'CEMADEN':
        print("Processando dados do CEMADEN...")
        
        CEMADEN_df = None
        for i in range(0, 62):
            df = pd.read_csv(f'datasets/CEMADEN/data ({i}).csv', sep=';')
            if i == 0:
                CEMADEN_df = df
            else:
                CEMADEN_df = pd.concat([CEMADEN_df, df], ignore_index=True, sort=False)

        CEMADEN_df.columns = ['1', '2', '3', '4', '5', '6', '7', '8']
        CEMADEN_df = CEMADEN_df[['5', '6', '7']]
        CEMADEN_df.columns = ['Site', 'Date', 'Precipitation']
        CEMADEN_df['Precipitation'] = CEMADEN_df['Precipitation'].str.replace(',', '.')
        CEMADEN_df['Site'] = CEMADEN_df['Site'].str.replace('-22,031', 'Jd_Sao_Paulo')
        CEMADEN_df['Site'] = CEMADEN_df['Site'].str.replace('-21,997', 'Cidade_Jardim')
        CEMADEN_df['Site'] = CEMADEN_df['Site'].str.replace('-21,898', 'Agua_Vermelha')

        CEMADEN_df[['Year', 'Month', 'Day_hour']] = CEMADEN_df.Date.str.split("-", expand=True)
        CEMADEN_df[['Day', 'Hour_min']] = CEMADEN_df.Day_hour.str.split(" ", expand=True)
        CEMADEN_df[['Hour', 'Min', 'Sec']] = CEMADEN_df.Hour_min.str.split(":", expand=True)
        CEMADEN_df = CEMADEN_df[['Site', 'Year', 'Month', 'Day', 'Hour', 'Precipitation']]

        jd_sp = CEMADEN_df[CEMADEN_df['Site']=='Jd_Sao_Paulo']
        cidade_jardim = CEMADEN_df[CEMADEN_df['Site']=='Cidade_Jardim']
        agua_vermelha = CEMADEN_df[CEMADEN_df['Site']=='Agua_Vermelha']

        return jd_sp, cidade_jardim, agua_vermelha

    elif source == 'INMET':
        print("Processando dados do INMET...")
        
        if mode == 'hourly':
            INMET_aut_df = pd.read_csv('datasets/INMET/data_aut_8h.csv', sep=';')
            INMET_aut_df.columns = ['Date', 'Hour', 'Precipitation', 'Null']
            INMET_aut_df = INMET_aut_df[['Date', 'Hour', 'Precipitation']]
            INMET_aut_df[['Year', 'Month', 'Day']] = INMET_aut_df.Date.str.split("-", expand=True)
            INMET_aut_df['Hour'] = pd.to_numeric(INMET_aut_df['Hour'] / 100, downcast='integer')

            INMET_conv_df = pd.read_csv('datasets/INMET/data_conv_8h.csv', sep=';')
            INMET_conv_df.columns = ['Date', 'Hour', 'Precipitation', 'Null']
            INMET_conv_df = INMET_conv_df[['Date', 'Hour', 'Precipitation']]
            INMET_conv_df[['Year', 'Month', 'Day']] = INMET_conv_df.Date.str.split("-", expand=True)
            INMET_conv_df['Hour'] = pd.to_numeric(INMET_conv_df['Hour'] / 100, downcast='integer')

            return INMET_aut_df, INMET_conv_df
        
        elif mode == 'daily':
            INMET_aut_df = pd.read_csv('datasets/INMET/data_aut_daily.csv', sep=';')
            INMET_aut_df.columns = ['Date', 'Precipitation', 'Null']
            INMET_aut_df = INMET_aut_df[['Date', 'Precipitation']]
            INMET_aut_df[['Year', 'Month', 'Day']] = INMET_aut_df.Date.str.split("-", expand=True)

            INMET_conv_df = pd.read_csv('datasets/INMET/data_conv_daily.csv', sep=';')
            INMET_conv_df.columns = ['Date', 'Precipitation', 'Null']
            INMET_conv_df = INMET_conv_df[['Date', 'Precipitation']]
            INMET_conv_df[['Year', 'Month', 'Day']] = INMET_conv_df.Date.str.split("-", expand=True)

            return INMET_aut_df, INMET_conv_df

    elif source == 'MAPLU':
        print("Processando dados do MAPLU...")
        
        MAPLU_esc_df, MAPLU_post_df = None, None

        for i in range(year_start, year_end + 1):
            esc_df = pd.read_csv(f'datasets/MAPLU/escola{i}.csv')
            esc_df['Site'] = esc_df['Site'].str.replace(f'escola{i}', 'Escola Sao Bento')
            if MAPLU_esc_df is None:
                MAPLU_esc_df = esc_df
            else:
                MAPLU_esc_df = pd.concat([MAPLU_esc_df, esc_df], ignore_index=True, sort=False)

            post_df = pd.read_csv(f'datasets/MAPLU/postosaude{i}.csv')
            post_df['Site'] = post_df['Site'].str.replace(f'postosaude{i}', 'Posto Santa Felicia')
            if MAPLU_post_df is None:
                MAPLU_post_df = post_df
            else:
                MAPLU_post_df = pd.concat([MAPLU_post_df, post_df], ignore_index=True, sort=False)

        return MAPLU_esc_df, MAPLU_post_df

    elif source == 'MAPLU_USP':
        print("Processando dados do MAPLU_USP...")
        
        MAPLU_usp_df = pd.read_csv('datasets/MAPLU/USP2.csv')
        MAPLU_usp_df[['Hour', 'Min']] = MAPLU_usp_df.Time.str.split(":", expand=True)
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
def read_csv(name, var, directory='Results/testes'):
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
INMET_aut_df, INMET_conv_df = process_data('INMET', mode='hourly')
MAPLU_esc_df, MAPLU_post_df = process_data('MAPLU', year_start=2015, year_end=2018)

# Exibir os primeiros resultados de cada DataFrame
print("Jardim São Paulo:\n", jd_sp.head())
print("Cidade Jardim:\n", cidade_jardim.head())
print("Água Vermelha:\n", agua_vermelha.head())
print("Inmet Aut:\n", INMET_aut_df.head())
print("Inmet conv:\n", INMET_conv_df.head())
print("MAPLU Escola:\n", MAPLU_esc_df.head())
print("MAPLU Posto Saude:\n", MAPLU_post_df.head())

# Supondo que 'df' seja o DataFrame carregado e processado
aggregate_to_csv(jd_sp, 'dados_precipitacao')

# Para ler um arquivo CSV específico
df_yearly = read_csv('dados_precipitacao', 'yearly')
df_monthly = read_csv('dados_precipitacao', 'monthly')