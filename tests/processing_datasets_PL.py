import numpy as np
import pandas as pd
import os
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import scipy
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import pymannkendall as mk
import statsmodels.api as sm
from enum import Enum

class DataSource(Enum):
    """Enum para as fontes de dados meteorológicos."""
    
    CEMADEN = 'CEMADEN'
    INMET = 'INMET'
    INMET_DAILY = 'INMET_DAILY'
    MAPLU = 'MAPLU'
    MAPLU_USP = 'MAPLU_USP'

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


def process_data(source: DataSource, data_path, year_start=None, year_end=None):
    """
    Processa dados meteorológicos de diferentes fontes.

    Parâmetros:
        source (DataSource): Enumeração que define as fontes válidas: 'CEMADEN', 'INMET', 'INMET_DAILY', 'MAPLU', 'MAPLU_USP'.
        data_path (str): Caminho para a pasta onde os dados estão armazenados.
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
        df1, df2 = process_data('INMET', 'datasets/')
    """

    if source == DataSource.CEMADEN:
        print("Processando dados do CEMADEN...")

        # Lê e concatena os arquivos CSV em um único DataFrame
        CEMADEN_df = pd.concat(
            [pd.read_csv(f'{data_path}/CEMADEN/data ({i}).csv', sep=';') for i in range(62)],
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
        CEMADEN_df['Site'] = CEMADEN_df['Site'].replace(site_replacements)

        # Divide a coluna Date em Year, Month, Day, Hour
        CEMADEN_df[['Year', 'Month', 'Day_hour']] = CEMADEN_df.Date.str.split("-", expand=True)
        CEMADEN_df[['Day', 'Hour_min']] = CEMADEN_df.Day_hour.str.split(" ", expand=True)
        CEMADEN_df[['Hour', 'Min', 'Seg']] = CEMADEN_df.Hour_min.str.split(":", expand=True)

        # Seleciona as colunas relevantes para o DataFrame final
        CEMADEN_df = CEMADEN_df[['Site', 'Year', 'Month', 'Day', 'Hour', 'Precipitation']]

        # Converte as colunas especificadas para numérico
        CEMADEN_df = convert_to_numeric(CEMADEN_df, ['Year', 'Month', 'Day', 'Hour', 'Precipitation'])

        # Filtra os DataFrames por site
        jd_sp = CEMADEN_df[CEMADEN_df['Site'] == 'Jd_Sao_Paulo']
        cidade_jardim = CEMADEN_df[CEMADEN_df['Site'] == 'Cidade_Jardim']
        agua_vermelha = CEMADEN_df[CEMADEN_df['Site'] == 'Agua_Vermelha']

        return jd_sp, cidade_jardim, agua_vermelha

    elif source in {DataSource.INMET, DataSource.INMET_DAILY}:
        print(f"Processando dados do {source}...")

        def process_inmet_data(file_path):
            """Processa os dados do INMET."""
            df = pd.read_csv(file_path, sep=';')
            df.columns = ['Date', 'Hour', 'Precipitation', 'Null']
            df = df[['Date', 'Hour', 'Precipitation']]
            df[['Year', 'Month', 'Day']] = df.Date.str.split("-", expand=True)
            df['Hour'] = df['Hour'].astype(float) / 100  # Converte hora para formato decimal
            return convert_to_numeric(df, ['Year', 'Month', 'Day', 'Hour'])

        if source == DataSource.INMET:
            # Processa os dados de estações automáticas e convencionais
            aut_df = process_inmet_data(f'{data_path}/INMET/data_aut_8h.csv')
            conv_df = process_inmet_data(f'{data_path}/INMET/data_conv_8h.csv')
        else:  # INMET_DAILY
            # Processa os dados diários de estações automáticas e convencionais
            aut_df = process_inmet_data(f'{data_path}/INMET/data_aut_daily.csv')
            conv_df = process_inmet_data(f'{data_path}/INMET/data_conv_daily.csv')

        return aut_df, conv_df

    elif source == DataSource.MAPLU:
        print("Processando dados do MAPLU...")

        def process_maplu_data(file_path, site_name):
            """Processa dados específicos do MAPLU."""
            df = pd.read_csv(file_path)
            df.columns = ['Site', 'Date', 'Precipitation']
            df['Site'] = site_name
            df[['Year', 'Month', 'Day_hour']] = df.Date.str.split("-", expand=True)
            df[['Day', 'Hour_min']] = df.Day_hour.str.split(" ", expand=True)
            df[['Hour', 'Min']] = df.Hour_min.str.split(":", expand=True)
            df = df[['Site', 'Year', 'Month', 'Day', 'Hour', 'Min', 'Precipitation']]
            return convert_to_numeric(df, ['Year', 'Month', 'Day', 'Hour', 'Min', 'Precipitation'])

        # Processa os dados da Escola e do Posto de Saúde
        esc_df = pd.concat(
            [process_maplu_data(f'{data_path}/MAPLU/escola{i}.csv', 'Escola Sao Bento') for i in range(year_start, year_end + 1)],
            ignore_index=True
        )
        posto_df = pd.concat(
            [process_maplu_data(f'{data_path}/MAPLU/postosaude{i}.csv', 'Posto Santa Felicia') for i in range(year_start, year_end + 1)],
            ignore_index=True
        )

        return esc_df, posto_df

    elif source == DataSource.MAPLU_USP:
        print("Processando dados do MAPLU_USP...")

        # Lê e processa os dados da USP
        usp_df = pd.read_csv(f'{data_path}/MAPLU/USP2.csv')
        usp_df[['Hour', 'Min']] = usp_df.Time.str.split(":", expand=True)
        usp_df = usp_df[['Year', 'Month', 'Day', 'Hour', 'Min', 'Precipitation']]
        return convert_to_numeric(usp_df, ['Year', 'Month', 'Day', 'Hour', 'Min', 'Precipitation'])
    
    else:
        raise ValueError(f"Fonte '{source}' não suportada.")





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
    var (str): Variável de agregação (ex: 'yearly', 'monthly', 'daily', 'hourly', 'min').
    directory (str): Diretório onde os arquivos estão salvos.
    
    Retorna:
    DataFrame: O DataFrame lido do arquivo CSV.
    """
    file_path = os.path.join(directory, f'{name}_{var}.csv')
    return pd.read_csv(file_path)




def verification(df):
    """
    Verifica a integridade de uma série temporal de dados meteorológicos.

    Parâmetros:
        df (DataFrame): Um DataFrame contendo colunas 'Year', 'Month', 'Day'.

    A função calcula o número total de dias entre a primeira e a última data
    e compara esse valor com o número de entradas no DataFrame. Se houver dias
    faltando, uma mensagem de erro é exibida. Caso contrário, confirma que a
    série está completa.
    """
    if df.empty:
        print("Fail - DataFrame is empty.")
        return
    
    # Verifica se as colunas 'Month' e 'Day' estão presentes
    if 'Month' not in df.columns or 'Day' not in df.columns:
        print("Fail - 'Month' and 'Day' columns are required for verification.")
        return

    # Acessa os valores de data usando iloc
    year_0, month_0, day_0 = df['Year'].iloc[0], df['Month'].iloc[0], df['Day'].iloc[0]
    year_i, month_i, day_i = df['Year'].iloc[-1], df['Month'].iloc[-1], df['Day'].iloc[-1]

    d0, di = date(year_0, month_0, day_0), date(year_i, month_i, day_i)
    ndays_verification = (di - d0).days
    ndays_real = len(df)
    
    verif_number = ndays_verification - ndays_real
    if verif_number > 0:
        print(f'Fail - series incomplete / number of days missing = {verif_number}')
    elif verif_number == 0:
        print('Series complete')
    else:
        print('Fail - dataset inválido')
        
        
        
def set_date(df):
    """
    Cria uma coluna 'Date' a partir das colunas 'Year', 'Month' e 'Day', define-a como índice e retorna o DataFrame.

    Parâmetros:
    df (DataFrame): DataFrame contendo as colunas 'Year', 'Month' e 'Day'.

    Retorna:
    DataFrame: DataFrame atualizado com a nova coluna 'Date' e o índice configurado.
    """
    # Cria a coluna 'Date' combinando 'Year', 'Month' e 'Day', ignorando erros em datas inválidas
    df['Date'] = [date(y, m, d) if pd.notnull(y) and pd.notnull(m) and pd.notnull(d) else pd.NaT
                  for y, m, d in zip(df['Year'], df['Month'], df['Day'])]

    # Define 'Date' como índice e retorna o DataFrame atualizado
    df.set_index('Date', inplace=True)
    
    return df



def complete_date_series(name, var):
    """
    Completa uma série temporal, garantindo que todas as datas entre a primeira
    e a última entrada estejam presentes no DataFrame, preenchendo datas faltantes.
    
    Parâmetros:
    name (str): Nome base do arquivo CSV contendo os dados.
    var (str): Variável de agregação (ex: 'yearly', 'monthly').
    
    Retorna:
    DataFrame: DataFrame com a série temporal completa e as datas reindexadas.
    """
    # Lê o DataFrame do arquivo CSV e configura a coluna 'Date' como índice
    df = set_date(read_csv(name, var))

    # Cria uma faixa de datas completa entre a primeira e a última data usando o índice
    idx = pd.date_range(df.index[0], df.index[-1])
    
    # Reindexa o DataFrame para preencher as datas faltantes e recria a coluna 'Date'
    df = df.reindex(idx)
    df['Date'] = df.index

    return df



def pearsonr_pval(x, y):
    """
    Função auxiliar que retorna o p-valor da correlação de Pearson entre duas séries de dados.

    Parâmetros:
    x, y (Series): Duas séries de dados numéricos.

    Retorna:
    float: p-valor da correlação de Pearson entre x e y.
    """
    return pearsonr(x, y)[1]



def left_join_precipitation(left_df, *dfs):
    """
    Faz junção 'inner' entre o DataFrame à esquerda e múltiplos DataFrames
    com base na coluna 'Date', mantendo apenas as colunas de precipitação.

    Parâmetros:
    left_df (DataFrame): DataFrame principal com a coluna 'Precipitation'.
    *dfs (DataFrame): Múltiplos DataFrames que também possuem 'Date' e 'Precipitation'.

    Retorna:
    DataFrame: DataFrame resultante contendo a coluna 'Date' e as colunas de precipitação.
    """
    # Inicializa o DataFrame de saída como o DataFrame principal (left_df)
    result_df = left_df[['Date', 'Precipitation']].rename(columns={'Precipitation': 'P_left'})
    
    # Faz a junção com cada DataFrame adicional passado em dfs
    for i, df in enumerate(dfs, 1):
        # Mantém apenas 'Date' e 'Precipitation' de cada DataFrame
        df_filtered = df[['Date', 'Precipitation']].rename(columns={'Precipitation': f'P_right{i}'})
        result_df = result_df.merge(df_filtered, on='Date', how='inner')
    
    return result_df




def correlation_plots(*dfs):
    """
    Gera gráficos de dispersão (pairplots) e calcula a correlação de Pearson entre as colunas de precipitação
    de múltiplos DataFrames passados.

    Parâmetros:
    *dfs (DataFrame): Múltiplos DataFrames que contêm a coluna 'Precipitation'.

    Retorna:
    tuple: Matrizes de correlação e p-valores.
    """
    # Faz a junção dos DataFrames e seleciona apenas as colunas de precipitação
    df = left_join_precipitation(*dfs)
    df = df.drop(columns='Date')  # Remove a coluna 'Date' para a análise de correlação
    
    # Gera o gráfico de pairplot
    sns.pairplot(df)
    plt.show()
    
    # Calcula a correlação de Pearson e p-valores
    corr_pearson = df.corr(method='pearson')
    pvalues_pearson = df.corr(method=pearsonr_pval)
    
    # Exibe os resultados de forma clara
    print('----- Pearson Correlation Results -----\n')
    
    print('Correlation Coefficient Matrix (Pearson):')
    print(corr_pearson.to_string(float_format="%.4f"))  # Formatando para 4 casas decimais
    
    print('\nP-values Matrix:')
    print(pvalues_pearson.to_string(float_format="%.4e"))  # Formato científico para p-valores
    
    # Explicação adicional
    print("\nInterpretation of Results:")
    print("- Correlation values close to 1 or -1 indicate strong relationships.")
    print("- P-values below 0.05 suggest statistically significant correlations.")
    
    return corr_pearson, pvalues_pearson



def simple_linear_regression(left_df, right_df1, right_df2):
    """
    Realiza regressões lineares simples entre a precipitação de múltiplos DataFrames.
    
    Parâmetros:
    left_df (DataFrame): DataFrame contendo a coluna 'Precipitation' da fonte principal.
    right_df1 (DataFrame): DataFrame contendo a coluna 'Precipitation' da primeira fonte secundária.
    right_df2 (DataFrame): DataFrame contendo a coluna 'Precipitation' da segunda fonte secundária.
    
    Retorna:
    Nenhum: Exibe gráficos de dispersão e linhas de regressão.
    """
    # Junta os DataFrames e seleciona as colunas de precipitação
    df = left_join_precipitation(left_df, right_df1, right_df2)[['P_left', 'P_right1', 'P_right2']].dropna()

    # Função auxiliar para ajuste de regressão e plotagem
    def fit_and_plot(X, y, label):
        lr = LinearRegression().fit(X, y)
        print(f'R-Squared ({label}):', lr.score(X, y))
        plt.scatter(X, y, s=150, alpha=0.3, edgecolor='white')
        plt.plot(X, lr.predict(X), color='r', linewidth=3)
        plt.ylabel('P_left', fontsize=12)
        plt.xlabel(label, fontsize=12)
        plt.show()

    # Ajusta a regressão para 'P_right1' e 'P_right2'
    fit_and_plot(df[['P_right1']], df['P_left'], 'P_right1')
    fit_and_plot(df[['P_right2']], df['P_left'], 'P_right2')
    
    
    
def multiple_linear_regression(left_df, right_df1, right_df2):
    """
    Realiza uma regressão linear múltipla entre as colunas de precipitação de três DataFrames.

    Parâmetros:
    left_df (DataFrame): DataFrame da esquerda, contendo a coluna 'Precipitation'.
    right_df1 (DataFrame): Primeiro DataFrame à direita, contendo a coluna 'Precipitation'.
    right_df2 (DataFrame): Segundo DataFrame à direita, contendo a coluna 'Precipitation'.

    Retorna:
    None: Exibe o R² e os coeficientes do modelo.
    """
    # Junta os DataFrames e seleciona as colunas de precipitação
    df = left_join_precipitation(left_df, right_df1, right_df2)[['P_left', 'P_right1', 'P_right2']].dropna()

    # Define variáveis independentes (X) e dependentes (y)
    X = df[['P_right1', 'P_right2']]
    y = df['P_left']
    
    # Cria e ajusta o modelo de regressão linear
    lr = LinearRegression().fit(X, y)

    # Exibe o R² e os coeficientes
    print('R-Squared:', lr.score(X, y))
    print('Coefficients:', lr.coef_, 'Intercept:', lr.intercept_)
    
    

def trend_analysis(data, alpha_value, plot_graphs=True, site=''):
    """
    Realiza uma análise de tendência em uma série temporal de dados de precipitação
    utilizando múltiplos testes de Mann-Kendall.

    Parâmetros:
    data (pd.Series): Série temporal de dados a serem analisados.
    alpha_value (float): Nível de significância para os testes.
    plot_graphs (bool): Se True, plota os gráficos da análise (padrão: True).
    site (str): Nome do local, usado no título do gráfico (padrão: '').

    Retorna:
    None: Exibe os resultados dos testes e gráficos, se solicitado.
    """
    # Executa os testes de tendência
    
    if isinstance(data, pd.DataFrame):
        data = data['Precipitation'].dropna()
    
    tests = {
        'Original': mk.original_test(data, alpha=alpha_value),
        'Hamed-Rao': mk.hamed_rao_modification_test(data, alpha=alpha_value),
        'Yue-Wang': mk.yue_wang_modification_test(data, alpha=alpha_value),
        'Trend-Free': mk.trend_free_pre_whitening_modification_test(data, alpha=alpha_value),
        'Pre-Whitening': mk.pre_whitening_modification_test(data, alpha=alpha_value)
    }

    # Imprime os resultados dos testes
    for name, result in tests.items():
        print(f'{name}: {result}')

    # Plota os gráficos se solicitado
    if plot_graphs:
        trend_line = np.arange(len(data)) * tests['Yue-Wang'].slope + tests['Yue-Wang'].intercept
        plt.figure(figsize=(6, 4))
        plt.plot(data, label='Data')
        plt.plot(data.index, trend_line, label='Trend Line', color='red')
        plt.xlabel('Months')
        plt.ylabel('Precipitation (mm)')
        plt.title(f'Trend Analysis for {site}')
        plt.legend()
        plt.show()
        
        
        
def get_trend(var, sites_list, alpha_value, group, data_type='obs', plot_graphs=True):
    """
    Realiza a análise de tendência em dados de precipitação usando diferentes variações do 
    teste de Mann-Kendall. Os resultados são armazenados em um CSV para cada grupo e tipo de dado.

    Parâmetros:
    -----------
    var : str
        Indica a variável usada na análise ('Year' para dados anuais ou 'Max_daily' para dados diários).
    sites_list : list
        Lista de nomes dos sites para os quais a análise será feita.
    alpha_value : float
        Valor de significância para os testes (ex.: 0.05).
    group : str
        Nome do grupo ao qual os sites pertencem (usado para nomear arquivos de saída).
    data_type : str, opcional
        Tipo de dado: 'obs' para dados observados, 'mod' para dados modelados por GCM.
    plot_graphs : bool, opcional
        Indica se gráficos de tendência devem ser plotados (não implementado no exemplo atual).

    Retorno:
    --------
    Salva um arquivo CSV contendo os resultados da análise de tendência para cada site e teste.

    Exceções:
    ---------
    - ZeroDivisionError: Tratamento para evitar erro de divisão por zero em algumas análises.
    - Exception: Lança uma exceção se ocorrer um erro inesperado durante a execução.
    """
    
    print('Running get_trend...')
    
    # Dicionário para armazenar os resultados de todos os testes
    trend_results = {
        'Site': [], 'Test_Type': [], 'Tau': [], 'p_value': [], 'Trend': [], 'h': [],
        'z': [], 's': [], 'var_s': [], 'Slope': [], 'Intercept': []
    }
    
    def store_results(site, test_type, result):
        """
        Armazena os resultados de um teste específico no dicionário principal.
        
        Parâmetros:
        -----------
        site : str
            Nome do site para o qual o teste foi realizado.
        test_type : str
            Tipo do teste de Mann-Kendall (Original, Hamed-Rao, etc.).
        result : dict
            Dicionário com os resultados do teste.
        """
        trend_results['Site'].append(site)
        trend_results['Test_Type'].append(test_type)
        trend_results['Tau'].append(result['Tau'])
        trend_results['p_value'].append(result['p'])
        trend_results['Trend'].append(result['trend'])
        trend_results['h'].append(result['h'])
        trend_results['z'].append(result['z'])
        trend_results['s'].append(result['s'])
        trend_results['var_s'].append(result['var_s'])
        trend_results['Slope'].append(result['slope'])
        trend_results['Intercept'].append(result['intercept'])

    # Mapeamento dos testes de Mann-Kendall disponíveis para evitar repetição de código
    test_functions = {
        'Original': mk.original_test,
        'Hamed-Rao': mk.hamed_rao_modification_test,
        'Yue-Wang': mk.yue_wang_modification_test,
        'Trend-Free': mk.trend_free_pre_whitening_modification_test,
        'Pre-Whitening': mk.pre_whitening_modification_test
    }

    # Itera sobre todos os sites fornecidos
    for site in sites_list:
        # Define o caminho do arquivo CSV com base no tipo de dado e variável
        if data_type == 'obs':
            file_path = f'Results/tests/{site}_yearly.csv' if var == 'Year' else f'Results/max_daily_{site}.csv'
        elif data_type == 'mod':
            file_path = f'GCM_data/bias_correction/{site}_yearly.csv' if var == 'Year' else f'GCM_data/bias_correction/max_daily_{site}.csv'
        else:
            raise ValueError("Invalid data type. Use 'obs' or 'mod'.")
        
        # Lê o arquivo CSV e remove valores nulos
        df = pd.read_csv(file_path).dropna(subset=['Precipitation'])
        print(f'--- {group} / {site} ---\n')

        try:
            # Aplica todos os testes de Mann-Kendall em um loop
            for test_name, test_func in test_functions.items():
                # Executa o teste e armazena os resultados em um dicionário
                result = test_func(df[['Precipitation']], alpha=alpha_value)
                store_results(site, test_name, {
                    'Tau': result[4], 'p': result[2], 'trend': result[0], 'h': result[1],
                    'z': result[3], 's': result[5], 'var_s': result[6],
                    'slope': result[7], 'intercept': result[8]
                })
            print('')
        except ZeroDivisionError:
            print('Division by zero!! - Not possible to perform trend analysis\n')
        except Exception as e:
            print(f'ALERT!!! Something else went wrong: {e}')
            raise

    # Cria um DataFrame com os resultados de todos os testes
    df_trend_result = pd.DataFrame(trend_results)
    
    # Define o caminho do arquivo de saída com base no tipo de dado
    output_path = (
        f'Results/{group}_{var}_trend_result.csv'
        if data_type == 'obs'
        else f'GCM_data/bias_correction/{group}_{var}_trend_result.csv'
    )
    
    # Salva o DataFrame em um arquivo CSV
    df_trend_result.to_csv(output_path, index=False, encoding='latin1')
    
    
    
def calculate_p90(data):
    """
    Calcula o percentil de 90% (P90) para valores de precipitação, ou seja, o valor que é excedido em apenas 10% das observações.
    Também plota o gráfico da probabilidade acumulada de não excedência em função da precipitação.

    Parâmetros:
    data (pd.DataFrame): DataFrame com uma coluna 'Precipitation' contendo valores de precipitação.

    Retorna:
    float: O valor de precipitação correspondente ao percentil de 90% (P90).
    """
    
    # Filtra e ordena os valores de precipitação, excluindo zeros
    df = data[['Precipitation']].query('Precipitation != 0').sort_values('Precipitation').reset_index(drop=True)

    # Calcula a probabilidade de não excedência para cada valor em porcentagem
    df['Probability'] = (df.index + 1) / len(df) * 100
    
    # Filtra o valor de precipitação onde a probabilidade de não excedência é aproximadamente 90%
    p90_value = df.loc[df['Probability'] >= 90, 'Precipitation'].iloc[0]

    # Plota o gráfico da probabilidade acumulada de não excedência
    sns.lineplot(x='Probability', y='Precipitation', data=df, color='black')
    plt.ylabel('Precipitation (mm)', fontsize=12)
    plt.xlabel('Probability (%)', fontsize=12)
    plt.title("Probability of Non-Exceedence")
    plt.show()
    
    return p90_value



def distribution_plot(name, var):
    """
    Gera um gráfico de densidade dos dados de precipitação 
    contidos em um arquivo CSV.

    Parâmetros:
    name (str): Nome do arquivo CSV.
    var (str): Nome da coluna de precipitação a ser exibida no título do gráfico.

    Retorna:
    None: Exibe o gráfico de densidade.
    """
    
    # Lê o arquivo CSV e carrega a coluna especificada
    df = read_csv(name, var)
    
    # Remove valores ausentes
    df = df.dropna()
    
    # Gera o gráfico de densidade
    sns.kdeplot(df['Precipitation'], color='skyblue', fill=True)
    
    plt.title(f'{name} - {var}')
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Density')
    plt.show()
    
    

def distribution_plot_df(df):
    """
    Gera um gráfico de densidade dos dados de precipitação 
    a partir de um DataFrame.

    Parâmetros:
    df (DataFrame): Um DataFrame contendo uma coluna 'Precipitation' 
                    com os dados de precipitação.

    Retorna:
    None: Exibe o gráfico de densidade.
    """
    
    # Remove valores ausentes da coluna 'Precipitation'
    df = df.dropna(subset=['Precipitation'])
    
    # Gera o gráfico de densidade
    sns.kdeplot(df['Precipitation'], color='skyblue', fill=True)
    
    # Configurações do gráfico
    plt.title('Distribuição de Precipitação')
    plt.xlabel('Precipitação (mm)')
    plt.ylabel('Densidade')
    
    # Exibe o gráfico
    plt.show()
    
    
    
def aggregate_precipitation(df, interval, dt_min=None):
    """
    Agrega os dados de precipitação em intervalos especificados.

    Parâmetros:
    df (DataFrame): Um DataFrame contendo uma coluna 'Precipitation' 
                    com os dados de precipitação, indexados por tempo.
    interval (int): O intervalo de agregação desejado:
                    - Se `dt_min` for None, considera 'interval' em horas.
                    - Se `dt_min` for um valor conhecido, considera 'interval' em minutos.
    dt_min (int, opcional): A resolução temporal dos dados em minutos. 
                            Necessário se 'interval' for em minutos.

    Retorna:
    list: Uma lista contendo as somas de precipitação para cada intervalo 
          especificado.
    """
    
    # Verifica se o DataFrame contém a coluna 'Precipitation'
    if 'Precipitation' not in df.columns:
        raise ValueError("O DataFrame deve conter uma coluna 'Precipitation'.")
    
    # Remove valores ausentes da coluna 'Precipitation'
    df = df[['Precipitation']].dropna()
    
    # Lista para armazenar os resultados acumulados
    acum_list = []

    if dt_min is None:
        # Caso padrão: agregação em horas
        n = interval  # Intervalo em horas
        for i in range(len(df) - n + 1):
            # Soma os valores de 'Precipitation' nas 'n' horas atuais
            acum = df.iloc[i:n + i]['Precipitation'].sum()
            acum_list.append(acum)
    
    else:
        # Caso em que o intervalo é em minutos
        n = interval // dt_min  # Calcula quantas entradas de dados devem ser somadas
        for i in range(len(df) - n + 1):
            # Soma os valores de 'Precipitation' nas 'n' entradas atuais
            acum = df.iloc[i:n + i]['Precipitation'].sum()
            acum_list.append(acum)

    return acum_list





    
    
# Teste a função
#jd_sp, cidade_jardim, agua_vermelha = process_data(DataSource.CEMADEN,'datasets')
INMET_aut_df, INMET_conv_df = process_data(DataSource.INMET,'datasets')
#INMET_DAILY_aut_df, INMET_DAILY_conv_df = process_data('INMET_DAILY')
MAPLU_esc_df, MAPLU_post_df = process_data(DataSource.MAPLU, 'datasets', year_start=2015, year_end=2017)


# Exibir os primeiros resultados de cada DataFrame

#print("Jardim São Paulo:\n", jd_sp.head(), "\n")
#print("Cidade Jardim:\n", cidade_jardim.head(), "\n")
#print("Água Vermelha:\n", agua_vermelha.head(), "\n")
#print("Inmet Aut:\n", INMET_aut_df.head(), "\n")
#print("Inmet conv:\n", INMET_conv_df.head(), "\n")
#print("Inmet Daily Aut:\n", INMET_DAILY_aut_df.head(), "\n")
#print("Inmet Daily conv:\n", INMET_DAILY_conv_df.head(), "\n")
#print("MAPLU Escola:\n", MAPLU_esc_df.head(), "\n")
#print("MAPLU Posto Saude:\n", MAPLU_post_df.head(), "\n")


# Supondo que 'df' seja o DataFrame carregado e processado
aggregate_to_csv(INMET_aut_df, 'inmet')
aggregate_to_csv(MAPLU_esc_df, 'maplu')
#aggregate_to_csv(jd_sp, 'jardim')


# Para ler um arquivo CSV específico
#df_inmet = read_csv('inmet', 'hourly')
df_maplu = read_csv('maplu', 'min')

#print(df_inmet[:64]) 

#df_inmet_3hour = aggregate_precipitation(df_inmet,60)

#print(df_inmet_3hour[:64]) 

#print(df_maplu[1590:1651]) 

#df_maplu_10min = aggregate_precipitation(df_maplu,10,1)

#print(df_maplu_10min[1590:1651])

#calculate_p90(df_inmet)

#distribution_plot('inmet','yearly')
#distribution_plot_df(df_inmet)
#distribution_plot('inmet','monthly')
#distribution_plot('inmet','daily')


#df_inmet = complete_date_series('inmet', 'monthly')


#print("Jardim São Paulo:\n", df_jd.head(), "\n")
#print("Cidade Jardim:\n", df_cj.head(), "\n")
#print("Água Vermelha:\n", df_av.head(), "\n")

#print(df_inmet.head())
#print(df_inmet.tail())


#trend_analysis(df_inmet,0.05,site='INMET AUT')

sites = ['inmet', 'jardim']

# Variável que você quer analisar: 'Year' ou 'Max_daily'
var = 'Year'

# Valor de significância para os testes (ex.: 0.05 para 95% de confiança)
alpha = 0.05

# Nome do grupo ao qual os sites pertencem (usado para nomear os arquivos de saída)
group_name = 'Group_A'

# Tipo de dado: 'obs' (observado) ou 'mod' (modelado por GCM)
data_type = 'obs'

# Chamada da função
#get_trend(var=var, sites_list=sites, alpha_value=alpha, group=group_name, data_type=data_type, plot_graphs=True)








