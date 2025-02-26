import numpy as np
import pandas as pd
import os
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import pymannkendall as mk

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
def aggregate_to_csv(df, name, directory='Results'):
    """
    Agrega os dados e salva em arquivos CSV anuais, mensais, diários e por hora.
    
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
def read_csv(name, var, directory='Results'):
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
        print('Series complete!')
    else:
        print('Fail - invalid dataset')
        
        
        
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
    
    # Cria uma faixa de datas completa entre a primeira e a última data usando o índice
    idx = pd.date_range(df.index[0], df.index[-1])
    
    # Reindexa o DataFrame para preencher as datas faltantes e recria a coluna 'Date'
    df = df.reindex(idx)
    df['Date'] = df.index

    # Preenche as colunas 'Year', 'Month' e 'Day' com os valores corretos
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day

    return df



def fill_missing_data(name, var):
    """
    Preenche os valores faltantes na coluna 'Precipitation' de um DataFrame
    utilizando interpolação sazonal (baseada em grupos mensais).

    Parâmetros:
    ----------
    name : str
        Nome do arquivo ou base de dados a ser carregado.
    var : str
        Tipo de dados ou variável a ser processada (ex.: 'daily').

    Retorna:
    -------
    df : pandas.DataFrame
        DataFrame com os valores interpolados na coluna 'Precipitation'.
        Os índices do DataFrame permanecem alinhados com os valores originais.
    """
    df = set_date(read_csv(name, var))
    
    # Realiza a interpolação sazonal (por mês)
    interpolated = (
        df.groupby('Month')['Precipitation']
        .apply(lambda group: group.interpolate(method='linear'))
    )

    # Realinha os índices do resultado interpolado com o DataFrame original
    df['Precipitation'] = interpolated.reset_index(level=0, drop=True)

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
    



def get_trend(var, sites_list, alpha_value, group, data_type='obs', plot_graphs=True):
    """
    Realiza a análise de tendência em dados de precipitação usando diferentes variações do 
    teste de Mann-Kendall. Os resultados são armazenados em um CSV para cada grupo e tipo de dado,
    e gráficos de tendência são gerados se plot_graphs=True.

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
        Indica se gráficos de tendência devem ser plotados.

    Retorno:
    --------
    Salva um arquivo CSV contendo os resultados da análise de tendência para cada site e teste,
    além de gráficos de tendência, se plot_graphs=True.
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

    from scipy.stats import linregress

    def plot_trend_graph(df, site, test_name, slope, intercept, var):
        plt.figure(figsize=(10, 6))
        x_var = 'Year'
        y_var = 'Precipitation'

        # Plota os pontos de dados
        sns.scatterplot(data=df, x=x_var, y=y_var, color='blue', label='Observações')
        
        # Calcula a linha de regressão linear
        regression = linregress(df[x_var], df[y_var])
        y_values = regression.slope * df[x_var] + regression.intercept
        
        plt.plot(df[x_var], y_values, color='green', label='Linha de Regressão Linear')
        
        # Calcula a linha de tendência (Sen's Slope)
        trend_values = slope * df[x_var] + intercept
        plt.plot(df[x_var], trend_values, color='red', label=f'Linha de Tendência ({test_name})')
        
        # Configurações do gráfico
        plt.title(f'Tendência de Precipitação - {site} ({test_name})', fontsize=14)
        plt.xlabel(x_var, fontsize=12)
        plt.ylabel('Precipitação (mm)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'Graphs/{site}_{test_name}_{var}_trend_plot.png')
        plt.close()


    # Mapeamento dos testes de Mann-Kendall disponíveis
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
            file_path = f'Results/{site}_yearly.csv' if var == 'Year' else f'Results/max_daily_{site}.csv'
        elif data_type == 'mod':
            file_path = f'GCM_data/bias_correction/{site}_yearly.csv' if var == 'Year' else f'GCM_data/bias_correction/max_daily_{site}.csv'
        else:
            raise ValueError("Invalid data type. Use 'obs' or 'mod'.")
        
        # Lê o arquivo CSV e remove valores nulos
        df = pd.read_csv(file_path).dropna(subset=['Precipitation'])
        print(f'--- {group} / {site} ---\n')

        try:
            # Aplica todos os testes de Mann-Kendall
            for test_name, test_func in test_functions.items():
                result = test_func(df[['Precipitation']], alpha=alpha_value)
                store_results(site, test_name, {
                    'Tau': result[4], 'p': result[2], 'trend': result[0], 'h': result[1],
                    'z': result[3], 's': result[5], 'var_s': result[6],
                    'slope': result[7], 'intercept': result[8]
                })
                
                # Gera gráficos se plot_graphs=True
                if plot_graphs and result[7] is not None and result[8] is not None:
                    plot_trend_graph(df, site, test_name, result[7], result[8], var)
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
    
    
    
def aggregate_precipitation(df, interval, dt_min=False):
    """
    Agrega os dados de precipitação em intervalos especificados.

    Parâmetros:
    df (DataFrame): Um DataFrame contendo uma coluna 'Precipitation' 
                    com os dados de precipitação, indexados por tempo.
    interval (int): O intervalo de agregação desejado:
                    - Se `dt_min` for None, considera 'interval' em horas.
                    - Se `dt_min` for True, considera 'interval' em minutos.
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

    if not dt_min:
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




def get_subdaily_extremes(df, interval, dt_min=False, return_max_only=True):
    """
    Calcula os valores máximos e mínimos de precipitação acumulada em intervalos 
    especificados para cada ano presente em um DataFrame. Se return_max_only for True, 
    retorna apenas os máximos.

    Parâmetros:
    df (DataFrame): Um DataFrame que deve conter, pelo menos, uma coluna 'Year' 
                    e dados de precipitação em uma coluna separada.
    interval (int): O intervalo de agregação desejado:
                    - Se 'dt_min' for False, considera 'interval' em horas (para máximos).
                    - Se 'dt_min' for True, considera 'interval' em minutos (para máximos e mínimos).
    dt_min (int, opcional): A resolução temporal dos dados em minutos. 
                            Necessário se 'interval' for em minutos.
    return_max_only (bool, opcional): Se True, retorna apenas os máximos. O padrão é True.

    Retorna:
    DataFrame: Um DataFrame contendo os anos e, dependendo do parâmetro, 
               os máximos e mínimos ou apenas os máximos de precipitação acumulada.
    """
    
    # Obtém a lista de anos únicos do DataFrame
    years_list = df['Year'].unique()
    
    # Inicializa listas para armazenar os máximos e mínimos subdiários
    max_subdaily_list = []
    min_subdaily_list = []

    # Itera sobre cada ano para calcular os extremos de precipitação acumulada
    for year in years_list:
        # Filtra os dados para o ano atual
        df_new = df[df['Year'] == year]
        
        # Agrega a precipitação em intervalos subdiários
        if not dt_min:
            subdaily_list = aggregate_precipitation(df_new, interval)
        else:
            subdaily_list = aggregate_precipitation(df_new, interval, dt_min)

        # Adiciona o máximo e mínimo encontrados às respectivas listas
        max_subdaily_list.append(max(subdaily_list))
        min_subdaily_list.append(min(subdaily_list))

    # Cria um DataFrame resultante com os anos
    if return_max_only:
        df_result = pd.DataFrame({
            'Year': years_list,
            f'Max_{interval}{"h" if dt_min is None else "min"}': max_subdaily_list  # Apenas máximos
        })
    else:
        df_result = pd.DataFrame({
            'Year': years_list,
            f'Max_{interval}{"h" if dt_min is None else "min"}': max_subdaily_list,  # Máximos
            f'Min_{interval}{"h" if dt_min is None else "min"}': min_subdaily_list   # Mínimos
        })

    return df_result



def get_max_subdaily_table(name_file, directory='Results', dt_min=False):
    """
    Calcula os máximos de precipitação acumulada em intervalos subdiários 
    e salva os resultados em um arquivo CSV. O cálculo pode ser realizado 
    para dados horários ou de minutos, dependendo da presença do parâmetro dt_min.

    Parâmetros:
    name_file (str): Nome do arquivo sem extensão que contém dados de precipitação.
    directory (str): Diretório onde os arquivos estão localizados e onde o resultado será salvo.
    dt_min (int, opcional): A resolução temporal dos dados em minutos. Necessário se os dados forem em minutos.

    Retorna:
    None: Salva um arquivo CSV contendo os máximos acumulados por intervalo.
    """
    print('Getting maximum subdaily...')
    
    # Lê o arquivo CSV contendo dados
    if not dt_min:
        df = pd.read_csv(f'{directory}/{name_file}_hourly.csv')
        # Lista dos intervalos em horas
        intervals = [1, 3, 6, 8, 10, 12, 24]
    else:
        df = pd.read_csv(f'{directory}/{name_file}_min.csv')
        # Lista dos intervalos em minutos
        intervals = [5, 10, 15, 20, 25, 30]

    # Cria um DataFrame inicial para armazenar os resultados
    df_final = pd.DataFrame({'Year': df['Year'].unique()})

    # Calcula e mescla os máximos para cada intervalo
    for interval in intervals:
        if not dt_min:
            max_subdaily = get_subdaily_extremes(df, interval)
            print(f'{interval}h done!')
        else:
            max_subdaily = get_subdaily_extremes(df, interval, dt_min)
            print(f'{interval}min done!')
        
        # Mescla os resultados no DataFrame final
        df_final = df_final.merge(max_subdaily, on='Year', how='inner')

    # Exibe o DataFrame final
    print('\n', df_final, '\n')

    # Salva o DataFrame final em um arquivo CSV
    if not dt_min:
        df_final.to_csv(f'{directory}/max_subdaily_{name_file}.csv', index=False)
    else:
        df_final.to_csv(f'{directory}/max_subdaily_min_{name_file}.csv', index=False)

    print('Done!')
    
    return df_final
    
    

def merge_max_tables(name_file, directory='Results'):
    """
    Mescla tabelas de máximos de precipitação acumulada em intervalos de minutos e horas 
    e salva os resultados em um arquivo CSV.

    Parâmetros:
    name_file (str): Nome do arquivo sem extensão que contém os dados.
    directory (str): Diretório onde os arquivos estão localizados e onde o resultado será salvo.
    
    Retorna:
    None: Salva um arquivo CSV contendo os máximos acumulados por intervalo em minutos e horas.
    """
    # Lê os arquivos CSV que contêm os máximos acumulados
    df_min = pd.read_csv(f'{directory}/max_subdaily_min_{name_file}.csv')
    df_hour = pd.read_csv(f'{directory}/max_subdaily_{name_file}.csv')
    
    # Mescla os DataFrames com base na coluna 'Year'
    df_complete = df_min.merge(df_hour, on='Year', how='inner')
    
    # Salva o DataFrame resultante em um novo arquivo CSV
    df_complete.to_csv(f'{directory}/max_subdaily_complete_{name_file}.csv', index=False)
    
    print('Merge completo! Arquivo salvo em:', f'{directory}/max_subdaily_complete_{name_file}.csv')
    
    
    

def remove_outliers_from_max(df, duration=0):
    """
    Remove outliers da coluna 'Precipitation' de um DataFrame, sem agrupar os dados.

    Args:
        df (pd.DataFrame): DataFrame com coluna 'Precipitation'.
        duration (int, optional): Duração usada para renomear a coluna (padrão: 0, sem renomeação).

    Returns:
        pd.DataFrame: DataFrame filtrado sem outliers na coluna 'Precipitation'.
    """
    # Removendo valores nulos
    df_no_na = df.dropna()

    # Calcula os limites para remoção de outliers usando o IQR
    q1 = df_no_na['Precipitation'].quantile(0.25)
    q3 = df_no_na['Precipitation'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filtra os dados para remover os outliers
    df_filtered = df_no_na[(df_no_na['Precipitation'] > lower_bound) & 
                           (df_no_na['Precipitation'] < upper_bound)]
    
    # Caso uma duração seja passada, renomeia a coluna
    if duration != 0:
        df_filtered.columns = ['Max_{dur}'.format(dur=duration)]
    
    return df_filtered




def max_annual_precipitation(df, name_file, output_dir='Results'):
    """
    Calcula o valor máximo de precipitação anual para cada ano e remove os outliers.
    Em seguida, salva o resultado em um arquivo CSV no diretório especificado.

    Parâmetros:
    - df (DataFrame): DataFrame com colunas 'Year' e 'Precipitation'.
    - name_file (str): Nome base do arquivo de saída.
    - output_dir (str): Diretório onde o arquivo CSV será salvo (padrão: 'Results').

    Retorna:
    - DataFrame com os valores máximos de precipitação anual, excluindo outliers.
    """
    # Remover linhas com valores nulos
    df = df.dropna()
    
    # Agrupar por ano e calcular o valor máximo de precipitação anual
    df_new = df.groupby(['Year'])['Precipitation'].max().reset_index()
    
    # Remover outliers usando a função auxiliar
    df_new = remove_outliers_from_max(df_new)
    
    # Garantir que o diretório de saída exista
    os.makedirs(output_dir, exist_ok=True)
    
    # Caminho completo do arquivo
    output_path = os.path.join(output_dir, f'max_daily_{name_file}.csv')
    
    # Salvar o resultado em um arquivo CSV
    df_new.to_csv(output_path, index=False)
    
    print(f"Arquivo salvo em: {output_path}")
    return df_new


 

def get_disagregation_factors(var_value, filename='parameters/fatores_desagregacao.csv'):
    """
    Lê os fatores de desagregação de um arquivo CSV e calcula fatores 
    baseados em um valor de variável fornecido.

    Parâmetros:
    var_value (float): Valor utilizado para calcular os fatores de desagregação.
    filename (str): Nome do arquivo CSV contendo os fatores de desagregação (padrão é 'fatores_desagregacao.csv').
    
    Retorna:
    DataFrame: Um DataFrame contendo os fatores de desagregação calculados.
    """
    # Lê o arquivo CSV contendo os fatores de desagregação
    df_disagreg_factors = pd.read_csv(filename)
    
    # Calcula os fatores de desagregação
    df_disagreg_factors['CETESB_p{v}'.format(v=var_value)] = df_disagreg_factors['CETESB_ger'] * (1 + var_value)
    df_disagreg_factors['CETESB_m{v}'.format(v=var_value)] = df_disagreg_factors['CETESB_ger'] * (1 - var_value)
    
    return df_disagreg_factors



def get_subdaily_from_disagregation_factors(df, type_of_disagregator, var_value, name_file, directory='Results'):
    """
    Calcula os valores subdiários de precipitação baseados em fatores de desagregação.

    Parâmetros:
    df (DataFrame): DataFrame contendo os dados de precipitação.
    type_of_disagregator (str): Tipo de desagregador ('original', 'plus' ou 'minus').
    var_value (float): Valor utilizado para os fatores de desagregação.
    name_file (str): Nome do arquivo sem extensão onde os resultados serão salvos.
    directory (str): Diretório onde o arquivo será salvo (padrão é 'Results').
    
    Retorna:
    None: Salva um arquivo CSV contendo os valores subdiários calculados.
    """
    
    df_subdaily = df
    df_disagreg_factors = get_disagregation_factors(var_value)
    
    # Define o tipo de desagregação
    if type_of_disagregator == 'original':
        type = 'ger'
    elif type_of_disagregator == 'plus':
        type = f'p{var_value}'
    elif type_of_disagregator == 'minus':
        type = f'm{var_value}'
    
    # Lista de intervalos que serão utilizados
    intervals = [5, 10, 15, 20, 25, 30, 60, 360, 480, 600, 720, 1440]
    
    # Verifica se a coluna com o tipo existe em df_disagreg_factors
    col_name = f'CETESB_{type}'
    if col_name not in df_disagreg_factors.columns:
        raise ValueError(f"Coluna {col_name} não encontrada em df_disagreg_factors.")
    
    # Aplica os fatores de desagregação aos intervalos correspondentes
    for i, interval in enumerate(intervals):
        if i < len(df_disagreg_factors):
            factor = df_disagreg_factors[col_name].iloc[i]
            column_name = f'Max_{interval}min' if interval < 60 else f'Max_{interval//60}h'
            df_subdaily[column_name] = df_subdaily['Precipitation'] * factor
        else:
            print(f"Intervalo {interval} não encontrado em fatores de desagregação.")
    
    # Salva o resultado no CSV
    output_path = f'{directory}/max_subdaily_{name_file}_{type}.csv'
    df_subdaily.to_csv(output_path, index=False)
    print(f'Resultado salvo em {output_path}')



def plot_subdaily_maximum_absolute(name_file, variation_percentage=20):
    """
    Gera gráficos de barras para os máximos subdiários de precipitação (1h, 6h, 8h, 10h, 12h, 24h)
    comparando dados observados com referências CETESB (e variações de ±variation_percentage%).

    Parâmetros:
    name_file (str): Nome base do arquivo (sem extensão), usado para carregar os arquivos de dados e salvar os gráficos.
    variation_percentage (int): Percentagem para variação positiva e negativa em relação aos dados CETESB. Padrão é 20.

    Etapas:
    1. Carrega os arquivos de dados e concatena-os em um DataFrame único.
    2. Gera gráficos de barras para cada intervalo de tempo, comparando os tipos de dados.
    3. Salva cada gráfico em uma pasta específica para fácil acesso.

    Retorno:
    Nenhum. Gráficos são salvos em arquivos na pasta 'Graphs/subdaily'.
    """
    print('Starting plot of absolute subdaily maximums...\n')

    # Carrega os arquivos de dados e adiciona uma coluna para identificar o tipo de dados
    file_paths = {
        'Observed': f'Results/tests/max_subdaily_{name_file}.csv',
        'CETESB': f'Results/tests/max_subdaily_{name_file}_ger.csv',
        f'CETESB_-{variation_percentage}%': f'Results/tests/max_subdaily_{name_file}_m{variation_percentage/100:.1f}.csv',
        f'CETESB_+{variation_percentage}%': f'Results/tests/max_subdaily_{name_file}_p{variation_percentage/100:.1f}.csv'
    }
    
    # Concatena os dados em um único DataFrame com o tipo de dados como coluna
    data_frames = []
    for data_type, path in file_paths.items():
        df = pd.read_csv(path)
        df['Type'] = data_type
        data_frames.append(df)
    
    # Combina todos os DataFrames em um único DataFrame final
    df_final = pd.concat(data_frames, ignore_index=True, sort=False)
    df_final = df_final[['Year', 'Max_1h', 'Max_6h', 'Max_8h', 'Max_10h', 'Max_12h', 'Max_24h', 'Type']]

    # Lista dos intervalos de tempo para gerar gráficos
    intervals = ['Max_1h', 'Max_6h', 'Max_8h', 'Max_10h', 'Max_12h', 'Max_24h']
    y_limit = 170  # Limite do eixo Y para todos os gráficos

    # Função auxiliar para gerar e salvar o gráfico de cada intervalo
    def plot_interval(interval):
        print(f'Generating plot for {interval}...')
        
        # Cria o gráfico de barras para o intervalo especificado
        g = sns.catplot(
            x="Year", y=interval, hue='Type', data=df_final, 
            kind='bar', height=5, aspect=1.5
        )
        g.set_axis_labels('', 'Precipitation (mm)')
        fig = g.figure
        fig.subplots_adjust(bottom=0.15, top=0.9, left=0.07)
        plt.xticks(rotation=50)
        plt.title(f'Subdaily {name_file} - {interval.split("_")[1]}')
        plt.ylim(0, y_limit)
        
        # Salva o gráfico em um arquivo
        plt.savefig(f'Graphs/subdaily/{name_file}_{interval.lower()}.png')
        print(f'Graph for {interval} saved!\n')

    # Gera e salva os gráficos para cada intervalo
    for interval in intervals:
        plot_interval(interval)

    print('All subdaily maximum plots generated and saved!')

    


def plot_subdaily_maximum_relative(name_file, max_hour, var_value):
    """
    Gera um gráfico das diferenças relativas entre os valores máximos subdiários observados e as referências CETESB.
    
    Parâmetros:
    name_file (str): Nome base do arquivo (sem extensão), usado para carregar os dados e salvar os gráficos.
    max_hour (int): Intervalo de tempo em horas para o qual as diferenças serão calculadas.
    var_value (float): Valor de variação aplicado aos fatores CETESB (por exemplo, 0.2 para 20%).
    
    Etapas:
    1. Carrega os dados de precipitação máxima observada e as referências CETESB (original, +20%, -20%).
    2. Calcula a diferença entre os dados observados e cada referência para o intervalo especificado.
    3. Gera um gráfico de barras que mostra essas diferenças ao longo dos anos, destacando o desvio de cada referência.
    4. Salva o gráfico em um arquivo PNG para análise posterior.
    
    Retorno:
    Nenhum. Gráfico é salvo em 'Graphs/subdaily_variations' com nome baseado nos parâmetros.
    """
    
    print('Starting plotting relative subdaily maximums\n')
    
    # Carrega os arquivos de dados e adiciona a coluna de identificação do tipo de dados
    file_paths = {
        'Observed': f'Results/tests/max_subdaily_{name_file}.csv',
        'CETESB': f'Results/tests/max_subdaily_{name_file}_ger.csv',
        f'CETESB_-{var_value}': f'Results/tests/max_subdaily_{name_file}_m{var_value}.csv',
        f'CETESB_+{var_value}': f'Results/tests/max_subdaily_{name_file}_p{var_value}.csv'
    }
    
    # Concatena os dados em um único DataFrame com o tipo de dados como coluna
    data_frames = []
    for data_type, path in file_paths.items():
        df = pd.read_csv(path)
        df['Type'] = data_type
        data_frames.append(df)
    
    # Combina todos os DataFrames em um único DataFrame final
    df_final = pd.concat(data_frames, ignore_index=True, sort=False)
    df_final = df_final[['Year', f'Max_{max_hour}h', 'Type']]
    
    # Separação dos dados observados e referências para cálculo de diferenças
    df_observed = df_final[df_final['Type'] == 'Observed'].reset_index()
    reference_dfs = {
        'CETESB': df_final[df_final['Type'] == 'CETESB'].reset_index(),
        f'CETESB_-{var_value}': df_final[df_final['Type'] == f'CETESB_-{var_value}'].reset_index(),
        f'CETESB_+{var_value}': df_final[df_final['Type'] == f'CETESB_+{var_value}'].reset_index()
    }
    
    # Calcula as diferenças entre os valores observados e cada referência
    for ref_type, ref_df in reference_dfs.items():
        ref_df[f'Dif_{max_hour}h'] = df_observed[f'Max_{max_hour}h'] - ref_df[f'Max_{max_hour}h']
        reference_dfs[ref_type] = ref_df  # Atualiza com a nova coluna de diferença
    
    # Soma dos erros de cada referência (somatório das diferenças)
    for ref_type, ref_df in reference_dfs.items():
        sum_error = ref_df[f'Dif_{max_hour}h'].sum()
        print(f'sum_error for {ref_type} / {var_value}: ', sum_error)
        print('')
    
    # Concatena as diferenças em um DataFrame único para geração do gráfico
    df_graph = pd.concat(reference_dfs.values(), ignore_index=True, sort=False)
    
    # Gera o gráfico de diferenças relativas
    g = sns.catplot(
        x="Year", y=f'Dif_{max_hour}h', hue='Type', data=df_graph,
        kind='bar', height=5, aspect=1.5
    )
    g.set_axis_labels('', 'Precipitation (mm)')
    fig = g.figure
    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.07)
    plt.xticks(rotation=50)
    plt.ylim(-100, 50)
    plt.title(f'Subdaily {name_file} - {max_hour}h - {var_value}')
    
    # Salva o gráfico em um arquivo
    output_path = f'Graphs/subdaily_variations/{name_file}_max{max_hour}_{var_value}_relative.png'
    plt.savefig(output_path)
    print(f'Graph for relative Max_{max_hour}h done and saved at {output_path}!\n')



def plot_subdaily_maximum_BL(max_hour):
    """
    Plota comparações de máximas subdiárias de precipitação
    observadas e ajustadas utilizando o método Bartlett-Lewis (BL) e fatores CETESB.

    Args:
        max_hour (int): O número de horas para as quais a máxima subdiária é calculada.
    """

    print('Iniciando o plot das máximas subdiárias relativas BL e observadas\n')

    # Lendo dados de precipitação máxima subdiária
    data_sources = {
        'INMET_aut observed': 'Results/max_subdaily_INMET_aut.csv',
        'INMET_aut CETESB': 'Results/max_subdaily_INMET_aut_ger.csv',
        'MAPLU_usp observed': 'Results/max_subdaily_MAPLU_usp.csv',
        'MAPLU_usp CETESB': 'Results/max_subdaily_MAPLU_usp_ger.csv',
        'INMET_aut BL': 'bartlet_lewis/max_subdaily_INMET_bl.csv',
        'MAPLU_usp BL': 'bartlet_lewis/max_subdaily_MAPLU_usp_bl.csv'
    }

    # Criar um DataFrame vazio para armazenar os dados lidos
    df_list = []
    
    for source, file_path in data_sources.items():
        df = pd.read_csv(file_path)
        df['Type'] = source  # Adiciona coluna para identificar a origem dos dados
        df_list.append(df)

    # Concatenar todos os DataFrames em um único DataFrame
    df_final = pd.concat(df_list, ignore_index=True, sort=False)

    # Selecionar colunas relevantes
    df_final = df_final[['Year', 'Max_1', 'Max_6', 'Max_8', 'Max_10', 'Max_12', 'Max_24', 'Type']]

    # Gráfico de máximas absolutas
    g = sns.catplot(x="Year", y=f"Max_{max_hour}", hue='Type', data=df_final, kind='bar', height=5, aspect=1.5)
    g.set_axis_labels('', 'Precipitação')
    plt.title(f'Máximas Subdiárias INMET e MAPLU - Teste BL - {max_hour}h')
    plt.ylim(0, 170)
    plt.xticks(rotation=50)
    plt.savefig(f'Graphs/subdaily_bl/BL_max{max_hour}_absolute.png')
    print(f'Gráfico das máximas absolutas Max_{max_hour}h gerado com sucesso!\n')

    # Processando os DataFrames para calcular diferenças
    df_processed = {source: df_final[df_final['Type'] == source].reset_index(drop=True) for source in data_sources.keys()}

    # Calculando as diferenças entre máximas observadas e ajustadas
    for source in ['INMET_aut observed', 'MAPLU_usp observed']:
        for method in ['CETESB', 'BL']:
            key = f"{source.split()[0]} {method}"
            df_processed[key]['Dif_{max_hour}'] = df_processed[source]['Max_{max_hour}'] - df_processed[key]['Max_{max_hour}']

    # Somando os erros
    error_sums = {source: df_processed[source]['Dif_{max_hour}'].sum() for source in df_processed if 'CETESB' in source or 'BL' in source}

    # Imprimindo os erros acumulados
    for key, value in error_sums.items():
        print(f'Soma do erro para {key}: {value}\n')

    # Gráfico de diferenças
    df_graph = pd.concat([df_processed['INMET_aut CETESB'], df_processed['INMET_aut BL'],
                          df_processed['MAPLU_usp CETESB'], df_processed['MAPLU_usp BL']], ignore_index=True, sort=False)

    h = sns.catplot(x="Year", y=f"Dif_{max_hour}", hue='Type', data=df_graph, kind='bar', height=5, aspect=1.5)
    h.set_axis_labels('', 'Precipitação')
    plt.title(f'Diferenças Subdiárias INMET e MAPLU - Teste BL - {max_hour}h')
    plt.ylim(-100, 50)
    plt.xticks(rotation=50)
    plt.savefig(f'Graphs/subdaily_bl/BL_max{max_hour}_relative.png')
    print(f'Gráfico das diferenças Max_{max_hour}h gerado com sucesso!\n')
    


def plot_optimized_subdaily(name_file, max_hour):
    """
    Gera um gráfico comparativo das diferenças entre as máximas de precipitação
    observadas e as estimativas de duas fontes (CETESB e dados otimizados) para um intervalo de tempo específico.
    
    Args:
        name_file (str): Nome do arquivo (sem extensão) que será utilizado para ler os dados de precipitação.
        max_hour (int): Hora máxima a ser considerada nas diferenças de precipitação.
    """
    
    print('Iniciando a plotagem das máximas subdiárias relativas...')
    print('')

    # Leitura dos dados observados, CETESB e otimizados
    df_observed = pd.read_csv(f'Resultsmax_subdaily_{name_file}.csv')
    df_observed['Type'] = 'Observado'

    df_cetesb = pd.read_csv(f'Results/max_subdaily_{name_file}_ger.csv')
    df_cetesb['Type'] = 'CETESB'

    df_optimized = pd.read_csv(f'Results/max_subdaily_{name_file}_otimizado.csv')
    df_optimized['Type'] = 'CETESB_otimizado'
    
    # Combinação dos DataFrames em um único DataFrame
    df_final = pd.concat([df_observed, df_cetesb, df_optimized], ignore_index=True, sort=False)
    df_final = df_final[['Year', 'Max_1h', 'Max_6h', 'Max_8h', 'Max_10h', 'Max_12h', 'Max_24h', 'Type']]
    
    # Processamento dos dados para calcular as diferenças
    df_diff_cetesb = df_final[df_final['Type'] == 'CETESB'].reset_index()
    df_diff_optimized = df_final[df_final['Type'] == 'CETESB_otimizado'].reset_index()
    df_obs = df_final[df_final['Type'] == 'Observado'].reset_index()

    # Cálculo das diferenças
    df_diff_cetesb[f'Dif_{max_hour}h'] = df_obs[f'Max_{max_hour}h'] - df_diff_cetesb[f'Max_{max_hour}h']
    df_diff_optimized[f'Dif_{max_hour}h'] = df_obs[f'Max_{max_hour}h'] - df_diff_optimized[f'Max_{max_hour}h']

    # Soma dos erros
    sum_error_cetesb = df_diff_cetesb[f'Dif_{max_hour}h'].sum()
    sum_error_optimized = df_diff_optimized[f'Dif_{max_hour}h'].sum()
    
    print(f'Soma do erro CETESB: {sum_error_cetesb}\n')
    print(f'Soma do erro otimizado: {sum_error_optimized}\n')

    # Preparação dos dados para o gráfico
    df_graph = pd.concat([df_diff_cetesb, df_diff_optimized], ignore_index=True, sort=False)

    # Criação do gráfico de barras para as diferenças
    g = sns.catplot(x="Year", y=f'Dif_{max_hour}h', hue='Type', data=df_graph, kind='bar', height=5, aspect=1.5)
    g.set_axis_labels('', 'Precipitação')
    fig = g.figure
    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.07)    
    plt.xticks(rotation=50)
    plt.ylim(-100, 50)
    plt.title(f'Subdiário {name_file} - {max_hour}h - otimizados')

    # Salvando o gráfico
    plt.savefig(f'Graphs/subdaily_variations/{name_file}_max{max_hour}_opt_relative.png')
    print(f'Gráfico das diferenças máximas de {max_hour}h concluído!\n')
    


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
        ax.set_xlabel("Média Pacum (mm)")  # Define o rótulo do eixo X
        ax.set_ylabel(f"Pacum {name} (mm)")  # Define o rótulo do eixo Y
        ax.set_title(f"Dispersão de {name}")  # Define o título do subplot

    plt.tight_layout()  # Ajusta o layout para evitar sobreposição dos subplots
    plt.show()