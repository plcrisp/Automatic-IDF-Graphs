from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pymannkendall as mk
from utils.error_correction import verification, fill_missing_data
from utils.data_processing import read_csv

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