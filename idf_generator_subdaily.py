import pandas as pd
import math
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from utils.error_correction import remove_outliers_from_max
from utils.get_distribution import fit_data, get_top_fitted_distributions
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression



def calculate_theoretical_max_precipitations(
    file_name: str,
    duration: int,
    return_periods: list,
    results_dir: str = 'results',
    disag_factor: str = 'nan',
    plot: bool = False,
    return_distribution_name: bool = False
):
    """
    Calcula as precipitações máximas teóricas para diferentes períodos de retorno.

    Parâmetros:
        file_name (str): Nome base do arquivo CSV contendo os dados de precipitação máxima subdiária.
        duration (int): Duração específica do evento de precipitação (ex.: 1 hora, 6 horas).
        return_periods (list): Lista de períodos de retorno (em anos) para cálculo das precipitações máximas.
        results_dir (str, opcional): Diretório onde os arquivos CSV estão armazenados. Padrão: 'results'.
        disag_factor (str, opcional): Fator de desagregação usado para ajustar o nome do arquivo ou processamento. 
                                               Padrão: 'nan'.
        plot (bool, opcional): Indica se um gráfico deve ser gerado. Padrão: False.
        return_distribution_name (bool, opcional): Indica se o nome da distribuição utilizada deve ser retornado. 
                                                   Padrão: False.

    Retorna:
        np.ndarray: Array com as precipitações máximas teóricas calculadas para os períodos de retorno.
        str (opcional): Nome da distribuição utilizada, se `return_distribution_name=True`.
    """
    # Ajusta o fator de desagregação
    disag = (
        '' if disag_factor == 'nan'
        else '_ger' if disag_factor == 'original'
        else '_bl' if disag_factor == 'bl'
        else f'_{disag_factor}'
    )
    if disag_factor == 'bl':
        file_name = f'complete_{file_name}'

    # Lê os dados do arquivo CSV
    file_path = f'{results_dir}/max_subdaily_{file_name}{disag}.csv'
    data = pd.read_csv(file_path)

    # Ordena os dados e calcula frequência empírica e período de retorno
    max_col = f'Max_{duration}'
    sorted_data = (
        data[[max_col]]
        .sort_values(max_col, ascending=False)
        .reset_index()
    )
    sorted_data['RP'] = (sorted_data.index + 1) / (len(sorted_data) + 1)
    sorted_data['RP'] = 1 / sorted_data['RP']

    # Remove outliers e extrai valores de precipitação
    filtered = remove_outliers_from_max(data[[max_col]], max_col, duration)
    values = filtered.values.ravel()

    # Ajusta os dados à melhor distribuição
    fit_results = fit_data(values)
    best_params = get_top_fitted_distributions(values, fit_results, n=1).iloc[0]

    # Seleciona e instancia a distribuição ajustada
    dist = getattr(st, best_params['distribution'].lower())
    prob_func = (
        dist(best_params['loc'], best_params['scale']) if math.isnan(best_params['c'])
        else dist(best_params['c'], best_params['loc'], best_params['scale'])
    )

    # Calcula precipitações teóricas para os períodos de retorno
    probs = [1 - 1 / period for period in return_periods]
    theoretical = prob_func.ppf(probs)

    # Gera o gráfico, se necessário
    if plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(return_periods, theoretical, label='Teórico', color='blue')
        ax.plot(sorted_data['RP'], sorted_data[max_col], label='Observado', linestyle='--', color='orange')
        ax.set(ylabel=f'Precipitação em {duration} (mm)', xlabel='Período de Retorno (Anos)', title=f'Distribuição: {best_params["distribution"]}')
        ax.legend()
        plot_path = f'graphs/distributions/quantile_plot_{file_name}_{disag}_{best_params["distribution"]}_subdaily_{duration}.png'
        fig.savefig(plot_path)
        plt.show()
        plt.close(fig)

    # Retorna os resultados
    if return_distribution_name:
        return theoretical, best_params['distribution']
    return theoretical



def calculate_idf_table(
    file_name: str,
    directory: str = 'results',
    disag_factor: str = 'nan',
    save_table: bool = False
):
    """
    Gera uma tabela IDF (Intensity-Duration-Frequency) com base nos dados de precipitação.

    Parâmetros:
        name_file (str): Nome do arquivo base contendo os dados de precipitação.
        directory (str, opcional): Diretório onde os arquivos CSV serão salvos. Padrão: 'Results'.
        disag_factor (str, opcional): Fator de desagregação usado para ajustar os dados. Padrão: 'nan'.
        save_table (bool, opcional): Indica se a tabela IDF deve ser salva em um arquivo CSV. Padrão: False.

    Retorna:
        tuple: Uma tupla contendo:
            - duration_list_min (list): Lista de durações dos eventos de precipitação (em minutos).
            - ln_i_RP_2Years, ln_i_RP_5Years, ln_i_RP_10Years, ln_i_RP_25Years, ln_i_RP_50Years, ln_i_RP_100Years:
              Listas com os logaritmos naturais das intensidades de precipitação para diferentes períodos de retorno.
    """
    
    # Define as durações com base no fator de desagregação
    if disag_factor == 'nan':
        duration_list_min = [60, 180, 360, 480, 600, 720, 1440]
        duration_list = [1, 3, 6, 8, 10, 12, 24]
    else:
        duration_list_min = [5, 10, 20, 30, 60, 360, 480, 600, 720, 1440]
        duration_list = ['5min', '10min', '20min', '30min', '1h', '6h', '8h', '10h', '12h', '24h']

    # Define os períodos de retorno
    return_periods = [2, 5, 10, 25, 50, 100]
    P_RP_dict = {f"P_RP_{rp}Years": [] for rp in return_periods}

    # Calcula as precipitações para cada duração
    for duration in duration_list:
        precipitations, dist_name = calculate_theoretical_max_precipitations(
            file_name=file_name,
            duration=duration,
            return_periods=return_periods,
            results_dir=directory,
            disag_factor=disag_factor,
            plot=False,
            return_distribution_name=True
        )
        for idx, rp in enumerate(return_periods):
            P_RP_dict[f"P_RP_{rp}Years"].append(precipitations[idx])

    # Calcula as intensidades e seus logaritmos naturais
    i_RP_dict = {}
    ln_i_RP_dict = {}
    for rp in return_periods:
        i_RP_dict[f"i_RP_{rp}Years"] = [P * 60 / d for P, d in zip(P_RP_dict[f"P_RP_{rp}Years"], duration_list_min)]
        ln_i_RP_dict[f"ln_i_RP_{rp}Years"] = [np.log(i) for i in i_RP_dict[f"i_RP_{rp}Years"]]

    # Salva a tabela em um arquivo CSV, se necessário
    if save_table:
        data = {
            "duration": duration_list_min,
            **P_RP_dict,
            **i_RP_dict
        }
        df = pd.DataFrame(data)
        
        if disag_factor == 'bl':
            file_name = f"complete_{file_name}"
        
        file_path = f"{directory}/IDFsubdaily_table_{file_name}_{dist_name}_{disag_factor}.csv"
        df.to_csv(file_path, index=False)

    # Retorna os resultados necessários
    return (
        duration_list_min,
        *[ln_i_RP_dict[f"ln_i_RP_{rp}Years"] for rp in return_periods]
    )



def calculate_duration_based_parameters(t0, name_file, return_periods, directory='results', disag_factor='nan'):
    """
    Calcula os parâmetros da curva IDF linearizada, modelando a relação entre duração e 
    intensidade de precipitação para múltiplos períodos de retorno.

    Parâmetros:
        t0 (float): Constante usada na linearização da curva IDF.
        name_file (str): Nome do arquivo base contendo os dados de precipitação.
        return_periods (list): Lista de períodos de retorno (em anos) para cálculo das precipitações máximas.
        directory (str, opcional): Diretório onde os arquivos CSV estão armazenados. Padrão: 'Results'.
        disag_factor (str, opcional): Fator de desagregação usado para ajustar os dados. Padrão: 'nan'.

    Retorna:
        dict: Um dicionário contendo os resultados para cada período de retorno. As chaves são os 
              períodos de retorno (em anos), e os valores são tuplas com os seguintes elementos:
              
              - r_sq (float): Coeficiente de determinação (R²) do ajuste linear. Mede a qualidade do 
                              ajuste entre ln(d + t0) e ln(i), onde:
                                - d: Duração do evento de precipitação (minutos ou horas).
                                - i: Intensidade de precipitação (mm/h).
                              Valores próximos de 1 indicam um bom ajuste.
              
              - ln_cte2 (float): Logaritmo natural do intercepto do ajuste linear. Está relacionado à 
                                 magnitude das intensidades de precipitação para o período de retorno 
                                 considerado. Pode ser convertido para cte2 usando: cte2 = exp(ln_cte2).
              
              - n (float): Coeficiente angular do ajuste linear. Descreve como a intensidade de 
                           precipitação diminui com o aumento da duração. Quanto maior o valor de n, 
                           mais rapidamente a intensidade diminui à medida que a duração aumenta.
                           Tipicamente, n está na faixa de 0.5 a 1.0.
    """
    # Chama calculate_idf_table para obter os dados
    duration_list_min, *ln_i_RP_lists = calculate_idf_table(
        file_name=name_file,
        directory=directory,
        disag_factor=disag_factor,
        save_table=False
    )

    # Dicionário para armazenar os resultados
    results = {}

    # Itera sobre os períodos de retorno
    for rp in return_periods:
        rp_index = [2, 5, 10, 25, 50, 100].index(rp)  # Índice do período de retorno
        ln_i_RP_ = ln_i_RP_lists[rp_index]

        # Calcula ln(d + t0) para cada duração
        ln_cte_list = [np.log(t0 + d) for d in duration_list_min]  # cte = ln(d + t0)

        # Realiza o ajuste linear
        x = np.array(ln_cte_list).reshape((-1, 1))
        y = np.array(ln_i_RP_)
        model = LinearRegression().fit(x, y)
        r_sq = model.score(x, y)
        ln_cte2 = model.intercept_
        n = model.coef_[0]

        # Armazena os resultados no dicionário
        results[rp] = (r_sq, ln_cte2, n)

    return results



def find_optimal_t0(name_file, return_periods, directory='results', disag_factor='nan'):
    """
    Encontra o valor ótimo de t0 minimizando a soma negativa dos R² para múltiplos períodos de retorno.

    Parâmetros:
        name_file (str): Nome do arquivo base contendo os dados de precipitação.
        return_periods (list): Lista de períodos de retorno (em anos) para cálculo das precipitações máximas.
        directory (str, opcional): Diretório onde os arquivos CSV estão armazenados. Padrão: 'Results'.
        disag_factor (str, opcional): Fator de desagregação usado para ajustar os dados. Padrão: 'nan'.

    Retorna:
        float: Valor ótimo de t0 encontrado pela otimização.
    """
    # Função interna para calcular a soma negativa dos R²
    def min_sum_r_sq(t0):
        """
        Calcula a soma negativa dos coeficientes de determinação (R²) para múltiplos períodos de retorno.

        Parâmetros:
            t0 (float): Constante usada na linearização da curva IDF.

        Retorna:
            float: Soma negativa dos coeficientes de determinação (R²) para os períodos de retorno.
        """
        # Calcula os parâmetros IDF para todos os períodos de retorno
        idf_results = calculate_duration_based_parameters(t0, name_file, return_periods, directory, disag_factor)

        # Calcula a soma dos R²
        total_r_sq = sum(result[0] for result in idf_results.values())

        # Retorna a soma negativa (para otimização minimizante)
        return -total_r_sq

    # Executa a otimização para encontrar o valor ótimo de t0
    result = minimize_scalar(min_sum_r_sq, bounds=(0.1, 10.0), method='bounded')  # Limites razoáveis para t0

    # Retorna o valor ótimo de t0
    return result.x



def calculate_return_period_based_parameters(t0, name_file, directory='Results', disag_factor='nan'):
    """
    Calcula os parâmetros da curva IDF linearizada com base nos períodos de retorno, modelando a 
    relação entre o período de retorno e a intensidade de precipitação.

    Parâmetros:
        t0 (float): Constante usada na linearização da curva IDF.
        name_file (str): Nome do arquivo base contendo os dados de precipitação.
        directory (str, opcional): Diretório onde os arquivos CSV estão armazenados. Padrão: 'Results'.
        disag_factor (str, opcional): Fator de desagregação usado para ajustar os dados. Padrão: 'nan'.

    Retorna:
        tuple: Uma tupla contendo:
        - r_sq (float): Coeficiente de determinação (R²) do ajuste linear. Mede a qualidade do ajuste entre ln(RP) e ln(cte2), onde:
            - RP: Período de retorno (anos).
            - cte2: Intercepto relacionado à magnitude das intensidades de precipitação.
            
        - K (float): Coeficiente relacionado à magnitude das intensidades de precipitação. 
                        É obtido exponenciando o intercepto do ajuste linear (K = exp(ln_K)).
                        
        - m (float): Expoente que descreve como a intensidade de precipitação aumenta com o período de retorno. 
                        Quanto maior o valor de m, mais sensível a intensidade é ao aumento do período de retorno.
    """
    # Lista fixa de períodos de retorno
    return_periods = [2, 5, 10, 25, 50, 100]

    # Calcula os parâmetros baseados na duração para todos os períodos de retorno
    duration_based_results = calculate_duration_based_parameters(
        t0=t0,
        name_file=name_file,
        return_periods=return_periods,
        directory=directory,
        disag_factor=disag_factor
    )

    # Extrai ln(cte2) para cada período de retorno
    ln_cte2_list = [duration_based_results[rp][1] for rp in return_periods]

    # Calcula ln(RP) para cada período de retorno
    ln_RP_list = [np.log(rp) for rp in return_periods]

    # Ajuste linear entre ln(RP) e ln(cte2)
    x = np.array(ln_RP_list).reshape((-1, 1))
    y = np.array(ln_cte2_list)
    model = LinearRegression().fit(x, y)

    # Extrai os resultados do ajuste
    r_sq = model.score(x, y)
    ln_K = model.intercept_
    K = np.exp(ln_K)
    m = model.coef_[0]

    return r_sq, K, m

   

def get_final_idf_params(name_file, directory='Results', disag_factor='nan', save_file=False):
    """
    Calcula os parâmetros finais da curva IDF e, opcionalmente, salva os resultados em um arquivo CSV.

    Parâmetros:
        name_file (str): Nome do arquivo base contendo os dados de precipitação.
        MY_DISTRIBUTIONS (dict): Dicionário contendo as distribuições ajustadas para os dados.
        dist (str): Distribuição específica a ser usada no cálculo das precipitações.
        directory (str, opcional): Diretório onde os arquivos CSV estão armazenados. Padrão: 'Results'.
        disag_factor (str, opcional): Fator de desagregação usado para ajustar os dados. Padrão: 'nan'.
        save_file (bool, opcional): Indica se os parâmetros devem ser salvos em um arquivo CSV. Padrão: False.

    Retorna:
        tuple: Uma tupla contendo:
            - t0 (float): Constante usada na linearização da curva IDF.
            - n (float): Expoente que descreve como a intensidade varia com a duração.
            - K (float): Coeficiente relacionado à magnitude das intensidades de precipitação.
            - m (float): Expoente que descreve como a intensidade varia com o período de retorno.
    """
    # Calcula o valor ótimo de t0
    t0 = find_optimal_t0(name_file, [2, 5, 10, 25, 50, 100], directory, disag_factor)

    # Calcula o expoente n (relacionado à duração) usando o período de retorno de 2 anos
    duration_based_results = calculate_duration_based_parameters(t0, name_file, [2], directory, disag_factor)
    n = abs(duration_based_results[2][2])  # Acessa o valor de n para o período de retorno de 2 anos

    # Calcula K e m (relacionados ao período de retorno)
    _, K, m = calculate_return_period_based_parameters(t0, name_file, directory, disag_factor)

    # Salva os parâmetros em um arquivo CSV, se necessário
    if save_file:
        data = {
            't0': [t0],
            'n': [n],
            'K': [K],
            'm': [m]
        }
        df = pd.DataFrame(data)
        df.to_csv(f'{directory}/IDF_params_{name_file}.csv', index=False)

    return t0, n, K, m
    


if __name__ == '__main__':
    
    p_2y, p_5y, p_10y, p_25y, p_50y, p_100y = calculate_theoretical_max_precipitations(
        file_name='inmet',
        duration='1h',
        return_periods=[2, 5, 10, 25, 50, 100],
        results_dir='results',
        disag_factor='p0.2',
        plot=True
    )
    
    calculate_idf_table(
        file_name='inmet',
        disag_factor='p0.2',
        save_table=True
    )
    
    results = calculate_duration_based_parameters(
        t0=11.827,
        name_file='inmet',
        return_periods=[2],
        directory='results',
        disag_factor='p0.2'
    )
    
    r_sqq, K, m = calculate_return_period_based_parameters(
        t0=11.827,
        name_file='inmet',
        directory='results',
        disag_factor='p0.2'
    )
    
    t0, n, K, m = get_final_idf_params(
        name_file='inmet',
        directory='results',
        disag_factor='p0.2',
        save_file=True
    )
    
   
    
    print(results)
    
    print('----------')
    
    print(f"R²: {r_sqq}")
    print(f"K: {K}")
    print(f"m: {m}")
    
    
    # Imprime os resultados
    print(f"Precipitação para 2 anos: {p_2y}")
    print(f"Precipitação para 5 anos: {p_5y}")
    print(f"Precipitação para 10 anos: {p_10y}")
    print(f"Precipitação para 25 anos: {p_25y}")
    print(f"Precipitação para 50 anos: {p_50y}")
    print(f"Precipitação para 100 anos: {p_100y}")
