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


def get_idf_for_fit(name_file, MY_DISTRIBUTIONS, dist, return_period_list, directory = 'Results', disag_factor = 'nan'):
    if disag_factor == 'nan':   
        duration_list_min = [60, 180, 360, 480, 600, 720, 1440]
        duration_list = [1, 3, 6, 8, 10, 12, 24]
    
    else:
        duration_list_min = [5, 10, 20, 30, 60, 360, 480, 600, 720, 1440]
        duration_list = ['5min', '10min', '20min', '30min', '1', '6', '8', '10', '12', '24']

    P_RP_ = []
    
    for duration in duration_list:
        P_ = get_precipitation_byRP(name_file, duration, MY_DISTRIBUTIONS, dist, return_period_list, directory, disag_factor)
        P_RP_.append(P_)
    
    i_RP_ = [P*60/d for P, d in zip(P_RP_, duration_list_min)] 
    ln_i_RP_ = [np.log(i) for i in i_RP_]
    
    return duration_list_min, ln_i_RP_


def get_idf_params1(t0, name_file, MY_DISTRIBUTIONS, dist, return_period_list, directory = 'Results', disag_factor = 'nan'):
    duration, ln_i_RP_ = get_idf_for_fit(name_file, MY_DISTRIBUTIONS, dist, return_period_list, directory, disag_factor)
    
    ln_cte_list = [np.log(t0+d) for d in duration] #cte = ln(d * t0) (for IDF linearization)
    
    x = np.array(ln_cte_list).reshape((-1, 1))
    y = np.array(ln_i_RP_)
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    ln_cte2 = model.intercept_
    n = model.coef_[0] 
    return r_sq, ln_cte2, n

def min_sum_r_sq(t0):
    r_sq_2 = get_idf_params1(t0, name_file, MY_DISTRIBUTIONS, dist, [2.0], directory, disag_factor)[0]
    r_sq_5 = get_idf_params1(t0, name_file, MY_DISTRIBUTIONS, dist, [5.0], directory, disag_factor)[0]
    r_sq_10 = get_idf_params1(t0, name_file, MY_DISTRIBUTIONS, dist, [10.0], directory, disag_factor)[0]
    r_sq_25 = get_idf_params1(t0, name_file, MY_DISTRIBUTIONS, dist, [25.0], directory, disag_factor)[0]
    r_sq_50 = get_idf_params1(t0, name_file, MY_DISTRIBUTIONS, dist, [50.0], directory, disag_factor)[0]
    r_sq_100 = get_idf_params1(t0, name_file, MY_DISTRIBUTIONS, dist, [100.0], directory, disag_factor)[0]
    
    sum = r_sq_2 + r_sq_5 + r_sq_10 + r_sq_25 + r_sq_50 + r_sq_100
    return -sum

def get_t0(name_file, MY_DISTRIBUTIONS, dist, directory, disag_factor):
    res = minimize_scalar(min_sum_r_sq)
    return res.x

def get_idf_params2(t0, name_file, MY_DISTRIBUTIONS, dist, directory = 'Results', disag_factor = 'nan'):
    ln_cte2_list = []
    ln_cte2_2 = get_idf_params1(t0, name_file, MY_DISTRIBUTIONS, dist, [2.0], directory, disag_factor)[1]
    ln_cte2_list.append(ln_cte2_2)
    ln_cte2_5 = get_idf_params1(t0, name_file, MY_DISTRIBUTIONS, dist, [5.0], directory, disag_factor)[1]
    ln_cte2_list.append(ln_cte2_5)
    ln_cte2_10 = get_idf_params1(t0, name_file, MY_DISTRIBUTIONS, dist, [10.0], directory, disag_factor)[1]
    ln_cte2_list.append(ln_cte2_10)
    ln_cte2_25 = get_idf_params1(t0, name_file, MY_DISTRIBUTIONS, dist, [25.0], directory, disag_factor)[1]
    ln_cte2_list.append(ln_cte2_25)
    ln_cte2_50 = get_idf_params1(t0, name_file, MY_DISTRIBUTIONS, dist, [50.0], directory, disag_factor)[1]
    ln_cte2_list.append(ln_cte2_50)
    ln_cte2_100 = get_idf_params1(t0, name_file, MY_DISTRIBUTIONS, dist, [100.0], directory, disag_factor)[1]
    ln_cte2_list.append(ln_cte2_100)
    #print(ln_cte2_list)
    return_period_list = [2, 5, 10, 25, 50, 100]

    ln_RP_list = [np.log(RP) for RP in return_period_list] 
    #print(ln_RP_list)
    
    x = np.array(ln_RP_list).reshape((-1, 1))
    y = np.array(ln_cte2_list)
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    ln_K = model.intercept_
    K = np.exp(ln_K)
    m = model.coef_[0] 
    
    return r_sq, K, m    

def get_final_idf_params(name_file, MY_DISTRIBUTIONS, dist, directory = 'Results', disag_factor = 'nan', save_file = False):
    #t0 = 11.827
    t0 = get_t0(name_file, MY_DISTRIBUTIONS, dist, directory, disag_factor)
    n = get_idf_params1(t0, name_file, MY_DISTRIBUTIONS, dist, [2], directory, disag_factor)[2]
    n = abs(n)
    #print(t0, n)
    K = get_idf_params2(t0, name_file, MY_DISTRIBUTIONS, dist, directory, disag_factor)[1]
    m = get_idf_params2(t0, name_file, MY_DISTRIBUTIONS, dist, directory, disag_factor)[2]
    #print(K, m)
    
    if save_file == True:
        dict_ = {'t0' : t0,
                'n' : n, 
                'K' : K, 
                'm' : m
                }
        df = pd.DataFrame(dict_)
        df.to_csv('{d}/IDF_params_{n}.csv'.format(n = name_file, d = directory), index = False)        
    
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
    
    # Imprime os resultados
    print(f"Precipitação para 2 anos: {p_2y}")
    print(f"Precipitação para 5 anos: {p_5y}")
    print(f"Precipitação para 10 anos: {p_10y}")
    print(f"Precipitação para 25 anos: {p_25y}")
    print(f"Precipitação para 50 anos: {p_50y}")
    print(f"Precipitação para 100 anos: {p_100y}")
