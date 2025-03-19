"""
Este código implementa um conjunto de funções para gerar e analisar curvas IDF (Intensity-Duration-Frequency), 
que são amplamente utilizadas em hidrologia para modelar a relação entre a intensidade, a duração e a frequência 
de eventos de precipitação. O processo geral inclui os seguintes passos:

1. **Cálculo das Precipitações Máximas Teóricas**:
   - A função `calculate_theoretical_max_precipitations` ajusta distribuições estatísticas aos dados históricos 
     de precipitação máxima subdiária para diferentes períodos de retorno. Ela calcula as precipitações máximas 
     teóricas esperadas para cada período de retorno, utilizando a melhor distribuição ajustada.

2. **Geração da Tabela IDF**:
   - A função `calculate_idf_table` gera uma tabela IDF completa, contendo as intensidades de precipitação para 
     diferentes combinações de duração e período de retorno. As intensidades são calculadas a partir das 
     precipitações máximas teóricas, ajustando-se para a duração específica do evento.

3. **Cálculo dos Parâmetros Baseados na Duração**:
   - A função `calculate_duration_based_parameters` realiza ajustes lineares para modelar a relação entre a 
     duração e a intensidade de precipitação para múltiplos períodos de retorno. Ela retorna parâmetros como o 
     coeficiente angular (n) e o intercepto (ln_cte2), que descrevem como a intensidade diminui com o aumento da 
     duração.

4. **Otimização do Parâmetro t0**:
   - A função `find_optimal_t0` encontra o valor ótimo da constante t0, que é usada para linearizar a relação 
     entre duração e intensidade. Isso é feito minimizando a soma negativa dos coeficientes de determinação (R²) 
     para múltiplos períodos de retorno.

5. **Cálculo dos Parâmetros Baseados no Período de Retorno**:
   - A função `calculate_return_period_based_parameters` modela a relação entre o período de retorno e a 
     intensidade de precipitação, ajustando uma reta entre ln(RP) e ln(cte2). Ela retorna parâmetros como K 
     (magnitude) e m (expoente), que descrevem como a intensidade aumenta com o período de retorno.

6. **Obtenção dos Parâmetros Finais da Curva IDF**:
   - A função `get_final_idf_params` coordena todas as etapas anteriores para calcular os parâmetros finais da 
     curva IDF: t0, n, K e m. Esses parâmetros podem ser salvos em um arquivo CSV para uso posterior. Além disso, 
     ela possibilita a visualização das curvas IDF por meio de gráficos, que podem ser salvos ou exibidos diretamente.

7. **Geração de Tabelas e Gráficos IDF**:
   - A função `generate_idf_tables` utiliza os parâmetros IDF pré-calculados para gerar tabelas de intensidade 
     e precipitação acumulada para diferentes durações e períodos de retorno.

O código utiliza métodos estatísticos avançados, como ajuste de distribuições e regressão linear, para garantir 
que as curvas IDF sejam precisas e representativas dos dados observados. Ele também oferece flexibilidade para 
trabalhar com diferentes fatores de desagregação e opções de salvamento de resultados.

A saída final inclui os parâmetros necessários para expressar a relação IDF matematicamente, conforme abaixo:
    i = (K * RP^m) / (d + t0)^n
Onde:
    - i: Intensidade de precipitação (mm/h).
    - RP: Período de retorno (anos).
    - d: Duração do evento de precipitação (minutos ou horas).
    - K, m, n, t0: Parâmetros calculados pelo modelo.
"""

import pandas as pd
import math
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import os

from error_correction import remove_outliers_from_max
from get_distribution import fit_data, get_top_fitted_distributions
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
    Calcula as precipitações máximas teóricas para diferentes períodos de retorno, ajustando os dados históricos 
    a uma distribuição estatística e utilizando a função inversa da probabilidade acumulada (PPF) para estimar 
    valores extremos.

    Parâmetros:
        file_name (str): Nome base do arquivo CSV contendo os dados de precipitação máxima subdiária. O arquivo deve 
                         estar localizado no diretório especificado por `results_dir` e conter uma coluna nomeada 
                         como `Max_{duration}` para cada duração específica.
        duration (int): Duração específica do evento de precipitação em horas ou minutos (ex.: 1 hora, 6 horas). Deve 
                        corresponder ao nome da coluna `Max_{duration}` no arquivo CSV.
        return_periods (list): Lista de períodos de retorno (em anos) para os quais as precipitações máximas teóricas 
                               serão calculadas. Exemplo: [2, 5, 10, 25, 50, 100].
        results_dir (str, opcional): Diretório onde os arquivos CSV de precipitação máxima estão armazenados. Padrão: 'results'.
        disag_factor (str, opcional): Fator de desagregação usado para ajustar o nome do arquivo ou processamento. 
                                      Valores possíveis incluem 'nan' (padrão), 'original', 'bl' ou outros fatores personalizados.
        plot (bool, opcional): Indica se um gráfico comparativo entre os valores observados e teóricos deve ser gerado. 
                               Padrão: False.
        return_distribution_name (bool, opcional): Indica se o nome da distribuição utilizada no ajuste deve ser retornado 
                                                   junto com os resultados. Padrão: False.

    Retorna:
        np.ndarray: Array com as precipitações máximas teóricas calculadas para os períodos de retorno fornecidos. Cada valor 
                    corresponde à precipitação esperada para o respectivo período de retorno.
        str (opcional): Nome da distribuição utilizada no ajuste (ex.: 'gumbel_r', 'genextreme'), retornado apenas se 
                        `return_distribution_name=True`.

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
        plot_path = f'../graphs/distributions/quantile_plot_{file_name}_{disag}_{best_params["distribution"]}_subdaily_{duration}.png'
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
        
        file_path = f"{directory}/IDF_table_{file_name}_{dist_name}_{disag_factor}.csv"
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

   

def get_final_idf_params(
    name_file,
    directory='Results',
    disag_factor='nan',
    save_file=False,
    plot=False,
    durations=None,
    return_periods=None,
    save_plot=False,
    plot_directory='../graphs'
):
    """
    Calcula os parâmetros finais da curva IDF e, opcionalmente, salva os resultados em um arquivo CSV 
    e/ou plota as curvas IDF.

    Parâmetros:
        name_file (str): Nome do arquivo base contendo os dados de precipitação.
        directory (str, opcional): Diretório onde os arquivos CSV estão armazenados. Padrão: 'Results'.
        disag_factor (str, opcional): Fator de desagregação usado para ajustar os dados. Padrão: 'nan'.
        save_file (bool, opcional): Indica se os parâmetros devem ser salvos em um arquivo CSV. Padrão: False.
        plot (bool, opcional): Indica se as curvas IDF devem ser plotadas. Padrão: False.
        durations (list, opcional): Lista de durações dos eventos de precipitação (em minutos). 
                                    Padrão: [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 720, 1440].
        return_periods (list, opcional): Lista de períodos de retorno (em anos). 
                                         Padrão: [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000].
        save_plot (bool, opcional): Indica se o gráfico deve ser salvo em um arquivo. Padrão: False.
        plot_directory (str, opcional): Diretório onde o gráfico será salvo, se `save_plot=True`. 
                                        Padrão: '../graphs'.

    Retorna:
        tuple: Uma tupla contendo:
            - t0 (float): Constante usada na linearização da curva IDF.
            - n (float): Expoente que descreve como a intensidade varia com a duração.
            - K (float): Coeficiente relacionado à magnitude das intensidades de precipitação.
            - m (float): Expoente que descreve como a intensidade varia com o período de retorno.
    """

    # Calcula o valor ótimo de t0
    t0 = find_optimal_t0(name_file, [2, 5, 10, 25, 50, 100], directory, disag_factor)

    # Calcula o expoente n (relacionado à duração) usando o período de retorno de 2 anos (convenção)
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
        df.to_csv(f'{directory}/IDF_params_{name_file}_{disag_factor}.csv', index=False)

    # Plota as curvas IDF, se necessário
    if plot:
        plot_idf_curves(
            t0, n, K, m,
            durations=durations,
            return_periods=return_periods,
            save_plot=save_plot,
            plot_directory=plot_directory,
            name_file=name_file,
        )

    return t0, n, K, m



def plot_idf_curves(
    t0, n, K, m,
    name_file,
    durations=None,
    return_periods=None,
    save_plot=False,
    plot_directory='../graphs'
):
    """
    Gera e exibe as curvas IDF (Intensity-Duration-Frequency) com base nos parâmetros fornecidos.

    Parâmetros:
        t0 (float): Constante usada na linearização da curva IDF.
        n (float): Expoente que descreve como a intensidade varia com a duração.
        K (float): Coeficiente relacionado à magnitude das intensidades de precipitação.
        m (float): Expoente que descreve como a intensidade varia com o período de retorno.
        name_file (str): Nome do arquivo base contendo os dados de precipitação.
        durations (list, opcional): Lista de durações dos eventos de precipitação (em minutos). 
                                    Padrão: [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 720, 1440].
        return_periods (list, opcional): Lista de períodos de retorno (em anos). 
                                         Padrão: [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000].
        save_plot (bool, opcional): Indica se o gráfico deve ser salvo em um arquivo. Padrão: False.
        plot_directory (str, opcional): Diretório onde o gráfico será salvo, se `save_plot=True`. 
                                        Padrão: '../graphs'.
    """
    # Define valores padrão para durations e return_periods, se não forem fornecidos
    if durations is None:
        durations = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 720, 1440]
    if return_periods is None:
        return_periods = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]

    # Gera as curvas IDF
    idf_curves = {}
    for rp in return_periods:
        intensities = [K * (rp ** m) / ((d + t0) ** n) for d in durations]
        idf_curves[rp] = intensities

    # Plota as curvas
    plt.figure(figsize=(10, 6))
    for rp, intensities in idf_curves.items():
        plt.plot(durations, intensities, label=f'RP = {rp} anos')
    plt.xlabel('Duração (minutos)')
    plt.ylabel('Intensidade (mm/h)')
    plt.title('Curvas IDF (Intensity-Duration-Frequency)')
    plt.legend()
    plt.grid(True)

    # Salva o gráfico, se necessário
    if save_plot:
        os.makedirs(plot_directory, exist_ok=True)  # Cria o diretório, se ele não existir
        plot_path = f'{plot_directory}/{name_file}_IDF_curves.png'
        plt.savefig(plot_path)

    # Exibe o gráfico
    plt.show()
    plt.close()
    
    
    
def generate_idf_tables(
    name_file: str,
    disag_factor: str,
    directory: str = '../results',
    durations=None,
    return_periods=None,    
    save_tables: bool = True
) -> tuple:
    """
    Gera tabelas de intensidade e precipitação para diferentes durações e períodos de retorno 
    com base nos parâmetros IDF pré-calculados.
    
    Parâmetros:
        name_file (str): Nome da estação meteorológica ou identificador da série de dados.
        disag_factor (str): Fator de desagregação usado para ajustar os dados.
        directory (str, opcional): Diretório onde os arquivos IDF estão armazenados. Padrão: 'results'.
        durations (list, opcional): Lista de durações dos eventos de precipitação (em minutos). 
                                    Padrão: [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 720, 1440].
        return_periods (list, opcional): Lista de períodos de retorno (em anos). 
                                         Padrão: [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000].
        save_tables (bool, opcional): Indica se as tabelas geradas devem ser salvas em arquivos CSV. 
                                      Padrão: True.
    
    Retorna:
        tuple: Uma tupla contendo:
            - df_intensity (pd.DataFrame): Tabela de intensidades de precipitação para diferentes 
                                           durações e períodos de retorno.
            - df_precipitation (pd.DataFrame): Tabela de precipitações acumuladas para diferentes 
                                               durações e períodos de retorno.
    """

    # Constrói o caminho do arquivo IDF com base no nome da estação e no diretório fornecido
    idf_file_path = f'{directory}/IDF_params_{name_file}_{disag_factor}.csv'

   # Lê os parâmetros IDF do arquivo CSV correspondente à estação
    try:
        df_IDF_params = pd.read_csv(idf_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo IDF não encontrado: {idf_file_path}")

    # Verifica se há dados correspondentes
    if df_IDF_params.empty:
        raise ValueError("Nenhum parâmetro IDF encontrado para os critérios fornecidos.")

    # Extrai os parâmetros IDF
    K = df_IDF_params['K'][0]
    t0 = df_IDF_params['t0'][0]
    m = df_IDF_params['m'][0]
    n = df_IDF_params['n'][0]
    
    # Define valores padrão para durations e return_periods, se não forem fornecidos
    if durations is None:
        durations = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 720, 1440]
    if return_periods is None:
        return_periods = [2, 5, 10, 25, 50, 100]


    # Calcula as intensidades e precipitações para cada período de retorno e duração
    i_final, P_final = [], []
    for RP in return_periods:
        i_list, P_list = [], []
        for d in durations:
            i_prec = K * (RP ** m) / ((d + t0) ** n)
            i_list.append(i_prec)
            P_list.append(i_prec * d / 60)  # Precipitação acumulada em mm
        i_final.append(i_list)
        P_final.append(P_list)

    # Cria os DataFrames para intensidade e precipitação
    columns = [f'i_RP_{rp}' for rp in return_periods]
    df_intensity = pd.DataFrame(i_final).transpose()
    df_intensity.columns = columns
    df_intensity['d'] = durations

    columns = [f'P_RP_{rp}' for rp in return_periods]
    df_precipitation = pd.DataFrame(P_final).transpose()
    df_precipitation.columns = columns
    df_precipitation['d'] = durations

    # Salva as tabelas em arquivos CSV, se necessário
    if save_tables:
        output_dir = f'{directory}'
        intensity_file = f'{output_dir}/{name_file}_{disag_factor}_intensityfromIDF_subdaily.csv'
        precipitation_file = f'{output_dir}/{name_file}_{disag_factor}_precipitationfromIDF_subdaily.csv'
        
        df_intensity.to_csv(intensity_file, index=False)
        df_precipitation.to_csv(precipitation_file, index=False)

    return df_intensity, df_precipitation
    


if __name__ == '__main__':
    
    t0, n, K, m = get_final_idf_params(
        name_file='inmet',
        directory='../results',
        disag_factor='p0.2',
        save_file=True,
        plot=True,
        durations=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        return_periods=[2, 5, 10, 25, 50, 100, 200, 500, 1000],
        save_plot=True,
    )
    
    df_intensity, df_precipitation = generate_idf_tables(
        name_file='inmet',
        disag_factor='p0.2', 
        save_tables=True
    )
