import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
import random
from cmath import nan



def dano_es_formula(data):
    """
    Calcula o número ideal de bins para um histograma usando a fórmula de Doane.

    A fórmula de Doane é uma modificação da regra de Sturges, ajustando o cálculo para
    levar em conta a assimetria dos dados (skewness). É especialmente útil para conjuntos
    de dados que não seguem uma distribuição normal.

    Referência:
    https://en.wikipedia.org/wiki/Histogram#Doane's_formula

    Parâmetros:
        data (list ou array-like): Conjunto de dados numéricos.

    Retorna:
        int: Número ideal de bins para o histograma.
    """
    # Número de observações
    N = len(data)
    
    if N <= 1:
        raise ValueError("O conjunto de dados deve conter pelo menos 2 elementos.")
    
    # Calcula o coeficiente de assimetria (skewness)
    skewness = st.skew(data)
    
    # Calcula o desvio padrão do coeficiente de assimetria
    sigma_g1 = math.sqrt((6 * (N - 2)) / ((N + 1) * (N + 3)))
    
    # Aplica a fórmula de Doane
    num_bins = 1 + math.log2(N) + math.log2(1 + abs(skewness) / sigma_g1)
    
    # Retorna o número de bins arredondado para o inteiro mais próximo
    return round(num_bins)



def plot_histogram(data, results, n):
    """
    Plota um histograma dos dados fornecidos e sobrepõe as distribuições ajustadas.

    Esta função exibe um histograma representando os dados fornecidos e traça as primeiras
    'n' distribuições ajustadas a partir do ranking fornecido em 'results'.

    Parâmetros:
        data (array-like): Conjunto de dados numéricos para o histograma.
        results (dict): Dicionário contendo distribuições ajustadas. 
            A estrutura esperada é {distribuição: (SSE, arg, loc, scale)}.
        n (int): Número de distribuições do ranking a serem sobrepostas no gráfico.

    Retorna:
        None: A função exibe o gráfico, mas não retorna valores.
    """
    if n <= 0:
        raise ValueError("O número de distribuições (n) deve ser maior que zero.")
    if len(results) < n:
        raise ValueError(f"O número de distribuições disponíveis ({len(results)}) é menor que n ({n}).")
    
    # Seleciona as primeiras 'n' distribuições do ranking
    selected_distributions = {k: results[k] for k in list(results)[:n]}
    
    # Configurações do histograma
    plt.figure(figsize=(10, 5))
    plt.hist(data, density=True, bins='auto', ec='white', 
             color=(63/235, 149/235, 170/235), alpha=0.75)
    plt.title('Histograma e Distribuições Ajustadas')
    plt.xlabel('Valores')
    plt.ylabel('Frequências')

    # Plota as distribuições ajustadas
    for distribution, result in selected_distributions.items():
        sse, arg, loc, scale = result  # Descompacta os parâmetros da distribuição
        x_plot = np.linspace(min(data), max(data), 1000)  # Intervalo de valores para o eixo x
        y_plot = distribution.pdf(x_plot, loc=loc, scale=scale, *arg)  # Calcula a PDF
        
        # Plota a distribuição com uma cor aleatória
        plt.plot(
            x_plot, y_plot, 
            label=f"{str(distribution.name).capitalize()}: SSE = {sse:.4f}",
            color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        )

    # Configuração da legenda
    plt.legend(title='Distribuições', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


    
def goodness_of_fit(data, results, n, mean, plot=True):
    """
    Avalia o ajuste de distribuições aos dados e, opcionalmente, plota as funções de distribuição cumulativa (CDF).

    Parâmetros:
        data (array-like): Conjunto de dados numéricos.
        results (dict): Dicionário contendo distribuições ajustadas. 
            Estrutura esperada: {distribuição: (SSE, arg, loc, scale)}.
        n (int): Número de distribuições do ranking a serem avaliadas.
        mean (float): Média teórica ou esperada para testes de ajuste.
        plot (bool): Se True, plota as CDFs das distribuições. Default é True.

    Retorna:
        None: A função realiza o gráfico opcional e não retorna valores.
    """
    # Validações de entrada
    if n <= 0:
        raise ValueError("O número de distribuições (n) deve ser maior que zero.")
    if len(results) < n:
        raise ValueError(f"O número de distribuições disponíveis ({len(results)}) é menor que n ({n}).")
    if not isinstance(plot, bool):
        raise TypeError("O parâmetro 'plot' deve ser um valor booleano (True ou False).")
    
    # Seleciona as primeiras 'n' distribuições do ranking
    selected_distributions = {k: results[k] for k in list(results)[:n]}
    
    # Plota as CDFs das distribuições se plot=True
    if plot:
        plt.figure(figsize=(10, 5))
        for distribution, result in selected_distributions.items():
            # Descompacta os parâmetros da distribuição
            sse, arg, loc, scale = result
            
            # Gera valores aleatórios a partir da distribuição ajustada
            simulated_data = distribution.rvs(loc=loc, scale=scale, *arg, size=(50, 2))
            
            # Cria os valores para o eixo x e calcula a CDF
            x_plot = np.linspace(min(data), max(data), 1000)
            y_plot = distribution.cdf(x_plot, loc=loc, scale=scale, *arg)
            
            # Plota a CDF da distribuição com uma cor aleatória
            plt.plot(
                x_plot, y_plot, 
                label=f"{str(distribution.name).capitalize()}: SSE = {sse:.4f}",
                color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
            )
        
        # Configurações do gráfico
        plt.title("Funções de Distribuição Cumulativa (CDF)")
        plt.xlabel("Valores")
        plt.ylabel("Probabilidade acumulada")
        plt.legend(title="Distribuições", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

 

def fit_data(data, MY_DISTRIBUTIONS):
    """
    Ajusta distribuições teóricas aos dados fornecidos, calcula o erro de ajuste (SSE) e retorna os resultados ordenados pelo erro.

    Parâmetros:
        data (array-like): Conjunto de dados numéricos a serem ajustados.
        MY_DISTRIBUTIONS (list): Lista de distribuições do Scipy para ajuste aos dados. As distribuições devem ser passadas como objetos de distribuição do Scipy (e.g., st.norm, st.expon).

    Retorna:
        dict: Dicionário contendo as distribuições ajustadas com os erros (SSE), parâmetros de ajuste (loc, scale, arg).
    """
    # Desabilitado devido à versão atual do SciPy
    # st.frechet_r, st.frechet_l: desabilitado na versão atual do SciPy
    # st.levy_stable: leva muito tempo para estimar os parâmetros

    # Validação do parâmetro 'data'
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("O parâmetro 'data' deve ser uma lista ou um array NumPy.")
    if len(data) < 2:
        raise ValueError("O conjunto de dados 'data' deve ter pelo menos dois elementos.")
    
    # Validação das distribuições fornecidas
    if not isinstance(MY_DISTRIBUTIONS, list):
        raise TypeError("O parâmetro 'MY_DISTRIBUTIONS' deve ser uma lista de distribuições.")
    if not all(isinstance(dist, st.rv_continuous) for dist in MY_DISTRIBUTIONS):
        raise TypeError("Cada item de 'MY_DISTRIBUTIONS' deve ser uma distribuição contínua do Scipy.")
    
    # Cálculo do histograma com o número de bins estimado pela fórmula de Danoe
    num_bins = dano_es_formula(data)
    frequencies, bin_edges = np.histogram(data, num_bins, density=True)
    central_values = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]

    # Inicialização do dicionário de resultados
    results = {}

    # Ajuste de distribuições e cálculo do erro (SSE)
    for distribution in MY_DISTRIBUTIONS:
        # Ajusta os parâmetros da distribuição aos dados
        params = distribution.fit(data)
        
        # Separa os parâmetros
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        
        # Calcula a função de densidade de probabilidade (PDF) ajustada
        pdf_values = [distribution.pdf(c, loc=loc, scale=scale, *arg) for c in central_values]
        
        # Calcula o erro (SSE) entre as frequências do histograma e a PDF ajustada
        sse = np.sum(np.power(frequencies - pdf_values, 2.0))
        
        # Armazena os resultados
        results[distribution] = [sse, arg, loc, scale]
    
    # Ordena os resultados pelo erro (SSE) de forma crescente
    sorted_results = {k: results[k] for k in sorted(results, key=lambda x: results[x][0])}

    return sorted_results




def get_parameters(data, results, n):
    """
    Extrai os parâmetros das 'n' distribuições mais ajustadas aos dados, com base no erro SSE, e retorna um DataFrame com as informações.

    Parâmetros:
        data (array-like): Conjunto de dados numéricos.
        results (dict): Dicionário de resultados, onde as chaves são distribuições e os valores são listas contendo SSE, parâmetros da distribuição (arg, loc, scale).
        n (int): Número de distribuições mais ajustadas a serem retornadas.

    Retorna:
        pd.DataFrame: Um DataFrame contendo as distribuições, seus erros (SSE), e os parâmetros (c, loc, scale).
    """
    
    # Verifica se 'n' é um número positivo
    if not isinstance(n, int) or n <= 0:
        raise ValueError("O parâmetro 'n' deve ser um número inteiro positivo.")
    
    # Filtra as n distribuições mais ajustadas com base no erro (SSE)
    N_DISTRIBUTIONS = {k: results[k] for k in list(results)[:n]}

    # Listas para armazenar os parâmetros e resultados
    Distributions_list = []
    SSE_list = []
    c_list = []
    loc_list = []
    scale_list = []

    # Mapeamento de distribuições para nomes legíveis
    dist_names = {
        st.gumbel_r: 'Gumbel',
        st.lognorm: 'Lognormal',
        st.genextreme: 'GEV',
        st.norm: 'Normal',
        st.genlogistic: 'Generalized Logistic'
    }

    # Itera sobre as distribuições mais ajustadas
    for distribution, result in N_DISTRIBUTIONS.items():
        # Atribui o nome legível para a distribuição
        dist = dist_names.get(distribution, distribution.name)

        # Extrai os resultados de SSE e parâmetros
        sse = result[0]
        arg = result[1]
        
        # Se o argumento não for vazio, pega o primeiro valor de 'arg', caso contrário, define como 'nan'
        c = arg[0] if len(arg) > 0 else float('nan')
        
        loc = result[2]
        scale = result[3]

        # Armazena os resultados nas listas correspondentes
        Distributions_list.append(dist)
        SSE_list.append(sse)
        c_list.append(c)
        loc_list.append(loc)
        scale_list.append(scale)

    # Cria um dicionário para formar o DataFrame
    result_dict = {
        'distribution': Distributions_list,
        'sse': SSE_list,
        'c': c_list,
        'loc': loc_list,
        'scale': scale_list
    }

    # Cria o DataFrame com os resultados
    df_result = pd.DataFrame(result_dict)

    return df_result



def main_daily(name_file='inmet_conv'):
    """
    Função principal para carregar dados de precipitação, ajustar distribuições, 
    realizar testes de ajuste e salvar os resultados.

    O processo inclui:
    1. Carregar o arquivo CSV com os dados de precipitação.
    2. Ajustar distribuições selecionadas aos dados.
    3. Gerar histogramas e realizar o teste de bondade de ajuste.
    4. Obter parâmetros das distribuições ajustadas e salvar os resultados.
    
    Parâmetro:
    - name_file (str): Nome do arquivo base (sem a extensão). Exemplo: 'inmet_conv'.
    """
    
    # Tentativa de leitura do arquivo de dados
    try:
        data_df_original = pd.read_csv(f'Results/max_daily_{name_file}.csv')
    except FileNotFoundError:
        print(f"Erro: O arquivo 'Results/max_daily_{name_file}.csv' não foi encontrado.")
        return
    
    # Verifica se a coluna 'Precipitation' existe no DataFrame
    if 'Precipitation' not in data_df_original.columns:
        print("Erro: A coluna 'Precipitation' não foi encontrada no arquivo.")
        return
    
    # Filtra a coluna de precipitação e calcula a média
    data_df = data_df_original[['Precipitation']]
    mean = data_df.iloc[:, 0].mean()
    
    data = data_df.values.ravel()  # Converte os dados de precipitação para um array numpy
    
    MY_DISTRIBUTIONS = [st.norm, st.lognorm, st.genextreme, st.gumbel_r, st.genlogistic]  # Definição das distribuições a serem ajustadas
    
    results = fit_data(data, MY_DISTRIBUTIONS)  # Ajuste de distribuições aos dados
    
    plot_histogram(data, results, 5)  # Plota o histograma e as distribuições ajustadas
    
    goodness_of_fit(data, results, 5, mean)  # Realiza o teste de bondade de ajuste
    
    df_parameters = get_parameters(data, results, 5)  # Obtém os parâmetros das distribuições ajustadas e cria um DataFrame
    
    df_parameters.to_csv(f'Results/{name_file}_dist_params.csv', index=False)  # Salva os parâmetros em um arquivo CSV
    
    print(f"Processamento concluído. Parâmetros das distribuições salvos em 'Results/{name_file}_dist_params.csv'.")



def main_subdaily(name_file, disag_factor, duration, directory='Results'):
    """
    Função principal para carregar dados subdiários de precipitação, ajustar distribuições, 
    realizar testes de bondade de ajuste e salvar os resultados.

    O processo inclui:
    1. Carregar o arquivo CSV com os dados subdiários de precipitação.
    2. Ajustar distribuições selecionadas aos dados.
    3. Gerar histogramas e realizar o teste de bondade de ajuste.
    4. Obter parâmetros das distribuições ajustadas e exibir os resultados.

    Parâmetros:
    - name_file: Nome base do arquivo de dados (sem extensão).
    - disag_factor: Fator de desacordo para ajustar o nome do arquivo de entrada.
    - duration: A duração do evento de precipitação que será analisada.
    - directory: Diretório onde o arquivo de dados está localizado (padrão: 'Results').
    """
    
    # Formatação do nome do arquivo de dados
    file_path = f'{directory}/max_subdaily_{name_file}{disag_factor}.csv'
    
    # Tentativa de leitura do arquivo de dados
    try:
        data_df_original = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
        return
    
    # Verifica se a coluna de dados esperada existe no DataFrame
    column_name = f'Max_{duration}'
    if column_name not in data_df_original.columns:
        print(f"Erro: A coluna '{column_name}' não foi encontrada no arquivo.")
        return
    
    # Filtra a coluna de precipitação e calcula a média
    data_df = data_df_original[[column_name]]
    mean = data_df.iloc[:, 0].mean()
    
    
    data = data_df.values.ravel() # Converte os dados para um array numpy

    MY_DISTRIBUTIONS = [st.norm, st.lognorm, st.genextreme, st.gumbel_r, st.genlogistic] # Definição das distribuições a serem ajustadas
    
    results = fit_data(data, MY_DISTRIBUTIONS) # Ajuste de distribuições aos dados
    
    
    plot_histogram(data, results, 5) # Plota o histograma e as distribuições ajustadas
    
    
    goodness_of_fit(data, results, 5, mean) # Realiza o teste de bondade de ajuste
    
    
    df_parameters = get_parameters(data, results, 5) # Obtém os parâmetros das distribuições ajustadas e cria um DataFrame
    
    
    print(df_parameters) # Exibe os parâmetros ajustados
    
    
    df_parameters.to_csv(f'{directory}/{name_file}_dist_params.csv', index=False) # Salva os parâmetros em um arquivo CSV
     
    print(f"Processamento concluído. Parâmetros das distribuições salvos em '{directory}/{name_file}_dist_params.csv'.")


if __name__ == "__main__":
#     name_file = 'complete_INMET_conv_bl'
#     directory = 'bartlet_lewis'
#     disag_factor = ''
#     duration = '20min'

    # EQM
    name_file = 'complete_HADGEM_baseline_MD'
    directory = 'GCM_data//bias_correction//gcm'
    disag_factor = ''
    duration = '24'  
        
    main_daily()
    main_subdaily('inmet','_p0.2','8h')
    
    
