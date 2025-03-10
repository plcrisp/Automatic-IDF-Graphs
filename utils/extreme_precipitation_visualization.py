import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    