import pandas as pd

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


