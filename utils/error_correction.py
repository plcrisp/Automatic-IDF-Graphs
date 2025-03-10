import pandas as pd
from datetime import date
from utils.data_processing import *

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
