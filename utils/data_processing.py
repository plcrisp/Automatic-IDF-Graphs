import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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



# Função para observar a distribuição dos dados
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