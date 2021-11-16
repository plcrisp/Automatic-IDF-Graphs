import numpy as np
import pandas as pd
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import scipy
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import pymannkendall as mk
import statsmodels.api as sm

def max_annual_daily_precipitation(df, name_file, directory = 'Results'):
    df = df.dropna()
    df_new = df.groupby(['Year'])['Precipitation'].max().reset_index()    

    df_new.to_csv('{d}/max_daily_{n}.csv'.format(d = directory, n = name_file), index = False)

    return df_new


def get_disagregation_factors(var_value):
    df_disagreg_factors = pd.read_csv('fatores_desagregacao.csv')
    df_disagreg_factors['CETESB_p{v}'.format(v = var_value)] = df_disagreg_factors['CETESB_ger']*(1+var_value)
    df_disagreg_factors['CETESB_m{v}'.format(v = var_value)] = df_disagreg_factors['CETESB_ger']*(1-var_value)
    return df_disagreg_factors

def get_subdaily_from_disagregation_factors(df, type_of_disagregator, var_value, name_file, directory = 'Results'):
    df_subdaily = df
    
    df_disagreg_factors = get_disagregation_factors(var_value)
    
    if type_of_disagregator == 'original':
        type = 'ger'
    if type_of_disagregator == 'plus':
        type = 'p{v}'.format(v = var_value)
    if type_of_disagregator == 'minus':
        type = 'm{v}'.format(v = var_value)
    
    df_subdaily['Max_5min'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_{t}'.format(t = type)][0]
    df_subdaily['Max_10min'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_{t}'.format(t = type)][1]
    df_subdaily['Max_15min'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_{t}'.format(t = type)][2]
    df_subdaily['Max_20min'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_{t}'.format(t = type)][3]
    df_subdaily['Max_25min'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_{t}'.format(t = type)][4]
    df_subdaily['Max_30min'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_{t}'.format(t = type)][5]
    df_subdaily['Max_1'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_{t}'.format(t = type)][6]
    df_subdaily['Max_6'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_{t}'.format(t = type)][7]
    df_subdaily['Max_8'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_{t}'.format(t = type)][8]
    df_subdaily['Max_10'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_{t}'.format(t = type)][9]
    df_subdaily['Max_12'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_{t}'.format(t = type)][10]
    df_subdaily['Max_24'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_{t}'.format(t = type)][11]

    df_subdaily.to_csv('{d}/max_subdaily_{n}_{t}.csv'.format(d = directory, n = name_file, t = type), index = False)
    
if __name__ == '__main__':
    print('Starting..')
    
    df_1 = pd.read_csv('Results/INMET_conv_daily.csv')
    df_2 = max_annual_daily_precipitation(df_1, 'INMET_conv', 'Results')
    get_subdaily_from_disagregation_factors(df_2, 'original', 0.1, 'INMET_conv', 'Results_2')
    get_subdaily_from_disagregation_factors(df_2, 'plus', 0.1, 'INMET_conv', 'Results_2')
    get_subdaily_from_disagregation_factors(df_2, 'minus', 0.1, 'INMET_conv', 'Results_2')
    
    