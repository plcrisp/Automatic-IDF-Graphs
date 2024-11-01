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

def process_CEMADEN():
    for i in range(0, 62):
        if i == 0:
            CEMADEN_df = pd.read_csv('CEMADEN/data ({n}).csv'.format(n = i), sep = ';')
            #print(CEMADEN_df)
            #input()
        else:
            df = pd.read_csv('CEMADEN/data ({n}).csv'.format(n = i), sep = ';')
            CEMADEN_df = pd.concat([CEMADEN_df, df], ignore_index=True, sort=False)
            #print(CEMADEN_df)
            #input()
    
    CEMADEN_df.columns = ['1', '2', '3', '4', '5', '6', '7', '8']
    CEMADEN_df = CEMADEN_df[['5', '6', '7']]
    CEMADEN_df.columns = ['Site', 'Date', 'Precipitation']
    CEMADEN_df['Precipitation'] = CEMADEN_df['Precipitation'].str.replace(',', '.')
    CEMADEN_df['Site'] = CEMADEN_df['Site'].str.replace('-22,031', 'Jd_Sao_Paulo')
    CEMADEN_df['Site'] = CEMADEN_df['Site'].str.replace('-21,997', 'Cidade_Jardim')
    CEMADEN_df['Site'] = CEMADEN_df['Site'].str.replace('-21,898', 'Agua_Vermelha')
    CEMADEN_df[['Year', 'Month', 'Day_hour']] = CEMADEN_df.Date.str.split("-", expand=True)
    CEMADEN_df[['Day', 'Hour_min']] = CEMADEN_df.Day_hour.str.split(" ", expand=True)
    CEMADEN_df[['Hour', 'Min', 'Seg']] = CEMADEN_df.Hour_min.str.split(":", expand=True)
    CEMADEN_df = CEMADEN_df[['Site', 'Year', 'Month', 'Day', 'Hour', 'Precipitation']]
    CEMADEN_df['Precipitation'] = pd.to_numeric(CEMADEN_df['Precipitation'])
    CEMADEN_df['Year'] = pd.to_numeric(CEMADEN_df['Year'], downcast='integer')
    CEMADEN_df['Month'] = pd.to_numeric(CEMADEN_df['Month'], downcast='integer')
    CEMADEN_df['Day'] = pd.to_numeric(CEMADEN_df['Day'], downcast='integer')
    CEMADEN_df['Hour'] = pd.to_numeric(CEMADEN_df['Hour'], downcast='integer')
    
    #print(CEMADEN_df.head())
    #print(is_string_dtype(CEMADEN_df['Year']))
    
    jd_sp = CEMADEN_df[CEMADEN_df['Site']=='Jd_Sao_Paulo']
    cidade_jardim = CEMADEN_df[CEMADEN_df['Site']=='Cidade_Jardim']
    agua_vermelha = CEMADEN_df[CEMADEN_df['Site']=='Agua_Vermelha']
    
    return jd_sp, cidade_jardim, agua_vermelha

def process_INMET():
    INMET_aut_df = pd.read_csv('INMET/data_aut_8h.csv', sep = ';')
    INMET_aut_df.columns = ['Date', 'Hour', 'Precipitation', 'Null']
    INMET_aut_df = INMET_aut_df[['Date', 'Hour', 'Precipitation']]
    INMET_aut_df[['Year', 'Month', 'Day']] = INMET_aut_df.Date.str.split("-", expand=True)
    INMET_aut_df['Hour'] = (INMET_aut_df['Hour']/100)
    INMET_aut_df = INMET_aut_df[['Year', 'Month', 'Day', 'Hour', 'Precipitation']]
    INMET_aut_df['Hour'] = pd.to_numeric(INMET_aut_df['Hour'], downcast='integer')
    INMET_aut_df['Year'] = pd.to_numeric(INMET_aut_df['Year'], downcast='integer')
    INMET_aut_df['Month'] = pd.to_numeric(INMET_aut_df['Month'], downcast='integer')
    INMET_aut_df['Day'] = pd.to_numeric(INMET_aut_df['Day'], downcast='integer')
    #print(INMET_aut_df)

    INMET_conv_df = pd.read_csv('INMET/data_conv_8h.csv', sep = ';')
    INMET_conv_df.columns = ['Date', 'Hour', 'Precipitation', 'Null']
    INMET_conv_df = INMET_conv_df[['Date', 'Hour', 'Precipitation']]
    INMET_conv_df[['Year', 'Month', 'Day']] = INMET_conv_df.Date.str.split("-", expand=True)
    INMET_conv_df['Hour'] = (INMET_conv_df['Hour']/100)
    INMET_conv_df = INMET_conv_df[['Year', 'Month', 'Day', 'Hour', 'Precipitation']]
    INMET_conv_df['Hour'] = pd.to_numeric(INMET_conv_df['Hour'], downcast='integer')
    INMET_conv_df['Year'] = pd.to_numeric(INMET_conv_df['Year'], downcast='integer')
    INMET_conv_df['Month'] = pd.to_numeric(INMET_conv_df['Month'], downcast='integer')
    INMET_conv_df['Day'] = pd.to_numeric(INMET_conv_df['Day'], downcast='integer')
    #print(INMET_conv_df)
    
    return INMET_aut_df, INMET_conv_df

def process_INMET_daily():
    INMET_aut_df = pd.read_csv('INMET/data_aut_daily.csv', sep = ';') 
    INMET_aut_df.columns = ['Date', 'Precipitation', 'Null']
    INMET_aut_df = INMET_aut_df[['Date', 'Precipitation']]
    INMET_aut_df[['Year', 'Month', 'Day']] = INMET_aut_df.Date.str.split("-", expand=True)
    INMET_aut_df['Year'] = pd.to_numeric(INMET_aut_df['Year'], downcast='integer')
    INMET_aut_df['Month'] = pd.to_numeric(INMET_aut_df['Month'], downcast='integer')
    INMET_aut_df['Day'] = pd.to_numeric(INMET_aut_df['Day'], downcast='integer')
    
    INMET_conv_df = pd.read_csv('INMET/data_conv_daily.csv', sep = ';') 
    INMET_conv_df.columns = ['Date', 'Precipitation', 'Null']
    INMET_conv_df = INMET_conv_df[['Date', 'Precipitation']]
    INMET_conv_df[['Year', 'Month', 'Day']] = INMET_conv_df.Date.str.split("-", expand=True)
    INMET_conv_df['Year'] = pd.to_numeric(INMET_conv_df['Year'], downcast='integer')
    INMET_conv_df['Month'] = pd.to_numeric(INMET_conv_df['Month'], downcast='integer')
    INMET_conv_df['Day'] = pd.to_numeric(INMET_conv_df['Day'], downcast='integer')
    #print(INMET_conv_df)
    
    return INMET_aut_df, INMET_conv_df

def process_MAPLU():
    for i in range(2015, 2019):
        if i == 2015:
            MAPLU_esc_df = pd.read_csv('MAPLU/escola{n}.csv'.format(n = i))
            #print(MAPLU_esc_df)
            #input()
            MAPLU_esc_df['Site'] = MAPLU_esc_df['Site'].str.replace('escola{n}'.format(n = i), 'Escola Sao Bento')
            
        else:
            df = pd.read_csv('MAPLU/escola{n}.csv'.format(n = i))
            MAPLU_esc_df = pd.concat([MAPLU_esc_df, df], ignore_index=True, sort=False)
            MAPLU_esc_df['Site'] = MAPLU_esc_df['Site'].str.replace('escola{n}'.format(n = i), 'Escola Sao Bento')
            #print(MAPLU_esc_df)
            #input()
    
    MAPLU_esc_df.columns = ['Site', 'Date', 'Precipitation']
    MAPLU_esc_df[['Year', 'Month', 'Day_hour']] = MAPLU_esc_df.Date.str.split("-", expand=True)
    MAPLU_esc_df[['Day', 'Hour_min']] = MAPLU_esc_df.Day_hour.str.split(" ", expand=True)
    MAPLU_esc_df[['Hour', 'Min']] = MAPLU_esc_df.Hour_min.str.split(":", expand=True)
    MAPLU_esc_df = MAPLU_esc_df[['Site', 'Year', 'Month', 'Day', 'Hour', 'Min', 'Precipitation']]
    MAPLU_esc_df['Precipitation'] = pd.to_numeric(MAPLU_esc_df['Precipitation'])
    MAPLU_esc_df['Year'] = pd.to_numeric(MAPLU_esc_df['Year'], downcast='integer')
    MAPLU_esc_df['Month'] = pd.to_numeric(MAPLU_esc_df['Month'], downcast='integer')
    MAPLU_esc_df['Day'] = pd.to_numeric(MAPLU_esc_df['Day'], downcast='integer')
    MAPLU_esc_df['Hour'] = pd.to_numeric(MAPLU_esc_df['Hour'], downcast='integer')
    MAPLU_esc_df['Min'] = pd.to_numeric(MAPLU_esc_df['Min'], downcast='integer')
    #print(MAPLU_esc_df)
    
    for i in range(2015, 2019):
        if i == 2015:
            MAPLU_post_df = pd.read_csv('MAPLU/postosaude{n}.csv'.format(n = i))
            #print(MAPLU_post_df)
            #input()
            MAPLU_post_df['Site'] = MAPLU_post_df['Site'].str.replace('postosaude{n}'.format(n = i), 'Posto Santa Felicia')
            
        else:
            df = pd.read_csv('MAPLU/postosaude{n}.csv'.format(n = i))
            MAPLU_post_df = pd.concat([MAPLU_post_df, df], ignore_index=True, sort=False)
            MAPLU_post_df['Site'] = MAPLU_post_df['Site'].str.replace('postosaude{n}'.format(n = i), 'Posto Santa Felicia')
            #print(MAPLU_post_df)
            #input()
    
    MAPLU_post_df.columns = ['Site', 'Date', 'Precipitation']
    MAPLU_post_df[['Year', 'Month', 'Day_hour']] = MAPLU_post_df.Date.str.split("-", expand=True)
    MAPLU_post_df[['Day', 'Hour_min']] = MAPLU_post_df.Day_hour.str.split(" ", expand=True)
    MAPLU_post_df[['Hour', 'Min']] = MAPLU_post_df.Hour_min.str.split(":", expand=True)
    MAPLU_post_df = MAPLU_post_df[['Site', 'Year', 'Month', 'Day', 'Hour', 'Min', 'Precipitation']]
    MAPLU_post_df['Precipitation'] = pd.to_numeric(MAPLU_post_df['Precipitation'])
    MAPLU_post_df['Year'] = pd.to_numeric(MAPLU_post_df['Year'], downcast='integer')
    MAPLU_post_df['Month'] = pd.to_numeric(MAPLU_post_df['Month'], downcast='integer')
    MAPLU_post_df['Day'] = pd.to_numeric(MAPLU_post_df['Day'], downcast='integer')
    MAPLU_post_df['Hour'] = pd.to_numeric(MAPLU_post_df['Hour'], downcast='integer')
    MAPLU_post_df['Min'] = pd.to_numeric(MAPLU_post_df['Min'], downcast='integer')
    #print(MAPLU_post_df)
    
    return MAPLU_esc_df, MAPLU_post_df

def process_MAPLU_USP():
    MAPLU_usp_df = pd.read_csv('MAPLU/USP2.csv')
    
    MAPLU_usp_df[['Hour', 'Min']] = MAPLU_usp_df.Time.str.split(":", expand=True)
    MAPLU_usp_df = MAPLU_usp_df[['Year', 'Month', 'Day', 'Hour', 'Min', 'Precipitation']]
    MAPLU_usp_df['Precipitation'] = pd.to_numeric(MAPLU_usp_df['Precipitation'])
    MAPLU_usp_df['Year'] = pd.to_numeric(MAPLU_usp_df['Year'], downcast='integer')
    MAPLU_usp_df['Month'] = pd.to_numeric(MAPLU_usp_df['Month'], downcast='integer')
    MAPLU_usp_df['Day'] = pd.to_numeric(MAPLU_usp_df['Day'], downcast='integer')
    MAPLU_usp_df['Hour'] = pd.to_numeric(MAPLU_usp_df['Hour'], downcast='integer')
    MAPLU_usp_df['Min'] = pd.to_numeric(MAPLU_usp_df['Min'], downcast='integer')
    #print(MAPLU_usp_df)
    
    return MAPLU_usp_df

def aggregate(df, var):
    if var == 'Year':
        df_new = df.groupby([var]).Precipitation.sum().reset_index()
    if var == 'Month':
        df_new = df.groupby(['Year',var]).Precipitation.sum().reset_index()
    if var == 'Day':
        df_new = df.groupby(['Year', 'Month', var]).Precipitation.sum().reset_index()
    if var == 'Hour':
        df_new = df.groupby(['Year', 'Month', 'Day', var]).Precipitation.sum().reset_index()
    
    return df_new

def aggregate_to_csv(df, name, directory = 'Results'):
    df_yearly = aggregate(df, 'Year')
    df_yearly.to_csv('{d}/{n}_yearly.csv'.format(n = name, d = directory), index = False)
    df_monthly = aggregate(df, 'Month')
    df_monthly.to_csv('{d}/{n}_monthly.csv'.format(n = name, d = directory), index = False)
    df_daily = aggregate(df, 'Day')
    df_daily.to_csv('{d}/{n}_daily.csv'.format(n = name, d = directory), index = False)
    df.to_csv('{d}/{n}_hourly.csv'.format(n = name, d = directory), index = False)

def aggregate_hourly_to_csv(df, name, directory = 'Results'):
    df_hourly = aggregate(df, 'Hour')
    df_hourly.to_csv('{d}/{n}_hourly.csv'.format(n = name, d = directory), index = False)
    df.to_csv('{d}/{n}_min.csv'.format(n = name, d = directory), index = False)
    
def read_csv(name, var):
    df = pd.read_csv('Results/{n}_{v}.csv'.format(n = name, v = var))
    
    return df

def verification(name):
    df = name
    year_0 = df['Year'][0]
    year_i = df['Year'][len(df)-1]
    month_0 = df['Month'][0]
    month_i = df['Month'][len(df)-1]
    day_0 = df['Day'][0]
    day_i = df['Day'][len(df)-1]
    
    d0 = date(year_0, month_0, day_0)
    di = date(year_i, month_i, day_i)
    #print(d0)
    delta = di - d0
    ndays_verification = delta.days
    #print(ndays_verification)
    ndays_real = len(df)
    #ndays_real2 = df['Precipitation'].isnull().sum()
    #print(ndays_real)
    
    verif_number = ndays_verification - ndays_real
    if verif_number > 0:
        print('Fail - series incomplete / number of days missing = {d}'.format(d = verif_number))
    else:
        print('Series complete')
        
def set_date(name):
    df = name
    
    date_list = []
    for i in range(0, len(df)):
        year_i = df['Year'][i]
        #print(year_i)
        month_i = df['Month'][i]
        #print(month_i)
        day_i = df['Day'][i]
        #print(day_i)           
        try:
            di = date(year_i, month_i, day_i)
        except:
            #print('ALERT!!!')
            #raise Exception
            pass
                    
        #print(di)
        date_list.append(di)
         
    #print(date_list)
    #input()
    df['Date'] = date_list
    #print(df)
    df = df.set_index('Date')
    df['Date'] = date_list
    #print(df)
     
    return df

def complete_date_series(name, var):
    df_original = read_csv('{n}'.format(n = name), '{v}'.format(v = var))
    df = set_date(df_original)
    #print(df)
    
    d0 = df['Date'][0]
    di = df['Date'][len(df)-1]
    #print(d0, di)
    idx = pd.date_range(d0, di)
    
    df.index = pd.DatetimeIndex(df.index)
    df = df.reindex(idx)
    df['Date'] = df.index
    return df

def left_join_precipitation(left_df, right_df1, right_df2):
    left_df2 = left_df.merge(right_df1, on = 'Date', how = 'inner')
    #print(left_df2)
    left_df_final = left_df2.merge(right_df2, on = 'Date', how = 'inner')
    #print(left_df_final)
    df = left_df_final[['Date', 'Precipitation_x', 'Precipitation_y', 'Precipitation']]
    df.columns = ['Date', 'P_left', 'P_right1', 'P_right2']
    #print(df)
    return df


def correlation_plots(left_df, right_df1, right_df2):
    df = left_join_precipitation(left_df, right_df1, right_df2)
    df = df[['P_left', 'P_right1', 'P_right2']]
    sns.pairplot(df)
    plt.show()
    corr_pearson = df.corr(method = 'pearson')
    pvalues_pearson = df.corr(method = pearsonr_pval)
    print('')
    print('----- Pearson correlation -----')
    print('Correlation coefficient matrix')
    print(corr_pearson)
    print('')
    print('P-values matrix')
    print(pvalues_pearson)

def correlation_plots_2(df):
    sns.pairplot(df)
    plt.show()
    corr_pearson = df.corr(method = 'pearson')
    pvalues_pearson = df.corr(method = pearsonr_pval)
    print('')
    print('----- Pearson correlation -----')
    print('Correlation coefficient matrix')
    print(corr_pearson)
    print('')
    print('P-values matrix')
    print(pvalues_pearson)
    
    return corr_pearson, pvalues_pearson

def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]    

def simple_linear_regression(left_df, right_df1, right_df2):
    df = left_join_precipitation(left_df, right_df1, right_df2)
    df = df[['P_left', 'P_right1', 'P_right2']]
    df = df.dropna()
    X1 = df[['P_right1']]
    X2 = df[['P_right2']]
    y = df['P_left']

    
    lr = LinearRegression()
    lr.fit(X1, y)
    
    yhat = lr.predict(X1)
    
    slr_slope = lr.coef_
    slr_intercept = lr.intercept_
    print('R-Squared_1 :', lr.score(X1, y))
    
    sns.scatterplot(x = 'P_right1', y = 'P_left', data = df, s = 150, alpha = 0.3, edgecolor = 'white')
    plt.plot(df['P_right1'], slr_slope*df['P_right1'] + slr_intercept, color = 'r', linewidth = 3)
    plt.ylabel('P_left', fontsize = 12)
    plt.xlabel('P_right1', fontsize = 12)    
    plt.show()
    
    lr.fit(X2, y)
    
    yhat = lr.predict(X1)
    
    slr_slope = lr.coef_
    slr_intercept = lr.intercept_
    print('R-Squared_2 :', lr.score(X1, y))
    
    sns.scatterplot(x = 'P_right1', y = 'P_left', data = df, s = 150, alpha = 0.3, edgecolor = 'white')
    plt.plot(df['P_right1'], slr_slope*df['P_right1'] + slr_intercept, color = 'r', linewidth = 3)
    plt.ylabel('P_left', fontsize = 12)
    plt.xlabel('P_right1', fontsize = 12)    
    plt.show()
    
def multiple_linear_regression(left_df, right_df1, right_df2):
    df = left_join_precipitation(left_df, right_df1, right_df2)
    df = df[['P_left', 'P_right1', 'P_right2']]
    df = df.dropna()
    X1 = df[['P_right1', 'P_right2']]
    y = df['P_left']
    
    lr = LinearRegression()
    lr.fit(X1, y)

    yhat = lr.predict(X1)
    print('R-Squared_2 :', lr.score(X1, y))
    
    mlr_slope = lr.coef_
    mlr_intercept = lr.intercept_
    print(mlr_slope, mlr_intercept)        

    
def trend_analysis(data, alpha_value, plot_graphs = True, site = ''):
#     fig, ax = plt.subplots(figsize=(12, 8))
#     sm.graphics.tsa.plot_acf(data, lags=20, ax=ax)
#     plt.show()
    
    result_original = mk.original_test(data, alpha = alpha_value)
    result_hamed_rao = mk.hamed_rao_modification_test(data, alpha = alpha_value)
    result_yue_wang = mk.yue_wang_modification_test(data, alpha = alpha_value)
    result_trend_free = mk.trend_free_pre_whitening_modification_test(data, alpha = alpha_value)
    result_pre_whitening = mk.pre_whitening_modification_test(data, alpha = alpha_value)
    
    print('Original: ', result_original)
    print('Hamed_rao: ', result_hamed_rao)
    print('Yue_wang: ', result_yue_wang)
    print('Trend_free:', result_trend_free)
    print('Pre_whitening: ', result_pre_whitening)
    
    if plot_graphs == True:
        fig, ax = plt.subplots(figsize=(6, 4))
        #res = mk.hamed_rao_modification_test(data)
        #res = mk.pre_whitening_modification_test(data, alpha = alpha_value)
        res = mk.original_test(data, alpha = alpha_value)
        trend_line = np.arange(len(data)) * res.slope + res.intercept
        
        ax.plot(data)
        ax.plot(data.index, trend_line)
        ax.legend(['data', 'trend line'])
        ax.set_xlabel('Years')
        ax.set_ylabel('Precipitation (mm)')
        fig.suptitle('{s}'.format(s = site))
        plt.show()
    

def get_trend(var, sites_list, alpha_value, group, type = 'obs', plot_graphs = True):
    #Type can be 'obs' for observed data or 'mod' for modeled data from gcm.
    #Var can be 'Year' or 'Max_daily'
    
    print('Running get_trend..')
    
    site_trend_list = []
    
    Tau_original_list = []
    p_original_list = []
    trend_original_list = []
    h_original_list = []
    z_original_list = []
    s_original_list = []
    var_s_original_list = []
    slope_original_list = []
    intercept_original_list =[]
    
    Tau_hamed_rao_list = []
    p_hamed_rao_list = []
    trend_hamed_rao_list = []
    h_hamed_rao_list = []
    z_hamed_rao_list = []
    s_hamed_rao_list = []
    var_s_hamed_rao_list = []
    slope_hamed_rao_list = []
    intercept_hamed_rao_list =[]
    
    Tau_yue_wang_list = []
    p_yue_wang_list = []
    trend_yue_wang_list = []
    h_yue_wang_list = []
    z_yue_wang_list = []
    s_yue_wang_list = []
    var_s_yue_wang_list = []
    slope_yue_wang_list = []
    intercept_yue_wang_list =[]

    Tau_trend_free_list = []
    p_trend_free_list = []
    trend_trend_free_list = []
    h_trend_free_list = []
    z_trend_free_list = []
    s_trend_free_list = []
    var_s_trend_free_list = []
    slope_trend_free_list = []
    intercept_trend_free_list =[]

    Tau_pre_whitening_list = []
    p_pre_whitening_list = []
    trend_pre_whitening_list = []
    h_pre_whitening_list = []
    z_pre_whitening_list = []
    s_pre_whitening_list = []
    var_s_pre_whitening_list = []
    slope_pre_whitening_list = []
    intercept_pre_whitening_list =[]
    
    for site in sites_list:
        if type == 'obs':
            if var == 'Year':
                df = pd.read_csv('Results/{s}_yearly.csv'.format(s = site))          
                
            if var == 'Max_daily':
                df = pd.read_csv('Results/max_daily_{s}.csv'.format(s = site))
        
        elif type == 'mod':
            if var == 'Year':
                df = pd.read_csv('GCM_data/bias_correction/{s}_yearly.csv'.format(s = site))          
                
            if var == 'Max_daily':
                df = pd.read_csv('GCM_data/bias_correction/max_daily_{s}.csv'.format(s = site))
        
        else:
            print('ALERT!! Type not specified in this function...')
        
        df_mk = df[['Precipitation']].dropna()
        print('--- {g} / {s} ----'.format(g = group, s = site))
        print('')
        try:
            trend_analysis(df_mk, alpha_value, plot_graphs, site)
            trend_original, h_original, p_original, z_original, Tau_original, s_original , var_s_original, slope_original, intercept_original = mk.original_test(df_mk, alpha = alpha_value)
            trend_hamed_rao, h_hamed_rao, p_hamed_rao, z_hamed_rao, Tau_hamed_rao, s_hamed_rao , var_s_hamed_rao, slope_hamed_rao, intercept_hamed_rao = mk.hamed_rao_modification_test(df_mk, alpha = alpha_value)
            trend_yue_wang, h_yue_wang, p_yue_wang, z_yue_wang, Tau_yue_wang, s_yue_wang , var_s_yue_wang, slope_yue_wang, intercept_yue_wang = mk.yue_wang_modification_test(df_mk, alpha = alpha_value)
            trend_trend_free, h_trend_free, p_trend_free, z_trend_free, Tau_trend_free, s_trend_free , var_s_trend_free, slope_trend_free, intercept_trend_free = mk.trend_free_pre_whitening_modification_test(df_mk, alpha = alpha_value)
            trend_pre_whitening, h_pre_whitening, p_pre_whitening, z_pre_whitening, Tau_pre_whitening, s_pre_whitening , var_s_pre_whitening, slope_pre_whitening, intercept_pre_whitening = mk.pre_whitening_modification_test(df_mk, alpha = alpha_value)

            site_trend_list.append(site)
            
            Tau_original_list.append(Tau_original)
            p_original_list.append(p_original)
            trend_original_list.append(trend_original)
            h_original_list.append(h_original)
            z_original_list.append(z_original)
            s_original_list.append(s_original)
            var_s_original_list.append(var_s_original)
            slope_original_list.append(slope_original)
            intercept_original_list.append(intercept_original)
            
            Tau_hamed_rao_list.append(Tau_hamed_rao)
            p_hamed_rao_list.append(p_hamed_rao)
            trend_hamed_rao_list.append(trend_hamed_rao)
            h_hamed_rao_list.append(h_hamed_rao)
            z_hamed_rao_list.append(z_hamed_rao)
            s_hamed_rao_list.append(s_hamed_rao)
            var_s_hamed_rao_list.append(var_s_hamed_rao)
            slope_hamed_rao_list.append(slope_hamed_rao)
            intercept_hamed_rao_list.append(intercept_hamed_rao)
            
            Tau_yue_wang_list.append(Tau_yue_wang)
            p_yue_wang_list.append(p_yue_wang)
            trend_yue_wang_list.append(trend_yue_wang)
            h_yue_wang_list.append(h_yue_wang)
            z_yue_wang_list.append(z_yue_wang)
            s_yue_wang_list.append(s_yue_wang)
            var_s_yue_wang_list.append(var_s_yue_wang)
            slope_yue_wang_list.append(slope_yue_wang)
            intercept_yue_wang_list.append(intercept_yue_wang)
        
            Tau_trend_free_list.append(Tau_trend_free)
            p_trend_free_list.append(p_trend_free)
            trend_trend_free_list.append(trend_trend_free)
            h_trend_free_list.append(h_trend_free)
            z_trend_free_list.append(z_trend_free)
            s_trend_free_list.append(s_trend_free)
            var_s_trend_free_list.append(var_s_trend_free)
            slope_trend_free_list.append(slope_trend_free)
            intercept_trend_free_list.append(intercept_trend_free)
        
            Tau_pre_whitening_list.append(Tau_pre_whitening)
            p_pre_whitening_list.append(p_pre_whitening)
            trend_pre_whitening_list.append(trend_pre_whitening)
            h_pre_whitening_list.append(h_pre_whitening)
            z_pre_whitening_list.append(z_pre_whitening)
            s_pre_whitening_list.append(s_pre_whitening)
            var_s_pre_whitening_list.append(var_s_pre_whitening)
            slope_pre_whitening_list.append(slope_pre_whitening)
            intercept_pre_whitening_list.append(intercept_pre_whitening)            
            
            #input()
            print('')
        except ZeroDivisionError:
            print('Division by zero!! - Not possible to perform trend analysis')
            print('')
            pass
        except:
            print('ALERT!!! Something else went wrong')
            raise Exception
        
    trend_dict ={
            'Site': site_trend_list,
            'Tau_original': Tau_original_list,
            'p_value_original': p_original_list,
            'trend_original': trend_original_list,
            'h_original': h_original_list,
            'z_original': z_original_list,
            's_original': s_original_list,
            'var_s_original': var_s_original_list,
            'slope_original': slope_original_list,
            'intercept_original': intercept_original_list,
            'Tau_hamed_rao': Tau_hamed_rao_list,
            'p_value_hamed_rao': p_hamed_rao_list,
            'trend_hamed_rao': trend_hamed_rao_list,
            'h_hamed_rao': h_hamed_rao_list,
            'z_hamed_rao': z_hamed_rao_list,
            's_hamed_rao': s_hamed_rao_list,
            'var_s_hamed_rao': var_s_hamed_rao_list,
            'slope_hamed_rao': slope_hamed_rao_list,
            'interceped_hamed_rao': intercept_hamed_rao_list,
            'Tau_yue_wang': Tau_yue_wang_list,
            'p_value_yue_wang': p_yue_wang_list,
            'trend_yue_wang': trend_yue_wang_list,
            'h_yue_wang': h_yue_wang_list,
            'z_yue_wang': z_yue_wang_list,
            's_yue_wang': s_yue_wang_list,
            'var_s_yue_wang': var_s_yue_wang_list,
            'slope_yue_wang': slope_yue_wang_list,
            'intercept_yue_wang': intercept_yue_wang_list,
            'Tau_trend_free': Tau_trend_free_list,
            'p_value_trend_free': p_trend_free_list,
            'trend_trend_free': trend_trend_free_list,
            'h_trend_free': h_trend_free_list,
            'z_trend_free': z_trend_free_list,
            's_trend_free': s_trend_free_list,
            'var_s_trend_free': var_s_trend_free_list,
            'slope_trend_free': slope_trend_free_list,
            'intercept_trend_free': intercept_trend_free_list,
            'Tau_pre_whitening': Tau_pre_whitening_list,
            'p_value_pre_whitening': p_pre_whitening_list,
            'trend_pre_whitening': trend_pre_whitening_list,
            'h_pre_whitening': h_pre_whitening_list,
            'z_pre_whitening': z_pre_whitening_list,
            's_pre_whitening': s_pre_whitening_list,
            'var_s_pre_whitening': var_s_pre_whitening_list,
            'slope_pre_whitening': slope_pre_whitening_list,
            'intercept_pre_whitening': intercept_pre_whitening_list        
            }
    
    df_trend_result = pd.DataFrame(trend_dict)
    
    if type == 'obs':
        df_trend_result.to_csv('Results/{g}_{v}_trend_result.csv'.format(g = group, v = var), index = False, encoding = 'latin1')
    elif type == 'mod':
        df_trend_result.to_csv('GCM_data/bias_correction/{g}_{v}_trend_result.csv'.format(g = group, v = var), index = False, encoding = 'latin1')
    else:
        print('ALERT!! Type not specified in this function..')
    
def p90_function(data):
    df = data[['Precipitation']]
    df = df[df.Precipitation != 0]
    df = df.sort_values('Precipitation')
    df = df.reset_index(drop = True)
    df['Order'] = list(range(1,len(df)+1))
    max_order = df['Order'][len(df)-1]
    df['Probability'] = (df['Order']/max_order)*100
    df['Probability_round'] = round(df['Probability'], 0)
    df_new = df.query('Probability_round == 90')
    P90 = (df_new['Precipitation'].iloc[0])
    
    sns.lineplot(x = 'Probability', y = 'Precipitation', data = df, palette = 'Black')
    plt.ylabel('Precipitation (mm)', fontsize = 12)
    plt.xlabel('Probability (%)', fontsize = 12)    
    plt.show()
    plt.title("Probability of non-exceedence")
    
    return P90

def distribution_plot(name, var):
    df = read_csv('{n}'.format(n = name), '{v}'.format(v = var))
    df = df.dropna()
    sns.distplot(df['Precipitation'], color = 'skyblue')
    plt.title('{n} - {v}'.format(n = name, v = var))
    plt.show()
    
def distribution_plot_df(df):
    df = df.dropna()
    sns.distplot(df['Precipitation'], color = 'skyblue')
    plt.show()
    
def aggregate_subdaily(df, hours):
    df = df[['Precipitation']]
    acum_list = []
    n = hours
    
    for i in range(len(df)-n+1):
        acum = df.iloc[0+i:n+i].sum()[0]
        #print(acum)
        acum_list.append(acum)
        i = i+1
    
    
    return acum_list

def aggregate_subdaily_minutes(df_min, min_agg, dt_min):
    #print(df_min)
    df = df_min[['Precipitation']]
    acum_list = []
    n_float = min_agg/dt_min
    n = int(n_float)
    #print(n)
    #list = range(len(df)-n+1)
    #print(list)
    
    for i in range(len(df)-n+1):
        #print(0+i, n+i)
        acum = df.iloc[0+i:n+i].sum()[0]
        #print(acum)
        #input()
        acum_list.append(acum)
        i = i+1
    
    #print(acum_list)
    return acum_list

def get_subdaily_max(df, hours):
    years_list = df['Year'].unique()
    max_subdaily_list = []
    
    for year in years_list:
        #print(year)
        df_new = df.loc[df['Year'] == year]
        #print(df_new)
        subdaily_list = aggregate_subdaily(df_new, hours)
        #print(subdaily_list)
        #input()
        max_subdaily = max(subdaily_list)
        #print(max_subdaily)
        #input()
        max_subdaily_list.append(max_subdaily)
        
    dict = {'Year': years_list,
            'Max_{h}'.format(h = hours): max_subdaily_list}
    
    df_result = pd.DataFrame(dict)
    
    return df_result

def get_subdaily_min_max(df_min, min_agg, dt_min):
    years_list = df_min['Year'].unique()
    #print(years_list)
    max_subdaily_list = []
    
    for year in years_list:
        #print(year)
        df_new = df_min.loc[df_min['Year'] == year]
        #print(df_new)
        subdaily_list = aggregate_subdaily_minutes(df_new, min_agg, dt_min)
        #print(subdaily_list)
        #input()
        max_subdaily = max(subdaily_list)
        #print(max_subdaily)
        #input()
        max_subdaily_list.append(max_subdaily)
        
    dict = {'Year': years_list,
            'Max_{min}min'.format(min = min_agg): max_subdaily_list}
    
    df_result = pd.DataFrame(dict)
    
    return df_result

def get_max_subdaily_table(name_file, directory = 'Results'):
    print('Getting maximum subdaily..')
    df = pd.read_csv('{d}/{n}_hourly.csv'.format(d = directory, n = name_file))
    df_1h = get_subdaily_max(df, 1)
    print('')
    print('1h done!..')
    df_3h = get_subdaily_max(df, 3)
    print('')
    print('3h done!..')
    df_final = df_1h.merge(df_3h, on = 'Year', how = 'inner')
     
    df_6h = get_subdaily_max(df, 6)
    print('')
    print('6h done!..')
    df_final = df_final.merge(df_6h, on = 'Year', how = 'inner')
     
    df_8h = get_subdaily_max(df, 8)
    print('')
    print('8h done!..')
    df_final = df_final.merge(df_8h, on = 'Year', how = 'inner')
     
    df_10h = get_subdaily_max(df, 10)
    print('')
    print('10h done!..')
    df_final = df_final.merge(df_10h, on = 'Year', how = 'inner')
     
    df_12h = get_subdaily_max(df, 12)
    print('')
    print('12h done!..')
    df_final = df_final.merge(df_12h, on = 'Year', how = 'inner')
     
    df_24h = get_subdaily_max(df, 24)
    print('')
    print('24h done!..')
    df_final = df_final.merge(df_24h, on = 'Year', how = 'inner')
    print('')
    print(df_final)
    print('')
    df_final.to_csv('{d}/max_subdaily_{n}.csv'.format(d = directory, n = name_file), index = False)
    print('Done!')

def get_max_subdaily_min_table(name_file, dt_min, directory = 'Results'):
    print('Getting maximum subdaily..')
    df_min = pd.read_csv('{d}/{n}_min.csv'.format(d = directory, n = name_file))
    df_5min = get_subdaily_min_max(df_min, 5, dt_min)
    print(df_5min)
    print('')
    print('5min done!..')
    df_10min = get_subdaily_min_max(df_min, 10, dt_min)
    print('')
    print('10min done!..')
    df_final = df_5min.merge(df_10min, on = 'Year', how = 'inner')
     
    df_15min = get_subdaily_min_max(df_min, 15, dt_min)
    print('')
    print('15min done!..')
    df_final = df_final.merge(df_15min, on = 'Year', how = 'inner')
     
    df_20min = get_subdaily_min_max(df_min, 20, dt_min)
    print('')
    print('20min done!..')
    df_final = df_final.merge(df_20min, on = 'Year', how = 'inner')
     
    df_25min = get_subdaily_min_max(df_min, 25, dt_min)
    print('')
    print('25min done!..')
    df_final = df_final.merge(df_25min, on = 'Year', how = 'inner')
     
    df_30min = get_subdaily_min_max(df_min, 30, dt_min)
    print('')
    print('30min done!..')
    df_final = df_final.merge(df_30min, on = 'Year', how = 'inner')
    print('')


    print(df_final)
    print('')
    df_final.to_csv('{d}/max_subdaily_min_{n}.csv'.format(d = directory, n = name_file), index = False)
    print('Done!')

def merge_max_tables(name_file, directory='Results'):
    df_min = pd.read_csv('{d}/max_subdaily_min_{n}.csv'.format(d = directory, n=name_file))
    df_hour = pd.read_csv('{d}/max_subdaily_{n}.csv'.format(d = directory, n=name_file))
    
    df_complete = df_min.merge(df_hour, on = 'Year', how = 'inner')
    df_complete.to_csv('{d}/max_subdaily_complete_{n}.csv'.format(d = directory, n = name_file), index = False)


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
    
def plot_subdaily_maximum_absolute(name_file):
    print('Starting ploting absolute subdaily maximums..')
    print('')
    df = pd.read_csv('Results/max_subdaily_{n}.csv'.format(n = name_file))
    df['Type'] = 'Observed'
    df_ger = pd.read_csv('Results/max_subdaily_{n}_df_ger.csv'.format(n = name_file))
    df_ger['Type'] = 'CETESB'
    df_m20 = pd.read_csv('Results/max_subdaily_{n}_df_m20.csv'.format(n = name_file))
    df_m20['Type'] = 'CETESB_-20%'
    df_p20 = pd.read_csv('Results/max_subdaily_{n}_df_p20.csv'.format(n = name_file))
    df_p20['Type'] = 'CETESB_+20%'
    
    df_final = pd.concat([df, df_ger, df_m20, df_p20], ignore_index = True, sort = False)
    df_final = df_final[['Year', 'Max_1', 'Max_6', 'Max_8', 'Max_10', 'Max_12', 'Max_24', 'Type']]
    #print(df_final)
    
    g = sns.catplot(x="Year", y="Max_1", hue = 'Type', data=df_final, kind = 'bar', height = 5, aspect = 1.5)
    g.set_axis_labels('', 'Precipitation')
    fig = g.fig
    fig.subplots_adjust(bottom = 0.15, top = 0.9, left = 0.07)
    plt.xticks(rotation=50)
    plt.title('Subdaily {n} - 1h'.format(n = name_file))
    plt.ylim(0, 170)
    #plt.show()
    #input()
    plt.savefig('Graphs/subdaily/{n}_max1.png'.format(n = name_file))
    print('Graph absolute Max_1h done!..')
    print('')

    g = sns.catplot(x="Year", y="Max_6", hue = 'Type', data=df_final, kind = 'bar', height = 5, aspect = 1.5)
    g.set_axis_labels('', 'Precipitation')
    fig = g.fig
    fig.subplots_adjust(bottom = 0.15, top = 0.9, left = 0.07)   
    plt.xticks(rotation=50)
    plt.title('Subdaily {n} - 6h'.format(n = name_file))
    plt.ylim(0, 170)
    #plt.show()
    plt.savefig('Graphs/subdaily/{n}_max6.png'.format(n = name_file))
    print('Graph absolute Max_6h done!..')
    print('')

    g = sns.catplot(x="Year", y="Max_8", hue = 'Type', data=df_final, kind = 'bar', height = 5, aspect = 1.5)
    g.set_axis_labels('', 'Precipitation')
    fig = g.fig
    fig.subplots_adjust(bottom = 0.15, top = 0.9, left = 0.07)
    plt.xticks(rotation=50)
    plt.title('Subdaily {n} - 8h'.format(n = name_file))
    plt.ylim(0, 170)
    #plt.show()
    plt.savefig('Graphs/subdaily/{n}_max8.png'.format(n = name_file))
    print('Graph absolute Max_8h done!..')
    print('')

    g = sns.catplot(x="Year", y="Max_10", hue = 'Type', data=df_final, kind = 'bar', height = 5, aspect = 1.5)
    g.set_axis_labels('', 'Precipitation')
    fig = g.fig
    fig.subplots_adjust(bottom = 0.15, top = 0.9, left = 0.07)
    plt.xticks(rotation=50)
    plt.title('Subdaily {n} - 10h'.format(n = name_file))
    plt.ylim(0, 170)
    #plt.show()
    plt.savefig('Graphs/subdaily/{n}_max10.png'.format(n = name_file))
    print('Graph absolute Max_10h done!..')
    print('')
        
    g = sns.catplot(x="Year", y="Max_12", hue = 'Type', data=df_final, kind = 'bar', height = 5, aspect = 1.5)
    g.set_axis_labels('', 'Precipitation')
    fig = g.fig
    fig.subplots_adjust(bottom = 0.15, top = 0.9, left = 0.07)    
    plt.xticks(rotation=50)
    plt.title('Subdaily {n} - 12h'.format(n = name_file))
    plt.ylim(0, 170)
    #plt.show()
    plt.savefig('Graphs/subdaily/{n}_max12.png'.format(n = name_file))
    print('Graph absolute Max_12h done!..')
    print('')
    
    g = sns.catplot(x="Year", y="Max_24", hue = 'Type', data=df_final, kind = 'bar', height = 5, aspect = 1.5)
    g.set_axis_labels('', 'Precipitation')
    fig = g.fig
    fig.subplots_adjust(bottom = 0.15, top = 0.9, left = 0.07)    
    plt.xticks(rotation=50)
    plt.title('Subdaily {n} - 24h'.format(n = name_file))
    plt.ylim(0, 170)
    #plt.show()
    plt.savefig('Graphs/subdaily/{n}_max24.png'.format(n = name_file))
    print('Graph absolute Max_24h done!..')
    print('')
    print('Done ploting absolute maximums!')
    print('')
    
def plot_subdaily_maximum_relative(name_file, max_hour, var_value):
    print('Starting ploting relative subdaily maximums')
    print('')
    df = pd.read_csv('Results/max_subdaily_{n}.csv'.format(n = name_file))
    df['Type'] = 'Observed'
    df_ger = pd.read_csv('Results/max_subdaily_{n}_ger.csv'.format(n = name_file))
    df_ger['Type'] = 'CETESB'
    df_m20 = pd.read_csv('Results/max_subdaily_{n}_m{v}.csv'.format(n = name_file, v = var_value))
    df_m20['Type'] = 'CETESB_-{v}'.format(v = var_value)
    df_p20 = pd.read_csv('Results/max_subdaily_{n}_p{v}.csv'.format(n = name_file, v = var_value))
    df_p20['Type'] = 'CETESB_+{v}'.format(v = var_value)

    df_final = pd.concat([df, df_ger, df_m20, df_p20], ignore_index = True, sort = False)
    df_final = df_final[['Year', 'Max_1', 'Max_6', 'Max_8', 'Max_10', 'Max_12', 'Max_24', 'Type']]
    
    df_obs_process = df_final.loc[df_final['Type'] == 'Observed'].reset_index()
    df_ger_process = df_final.loc[df_final['Type'] == 'CETESB'].reset_index()
    df_mvar_process = df_final[df_final['Type'] == 'CETESB_-{v}'.format(v = var_value)].reset_index()
    df_pvar_process = df_final[df_final['Type'] == 'CETESB_+{v}'.format(v = var_value)].reset_index()
    
    df_ger_process['Dif_{m}'.format(m = max_hour)] = df_obs_process['Max_{m}'.format(m = max_hour)] - df_ger_process['Max_{m}'.format(m = max_hour)]
    df_mvar_process['Dif_{m}'.format(m = max_hour)] = df_obs_process['Max_{m}'.format(m = max_hour)] - df_mvar_process['Max_{m}'.format(m = max_hour)]
    df_pvar_process['Dif_{m}'.format(m = max_hour)] = df_obs_process['Max_{m}'.format(m = max_hour)] - df_pvar_process['Max_{m}'.format(m = max_hour)]
    
    sum_error_ger = df_ger_process['Dif_{m}'.format(m = max_hour)].sum()
    sum_error_mvar = df_mvar_process['Dif_{m}'.format(m = max_hour)].sum()
    sum_error_pvar = df_pvar_process['Dif_{m}'.format(m = max_hour)].sum()
    
    print('sum_error_ger / {v}: '.format(v = var_value), sum_error_ger)
    print('')
    print('sum_error_mvar / {v}: '.format(v = var_value), sum_error_mvar)
    print('')
    print('sum_error_pvar / {v}: '.format(v = var_value), sum_error_pvar)    
    print('')
    
    df_graph = pd.concat([df_ger_process, df_mvar_process, df_pvar_process], ignore_index = True, sort = False)
    
    g = sns.catplot(x="Year", y="Dif_{m}".format(m = max_hour), hue = 'Type', data=df_graph, kind = 'bar', height = 5, aspect = 1.5)
    g.set_axis_labels('', 'Precipitation')
    fig = g.fig
    fig.subplots_adjust(bottom = 0.15, top = 0.9, left = 0.07)    
    plt.xticks(rotation=50)
    plt.ylim(-100, 50)
    plt.title('Subdaily {n} - {m}h - {v}'.format(n = name_file, m = max_hour, v = var_value))
    #plt.show()
    plt.savefig('Graphs/subdaily_variations/{n}_max{m}_{v}_relative.png'.format(n = name_file, m = max_hour, v = var_value))
    print('Graph relative Max_{m}h done!..'.format(m = max_hour))
    print('')
    
def plot_subdaily_maximum_BL(max_hour):
    #Comparar INMET_aut e MAPLU_usp observados com INMET_aut e MAPLU_usp desagregados por BL e por desagregador
    print('Starting ploting relative subdaily maximums BL and observed')
    print('')
    df_inmet = pd.read_csv('Results/max_subdaily_INMET_aut.csv')
    df_inmet['Type'] = 'INMET_aut observed'
    df_inmet_ger = pd.read_csv('Results/max_subdaily_INMET_aut_ger.csv')
    df_inmet_ger['Type'] = 'INMET_aut CETESB'
    df_maplu = pd.read_csv('Results/max_subdaily_MAPLU_usp.csv')
    df_maplu['Type'] = 'MAPLU_usp observed'
    df_maplu_ger = pd.read_csv('Results/max_subdaily_MAPLU_usp_ger.csv')
    df_maplu_ger['Type'] = 'MAPLU_usp CETESB'
    df_inmet_bl = pd.read_csv('bartlet_lewis/max_subdaily_INMET_bl.csv')
    df_inmet_bl['Type'] = 'INMET_aut BL'
    df_maplu_bl = pd.read_csv('bartlet_lewis/max_subdaily_MAPLU_usp_bl.csv')
    df_maplu_bl['Type'] = 'MAPLU_usp BL'
    

    df_final = pd.concat([df_inmet, df_inmet_ger, df_inmet_bl, df_maplu, df_maplu_ger, df_maplu_bl], ignore_index = True, sort = False)
    df_final = df_final[['Year', 'Max_1', 'Max_6', 'Max_8', 'Max_10', 'Max_12', 'Max_24', 'Type']]
    
    g = sns.catplot(x="Year", y="Max_{m}".format(m = max_hour), hue = 'Type', data=df_final, kind = 'bar', height = 5, aspect = 1.5)
    g.set_axis_labels('', 'Precipitation')
    fig = g.fig
    fig.subplots_adjust(bottom = 0.15, top = 0.9, left = 0.07)
    plt.xticks(rotation=50)
    plt.title('Subdaily INMET and MAPLU - BL test - {m}h'.format(m = max_hour))
    plt.ylim(0, 170)
    #plt.show()
    #input()
    plt.savefig('Graphs/subdaily_bl/BL_max{m}_absolute.png'.format(m = max_hour))
    print('Graph absolute Max_{m}h done!..'.format(m = max_hour))
    print('')
    
    df_inmet_obs_process = df_final.loc[df_final['Type'] == 'INMET_aut observed'].reset_index()
    df_inmet_ger_process = df_final.loc[df_final['Type'] == 'INMET_aut CETESB'].reset_index()
    df_inmet_bl_process = df_final[df_final['Type'] == 'INMET_aut BL'].reset_index()
    df_maplu_obs_process = df_final.loc[df_final['Type'] == 'MAPLU_usp observed'].reset_index()
    df_maplu_ger_process = df_final.loc[df_final['Type'] == 'MAPLU_usp CETESB'].reset_index()
    df_maplu_bl_process = df_final[df_final['Type'] == 'MAPLU_usp BL'].reset_index()
    
    df_inmet_ger_process['Dif_{m}'.format(m = max_hour)] = df_inmet_obs_process['Max_{m}'.format(m = max_hour)] - df_inmet_ger_process['Max_{m}'.format(m = max_hour)]
    df_inmet_bl_process['Dif_{m}'.format(m = max_hour)] = df_inmet_obs_process['Max_{m}'.format(m = max_hour)] - df_inmet_bl_process['Max_{m}'.format(m = max_hour)]
    df_maplu_ger_process['Dif_{m}'.format(m = max_hour)] = df_maplu_obs_process['Max_{m}'.format(m = max_hour)] - df_maplu_ger_process['Max_{m}'.format(m = max_hour)]
    df_maplu_bl_process['Dif_{m}'.format(m = max_hour)] = df_maplu_obs_process['Max_{m}'.format(m = max_hour)] - df_maplu_bl_process['Max_{m}'.format(m = max_hour)]
    
    
    sum_error_inmet_ger = df_inmet_ger_process['Dif_{m}'.format(m = max_hour)].sum()
    sum_error_inmet_bl = df_inmet_bl_process['Dif_{m}'.format(m = max_hour)].sum()
    sum_error_maplu_ger = df_maplu_ger_process['Dif_{m}'.format(m = max_hour)].sum()
    sum_error_maplu_bl = df_maplu_bl_process['Dif_{m}'.format(m = max_hour)].sum()

    
    print('sum_error_inmet_ger: ', sum_error_inmet_ger)
    print('')
    print('sum_error_inmet_bl: ', sum_error_inmet_bl)
    print('')
    print('sum_error_maplu_ger: ', sum_error_maplu_ger)
    print('')
    print('sum_error_maplu_bl: ', sum_error_maplu_bl)
    print('')


    df_graph = pd.concat([df_inmet_ger_process, df_inmet_bl_process, df_maplu_ger_process, df_maplu_bl_process], ignore_index = True, sort = False)
    
    h = sns.catplot(x="Year", y="Dif_{m}".format(m = max_hour), hue = 'Type', data=df_graph, kind = 'bar', height = 5, aspect = 1.5)
    h.set_axis_labels('', 'Precipitation')
    fig2 = h.fig
    fig2.subplots_adjust(bottom = 0.15, top = 0.9, left = 0.07)    
    plt.xticks(rotation=50)
    plt.ylim(-100, 50)
    plt.title('Subdaily INMET and MAPLU - BL test - {m}h'.format(m = max_hour))
    #plt.show()
    plt.savefig('Graphs/subdaily_bl/BL_max{m}_relative.png'.format(m = max_hour))
    print('Graph relative Max_{m}h done!..'.format(m = max_hour))
    print('')

def get_subdaily_optimized(name_file):
    df_disagreg_factors = pd.read_csv('fatores_desagregacao.csv')
    df_aut = pd.read_csv('Results/max_daily_{n}_2.csv'.format(n = name_file))

    df_subdaily = df_aut.copy()
    
    df_subdaily['Max_5min'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_ger'][0]
    df_subdaily['Max_10min'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_ger'][1]
    df_subdaily['Max_15min'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_ger'][2]
    df_subdaily['Max_20min'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_ger'][3]
    df_subdaily['Max_25min'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_ger'][4]
    df_subdaily['Max_30min'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_ger'][5]
    df_subdaily['Max_1'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_ger'][6]*(1+0.05)
    df_subdaily['Max_6'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_ger'][7]*(1-0.05)
    df_subdaily['Max_8'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_ger'][8]*(1-0.1)
    df_subdaily['Max_10'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_ger'][9]*(1-0.1)
    df_subdaily['Max_12'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_ger'][10]*(1-0.1)
    df_subdaily['Max_24'] = df_subdaily['Precipitation']*df_disagreg_factors['CETESB_ger'][11]*(1-0.01)
    
    df_subdaily.to_csv('Results/max_subdaily_{n}_otimizado.csv'.format(n = name_file, t = type), index = False)

def plot_optimized_subdaily(name_file, max_hour):
    print('Starting ploting relative subdaily maximums')
    print('')
    df = pd.read_csv('Results/max_subdaily_{n}.csv'.format(n = name_file))
    df['Type'] = 'Observed'
    df_ger = pd.read_csv('Results/max_subdaily_{n}_ger.csv'.format(n = name_file))
    df_ger['Type'] = 'CETESB'
    df_opt = pd.read_csv('Results/max_subdaily_{n}_otimizado.csv'.format(n = name_file))
    df_opt['Type'] = 'CETESB_otimizado'
    
    df_final = pd.concat([df, df_ger, df_opt], ignore_index = True, sort = False)
    df_final = df_final[['Year', 'Max_1', 'Max_6', 'Max_8', 'Max_10', 'Max_12', 'Max_24', 'Type']]
    
    df_obs_process = df_final.loc[df_final['Type'] == 'Observed'].reset_index()
    df_ger_process = df_final.loc[df_final['Type'] == 'CETESB'].reset_index()
    df_opt_process = df_final[df_final['Type'] == 'CETESB_otimizado'].reset_index()
    
    df_ger_process['Dif_{m}'.format(m = max_hour)] = df_obs_process['Max_{m}'.format(m = max_hour)] - df_ger_process['Max_{m}'.format(m = max_hour)]
    df_opt_process['Dif_{m}'.format(m = max_hour)] = df_obs_process['Max_{m}'.format(m = max_hour)] - df_opt_process['Max_{m}'.format(m = max_hour)]
    
    sum_error_ger = df_ger_process['Dif_{m}'.format(m = max_hour)].sum()
    sum_error_mvar = df_opt_process['Dif_{m}'.format(m = max_hour)].sum()
    
    print('sum_error_ger : ', sum_error_ger)
    print('')
    print('sum_error_opt: ', sum_error_mvar)
    print('')
    
    df_graph = pd.concat([df_ger_process, df_opt_process], ignore_index = True, sort = False)
    
    g = sns.catplot(x="Year", y="Dif_{m}".format(m = max_hour), hue = 'Type', data=df_graph, kind = 'bar', height = 5, aspect = 1.5)
    g.set_axis_labels('', 'Precipitation')
    fig = g.fig
    fig.subplots_adjust(bottom = 0.15, top = 0.9, left = 0.07)    
    plt.xticks(rotation=50)
    plt.ylim(-100, 50)
    plt.title('Subdaily {n} - {m}h - opt'.format(n = name_file, m = max_hour))
    #plt.show()
    plt.savefig('Graphs/subdaily_variations/{n}_max{m}_opt_relative.png'.format(n = name_file, m = max_hour))
    print('Graph relative Max_{m}h done!..'.format(m = max_hour))
    print('')
    
def max_annual_precipitation(df):
    df = df.dropna()
    df_new = df.groupby(['Year'])['Precipitation'].max().reset_index()
    #med = df_new['Precipitation'].median()
    q1 = df_new['Precipitation'].quantile(0.25)
    q3 = df_new['Precipitation'].quantile(0.75)
    IQR =  q3 - q1
    L0 = IQR*1.5
    L_low = q1 - L0
    L_high = q3 + L0
    #print(L_low, L_high)
    df_new = df_new[df_new.Precipitation > L_low]
    df_new = df_new[df_new.Precipitation < L_high]
    
    return df_new

def remove_outliers_from_max(df):
    df_new = df.dropna()
    q1 = df_new['Precipitation'].quantile(0.25)
    q3 = df_new['Precipitation'].quantile(0.75)
    IQR =  q3 - q1
    L0 = IQR*1.5
    L_low = q1 - L0
    L_high = q3 + L0
    #print(L_low, L_high)
    df_new = df_new[df_new.Precipitation > L_low]
    df_new = df_new[df_new.Precipitation < L_high]
    
    return df_new



if __name__ == '__main__':
    print('Starting..')
#     jd_sp, cidade_jardim, agua_vermelha = process_CEMADEN()
#     INMET_aut, INMET_conv = process_INMET()
#     MAPLU_esc, MAPLU_post = process_MAPLU()
#        
#     aggregate_to_csv(jd_sp, 'jd_sp') 
#     aggregate_to_csv(cidade_jardim, 'cidade_jardim') 
#     aggregate_to_csv(agua_vermelha, 'agua_vermelha') 
#     aggregate_to_csv(INMET_aut, 'INMET_aut') 
#     aggregate_to_csv(INMET_conv, 'INMET_conv')        
#     aggregate_to_csv(MAPLU_esc, 'MAPLU_esc') 
#     aggregate_to_csv(MAPLU_post, 'MAPLU_post')
#     aggregate_hourly_to_csv(MAPLU_esc, 'MAPLU_esc') 
#     aggregate_hourly_to_csv(MAPLU_post, 'MAPLU_post')
#     
#     MAPLU_usp = process_MAPLU_USP()
#     aggregate_to_csv(MAPLU_usp, 'MAPLU_usp')
#     aggregate_hourly_to_csv(MAPLU_usp, 'MAPLU_usp')
# 
#     MAPLU_usp = pd.read_csv('bartlet_lewis/MAPLU_disag.csv')
#     aggregate_to_csv(MAPLU_usp, 'MAPLU_usp_bl', directory='bartlet_lewis')
#     aggregate_hourly_to_csv(MAPLU_usp, 'MAPLU_usp_bl', directory='bartlet_lewis')
# 
#     INMET = pd.read_csv('bartlet_lewis/INMET_disag.csv')
#     aggregate_to_csv(INMET, 'INMET_bl', directory='bartlet_lewis')
#     aggregate_hourly_to_csv(INMET, 'INMET_bl', directory='bartlet_lewis')

    INMET_conv = pd.read_csv('bartlet_lewis/INMETconv_disag.csv') #mudar o nome do caminho do arquivo de entrada
    df_novo_para_salvar = aggregate_subdaily(INMET_conv, 2) #se a planilha ja estiver agregada para horas. ja considerando dados a cada hora.
    df_novo_para_salvar = aggregate_subdaily_minutes(INMET_conv, 5, 5) #se a planilha estiver com os dados em minutos. dt_min eh o intervalo de minutos dos dados observados.
    df_novo_para_salvar.to_csv('GCM_data/bias_correction/test_params_sp/comparison_min.csv') #mudar o nome do caminho para onde vc quer que salva

    
    #aggregate_to_csv(INMET_conv, 'INMET_conv_bl', directory='bartlet_lewis')
    #aggregate_hourly_to_csv(INMET_conv, 'INMET_conv_bl', directory='bartlet_lewis')
        
    print('Done!')  
