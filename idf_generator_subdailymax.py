import pandas as pd
from functions_treatment import *
from functions_get_distribution import *
from scipy.optimize import minimize_scalar

def remove_outliers(df, duration):
    df_new = df.dropna()
    df_new.columns = ['Precipitation']
    #print(df_new)
    q1 = df_new['Precipitation'].quantile(0.25)
    q3 = df_new['Precipitation'].quantile(0.75)
    IQR =  q3 - q1
    L0 = IQR*1.5
    L_low = q1 - L0
    L_high = q3 + L0
    #print(L_low, L_high)
    df_new = df_new[df_new.Precipitation > L_low]
    df_new = df_new[df_new.Precipitation < L_high]
    df_new.columns = ['Max_{dur}'.format(dur = duration)]
    #print(df_new)
    #input()
    return df_new

def get_theoretical_max_precipitations(name_file, duration, MY_DISTRIBUTIONS, return_period_list, dist, directory = 'Results', disag_factor = 'nan', plot_graph = False):
    if disag_factor == 'nan':
        disag_factor = ''
    elif disag_factor == 'original':
        disag_factor = '_ger'
    elif disag_factor == 'bl':
        disag_factor = '_bl'
        name_file = 'complete_{name}'.format(name = name_file, disag = disag_factor)
        
    else:
        disag_factor = '{disag}'.format(disag = disag_factor)
        
    data_df_original = pd.read_csv('{d}/max_subdaily_{n}{disag}.csv'.format(n = name_file, d = directory, disag = disag_factor))
     
    data_df_2 = data_df_original.sort_values('Max_{dur}'.format(dur = duration), ascending=False).reset_index()[['Year', 'Max_{dur}'.format(dur = duration)]].reset_index()
    data_df_2['Frequency'] = (data_df_2['index'] + 1)/(len(data_df_2)+1)
    data_df_2['RP'] = 1/data_df_2['Frequency']
    #print(data_df_2)
     
    data_df = data_df_original[['Max_{dur}'.format(dur = duration)]]
    data_df = remove_outliers(data_df, duration)
    mean = data_df.iloc[:,0].mean()
    data = data_df.values.ravel()
    #print(data)
     
    #MY_DISTRIBUTIONS = [st.norm, st.lognorm, st.genextreme, st.gumbel_r]
    
    results = fit_data(data, MY_DISTRIBUTIONS) ## Usar qdo nao eh GEV para INMET_aut
    df_parameters = get_parameters(data, results, 5) ## Usar qdo nao eh GEV para INMET_aut
    #df_parameters = pd.read_csv('Results/INMET_aut_GEV_params.csv') ##Usar qdo eh GEV para INMET_aut
    #print(df_parameters)
    #input()
    
    dist_n = dist 
    dist = MY_DISTRIBUTIONS[0]
    c = df_parameters['c'][0]
    loc = df_parameters['loc'][0]
    scale = df_parameters['scale'][0]
    #print(c, loc, scale)
    
    if math.isnan(c) == True:
        prob_function_obj = dist(loc, scale)
    else:
        prob_function_obj = dist(c, loc, scale)

    #x_in = np.linspace(0,1,200)
    RP_list = return_period_list
    #print(RP_list)
    probabilities = [1 - 1/RP for RP in RP_list]
    #print(probabilities)
    precipitations_dist = prob_function_obj.ppf(probabilities)
    #print(y_out)
    if plot_graph == True:   
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(RP_list, precipitations_dist) # graphically check the results of the inverse CDF
        ax.plot(data_df_2['RP'], data_df_2['Max_{dur}'.format(dur = duration)])
        ax.set(ylabel = 'Precipitation (mm)', xlabel = 'Return Period (Years)', title = dist_n)
        #ax.grid(color = 'gray')
        #ax.set_facecolor('white')
        #plt.show()
        fig.savefig('Graphs/distributions/quantile_plot_{n}_{disag}_{dis}_subdaily_{dur}.png'.format(n = name_file, dis = dist_n, dur = duration, disag = disag_factor))

    return precipitations_dist


def get_precipitations_allRP(name_file, duration, MY_DISTRIBUTIONS, dist, directory, disag_factor = 'nan'):
    
    return_period_list = [1.1, 2, 5, 10, 25, 50, 100]
    
    precipitations_dist = get_theoretical_max_precipitations(name_file, duration, MY_DISTRIBUTIONS, return_period_list, dist, directory, disag_factor)
    
    P_2Years = precipitations_dist[1]   
    P_5Years = precipitations_dist[2]   
    P_10Years = precipitations_dist[3]
    P_25Years = precipitations_dist[4]
    P_50Years = precipitations_dist[5]
    P_100Years = precipitations_dist[6]
    
    return P_2Years, P_5Years, P_10Years, P_25Years, P_50Years, P_100Years


def get_precipitation_byRP(name_file, duration, MY_DISTRIBUTIONS, dist, return_period_list, directory, disag_factor = 'nan'):
    
    precipitations_dist = get_theoretical_max_precipitations(name_file, duration, MY_DISTRIBUTIONS, return_period_list, dist, directory, disag_factor)
    P_ = precipitations_dist[0]
    
    return P_  


def get_idf_table(name_file, MY_DISTRIBUTIONS, dist, directory = 'Results', disag_factor = 'nan', save_table = False):
    
    if disag_factor == 'nan':   
        duration_list_min = [60, 180, 360, 480, 600, 720, 1440]
        duration_list = [1, 3, 6, 8, 10, 12, 24]
    
    else:
        duration_list_min = [5, 10, 20, 30, 60, 360, 480, 600, 720, 1440]
        duration_list = ['5min', '10min', '20min', '30min', '1', '6', '8', '10', '12', '24']

    P_RP_2Years = []
    P_RP_5Years = []
    P_RP_10Years = []
    P_RP_25Years = []
    P_RP_50Years = []
    P_RP_100Years = []
    
    for duration in duration_list:
        P_2Years, P_5Years, P_10Years, P_25Years, P_50Years, P_100Years = get_precipitations_allRP(name_file, duration, MY_DISTRIBUTIONS, dist, directory, disag_factor)
        P_RP_2Years.append(P_2Years)
        P_RP_5Years.append(P_5Years)
        P_RP_10Years.append(P_10Years)
        P_RP_25Years.append(P_25Years)
        P_RP_50Years.append(P_50Years)
        P_RP_100Years.append(P_100Years)

    #print(P_RP_2Years)
    i_RP_2Years = [P*60/d for P, d in zip(P_RP_2Years, duration_list_min)] 
    #print(i_RP_2Years)
    #input()
    ln_i_RP_2Years = [np.log(i) for i in i_RP_2Years]
    #print(ln_i_RP_2Years)
    #input()
    i_RP_5Years = [P*60/d for P, d in zip(P_RP_5Years, duration_list_min)] 
    ln_i_RP_5Years = [np.log(i) for i in i_RP_5Years]
    
    i_RP_10Years = [P*60/d for P, d in zip(P_RP_10Years, duration_list_min)] 
    ln_i_RP_10Years = [np.log(i) for i in i_RP_10Years]
    
    i_RP_25Years = [P*60/d for P, d in zip(P_RP_25Years, duration_list_min)] 
    ln_i_RP_25Years = [np.log(i) for i in i_RP_25Years]
    
    i_RP_50Years = [P*60/d for P, d in zip(P_RP_50Years, duration_list_min)] 
    ln_i_RP_50Years = [np.log(i) for i in i_RP_50Years]
    
    i_RP_100Years = [P*60/d for P, d in zip(P_RP_100Years, duration_list_min)] 
    ln_i_RP_100Years = [np.log(i) for i in i_RP_100Years]
    
    if save_table == True:  ##Quando for arrumar o codigo eu posso tirar o ln, o return e o save_table pq essa funcao serve soh para me dar a tabela. Nao uso ela dps.
        dict_ = {'duration' : duration_list_min,
                'P_RP_2Years' : P_RP_2Years, 
                'P_RP_5Years' : P_RP_5Years, 
                'P_RP_10Years' : P_RP_10Years, 
                'P_RP_25Years' : P_RP_25Years,
                'P_RP_50Years' : P_RP_50Years,
                'P_RP_100Years' : P_RP_100Years, 
                'i_RP_2Years' : i_RP_2Years,
                'i_RP_5Years' : i_RP_5Years,
                'i_RP_10Years' : i_RP_10Years,
                'i_RP_25Years' : i_RP_25Years,
                'i_RP_50Years' : i_RP_50Years,
                'i_RP_100Years' : i_RP_100Years
            }
        
        df = pd.DataFrame(dict_)
        #print(df)
        if disag_factor == 'bl':
            name_file = 'complete_{n}'.format(n = name_file)
            
        df.to_csv('{d}/IDFsubdaily_table_{n}_{dis}_{disag}.csv'.format(n = name_file, dis = dist, d = directory, disag = disag_factor), index = False)
    
    return duration_list_min, ln_i_RP_2Years, ln_i_RP_5Years, ln_i_RP_10Years, ln_i_RP_25Years, ln_i_RP_50Years, ln_i_RP_100Years


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
    name_file = 'complete_MIROC5_baseline_MD'
    print(name_file)
    directory = 'GCM_data//bias_correction//gcm'
    dist_list = ['Lognormal', 'GEV', 'Gumbel', 'GenLogistic']
    disag_factor = ''

    for dist in dist_list:
        print('--> ', dist, disag_factor)
        if dist == 'Normal':
            MY_DISTRIBUTIONS = [st.norm]  ##Para INMET_aut GEV peguei os valores dos parametros do R
        if dist == 'GEV':
            MY_DISTRIBUTIONS = [st.genextreme]
        if dist == 'Lognormal':
            MY_DISTRIBUTIONS = [st.lognorm]
        if dist == 'GenLogistic':
            MY_DISTRIBUTIONS = [st.genlogistic]
        if dist == 'Gumbel':
            MY_DISTRIBUTIONS = [st.gumbel_r]
                    
        return_period_list = [1.1, 2, 5, 10, 25, 50, 100]
        return_period_list = [1.1, 2, 5, 10, 25, 50, 100]  ## Default return_period_list for get_idf_table
         
        get_theoretical_max_precipitations(name_file, 1, MY_DISTRIBUTIONS, return_period_list, dist, directory, disag_factor, plot_graph = True)
        #get_theoretical_max_precipitations(name_file, 3, MY_DISTRIBUTIONS, return_period_list, dist, disag_factor, plot_graph = True)
        get_theoretical_max_precipitations(name_file, 6, MY_DISTRIBUTIONS, return_period_list, dist, directory, disag_factor, plot_graph = True)
        get_theoretical_max_precipitations(name_file, 8, MY_DISTRIBUTIONS, return_period_list, dist, directory, disag_factor, plot_graph = True)
        get_theoretical_max_precipitations(name_file, 10, MY_DISTRIBUTIONS, return_period_list, dist, directory, disag_factor, plot_graph = True)
        get_theoretical_max_precipitations(name_file, 12, MY_DISTRIBUTIONS, return_period_list, dist, directory, disag_factor, plot_graph = True)
        get_theoretical_max_precipitations(name_file, 24, MY_DISTRIBUTIONS, return_period_list, dist, directory, disag_factor, plot_graph = True)
     
        get_idf_table(name_file, MY_DISTRIBUTIONS, dist, directory, disag_factor, save_table = True)
     
        t0, n, K, m = get_final_idf_params(name_file, MY_DISTRIBUTIONS, dist, directory, disag_factor)    
        print('K: ', K)
        print('t0: ', t0)
        print(' m: ', m)
        print('n: ', n)
        print('')


