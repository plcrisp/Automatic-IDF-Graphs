import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import bias_correction

def get_idf_table_calc_cc(station_name, disag_factor, dist_type, downscaling_method, save_table = True):

    if downscaling_method == 'EQM':
        df_IDF_params = pd.read_csv('IDF_params_subdaily.csv')
        disag_factor = '_' + disag_factor
        station_name = station_name + '_EQM_baseline'

    else:
        if disag_factor == 'bl':
            df_IDF_params = pd.read_csv('IDF_params_subdaily.csv')
            disag_factor = '_' + disag_factor
            station_name = station_name + '_' + downscaling_method + '_baseline'
            
        else:
            df_IDF_params = pd.read_csv('IDF_params_subdaily.csv')
            disag_factor = '_' + disag_factor
            station_name = station_name + '_' + downscaling_method + '_baseline'
    
    #print(df_IDF_params)
    #print(station_name)
    #print(disag_factor)
    #print(dist_type)
    df_selected = df_IDF_params.loc[df_IDF_params['Station'] == station_name]
    df_selected = df_selected.loc[df_selected['Disag_factors'] == disag_factor]
    df_selected = df_selected.loc[df_selected['Dist_Type'] == dist_type]
    df_selected = df_selected.reset_index()
    #print(disag_factor)
    #print(dist_type)
    #print(df_selected)
    K = df_selected['K'][0]
    t0 = df_selected['t0'][0]
    m = df_selected['m'][0]
    n = df_selected['n'][0]
    #input()
    return_period_list = [2, 5, 10, 25, 50, 100]
    if disag_factor == 'nan':
        duration_list = [10, 20, 30, 60, 180, 360, 480, 600, 720, 1440]
    else:
        duration_list = [5, 10, 20, 30, 60, 180, 360, 480, 600, 720, 1440]
    
    i_final = []
    P_final = []
    for RP in return_period_list:
        #print(RP)
        i_list = []
        P_list = []
        for d in duration_list:
            i_prec =  K*RP**m/(d + t0)**n
            i_list.append(i_prec)
            P = i_prec * d/60
            P_list.append(P)
        #print(i_list)
        #print(P_list)
        i_final.append(i_list)
        P_final.append(P_list)
     
    df_intensity = pd.DataFrame(i_final)
    df_intensity = df_intensity.transpose()
    df_intensity.columns = ['i_RP_2', 'i_RP_5', 'i_RP_10', 'i_RP_25', 'i_RP_50', 'i_RP_100']
    df_intensity['d'] = duration_list
     
    df_precipitation = pd.DataFrame(P_final)
    df_precipitation = df_precipitation.transpose()
    df_precipitation.columns = ['P_RP_2', 'P_RP_5', 'P_RP_10', 'P_RP_25', 'P_RP_50', 'P_RP_100']
    df_precipitation['d'] = duration_list
    
    #print(df_idf_table)
    if save_table == True:
        df_precipitation.to_csv('GCM_data/IDF_tables_calc/{n}{dis}_{dist}_precipitationfromIDF_subdaily.csv'.format(n = station_name, dis = disag_factor, dist = dist_type), index = False)
        df_intensity.to_csv('GCM_data/IDF_tables_calc/{n}{dis}_{dist}_intensityfromIDF_subdaily.csv'.format(n = station_name, dis = disag_factor, dist = dist_type), index = False)
    
    return df_intensity, df_precipitation

def get_idf_table_calc_hist(station_name, disag_factor, dist_type, data_type, save_table = True):
    
    if data_type == 'subdaily':
        df_IDF_params = pd.read_csv('IDF_params_subdaily.csv')
        disag_factor2 = '_' + disag_factor
        
    else:
        if disag_factor == 'ger':
            df_IDF_params = pd.read_csv('IDF_params.csv')
            disag_factor2 = 'original'
            
        elif disag_factor == 'p0.2':
            df_IDF_params = pd.read_csv('IDF_params.csv')
            disag_factor2 = 'p_20'
            
        elif disag_factor == 'm0.2':
            df_IDF_params = pd.read_csv('IDF_params.csv')
            disag_factor2 = 'm_20'
            
        elif disag_factor == 'otimizado':
            df_IDF_params = pd.read_csv('IDF_params.csv')
            disag_factor2 = 'opt'
                            
    #print(df_IDF_params)
    #print(station_name)
    #print(disag_factor)
    #print(dist_type)
    df_selected = df_IDF_params.loc[df_IDF_params['Station'] == station_name]
    df_selected = df_selected.loc[df_selected['Disag_factors'] == disag_factor2]
    df_selected = df_selected.loc[df_selected['Dist_Type'] == dist_type]
    df_selected = df_selected.reset_index()
    #print(disag_factor)
    #print(dist_type)
    #print(df_selected)
    K = df_selected['K'][0]
    t0 = df_selected['t0'][0]
    m = df_selected['m'][0]
    n = df_selected['n'][0]
    #input()
    return_period_list = [2, 5, 10, 25, 50, 100]
    if disag_factor == 'nan':
        duration_list = [10, 20, 30, 60, 180, 360, 480, 600, 720, 1440]
    else:
        duration_list = [5, 10, 20, 30, 60, 180, 360, 480, 600, 720, 1440]
    
    i_final = []
    P_final = []
    for RP in return_period_list:
        #print(RP)
        i_list = []
        P_list = []
        for d in duration_list:
            i_prec =  K*RP**m/(d + t0)**n
            i_list.append(i_prec)
            P = i_prec * d/60
            P_list.append(P)
        #print(i_list)
        #print(P_list)
        i_final.append(i_list)
        P_final.append(P_list)
     
    df_intensity = pd.DataFrame(i_final)
    df_intensity = df_intensity.transpose()
    df_intensity.columns = ['i_RP_2', 'i_RP_5', 'i_RP_10', 'i_RP_25', 'i_RP_50', 'i_RP_100']
    df_intensity['d'] = duration_list
     
    df_precipitation = pd.DataFrame(P_final)
    df_precipitation = df_precipitation.transpose()
    df_precipitation.columns = ['P_RP_2', 'P_RP_5', 'P_RP_10', 'P_RP_25', 'P_RP_50', 'P_RP_100']
    df_precipitation['d'] = duration_list
    
    #print(df_idf_table)
    if save_table == True:
        df_precipitation.to_csv('Results/IDF_tables_calc/{n}_{dis}_{dist}_precipitationfromIDF_{dtype}.csv'.format(n = station_name, dis = disag_factor, dist = dist_type, dtype = data_type), index = False)
        df_intensity.to_csv('Results/IDF_tables_calc/{n}_{dis}_{dist}_intensityfromIDF_{dtype}.csv'.format(n = station_name, dis = disag_factor, dist = dist_type, dtype = data_type), index = False)
    
    return df_intensity, df_precipitation

if __name__ == '__main__':  

# # HISTORICAL DATA
#     station_name_list = ['INMET_conv', 'INMET_aut', 'Base_IDF', 'Average']
# 
#     dist_type_list = ['Gumbel', 'GenLogistic', 'Normal', 'GEV', 'Lognormal']
# 
#     disag_factor_list = ['bl', 'ger', 'm0.2', 'p0.2', 'otimizado', 'nan']
#     
#     data_type_list = ['daily', 'subdaily']
# 
#     for station_name in station_name_list:
#         for dist_type in dist_type_list:
#             for disag_factor in disag_factor_list:
#                 for data_type in data_type_list:
#                     try:
#                         i_observed, p_observed = get_idf_table_calc_hist(station_name, disag_factor, dist_type, data_type, save_table = True)
#                     except:
#                         pass

## CLIMATE CHANGE
    station_name_list = ['HADGEM', 'MIROC5']
 
    dist_type_list = ['Gumbel', 'GenLogistic', 'Normal', 'GEV', 'Lognormal']
 
    disag_factor_list = ['bl', 'ger', 'm0.2', 'p0.2']
     
    downscaling_method_list = ['EQM', 'DBC', 'MD', 'PT', 'QM']
 
    for station_name in station_name_list:
        for dist_type in dist_type_list:
            for disag_factor in disag_factor_list:
                for downscaling_method in downscaling_method_list:
                    try:
                        i_observed, p_observed = get_idf_table_calc_cc(station_name, disag_factor, dist_type, downscaling_method, save_table = True)
                    except:
                        pass
    
#     station_name = 'HADGEM'
#     disag_factor = 'bl'
#     dist_type = 'Gumbel'
#     downscaling_method = 'QM'
#     
#     i_observed, p_observed = get_idf_table_calc_cc(station_name, disag_factor, dist_type, downscaling_method, save_table = True)

    print('Done!')
