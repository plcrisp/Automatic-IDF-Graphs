import pandas as pd

def get_RMSE(y_obs_list, y_mod_list):
    #root mean square error
    
    MSE_list = []   
    for i in range(len(y_obs_list)):
        y_mod = y_mod_list[i]
        y_obs = y_obs_list[i]
        MSE_i = ((y_mod-y_obs)**2)
        MSE_list.append(MSE_i)
    
    RMSE = (sum(MSE_list)/(len(y_obs_list)))**(1/2)
    return RMSE

def get_MARE(y_obs_list, y_mod_list):
    #mean absolute relative error
    
    ARE_list = []
    for i in range(len(y_obs_list)):
        y_mod = y_mod_list[i]
        y_obs = y_obs_list[i]
        ARE_i = abs(y_mod - y_obs)/y_obs
        ARE_list.append(ARE_i)
    
    MARE = sum(ARE_list)/len(y_obs_list)
    return MARE
    
def get_MBE(y_obs_list, y_mod_list):
    #mean bias error
    
    BE_list = []
    for i in range(len(y_obs_list)):
        y_mod = y_mod_list[i]
        y_obs = y_obs_list[i]
        BE_i = (y_mod - y_obs)
        BE_list.append(BE_i)
    
    MBE = sum(BE_list)/len(y_obs_list)
    return MBE

def get_R(y_obs_list, y_mod_list):
    #correlation coefficient
    
    num1 = []
    den1 = []
    #num2 = []
    #den2 = []
    
    mean_y_obs = sum(y_obs_list)/len(y_obs_list)
    for i in range(len(y_obs_list)):
        y_mod = y_mod_list[i]
        y_obs = y_obs_list[i]
        num1_i = (y_obs - y_mod)**2
        den1_i = (y_obs - mean_y_obs)**2
        num1.append(num1_i)
        den1.append(den1_i)
        #num2_i = (y_mod - mean_y_obs)**2
        #den2_i = (y_obs - mean_y_obs)**2
        #num2.append(num2_i)
        #den2.append(den2_i)
        
    rsquare1 = 1 - (sum(num1)/sum(den1))
    r = rsquare1**(1/2)
    #r = rsquare1
    #rsquare2 = sum(num2)/sum(den2)  
    #r2 = rsquare2**(1/2)
    return r

def get_R_2(y_obs_list, y_mod_list):
    # correlation coefficient calculation 2
    
    num1 = []
    den1 = []
    #num2 = []
    den2 = []
    
    mean_y_obs = sum(y_obs_list)/len(y_obs_list)
    mean_y_mod = sum(y_mod_list)/len(y_mod_list)
    for i in range(len(y_obs_list)):
        y_mod = y_mod_list[i]
        y_obs = y_obs_list[i]
        num1_i = (y_obs - mean_y_obs)*(y_mod - mean_y_mod)
        den1_i = (y_obs - mean_y_obs)**2
        den2_i = (y_mod - mean_y_mod)**2
        num1.append(num1_i)
        den1.append(den1_i)
        #num2_i = (y_mod - mean_y_obs)**2
        #den2_i = (y_obs - mean_y_obs)**2
        #num2.append(num2_i)
        den2.append(den2_i)


        
    den_final = ((sum(den1))**(1/2))*(sum(den2)**(1/2))    
    r = sum(num1)/den_final

#     print(y_obs_list)
#     print(y_mod_list)
#     print(num1)
#     print(den1)
#     print(den2)    
#     print(den_final)
#     print(sum(num1))
#     print(r)
#     input()
    
    return r    

def get_IPE(RMSE, RMSE_max, MARE, MARE_max, MBE, MBE_max, R, R_min):
    #Ideal point error
    
    IPE = (0.25*((RMSE/RMSE_max)**2 + (MARE/MARE_max)**2 + abs(MBE/abs(MBE_max))**2 + ((R - 1)/R_min)**2))**(1/2)
    return IPE

def get_IPE2(RMSE, RMSE_max, MARE, MARE_max, MBE, MBE_max, R, R_max):
    #Ideal point error
    
    IPE = (0.25*((RMSE/RMSE_max)**2 + (MARE/MARE_max)**2 + abs(MBE/abs(MBE_max))**2 + ((R - 1)/(1/R_max))**2))**(1/2)
    return IPE


def get_list_to_compare(idf_to_compare, return_period, station_name, disag_factor, dist_type, data_type, period):
    if idf_to_compare == 'base':
        df_intensity_base = pd.read_csv('Results/IDF_tables_calc/Base_IDF_ger_Gumbel_intensityfromIDF_daily.csv')
    elif idf_to_compare == 'average':
        df_intensity_base = pd.read_csv('Results/IDF_tables_calc/Average_ger_GenLogistic_intensityfromIDF_daily.csv')
    elif idf_to_compare == 'inmet_aut_nan':
        df_intensity_base = pd.read_csv('Results/IDF_tables_calc/INMET_aut_nan_GenLogistic_intensityfromIDF_subdaily.csv')        
    else:
        print('IDF to compare not defined..')

    if period == 'historical_obs':
        df_intensity_calc = pd.read_csv('Results/IDF_tables_calc_correct/{n}_{disag}_{dist}_intensityfromIDF_{dtype}.csv'.format(n = station_name, disag = disag_factor, dist = dist_type, dtype = data_type))
    elif period == 'historical_proj':
        df_intensity_calc = pd.read_csv('GCM_data/IDF_tables_calc/{n}_{disag}_{dist}_intensityfromIDF_{dtype}.csv'.format(n = station_name, disag = disag_factor, dist = dist_type, dtype = data_type))
    else:
        print('Period not defined..')
    #print(df_intensity_base)
    #print(df_intensity_calc)
    #input()
    name_column_selection = 'i_RP_' + return_period

    y_obs_list = df_intensity_base[name_column_selection].to_list()
    y_mod_list = df_intensity_calc[name_column_selection].to_list()
    
    if idf_to_compare == 'inmet_aut_nan':
        #print('caiu aqui')
        y_obs_list = y_obs_list[3:].copy()
        y_mod_list = y_mod_list[4:].copy()
        
    #print(y_obs_list)
    #print(y_mod_list)
    #print(len(y_obs_list))
    #print(len(y_mod_list))
    #input()
    return y_obs_list, y_mod_list


def get_errors_df(idf_to_compare, station_name_list, disag_factor_list, dist_type_list, data_type_list, period_list, return_period_list):
    RMSE_list = []
    MARE_list = []
    MBE_list = []
    R_list = []
    station_name_list_todf = []
    disag_factor_list_todf= []
    dist_type_list_todf = []
    data_type_list_todf = []
    period_list_todf = []
    return_period_list_todf = []
    
    for station_name in station_name_list:
        for disag_factor in disag_factor_list:
            for dist_type in dist_type_list:
                for data_type in data_type_list:
                    for period in period_list:
    
                        for return_period in return_period_list:
                            try:
                                y_mod_list, y_obs_list = get_list_to_compare(idf_to_compare, return_period, station_name, disag_factor, dist_type, data_type, period) 
                                 
                                RMSE = get_RMSE(y_mod_list, y_obs_list)
                                MARE = get_MARE(y_obs_list, y_mod_list)
                                MBE = get_MBE(y_obs_list, y_mod_list)
                                #R = get_R(y_obs_list, y_mod_list)
                                R = get_R_2(y_obs_list, y_mod_list)
                                #print(RMSE, MARE, MBE, R)
                                RMSE_list.append(RMSE)
                                MARE_list.append(MARE)
                                MBE_list.append(MBE)
                                R_list.append(R)
                                station_name_list_todf.append(station_name)
                                disag_factor_list_todf.append(disag_factor)
                                dist_type_list_todf.append(dist_type)
                                data_type_list_todf.append(data_type)
                                period_list_todf.append(period)
                                return_period_list_todf.append(return_period)
                            except:
                                #print('pass')
                                pass
    
        
    error_dict = {'station_name': station_name_list_todf,
            'disag_factor': disag_factor_list_todf,
            'dist_type' : dist_type_list_todf,
            'data_type': data_type_list_todf,
            'period': period_list_todf,
            'return_period': return_period_list_todf,
            'RMSE': RMSE_list,
            'MARE': MARE_list,
            'MBE': MBE_list,
            'R': R_list
        }
    
    df_errors = pd.DataFrame(error_dict)
    
    return df_errors


if __name__ == '__main__':
    
#     y_obs_list = [1,2,3,4,5]
#     y_obs_list_2 = [1.5 ,2.5, 3.8, 4,5]
#     y_mod_list = [1.5, 1.5, 3.2, 4.8, 6]
#     
#     R = get_R(y_obs_list, y_mod_list)
#     R2 = get_R_2(y_obs_list, y_mod_list)
#     R_2 = get_R(y_obs_list_2, y_mod_list)
#     R2_2 = get_R_2(y_obs_list_2, y_mod_list)
# 
#     
#     print(R, R2, R_2, R2_2)
    
    
    ## Creating errors dataframe        
    idf_to_compare = 'base'
    #idf_to_compare = 'average'
    #idf_to_compare = 'inmet_aut_nan'
    #station_name_list = ['INMET_conv', 'INMET_aut']
    station_name_list = ['HADGEM_DBC_baseline', 'HADGEM_EQM_baseline', 'HADGEM_MD_baseline', 'HADGEM_PT_baseline', 'HADGEM_QM_baseline', 'MIROC5_DBC_baseline', 'MIROC5_EQM_baseline', 'MIROC5_MD_baseline', 'MIROC5_PT_baseline', 'MIROC5_QM_baseline']    
    dist_type_list = ['Gumbel', 'GenLogistic', 'Normal', 'GEV', 'Lognormal']
    #disag_factor_list = ['bl', 'ger', 'm0.2', 'p0.2', 'otimizado', 'nan']   
    disag_factor_list = ['bl', 'ger', 'm0.2', 'p0.2', 'otimizado']   
    data_type_list = ['daily', 'subdaily']
    period_list = ['historical_proj']
    #period_list = ['historical_obs']
    return_period_list = ['2', '5', '10', '25', '50', '100']
     
    df_errors = get_errors_df(idf_to_compare, station_name_list, disag_factor_list, dist_type_list, data_type_list, period_list, return_period_list)
    print(df_errors)

    # calcule of IPE
    RMSE_list = df_errors['RMSE'].to_list()
    MARE_list = df_errors['MARE'].to_list()
    MBE_list = df_errors['MBE'].to_list()
    R_list = df_errors['R'].to_list()
    
    RMSE_max = max(RMSE_list)
    MARE_max = max(MARE_list)
    MBE_max = max(MBE_list, key = abs)
    R_max = max(R_list)
    R_min = min(R_list)
    
    IPE_list = []
    #IPE2_list = []
    
    for i in range(len(RMSE_list)):
        RMSE = RMSE_list[i]
        MARE = MARE_list[i]
        MBE = MBE_list[i]
        R = R_list[i]
        IPE = get_IPE(RMSE, RMSE_max, MARE, MARE_max, MBE, MBE_max, R, R_min)
        #IPE2 = get_IPE2(RMSE, RMSE_max, MARE, MARE_max, MBE, MBE_max, R, R_max)
        IPE_list.append(IPE)
        #IPE2_list.append(IPE2)
        
    df_errors['IPE'] = IPE_list
    #df_errors['IPE2'] = IPE2_list
    print(df_errors)    
    
    
    df_errors.to_csv('error_IDF_{period}_{comparison}.csv'.format(period = period_list[0], comparison = idf_to_compare), index = False, encoding = 'latin1')
        
    print('Done!')
 

