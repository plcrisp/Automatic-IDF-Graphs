from functions_treatment import *
from numpy import dtype

# ##BASELINE ANALYSIS
# print('---Baseline---')
# df_hadgem_md = pd.read_csv('GCM_data/bias_correction/HADGEM_baseline_MD_daily.csv')
# df_hadgem_pt = pd.read_csv('GCM_data/bias_correction/HADGEM_baseline_PT_daily.csv')
# df_hadgem_qm = pd.read_csv('GCM_data/bias_correction/HADGEM_baseline_QM_daily.csv')
# df_hadgem_qm_adj = pd.read_csv('GCM_data/bias_correction/HADGEM_baseline_QM_adj_daily.csv')
# df_hadgem_dbc = pd.read_csv('GCM_data/bias_correction/HADGEM_baseline_DBC_daily.csv')

# df_miroc_qm_adj = pd.read_csv('GCM_data/bias_correction/MIROC5_baseline_QM_adj_daily.csv')
# df_miroc_dbc = pd.read_csv('GCM_data/bias_correction/MIROC5_baseline_DBC_daily.csv')
# df_miroc_md = pd.read_csv('GCM_data/bias_correction/MIROC5_baseline_MD_daily.csv')
# df_miroc_pt = pd.read_csv('GCM_data/bias_correction/MIROC5_baseline_PT_daily.csv')
# df_miroc_qm = pd.read_csv('GCM_data/bias_correction/MIROC5_baseline_QM_daily.csv')
# 
# #P90
# print('--> P90')
# print('P90 HADGEM_MD / baseline: ', p90_function(df_hadgem_md))
# print('P90 HADGEM_PT / baseline: ', p90_function(df_hadgem_pt))
# #print('P90 HADGEM_QM / baseline: ', p90_function(df_hadgem_qm))
# print('P90 HADGEM_QM_adj / baseline: ', p90_function(df_hadgem_qm_adj))
# print('P90 MIROC5_QM_adj / baseline: ', p90_function(df_miroc_qm_adj))
# print('P90 HADGEM_DBC / baseline: ', p90_function(df_hadgem_dbc))
# print('P90 MIROC5_DBC / baseline: ', p90_function(df_miroc_dbc))
# print('')
# print('P90 MIROC5_MD / baseline: ', p90_function(df_miroc_md))
# print('P90 MIROC5_PT / baseline: ', p90_function(df_miroc_pt))
# print('P90 MIROC5_QM / baseline: ', p90_function(df_miroc_qm))
  
# #Aggregate to year
# def aggregate_to_csv_year(df, name, directory = 'GCM_data/bias_correction'):
#     df_yearly = aggregate(df, 'Year')
#     df_yearly.to_csv('{d}/{n}_yearly.csv'.format(n = name, d = directory), index = False)
#   
# aggregate_to_csv_year(df_hadgem_md, 'HADGEM_baseline_MD')
# aggregate_to_csv_year(df_hadgem_pt, 'HADGEM_baseline_PT')
# #aggregate_to_csv_year(df_hadgem_qm, 'HADGEM_baseline_QM')
#   
# aggregate_to_csv_year(df_miroc_md, 'MIROC5_baseline_MD')
# aggregate_to_csv_year(df_miroc_pt, 'MIROC5_baseline_PT')
# aggregate_to_csv_year(df_miroc_qm, 'MIROC5_baseline_QM')
#  
# #Calculate annual max daily precipitation
# def get_max_daily(df, name, directory = 'GCM_data/bias_correction'):
#     df_max = max_annual_precipitation(df)
#     df_max.to_csv('{d}/max_daily_{n}.csv'.format(n = name, d = directory), index = False)
# 
# print('Getting max daily..') 
# get_max_daily(df_hadgem_md, 'HADGEM_baseline_MD')
# get_max_daily(df_hadgem_pt, 'HADGEM_baseline_PT')
# get_max_daily(df_hadgem_qm_adj, 'HADGEM_baseline_QM')
# get_max_daily(df_hadgem_dbc, 'HADGEM_baseline_DBC')
#   
# #get_max_daily(df_miroc_md, 'MIROC5_baseline_MD')
# #get_max_daily(df_miroc_pt, 'MIROC5_baseline_PT')
# get_max_daily(df_miroc_qm_adj, 'MIROC5_baseline_QM')
# get_max_daily(df_miroc_dbc, 'MIROC5_baseline_DBC')
#   
# #Trend analysis
# print('')
# print('--> Trend analysis')
# alpha_value = 0.05
# group = 'GCM_baseline'
# sites_list = ['HADGEM_baseline_MD', 'HADGEM_baseline_PT', 'HADGEM_baseline_QM', 'MIROC5_baseline_MD', 'MIROC5_baseline_PT', 'MIROC5_baseline_QM']
# print('- Annual precipitation')
# get_trend('Year', sites_list, alpha_value, group, 'mod')
# print('')
# print('- Max_daily')
# get_trend('Max_daily', sites_list, alpha_value, group, 'mod')
  
# #Calculate subdaily max from disagreggation factors
# print('')
# print('Getting max subdaily from disag factors..') 
# df_hadgem_md_dmax = pd.read_csv('GCM_data/bias_correction/max_daily_HADGEM_baseline_MD.csv')
# df_hadgem_pt_dmax = pd.read_csv('GCM_data/bias_correction/max_daily_HADGEM_baseline_PT.csv')
# df_hadgem_qm_dmax = pd.read_csv('GCM_data/bias_correction/max_daily_HADGEM_baseline_QM.csv')
# df_hadgem_dbc_dmax = pd.read_csv('GCM_data/bias_correction/max_daily_HADGEM_baseline_DBC.csv')
# 
# df_miroc_md_dmax = pd.read_csv('GCM_data/bias_correction/max_daily_MIROC5_baseline_MD.csv')
# df_miroc_pt_dmax = pd.read_csv('GCM_data/bias_correction/max_daily_MIROC5_baseline_PT.csv')
# df_miroc_qm_dmax = pd.read_csv('GCM_data/bias_correction/max_daily_MIROC5_baseline_QM.csv')
# df_miroc_dbc_dmax = pd.read_csv('GCM_data/bias_correction/max_daily_MIROC5_baseline_DBC.csv')
# 
# var_value = 0.2
# 
# get_subdaily_from_disagregation_factors(df_hadgem_md_dmax, 'original', var_value, 'HADGEM_baseline_MD', 'GCM_data/bias_correction')
# get_subdaily_from_disagregation_factors(df_hadgem_md_dmax, 'plus', var_value, 'HADGEM_baseline_MD', 'GCM_data/bias_correction')
# get_subdaily_from_disagregation_factors(df_hadgem_md_dmax, 'minus', var_value, 'HADGEM_baseline_MD', 'GCM_data/bias_correction')
# 
# get_subdaily_from_disagregation_factors(df_hadgem_pt_dmax, 'original', var_value, 'HADGEM_baseline_PT', 'GCM_data/bias_correction')
# get_subdaily_from_disagregation_factors(df_hadgem_pt_dmax, 'plus', var_value, 'HADGEM_baseline_PT', 'GCM_data/bias_correction')
# get_subdaily_from_disagregation_factors(df_hadgem_pt_dmax, 'minus', var_value, 'HADGEM_baseline_PT', 'GCM_data/bias_correction')
# 
# get_subdaily_from_disagregation_factors(df_hadgem_qm_dmax, 'original', var_value, 'HADGEM_baseline_QM', 'GCM_data/bias_correction')
# get_subdaily_from_disagregation_factors(df_hadgem_qm_dmax, 'plus', var_value, 'HADGEM_baseline_QM', 'GCM_data/bias_correction')
# get_subdaily_from_disagregation_factors(df_hadgem_qm_dmax, 'minus', var_value, 'HADGEM_baseline_QM', 'GCM_data/bias_correction')
# 
# get_subdaily_from_disagregation_factors(df_hadgem_dbc_dmax, 'original', var_value, 'HADGEM_baseline_DBC', 'GCM_data/bias_correction')
# get_subdaily_from_disagregation_factors(df_hadgem_dbc_dmax, 'plus', var_value, 'HADGEM_baseline_DBC', 'GCM_data/bias_correction')
# get_subdaily_from_disagregation_factors(df_hadgem_dbc_dmax, 'minus', var_value, 'HADGEM_baseline_DBC', 'GCM_data/bias_correction')
# 
# get_subdaily_from_disagregation_factors(df_miroc_md_dmax, 'original', var_value, 'MIROC5_baseline_MD', 'GCM_data/bias_correction')
# get_subdaily_from_disagregation_factors(df_miroc_md_dmax, 'plus', var_value, 'MIROC5_baseline_MD', 'GCM_data/bias_correction')
# get_subdaily_from_disagregation_factors(df_miroc_md_dmax, 'minus', var_value, 'MIROC5_baseline_MD', 'GCM_data/bias_correction')
# 
# get_subdaily_from_disagregation_factors(df_miroc_pt_dmax, 'original', var_value, 'MIROC5_baseline_PT', 'GCM_data/bias_correction')
# get_subdaily_from_disagregation_factors(df_miroc_pt_dmax, 'plus', var_value, 'MIROC5_baseline_PT', 'GCM_data/bias_correction')
# get_subdaily_from_disagregation_factors(df_miroc_pt_dmax, 'minus', var_value, 'MIROC5_baseline_PT', 'GCM_data/bias_correction')
# 
# get_subdaily_from_disagregation_factors(df_miroc_qm_dmax, 'original', var_value, 'MIROC5_baseline_QM', 'GCM_data/bias_correction')
# get_subdaily_from_disagregation_factors(df_miroc_qm_dmax, 'plus', var_value, 'MIROC5_baseline_QM', 'GCM_data/bias_correction')
# get_subdaily_from_disagregation_factors(df_miroc_qm_dmax, 'minus', var_value, 'MIROC5_baseline_QM', 'GCM_data/bias_correction')
# 
# get_subdaily_from_disagregation_factors(df_miroc_dbc_dmax, 'original', var_value, 'MIROC5_baseline_DBC', 'GCM_data/bias_correction')
# get_subdaily_from_disagregation_factors(df_miroc_dbc_dmax, 'plus', var_value, 'MIROC5_baseline_DBC', 'GCM_data/bias_correction')
# get_subdaily_from_disagregation_factors(df_miroc_dbc_dmax, 'minus', var_value, 'MIROC5_baseline_DBC', 'GCM_data/bias_correction')

#Calculate subdaily max from bartlett lewis
# print('')
# print('Getting subdaily max from bartlett lewis..')
#HADGEM_baseline_MD = pd.read_csv('GCM_data/bias_correction/gcm/HADGEM_baseline_MD_disag.csv')
#aggregate_to_csv(HADGEM_baseline_MD, 'HADGEM_baseline_MD', directory='GCM_data/bias_correction/gcm')
#aggregate_hourly_to_csv(HADGEM_baseline_md, 'HADGEM_baseline_MD', directory='GCM_data/bias_correction/gcm')
#get_max_subdaily_table('HADGEM_baseline_MD', directory = 'GCM_data/bias_correction/gcm')
#get_max_subdaily_min_table('HADGEM_baseline_MD', 5, directory='GCM_data/bias_correction/gcm')
#merge_max_tables('HADGEM_baseline_MD', directory = 'GCM_data/bias_correction/gcm')    
# 
# print('')
# HADGEM_baseline_PT = pd.read_csv('GCM_data/bias_correction/gcm/HADGEM_baseline_PT_disag.csv', na_values = {'     NA'})
# aggregate_to_csv(HADGEM_baseline_PT, 'HADGEM_baseline_PT', directory='GCM_data/bias_correction/gcm')
# aggregate_hourly_to_csv(HADGEM_baseline_PT, 'HADGEM_baseline_PT', directory='GCM_data/bias_correction/gcm')
# get_max_subdaily_table('HADGEM_baseline_PT', directory = 'GCM_data/bias_correction/gcm')
# get_max_subdaily_min_table('HADGEM_baseline_PT', 5, directory='GCM_data/bias_correction/gcm')
# merge_max_tables('HADGEM_baseline_PT', directory = 'GCM_data/bias_correction/gcm')    
# print('Done HADGEM PT') 
#  
# print('')
# MIROC5_baseline_MD = pd.read_csv('GCM_data/bias_correction/gcm/MIROC5_baseline_MD_disag.csv')
# aggregate_to_csv(MIROC5_baseline_MD, 'MIROC5_baseline_MD', directory='GCM_data/bias_correction/gcm')
# aggregate_hourly_to_csv(MIROC5_baseline_MD, 'MIROC5_baseline_MD', directory='GCM_data/bias_correction/gcm')
# get_max_subdaily_table('MIROC5_baseline_MD', directory = 'GCM_data/bias_correction/gcm')
# get_max_subdaily_min_table('MIROC5_baseline_MD', 5, directory='GCM_data/bias_correction/gcm')
# merge_max_tables('MIROC5_baseline_MD', directory = 'GCM_data/bias_correction/gcm')    
# print('Done MIROC5 MD')
# 
# print('') 
# MIROC5_baseline_PT = pd.read_csv('GCM_data/bias_correction/gcm/MIROC5_baseline_PT_disag.csv', dtype = {'Precipitation': np.float64}, na_values = {'     NA'} )
# # print(MIROC5_baseline_PT.dtypes)
# # input()
# aggregate_to_csv(MIROC5_baseline_PT, 'MIROC5_baseline_PT', directory='GCM_data/bias_correction/gcm')
# aggregate_hourly_to_csv(MIROC5_baseline_PT, 'MIROC5_baseline_PT', directory='GCM_data/bias_correction/gcm')
# get_max_subdaily_table('MIROC5_baseline_PT', directory = 'GCM_data/bias_correction/gcm')
# get_max_subdaily_min_table('MIROC5_baseline_PT', 5, directory='GCM_data/bias_correction/gcm')
# merge_max_tables('MIROC5_baseline_PT', directory = 'GCM_data/bias_correction/gcm')    
# print('Done MIROC5 PT')

# print('')
# HADGEM_baseline_DBC = pd.read_csv('GCM_data/bias_correction/gcm/HADGEM_baseline_DBC_disag.csv', na_values = {'     NA'})
# aggregate_to_csv(HADGEM_baseline_DBC, 'HADGEM_baseline_DBC', directory='GCM_data/bias_correction/gcm')
# aggregate_hourly_to_csv(HADGEM_baseline_DBC, 'HADGEM_baseline_DBC', directory='GCM_data/bias_correction/gcm')
# get_max_subdaily_table('HADGEM_baseline_DBC', directory = 'GCM_data/bias_correction/gcm')
# get_max_subdaily_min_table('HADGEM_baseline_DBC', 5, directory='GCM_data/bias_correction/gcm')
# merge_max_tables('HADGEM_baseline_DBC', directory = 'GCM_data/bias_correction/gcm')    
# print('Done HADGEM DBC') 
# 
# MIROC5_baseline_DBC = pd.read_csv('GCM_data/bias_correction/gcm/MIROC5_baseline_DBC_disag.csv', dtype = {'Precipitation': np.float64}, na_values = {'     NA'} )
# # print(MIROC5_baseline_DBC.dtypes)
# # input()
# aggregate_to_csv(MIROC5_baseline_DBC, 'MIROC5_baseline_DBC', directory='GCM_data/bias_correction/gcm')
# aggregate_hourly_to_csv(MIROC5_baseline_DBC, 'MIROC5_baseline_DBC', directory='GCM_data/bias_correction/gcm')
# get_max_subdaily_table('MIROC5_baseline_DBC', directory = 'GCM_data/bias_correction/gcm')
# get_max_subdaily_min_table('MIROC5_baseline_DBC', 5, directory='GCM_data/bias_correction/gcm')
# merge_max_tables('MIROC5_baseline_DBC', directory = 'GCM_data/bias_correction/gcm')    
# print('Done MIROC5 DBC')
#  
# print('')
# HADGEM_baseline_QM_adj = pd.read_csv('GCM_data/bias_correction/gcm/HADGEM_baseline_QM_adj_disag.csv', na_values = {'     NA'})
# aggregate_to_csv(HADGEM_baseline_QM_adj, 'HADGEM_baseline_QM_adj', directory='GCM_data/bias_correction/gcm')
# aggregate_hourly_to_csv(HADGEM_baseline_QM_adj, 'HADGEM_baseline_QM_adj', directory='GCM_data/bias_correction/gcm')
# get_max_subdaily_table('HADGEM_baseline_QM_adj', directory = 'GCM_data/bias_correction/gcm')
# get_max_subdaily_min_table('HADGEM_baseline_QM_adj', 5, directory='GCM_data/bias_correction/gcm')
# merge_max_tables('HADGEM_baseline_QM_adj', directory = 'GCM_data/bias_correction/gcm')    
# print('Done HADGEM QM_adj') 
 
# MIROC5_baseline_QM_adj = pd.read_csv('GCM_data/bias_correction/gcm/MIROC5_baseline_QM_adj_disag.csv', dtype = {'Precipitation': np.float64}, na_values = {'     NA'} )
# print(MIROC5_baseline_QM_adj.dtypes)
# input()
# aggregate_to_csv(MIROC5_baseline_QM_adj, 'MIROC5_baseline_QM_adj', directory='GCM_data/bias_correction/gcm')
# aggregate_hourly_to_csv(MIROC5_baseline_QM_adj, 'MIROC5_baseline_QM_adj', directory='GCM_data/bias_correction/gcm')
# get_max_subdaily_table('MIROC5_baseline_QM_adj', directory = 'GCM_data/bias_correction/gcm')
# get_max_subdaily_min_table('MIROC5_baseline_QM_adj', 5, directory='GCM_data/bias_correction/gcm')
# merge_max_tables('MIROC5_baseline_QM_adj', directory = 'GCM_data/bias_correction/gcm')    
# print('Done MIROC5 QM_adj')
# 
# print('')
# print('Done!')