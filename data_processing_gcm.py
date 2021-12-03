from functions_treatment import *
from numpy import dtype

##BASELINE ANALYSIS
print('---Baseline---')
df_hadgem_md = pd.read_csv('GCM_data/bias_correction/HADGEM_baseline_MD_daily.csv')
df_hadgem_pt = pd.read_csv('GCM_data/bias_correction/HADGEM_baseline_PT_daily.csv')
df_hadgem_qm = pd.read_csv('GCM_data/bias_correction/HADGEM_baseline_QM_daily.csv')
df_hadgem_dbc = pd.read_csv('GCM_data/bias_correction/HADGEM_baseline_DBC_daily.csv')

df_miroc_dbc = pd.read_csv('GCM_data/bias_correction/MIROC5_baseline_DBC_daily.csv')
df_miroc_md = pd.read_csv('GCM_data/bias_correction/MIROC5_baseline_MD_daily.csv')
df_miroc_pt = pd.read_csv('GCM_data/bias_correction/MIROC5_baseline_PT_daily.csv')
df_miroc_qm = pd.read_csv('GCM_data/bias_correction/MIROC5_baseline_QM_daily.csv')
 
#P90
print('--> P90')
print('P90 HADGEM_MD / baseline: ', p90_function(df_hadgem_md))
print('P90 HADGEM_PT / baseline: ', p90_function(df_hadgem_pt))
print('P90 HADGEM_QM / baseline: ', p90_function(df_hadgem_qm))
print('P90 HADGEM_DBC / baseline: ', p90_function(df_hadgem_dbc))
print('')
print('P90 MIROC5_DBC / baseline: ', p90_function(df_miroc_dbc))
print('P90 MIROC5_MD / baseline: ', p90_function(df_miroc_md))
print('P90 MIROC5_PT / baseline: ', p90_function(df_miroc_pt))
print('P90 MIROC5_QM / baseline: ', p90_function(df_miroc_qm))
  
#Aggregate to year
def aggregate_to_csv_year(df, name, directory = 'GCM_data/bias_correction'):
    df_yearly = aggregate(df, 'Year')
    df_yearly.to_csv('{d}/{n}_yearly.csv'.format(n = name, d = directory), index = False)
  
aggregate_to_csv_year(df_hadgem_md, 'HADGEM_baseline_MD')
aggregate_to_csv_year(df_hadgem_pt, 'HADGEM_baseline_PT')
aggregate_to_csv_year(df_hadgem_qm, 'HADGEM_baseline_QM')
  
aggregate_to_csv_year(df_miroc_md, 'MIROC5_baseline_MD')
aggregate_to_csv_year(df_miroc_pt, 'MIROC5_baseline_PT')
aggregate_to_csv_year(df_miroc_qm, 'MIROC5_baseline_QM')
  
#Calculate annual max daily precipitation
def get_max_daily(df, name, directory = 'GCM_data/bias_correction'):
    df_max = max_annual_precipitation(df)
    df_max.to_csv('{d}/max_daily_{n}.csv'.format(n = name, d = directory), index = False)

print('Getting max daily..') 
get_max_daily(df_hadgem_md, 'HADGEM_baseline_MD')
get_max_daily(df_hadgem_pt, 'HADGEM_baseline_PT')
get_max_daily(df_hadgem_qm, 'HADGEM_baseline_QM')
get_max_daily(df_hadgem_dbc, 'HADGEM_baseline_DBC')
 
get_max_daily(df_miroc_md, 'MIROC5_baseline_MD')
get_max_daily(df_miroc_pt, 'MIROC5_baseline_PT')
get_max_daily(df_miroc_qm, 'MIROC5_baseline_QM')
get_max_daily(df_miroc_dbc, 'MIROC5_baseline_DBC')
   
#Trend analysis
print('')
print('--> Trend analysis')
alpha_value = 0.05
group = 'GCM_baseline'
sites_list = ['HADGEM_baseline_MD', 'HADGEM_baseline_PT', 'HADGEM_baseline_QM', 'MIROC5_baseline_MD', 'MIROC5_baseline_PT', 'MIROC5_baseline_QM']
print('- Annual precipitation')
get_trend('Year', sites_list, alpha_value, group, 'mod')
print('')
print('- Max_daily')
get_trend('Max_daily', sites_list, alpha_value, group, 'mod')

print('')
print('Done!')
