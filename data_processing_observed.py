import pandas as pd
from functions_processing import *

## GAP CHECKING
INMET_aut = read_csv('INMET_aut', 'daily')   
verification(INMET_aut)

INMET_conv = read_csv('INMET_conv', 'daily')   
verification(INMET_conv)

## GAP FILLING
INMET_aut = complete_date_series('INMET_aut', 'daily')
INMET_conv = complete_date_series('INMET_conv', 'daily')

## CONSISTENCY TEST - DOUBLE MASS 
df = pd.merge(agua_vermelha, jd_sp, how='left', on=['Year', 'Month'])
df = pd.merge(df, INMET_aut, how='left', on=['Year', 'Month'])
df = pd.merge(df, INMET_conv, how='left', on=['Year', 'Month'])
df = pd.merge(df, MAPLU_esc, how='left', on=['Year', 'Month'])
df = pd.merge(df, MAPLU_usp, how = 'left', on=['Year', 'Month'])
df.columns = ['Year', 'Month', 'P_av', 'P_jdsp', 'P_inmet_aut', 'P_inmet_conv', 'P_maplu_esc', 'P_maplu_usp']
df = df.dropna()
df['P_average'] = df[['P_av', 'P_jdsp', 'P_inmet_aut', 'P_inmet_conv', 'P_maplu_esc']].mean(axis=1)
df['Pacum_av'] = np.nan
df['Pacum_jdsp'] = np.nan
df['Pacum_inmet_aut'] = np.nan
df['Pacum_inmet_conv'] = np.nan
df['Pacum_maplu_esc'] = np.nan
df['Pacum_maplu_usp'] = np.nan
df['Pacum_average'] = np.nan
     
for i in range(len(df)):
    if i == 0:
        df['Pacum_av'][i] = df['P_av'][i]
        df['Pacum_jdsp'][i] = df['P_jdsp'][i]
        df['Pacum_inmet_aut'][i] = df['P_inmet_aut'][i]
        df['Pacum_inmet_conv'][i] = df['P_inmet_conv'][i]
        df['Pacum_maplu_esc'][i] = df['P_maplu_esc'][i]
        df['Pacum_maplu_usp'][i] = df['P_maplu_usp'][i]
        df['Pacum_average'][i] = df['P_average'][i]
    else:
        df['Pacum_av'][i] = df['Pacum_av'][i-1]+df['P_av'][i]
        df['Pacum_jdsp'][i] = df['Pacum_jdsp'][i-1]+df['P_jdsp'][i]
        df['Pacum_inmet_aut'][i] = df['Pacum_inmet_aut'][i-1]+df['P_inmet_aut'][i]
        df['Pacum_inmet_conv'][i] = df['Pacum_inmet_conv'][i-1]+df['P_inmet_conv'][i]
        df['Pacum_maplu_esc'][i] = df['Pacum_maplu_esc'][i-1]+df['P_maplu_esc'][i]
        df['Pacum_maplu_usp'][i] = df['Pacum_maplu_usp'][i-1]+df['P_maplu_usp'][i]
        df['Pacum_average'][i] = df['Pacum_average'][i-1]+df['P_average'][i]

print(df)
df.to_csv('Results/dupla_massa_new.csv', index = False)
df = pd.read_csv('Results/dupla_massa_new.csv')
sns.set_context("talk", font_scale=0.8)
plt.figure(figsize=(8,6))
sns.scatterplot(x="Pacum_average", 
                y="Pacum",
                hue = 'Station', 
                data=df)
plt.xlabel("Pacum Average (mm)")
plt.ylabel("Pacum (mm)")
plt.title("Average")
# plt.savefig("default_legend_position_Seaborn_scatterplot.png",
#                     format='png',dpi=150)
plt.show()
   
sns.set_context("talk", font_scale=0.8)
plt.figure(figsize=(8,6))
sns.scatterplot(x="Pacum_inmet_aut", 
                y="Pacum",
                hue = 'Station', 
                data=df)
plt.xlabel("Pacum inmet_aut (mm)")
plt.ylabel("Pacum (mm)")
plt.title("INMET_aut")
   
plt.show()

## P90 ANALYSIS
print('P90 INMET / Automatic: ', p90_function(INMET_aut))
print('P90 INMET / Conventional: ', p90_function(INMET_conv))

## TREND ANALYSIS
alpha_value = 0.05
 
print('Annual precipitation')
print('')
group = 'INMET'
sites_list = ['INMET_aut', 'INMET_conv']
get_trend('Year', sites_list, alpha_value, group)
print('')
print('Daily maximum')
print('')
group = 'INMET'
sites_list = ['INMET_aut', 'INMET_conv']
get_trend('Max_daily', sites_list, alpha_value, group)
