import pandas as pd
from functions_treatment import *

# ## Calculo de maximos subdiarios para Bartlett Lewis
# get_max_subdaily_table('MAPLU_usp_bl', directory = 'bartlet_lewis')
# get_max_subdaily_table('INMET_bl', directory = 'bartlet_lewis')
# get_max_subdaily_table('INMET_conv_bl', directory = 'bartlet_lewis')
# get_max_subdaily_table('MAPLU_usp')

#get_max_subdaily_min_table('MAPLU_usp', 5)

#get_max_subdaily_min_table('MAPLU_usp_bl', 5, directory='bartlet_lewis')
#get_max_subdaily_min_table('INMET_bl', 5, directory='bartlet_lewis')
#get_max_subdaily_min_table('INMET_conv_bl', 5, directory='bartlet_lewis')

#merge_max_tables('MAPLU_usp_bl', directory = 'bartlet_lewis' )    
# merge_max_tables('INMET_bl', directory = 'bartlet_lewis' )    
# merge_max_tables('INMET_conv_bl', directory = 'bartlet_lewis' )    
# print('Done!')

# 
# ## Plot dos subdiarios com BL para comparacao
# plot_subdaily_maximum_BL(1)
# #plot_subdaily_maximum_BL(3)
# plot_subdaily_maximum_BL(6)
# plot_subdaily_maximum_BL(8)
# plot_subdaily_maximum_BL(10)
# plot_subdaily_maximum_BL(12)
# plot_subdaily_maximum_BL(24)

# ## Dados diarios INMET_aut
#  
# INMET_aut_df, INMET_conv_df = treat_INMET_daily()
#  
# INMET_conv_df = INMET_conv_df[['Day', 'Month', 'Year', 'Precipitation']]
# #INMET_aut_df = INMET_aut_df.dropna()
# INMET_conv_df = INMET_conv_df.fillna('NA')
# #print(INMET_aut_df)
#  
# INMET_conv_df.to_csv('INMET_conv_daily_test.csv', index = False)
# #INMET_conv_df.to_csv('INMET_conv_daily_test.csv', index = False)
# print('Done!')

# ## Calculo de maximos subdiarios por desagregadores otimizados
# 
# name_file = 'agua_vermelha'
# get_subdaily_optimized(name_file)
# 
# plot_optimized_subdaily(name_file, 1)
# plot_optimized_subdaily(name_file, 6)
# plot_optimized_subdaily(name_file, 8)
# plot_optimized_subdaily(name_file, 10)
# plot_optimized_subdaily(name_file, 12)
# plot_optimized_subdaily(name_file, 24)

# ## Calculo de maximos subdiarios por desagregadores
# df_conv = pd.read_csv('Results/max_daily_INMET_conv_2.csv')
# df_aut = pd.read_csv('Results/max_daily_INMET_aut_2.csv')
# df_agua_vermelha = pd.read_csv('Results/max_daily_agua_vermelha_2.csv')
# df_maplu = pd.read_csv('Results/max_daily_MAPLU_usp.csv')  
# var_value = 0.005
# 
# print('Getting subdaily for MAPLU..')
# get_subdaily_from_disagregation_factors(df_maplu, 'original', var_value, 'MAPLU_usp') 

# print('Getting subdaily for INMET_conv..')
# get_subdaily_from_disagregation_factors(df_conv, 'original', var_value, 'INMET_conv')
# get_subdaily_from_disagregation_factors(df_conv, 'plus', var_value, 'INMET_conv')
# get_subdaily_from_disagregation_factors(df_conv, 'minus',var_value, 'INMET_conv')
#   
# print('')
# print('Getting subdaily for INMET_aut..')
# get_subdaily_from_disagregation_factors(df_aut, 'original', var_value, 'INMET_aut')
# get_subdaily_from_disagregation_factors(df_aut, 'plus', var_value, 'INMET_aut')
# get_subdaily_from_disagregation_factors(df_aut, 'minus', var_value, 'INMET_aut')
#   
# print('')
# print('Getting subdaily for agua_vermelha..')
# get_subdaily_from_disagregation_factors(df_agua_vermelha, 'original', var_value, 'agua_vermelha')
# get_subdaily_from_disagregation_factors(df_agua_vermelha, 'plus', var_value, 'agua_vermelha')
# get_subdaily_from_disagregation_factors(df_agua_vermelha, 'minus', var_value,  'agua_vermelha')
#   
# print('')
# print('Done disagregation')
# 
# 
# plot_subdaily_maximum_relative('INMET_aut', 1, var_value)
# plot_subdaily_maximum_relative('INMET_aut', 6, var_value)
# plot_subdaily_maximum_relative('INMET_aut', 8, var_value)
# plot_subdaily_maximum_relative('INMET_aut', 10, var_value)
# plot_subdaily_maximum_relative('INMET_aut', 12, var_value)
# plot_subdaily_maximum_relative('INMET_aut', 24, var_value)

# 
# ## Calculo de maximos subdiarios para agua vermelha
# print('Starting..')
# df = pd.read_csv('Results/agua_vermelha_hourly.csv')
# df_1h = get_subdaily_max(df, 1)
# print('')
# print('1h done!..')
# df_3h = get_subdaily_max(df, 3)
# print('')
# print('3h done!..')
# df_final = df_1h.merge(df_3h, on = 'Year', how = 'inner')
# 
# df_6h = get_subdaily_max(df, 6)
# print('')
# print('6h done!..')
# df_final = df_final.merge(df_6h, on = 'Year', how = 'inner')
# 
# df_8h = get_subdaily_max(df, 8)
# print('')
# print('8h done!..')
# df_final = df_final.merge(df_8h, on = 'Year', how = 'inner')
# 
# df_10h = get_subdaily_max(df, 10)
# print('')
# print('10h done!..')
# df_final = df_final.merge(df_10h, on = 'Year', how = 'inner')
# 
# df_12h = get_subdaily_max(df, 12)
# print('')
# print('12h done!..')
# df_final = df_final.merge(df_12h, on = 'Year', how = 'inner')
# 
# df_24h = get_subdaily_max(df, 24)
# print('')
# print('24h done!..')
# df_final = df_final.merge(df_24h, on = 'Year', how = 'inner')
# print('')
# print(df_final)
# print('')
# df_final.to_csv('Results/max_subdaily_agua_vermelha.csv', index = False)
# print('Done!')


# ##Calculo de maximos subdiarios para INMET_aut
# print('Starting..')
# df = pd.read_csv('Results/INMET_aut_hourly.csv')
# df_1h = get_subdaily_max(df, 1)
# print('')
# print('1h done!..')
# df_3h = get_subdaily_max(df, 3)
# print('')
# print('3h done!..')
# df_final = df_1h.merge(df_3h, on = 'Year', how = 'inner')
# 
# df_6h = get_subdaily_max(df, 6)
# print('')
# print('6h done!..')
# df_final = df_final.merge(df_6h, on = 'Year', how = 'inner')
# 
# df_8h = get_subdaily_max(df, 8)
# print('')
# print('8h done!..')
# df_final = df_final.merge(df_8h, on = 'Year', how = 'inner')
# 
# df_10h = get_subdaily_max(df, 10)
# print('')
# print('10h done!..')
# df_final = df_final.merge(df_10h, on = 'Year', how = 'inner')
# 
# df_12h = get_subdaily_max(df, 12)
# print('')
# print('12h done!..')
# df_final = df_final.merge(df_12h, on = 'Year', how = 'inner')
# 
# df_24h = get_subdaily_max(df, 24)
# print('')
# print('24h done!..')
# df_final = df_final.merge(df_24h, on = 'Year', how = 'inner')
# print('')
# print(df_final)
# print('')
# df_final.to_csv('Results/max_subdaily_INMET_aut.csv', index = False)
# print('Done!')

# # Calculo de maximos diarios e remocao de outliers 
# df_aut = read_csv('MAPLU_usp', 'daily')
# df_aut, df_conv = treat_INMET_daily()
# df_aut.to_csv('Results/INMET_aut_daily_2.csv', index = False)
# df_conv.to_csv('Results/INMET_conv_daily_2.csv', index = False)
#          
# df = max_annual_precipitation(df_aut)
# df.to_csv('Results/max_daily_MAPLU_usp.csv')
# print(df)

# #Comparacao dos maximos anuais nas diferentes estacoes
# ## Pairplot
# df = pd.read_csv('Results/comparacoes_max_daily.csv')
# #df = df[['INMET_conv', 'INMET_aut', 'agua_vermelha', 'cidade_jardim', 'jd_sp', 'MAPLU_esc', 'MAPLU_post', 'MAPLU_usp']]
# #df = df[['MAPLU_esc', 'MAPLU_post']]
# #df = df.dropna()
# df = df[['INMET_conv', 'INMET_aut', 'MAPLU_usp']]
# df = df.dropna()
# #print(df)
# df_corr, df_pvalue = correlation_plots_2(df)
# #df_corr.to_csv('Results/correlation_matrix_everyone.csv', index = False)
# #df_pvalue.to_csv('Results/pvalue_matrix_everyone.csv', index = False)
# plt.show()


# ## Barplot
# df = pd.read_csv('Results/comparacoes_max_daily_barplot.csv')
# #df = df.loc[df['Type'].isin(['INMET_conv', 'INMET_aut'])]
# df_barplot = df.loc[df['Year'] >= 2015]
#  
# #print(df_barplot)
#  
# ax = sns.catplot(x="Year", y="Precipitation", hue = 'Type', data=df_barplot, kind = 'bar')
# plt.xticks(rotation=30)
# plt.show()

# ## Nova analise de tendencia
# alpha_value = 0.05
#  
# print('Annual precipitation')
# group = 'CEMADEN'
# sites_list = ['agua_vermelha', 'jd_sp', 'cidade_jardim']
# get_trend('Year', sites_list, alpha_value, group)
#  
# print('')
# group = 'INMET'
# sites_list = ['INMET_aut', 'INMET_conv']
# get_trend('Year', sites_list, alpha_value, group)
#  
# print('')
# group = 'MAPLU'
# sites_list = ['MAPLU_esc', 'MAPLU_post', 'MAPLU_usp']
# get_trend('Year', sites_list, alpha_value, group)
#  
# print('')
# print('Daily maximum')
# group = 'CEMADEN'
# sites_list = ['agua_vermelha']
# get_trend('Max_daily', sites_list, alpha_value, group)
#  
# print('')
# group = 'INMET'
# sites_list = ['INMET_aut', 'INMET_conv']
# get_trend('Max_daily', sites_list, alpha_value, group)
# 
# print('')
# group = 'MAPLU'
# sites_list = ['MAPLU_esc', 'MAPLU_post', 'MAPLU_usp']
# get_trend('Max_daily', sites_list, alpha_value, group)

# # dados com base mensal
# agua_vermelha = read_csv('agua_vermelha', 'monthly')
# jd_sp = read_csv('jd_sp', 'monthly')
# INMET_aut = read_csv('INMET_aut', 'monthly')
# INMET_conv = read_csv('INMET_conv', 'monthly')
# MAPLU_esc = read_csv('MAPLU_esc', 'monthly')
# MAPLU_usp = read_csv('MAPLU_usp', 'monthly')
    
# dados com base diaria
# agua_vermelha = read_csv('agua_vermelha', 'daily')
# jd_sp = read_csv('jd_sp', 'daily')
# INMET_aut = read_csv('INMET_aut', 'daily')
# INMET_conv = read_csv('INMET_conv', 'daily')
# MAPLU_esc = read_csv('MAPLU_esc', 'daily')
# MAPLU_usp = read_csv('MAPLU_usp', 'daily')

# ## Distribution plots
# 
# distribution_plot('agua_vermelha', 'hourly')
# distribution_plot('agua_vermelha', 'daily')
# distribution_plot('agua_vermelha', 'monthly')
# distribution_plot('agua_vermelha', 'yearly')
# 
# distribution_plot('jd_sp', 'hourly')
# distribution_plot('jd_sp', 'daily')
# distribution_plot('jd_sp', 'monthly')
# distribution_plot('jd_sp', 'yearly')
# 
# distribution_plot('INMET_aut', 'hourly')
# distribution_plot('INMET_aut', 'daily')
# distribution_plot('INMET_aut', 'monthly')
# distribution_plot('INMET_aut', 'yearly')
# 
# distribution_plot('INMET_conv', 'hourly')
# distribution_plot('INMET_conv', 'daily')
# distribution_plot('INMET_conv', 'monthly')
# distribution_plot('INMET_conv', 'yearly')
# 
# distribution_plot('MAPLU_esc', 'hourly')
# distribution_plot('MAPLU_esc', 'daily')
# distribution_plot('MAPLU_esc', 'monthly')
# distribution_plot('MAPLU_esc', 'yearly')

## P90 ANALYSIS
# print('P90 CEMADEN / Agua Vermelha: ', p90_function(agua_vermelha))
# print('P90 CEMADEN / Jardim Sao Paulo: ', p90_function(jd_sp))
# print('P90 INMET / Automatic: ', p90_function(INMET_aut))
# print('P90 INMET / Conventional: ', p90_function(INMET_conv))
# print('P90 MAPLU / Escola Sao Bento: ', p90_function(MAPLU_esc))
# print('P90 MAPLU / USP2: ', p90_function(MAPLU_usp))


# ## ANALISE DE TENDENCIA
# agua_vermelha_mk = agua_vermelha[['Precipitation']]
# jd_sp_mk = jd_sp[['Precipitation']]
# INMET_aut_mk = INMET_aut[['Precipitation']]
# INMET_conv_mk = INMET_conv[['Precipitation']]
# MAPLU_esc_mk = MAPLU_esc[['Precipitation']]
# 
# alpha_value = 0.05
# 
# print('--- CEMADEN / AGUA VERMELHA ----')
# print('')
# trend_analysis(agua_vermelha_mk, alpha_value)
# 
# print('')
# print('--- CEMADEN / JARDIM SAO PAULO ----')
# print('')
# trend_analysis(jd_sp_mk, alpha_value)
# 
# print('')
# print('--- INMET / AUTOMATICO ----')
# print('')
# trend_analysis(INMET_aut_mk, alpha_value)
# 
# print('')
# print('--- INMET / CONVENCIONAL ----')
# print('')
# trend_analysis(INMET_conv_mk, alpha_value)
# 
# print('')
# print('--- MAPLU / ESCOLA SAO BENTO ----')
# print('')
# trend_analysis(MAPLU_esc_mk, alpha_value)

# DUPLA MASSA - ANALISE DE CONSISTENCIA
    
# df = pd.merge(agua_vermelha, jd_sp, how='left', on=['Year', 'Month'])
# df = pd.merge(df, INMET_aut, how='left', on=['Year', 'Month'])
# df = pd.merge(df, INMET_conv, how='left', on=['Year', 'Month'])
# df = pd.merge(df, MAPLU_esc, how='left', on=['Year', 'Month'])
# df = pd.merge(df, MAPLU_usp, how = 'left', on=['Year', 'Month'])
# df.columns = ['Year', 'Month', 'P_av', 'P_jdsp', 'P_inmet_aut', 'P_inmet_conv', 'P_maplu_esc', 'P_maplu_usp']
# df = df.dropna()
# df['P_average'] = df[['P_av', 'P_jdsp', 'P_inmet_aut', 'P_inmet_conv', 'P_maplu_esc']].mean(axis=1)
# df['Pacum_av'] = np.nan
# df['Pacum_jdsp'] = np.nan
# df['Pacum_inmet_aut'] = np.nan
# df['Pacum_inmet_conv'] = np.nan
# df['Pacum_maplu_esc'] = np.nan
# df['Pacum_maplu_usp'] = np.nan
# df['Pacum_average'] = np.nan
#     
# for i in range(len(df)):
#     if i == 0:
#         df['Pacum_av'][i] = df['P_av'][i]
#         df['Pacum_jdsp'][i] = df['P_jdsp'][i]
#         df['Pacum_inmet_aut'][i] = df['P_inmet_aut'][i]
#         df['Pacum_inmet_conv'][i] = df['P_inmet_conv'][i]
#         df['Pacum_maplu_esc'][i] = df['P_maplu_esc'][i]
#         df['Pacum_maplu_usp'][i] = df['P_maplu_usp'][i]
#         df['Pacum_average'][i] = df['P_average'][i]
#     else:
#         df['Pacum_av'][i] = df['Pacum_av'][i-1]+df['P_av'][i]
#         df['Pacum_jdsp'][i] = df['Pacum_jdsp'][i-1]+df['P_jdsp'][i]
#         df['Pacum_inmet_aut'][i] = df['Pacum_inmet_aut'][i-1]+df['P_inmet_aut'][i]
#         df['Pacum_inmet_conv'][i] = df['Pacum_inmet_conv'][i-1]+df['P_inmet_conv'][i]
#         df['Pacum_maplu_esc'][i] = df['Pacum_maplu_esc'][i-1]+df['P_maplu_esc'][i]
#         df['Pacum_maplu_usp'][i] = df['Pacum_maplu_usp'][i-1]+df['P_maplu_usp'][i]
#         df['Pacum_average'][i] = df['Pacum_average'][i-1]+df['P_average'][i]
# 
# print(df)
# df.to_csv('Results/dupla_massa_new.csv', index = False)

# ## ONLY CEMADEN AND INMET
# df.columns = ['Year', 'Month', 'P_av', 'P_jdsp', 'P_inmet_aut', 'P_inmet_conv']
# df = df.dropna()
# df['P_average'] = df[['P_av', 'P_jdsp', 'P_inmet_aut', 'P_inmet_conv']].mean(axis=1)
# df['Pacum_av'] = np.nan
# df['Pacum_jdsp'] = np.nan
# df['Pacum_inmet_aut'] = np.nan
# df['Pacum_inmet_conv'] = np.nan
# df['Pacum_average'] = np.nan
# for i in range(len(df)):
#     if i == 0:
#         df['Pacum_av'][i] = df['P_av'][i]
#         df['Pacum_jdsp'][i] = df['P_jdsp'][i]
#         df['Pacum_inmet_aut'][i] = df['P_inmet_aut'][i]
#         df['Pacum_inmet_conv'][i] = df['P_inmet_conv'][i]
#         df['Pacum_average'][i] = df['P_average'][i]
#     else:
#         df['Pacum_av'][i] = df['Pacum_av'][i-1]+df['P_av'][i]
#         df['Pacum_jdsp'][i] = df['Pacum_jdsp'][i-1]+df['P_jdsp'][i]
#         df['Pacum_inmet_aut'][i] = df['Pacum_inmet_aut'][i-1]+df['P_inmet_aut'][i]
#         df['Pacum_inmet_conv'][i] = df['Pacum_inmet_conv'][i-1]+df['P_inmet_conv'][i]
#         df['Pacum_average'][i] = df['Pacum_average'][i-1]+df['P_average'][i]
#  
# print(df)
# df.to_csv('Results/dupla_massa_CEMADEN.csv', index = False)

# ## ONLY MAPLU AND INMET
# df = pd.merge(MAPLU_esc, INMET_aut, how='left', on=['Year', 'Month'])
# df = pd.merge(df, INMET_conv, how='left', on=['Year', 'Month'])
# df.columns = ['Year', 'Month', 'P_maplu', 'P_inmet_aut', 'P_inmet_conv']
# df = df.dropna()
# df['P_average'] = df[['P_maplu', 'P_inmet_aut', 'P_inmet_conv']].mean(axis=1)
# df['Pacum_maplu'] = np.nan
# df['Pacum_inmet_aut'] = np.nan
# df['Pacum_inmet_conv'] = np.nan
# df['Pacum_average'] = np.nan
#  
# for i in range(len(df)):
#     if i == 0:
#         df['Pacum_inmet_aut'][i] = df['P_inmet_aut'][i]
#         df['Pacum_inmet_conv'][i] = df['P_inmet_conv'][i]
#         df['Pacum_maplu'][i] = df['P_maplu'][i]
#         df['Pacum_average'][i] = df['P_average'][i]
#     else:
#         df['Pacum_inmet_aut'][i] = df['Pacum_inmet_aut'][i-1]+df['P_inmet_aut'][i]
#         df['Pacum_inmet_conv'][i] = df['Pacum_inmet_conv'][i-1]+df['P_inmet_conv'][i]
#         df['Pacum_maplu'][i] = df['Pacum_maplu'][i-1]+df['P_maplu'][i]
#         df['Pacum_average'][i] = df['Pacum_average'][i-1]+df['P_average'][i]
# 
# print(df)
# df.to_csv('Results/dupla_massa_MAPLU.csv', index = False)

# #df = pd.read_csv('Results/dupla_massa.csv')
# #df = pd.read_csv('Results/dupla_massa_CEMADEN.csv')
# #df = pd.read_csv('Results/dupla_massa_MAPLU.csv')
# df = pd.read_csv('Results/dupla_massa_new.csv')
# sns.set_context("talk", font_scale=0.8)
# plt.figure(figsize=(8,6))
# sns.scatterplot(x="Pacum_average", 
#                 y="Pacum",
#                 hue = 'Station', 
#                 data=df)
# plt.xlabel("Pacum Average (mm)")
# plt.ylabel("Pacum (mm)")
# plt.title("Average")
# # plt.savefig("default_legend_position_Seaborn_scatterplot.png",
# #                     format='png',dpi=150)
# plt.show()
#    
# sns.set_context("talk", font_scale=0.8)
# plt.figure(figsize=(8,6))
# sns.scatterplot(x="Pacum_inmet_aut", 
#                 y="Pacum",
#                 hue = 'Station', 
#                 data=df)
# plt.xlabel("Pacum inmet_aut (mm)")
# plt.ylabel("Pacum (mm)")
# plt.title("INMET_aut")
#    
# plt.show()


# ## COMPLETAR FALHAS
# INMET_aut = complete_date_series('INMET_aut', 'daily')
# INMET_conv = complete_date_series('INMET_conv', 'daily')
# 
# agua_vermelha = complete_date_series('agua_vermelha', 'daily')
# agua_vermelha.to_csv('Results/agua_vermelha_daily_completo.csv', index = False)
# jd_sp = complete_date_series('jd_sp', 'daily')
# jd_sp.to_csv('Results/jd_sp_daily_completo.csv', index = False)
# MAPLU_esc = complete_date_series('MAPLU_esc', 'daily')
# MAPLU_esc.to_csv('Results/MAPLU_esc_daily_completo.csv', index = False)
# 
# df = left_join_precipitation(agua_vermelha, INMET_aut, INMET_conv)
# df.to_csv('Results/VERIFICAR_agua_vermelha.csv', index = False)
# df = left_join_precipitation(jd_sp, INMET_aut, INMET_conv)
# df.to_csv('Results/VERIFICAR_jd_sp.csv', index = False)
# df = left_join_precipitation(MAPLU_esc, INMET_aut, INMET_conv)
# df.to_csv('Results/VERIFICAR_MAPLU_esc.csv', index = False)


#correlation_plots(agua_vermelha, INMET_aut, INMET_conv)
#multiple_linear_regression(agua_vermelha, INMET_aut, INMET_conv)

# ## VERIFICACAO DE FALHAS
# cidade_jardim = read_csv('cidade_jardim', 'daily')   
# verification(cidade_jardim) 
# 
# jd_sp = read_csv('jd_sp', 'daily')   
# verification(jd_sp) 
# 
# INMET_aut = read_csv('INMET_aut', 'daily')   
# verification(INMET_aut)
# 
# INMET_conv = read_csv('INMET_conv', 'daily')   
# verification(INMET_conv)
# 
# MAPLU_esc = read_csv('MAPLU_esc', 'daily')   
# verification(MAPLU_esc)
# 
# MAPLU_post = read_csv('MAPLU_post', 'daily')   
# verification(MAPLU_post)

# MAPLU_usp = read_csv('MAPLU_usp', 'daily')   
# verification(MAPLU_usp)




