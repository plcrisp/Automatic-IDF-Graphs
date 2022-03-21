import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import xticks

sns.set(style="ticks")

def plot_error_INMET(df):
    df_INMET_conv = df.loc[df['station_name'] == 'INMET_conv']
    df_inconv_subdaily = df_INMET_conv.loc[df_INMET_conv['data_type'] == 'subdaily_sao_paulo']
    df_inconv_daily = df_INMET_conv.loc[df_INMET_conv['data_type'] == 'daily']
    
    df_inconv_subdaily_gumbel = df_inconv_subdaily.loc[df_inconv_subdaily['dist_type'] == 'Gumbel']
    df_inconv_daily_gumbel = df_inconv_daily.loc[df_inconv_daily['dist_type'] == 'Gumbel']
    df_inconv_subdaily_genlog = df_inconv_subdaily.loc[df_inconv_subdaily['dist_type'] == 'GenLogistic']
    df_inconv_daily_genlog = df_inconv_daily.loc[df_inconv_daily['dist_type'] == 'GenLogistic']
    df_inconv_subdaily_gev = df_inconv_subdaily.loc[df_inconv_subdaily['dist_type'] == 'GEV']
    df_inconv_daily_gev = df_inconv_daily.loc[df_inconv_daily['dist_type'] == 'GEV']
    df_inconv_subdaily_norm = df_inconv_subdaily.loc[df_inconv_subdaily['dist_type'] == 'Normal']
    df_inconv_daily_norm = df_inconv_daily.loc[df_inconv_daily['dist_type'] == 'Normal']
    
    # print(df_inconv_daily['dist_type'])
    # print(df_inconv_daily_gumbel)
    # print(df_inconv_daily_gev)
    
    df_INMET_aut = df.loc[df['station_name'] == 'INMET_aut']
    df_inaut_subdaily = df_INMET_aut.loc[df_INMET_aut['data_type'] == 'subdaily_sao_paulo']
    df_inaut_daily = df_INMET_aut.loc[df_INMET_aut['data_type'] == 'daily']
    
    df_inaut_subdaily_gumbel = df_inaut_subdaily.loc[df_inaut_subdaily['dist_type'] == 'Gumbel']
    df_inaut_daily_gumbel = df_inaut_daily.loc[df_inaut_daily['dist_type'] == 'Gumbel']
    df_inaut_subdaily_genlog = df_inaut_subdaily.loc[df_inaut_subdaily['dist_type'] == 'GenLogistic']
    df_inaut_daily_genlog = df_inaut_daily.loc[df_inaut_daily['dist_type'] == 'GenLogistic']
    df_inaut_subdaily_gev = df_inaut_subdaily.loc[df_inaut_subdaily['dist_type'] == 'GEV']
    df_inaut_daily_gev = df_inaut_daily.loc[df_inaut_daily['dist_type'] == 'GEV']
    df_inaut_subdaily_norm = df_inaut_subdaily.loc[df_inaut_subdaily['dist_type'] == 'Normal']
    df_inaut_daily_norm = df_inaut_daily.loc[df_inaut_daily['dist_type'] == 'Normal']
    
    fig, axs = plt.subplots(nrows = 4, ncols = 2, sharex = True, figsize=(8, 8))
    fig.tight_layout()
    fig.align_ylabels()
    
    scatter = axs[0,0].scatter(x = df_inconv_daily_gumbel.IPE, 
                y = df_inconv_daily_gumbel.disag_factor, 
                s= df_inconv_daily_gumbel.return_period,
                marker = '^',
                color = 'peachpuff') 
    
    axs[1,0].scatter(x = df_inconv_daily_genlog.IPE, 
                y = df_inconv_daily_genlog.disag_factor, 
                s= df_inconv_daily_genlog.return_period,
                marker = '^',
                color = 'peachpuff') 
    
    axs[2,0].scatter(x = df_inconv_daily_gev.IPE, 
                y = df_inconv_daily_gev.disag_factor, 
                s= df_inconv_daily_gev.return_period,
                marker = '^',
                color = 'peachpuff') 
    
                
    axs[0,1].scatter(x = df_inconv_subdaily_gumbel.IPE, 
                y = df_inconv_subdaily_gumbel.disag_factor, 
                s = df_inconv_subdaily_gumbel.return_period,
                marker = '^',
                color = 'peachpuff') 
    
    axs[1,1].scatter(x = df_inconv_subdaily_genlog.IPE, 
                y = df_inconv_subdaily_genlog.disag_factor, 
                s = df_inconv_subdaily_genlog.return_period,
                marker = '^',
                color = 'peachpuff') 
    
    axs[2,1].scatter(x = df_inconv_subdaily_gev.IPE, 
                y = df_inconv_subdaily_gev.disag_factor, 
                s = df_inconv_subdaily_gev.return_period,
                marker = '^',
                color = 'peachpuff') 
    
    axs[3,1].scatter(x = df_inconv_subdaily_norm.IPE, 
                y = df_inconv_subdaily_norm.disag_factor, 
                s = df_inconv_subdaily_norm.return_period,
                marker = '^',
                color = 'peachpuff')
    
    scatter2 = axs[0,0].scatter(x = df_inaut_daily_gumbel.IPE, 
                y = df_inaut_daily_gumbel.disag_factor, 
                s= df_inaut_daily_gumbel.return_period,
                marker = 'o',
                color = 'thistle') 
    
    axs[1,0].scatter(x = df_inaut_daily_genlog.IPE, 
                y = df_inaut_daily_genlog.disag_factor, 
                s= df_inaut_daily_genlog.return_period,
                marker = 'o',
                color = 'thistle') 
    
    axs[2,0].scatter(x = df_inaut_daily_gev.IPE, 
                y = df_inaut_daily_gev.disag_factor, 
                s= df_inaut_daily_gev.return_period,
                marker = 'o',
                color = 'thistle') 
    
                
    axs[0,1].scatter(x = df_inaut_subdaily_gumbel.IPE, 
                y = df_inaut_subdaily_gumbel.disag_factor, 
                s = df_inaut_subdaily_gumbel.return_period,
                marker = 'o',
                color = 'thistle') 
    
    axs[1,1].scatter(x = df_inaut_subdaily_genlog.IPE, 
                y = df_inaut_subdaily_genlog.disag_factor, 
                s = df_inaut_subdaily_genlog.return_period,
                marker = 'o',
                color = 'thistle') 
    
    axs[2,1].scatter(x = df_inaut_subdaily_gev.IPE, 
                y = df_inaut_subdaily_gev.disag_factor, 
                s = df_inaut_subdaily_gev.return_period,
                marker = 'o',
                color = 'thistle') 
    
    axs[3,1].scatter(x = df_inaut_subdaily_norm.IPE, 
                y = df_inaut_subdaily_norm.disag_factor, 
                s = df_inaut_subdaily_norm.return_period,
                marker = 'o',
                color = 'thistle')  
    
    axs[3,0].set_xlabel('IPE')
    axs[3,1].set_xlabel('IPE')
    
    axs[0,0].set_ylabel('Gumbel')
    axs[1,0].set_ylabel('GenLogistic')
    axs[2,0].set_ylabel('GEV')
    axs[3,0].set_ylabel('Normal')           
    
    
    handles, labels = scatter.legend_elements(prop="sizes", num = [2, 5, 10, 25, 50, 100], alpha=0.6)
    legend = fig.legend(handles, labels, 
                        loc="upper right", 
                        bbox_to_anchor = (1, 0.98), 
                        title="RP")
    
    
    
    # axs[0,0].scatter(x = df_INMET_aut.IPE, 
    #             y = df_INMET_aut.disag_factor, 
    #             #c = df_INMET_conv.return_period, 
    #             #s= smax - (max - df_INMET_aut.return_period)*(smax - smin)/(max-min),
    #             s = df_INMET_aut.return_period,
    #             marker = 'o',
    #             color = 'powderblue') 
    #             #cmap = sns.diverging_palette(220, 10, as_cmap=True))
    
    
    # handles, labels = scatter.legend_elements(prop="sizes", func = lambda x: smax - (max - x)*(smax - smin)/(max-min), alpha=0.6)
    # legend = ax.legend(handles, labels, loc="upper right", title="Sizes")
    # plt.xlabel("IPE")
    # plt.ylabel("Disagregation factor")

def plot_error_GCM(df):
    df_HADGEM = df.loc[df['station_name'] == 'HADGEM']
    df_hadgem_dbc = df_HADGEM.loc[df_HADGEM['downscaling_type'] == 'DBC']
    df_hadgem_eqm = df_HADGEM.loc[df_HADGEM['downscaling_type'] == 'EQM']
    df_hadgem_md = df_HADGEM.loc[df_HADGEM['downscaling_type'] == 'MD']
    df_hadgem_pt = df_HADGEM.loc[df_HADGEM['downscaling_type'] == 'PT']
    df_hadgem_qm = df_HADGEM.loc[df_HADGEM['downscaling_type'] == 'QM']
    
    df_hadgem_dbc_gumbel = df_hadgem_dbc.loc[df_hadgem_dbc['dist_type'] == 'Gumbel']
    df_hadgem_dbc_genlog = df_hadgem_dbc.loc[df_hadgem_dbc['dist_type'] == 'GenLogistic']
    df_hadgem_dbc_gev = df_hadgem_dbc.loc[df_hadgem_dbc['dist_type'] == 'GEV']
    df_hadgem_dbc_norm = df_hadgem_dbc.loc[df_hadgem_dbc['dist_type'] == 'Normal']
    df_hadgem_dbc_lognorm = df_hadgem_dbc.loc[df_hadgem_dbc['dist_type'] == 'Lognormal']
    
    df_hadgem_eqm_gumbel = df_hadgem_eqm.loc[df_hadgem_eqm['dist_type'] == 'Gumbel']
    df_hadgem_eqm_genlog = df_hadgem_eqm.loc[df_hadgem_eqm['dist_type'] == 'GenLogistic']
    df_hadgem_eqm_gev = df_hadgem_eqm.loc[df_hadgem_eqm['dist_type'] == 'GEV']
    df_hadgem_eqm_norm = df_hadgem_eqm.loc[df_hadgem_eqm['dist_type'] == 'Normal']
    df_hadgem_eqm_lognorm = df_hadgem_eqm.loc[df_hadgem_eqm['dist_type'] == 'Lognormal']

    df_hadgem_md_gumbel = df_hadgem_md.loc[df_hadgem_md['dist_type'] == 'Gumbel']
    df_hadgem_md_genlog = df_hadgem_md.loc[df_hadgem_md['dist_type'] == 'GenLogistic']
    df_hadgem_md_gev = df_hadgem_md.loc[df_hadgem_md['dist_type'] == 'GEV']
    df_hadgem_md_norm = df_hadgem_md.loc[df_hadgem_md['dist_type'] == 'Normal']
    df_hadgem_md_lognorm = df_hadgem_md.loc[df_hadgem_md['dist_type'] == 'Lognormal']

    df_hadgem_pt_gumbel = df_hadgem_pt.loc[df_hadgem_pt['dist_type'] == 'Gumbel']
    df_hadgem_pt_genlog = df_hadgem_pt.loc[df_hadgem_pt['dist_type'] == 'GenLogistic']
    df_hadgem_pt_gev = df_hadgem_pt.loc[df_hadgem_pt['dist_type'] == 'GEV']
    df_hadgem_pt_norm = df_hadgem_pt.loc[df_hadgem_pt['dist_type'] == 'Normal']
    df_hadgem_pt_lognorm = df_hadgem_pt.loc[df_hadgem_pt['dist_type'] == 'Lognormal']

    df_hadgem_qm_gumbel = df_hadgem_qm.loc[df_hadgem_qm['dist_type'] == 'Gumbel']
    df_hadgem_qm_genlog = df_hadgem_qm.loc[df_hadgem_qm['dist_type'] == 'GenLogistic']
    df_hadgem_qm_gev = df_hadgem_qm.loc[df_hadgem_qm['dist_type'] == 'GEV']
    df_hadgem_qm_norm = df_hadgem_qm.loc[df_hadgem_qm['dist_type'] == 'Normal']
    df_hadgem_qm_lognorm = df_hadgem_qm.loc[df_hadgem_qm['dist_type'] == 'Lognormal']


    df_MIROC5 = df.loc[df['station_name'] == 'MIROC5']
    df_miroc_dbc = df_MIROC5.loc[df_MIROC5['downscaling_type'] == 'DBC']
    df_miroc_eqm = df_MIROC5.loc[df_MIROC5['downscaling_type'] == 'EQM']
    df_miroc_md = df_MIROC5.loc[df_MIROC5['downscaling_type'] == 'MD']
    df_miroc_pt = df_MIROC5.loc[df_MIROC5['downscaling_type'] == 'PT']
    df_miroc_qm = df_MIROC5.loc[df_MIROC5['downscaling_type'] == 'QM']
    
    df_miroc_dbc_gumbel = df_miroc_dbc.loc[df_miroc_dbc['dist_type'] == 'Gumbel']
    df_miroc_dbc_genlog = df_miroc_dbc.loc[df_miroc_dbc['dist_type'] == 'GenLogistic']
    df_miroc_dbc_gev = df_miroc_dbc.loc[df_miroc_dbc['dist_type'] == 'GEV']
    df_miroc_dbc_norm = df_miroc_dbc.loc[df_miroc_dbc['dist_type'] == 'Normal']
    df_miroc_dbc_lognorm = df_miroc_dbc.loc[df_miroc_dbc['dist_type'] == 'Lognormal']
    
    df_miroc_eqm_gumbel = df_miroc_eqm.loc[df_miroc_eqm['dist_type'] == 'Gumbel']
    df_miroc_eqm_genlog = df_miroc_eqm.loc[df_miroc_eqm['dist_type'] == 'GenLogistic']
    df_miroc_eqm_gev = df_miroc_eqm.loc[df_miroc_eqm['dist_type'] == 'GEV']
    df_miroc_eqm_norm = df_miroc_eqm.loc[df_miroc_eqm['dist_type'] == 'Normal']
    df_miroc_eqm_lognorm = df_miroc_eqm.loc[df_miroc_eqm['dist_type'] == 'Lognormal']

    df_miroc_md_gumbel = df_miroc_md.loc[df_miroc_md['dist_type'] == 'Gumbel']
    df_miroc_md_genlog = df_miroc_md.loc[df_miroc_md['dist_type'] == 'GenLogistic']
    df_miroc_md_gev = df_miroc_md.loc[df_miroc_md['dist_type'] == 'GEV']
    df_miroc_md_norm = df_miroc_md.loc[df_miroc_md['dist_type'] == 'Normal']
    df_miroc_md_lognorm = df_miroc_md.loc[df_miroc_md['dist_type'] == 'Lognormal']

    df_miroc_pt_gumbel = df_miroc_pt.loc[df_miroc_pt['dist_type'] == 'Gumbel']
    df_miroc_pt_genlog = df_miroc_pt.loc[df_miroc_pt['dist_type'] == 'GenLogistic']
    df_miroc_pt_gev = df_miroc_pt.loc[df_miroc_pt['dist_type'] == 'GEV']
    df_miroc_pt_norm = df_miroc_pt.loc[df_miroc_pt['dist_type'] == 'Normal']
    df_miroc_pt_lognorm = df_miroc_pt.loc[df_miroc_pt['dist_type'] == 'Lognormal']

    df_miroc_qm_gumbel = df_miroc_qm.loc[df_miroc_qm['dist_type'] == 'Gumbel']
    df_miroc_qm_genlog = df_miroc_qm.loc[df_miroc_qm['dist_type'] == 'GenLogistic']
    df_miroc_qm_gev = df_miroc_qm.loc[df_miroc_qm['dist_type'] == 'GEV']
    df_miroc_qm_norm = df_miroc_qm.loc[df_miroc_qm['dist_type'] == 'Normal']
    df_miroc_qm_lognorm = df_miroc_qm.loc[df_miroc_qm['dist_type'] == 'Lognormal']    

    
    fig, axs = plt.subplots(nrows = 5, ncols = 2, 
                            #sharex = True, 
                            figsize=(8, 8))
    fig.tight_layout()
    fig.align_ylabels()
    
    scatter = axs[0,0].scatter(x = df_hadgem_dbc_gumbel.IPE, 
                y = df_hadgem_dbc_gumbel.disag_factor, 
                s= df_hadgem_dbc_gumbel.return_period,
                marker = '^',
                color = 'peachpuff') 
    
    axs[1,0].scatter(x = df_hadgem_dbc_genlog.IPE, 
                y = df_hadgem_dbc_genlog.disag_factor, 
                s= df_hadgem_dbc_genlog.return_period,
                marker = '^',
                color = 'peachpuff') 
    
    axs[2,0].scatter(x = df_hadgem_dbc_gev.IPE, 
                y = df_hadgem_dbc_gev.disag_factor, 
                s= df_hadgem_dbc_gev.return_period,
                marker = '^',
                color = 'peachpuff') 

    axs[3,0].scatter(x = df_hadgem_dbc_norm.IPE, 
                y = df_hadgem_dbc_norm.disag_factor, 
                s= df_hadgem_dbc_norm.return_period,
                marker = '^',
                color = 'peachpuff')    

    axs[4,0].scatter(x = df_hadgem_dbc_lognorm.IPE, 
                y = df_hadgem_dbc_lognorm.disag_factor, 
                s= df_hadgem_dbc_lognorm.return_period,
                marker = '^',
                color = 'peachpuff')

    scatter2 = axs[0,0].scatter(x = df_hadgem_eqm_gumbel.IPE, 
                y = df_hadgem_eqm_gumbel.disag_factor, 
                s= df_hadgem_eqm_gumbel.return_period,
                marker = 'o',
                color = 'thistle') 
    
    axs[1,0].scatter(x = df_hadgem_eqm_genlog.IPE, 
                y = df_hadgem_eqm_genlog.disag_factor, 
                s= df_hadgem_eqm_genlog.return_period,
                marker = 'o',
                color = 'thistle') 
    
    axs[2,0].scatter(x = df_hadgem_eqm_gev.IPE, 
                y = df_hadgem_eqm_gev.disag_factor, 
                s= df_hadgem_eqm_gev.return_period,
                marker = 'o',
                color = 'thistle') 

    axs[3,0].scatter(x = df_hadgem_eqm_norm.IPE, 
                y = df_hadgem_eqm_norm.disag_factor, 
                s= df_hadgem_eqm_norm.return_period,
                marker = 'o',
                color = 'thistle')    

    axs[4,0].scatter(x = df_hadgem_eqm_lognorm.IPE, 
                y = df_hadgem_eqm_lognorm.disag_factor, 
                s= df_hadgem_eqm_lognorm.return_period,
                marker = 'o',
                color = 'thistle') 

    scatter3 = axs[0,0].scatter(x = df_hadgem_md_gumbel.IPE, 
                y = df_hadgem_md_gumbel.disag_factor, 
                s= df_hadgem_md_gumbel.return_period,
                marker = 's',
                color = 'powderblue') 
    
    axs[1,0].scatter(x = df_hadgem_md_genlog.IPE, 
                y = df_hadgem_md_genlog.disag_factor, 
                s= df_hadgem_md_genlog.return_period,
                marker = 's',
                color = 'powderblue') 
    
    axs[2,0].scatter(x = df_hadgem_md_gev.IPE, 
                y = df_hadgem_md_gev.disag_factor, 
                s= df_hadgem_md_gev.return_period,
                marker = 's',
                color = 'powderblue') 

    axs[3,0].scatter(x = df_hadgem_md_norm.IPE, 
                y = df_hadgem_md_norm.disag_factor, 
                s= df_hadgem_md_norm.return_period,
                marker = 's',
                color = 'powderblue')    

    axs[4,0].scatter(x = df_hadgem_md_lognorm.IPE, 
                y = df_hadgem_md_lognorm.disag_factor, 
                s= df_hadgem_md_lognorm.return_period,
                marker = 's',
                color = 'powderblue') 

    scatter4 = axs[0,0].scatter(x = df_hadgem_pt_gumbel.IPE, 
                y = df_hadgem_pt_gumbel.disag_factor, 
                s= df_hadgem_pt_gumbel.return_period,
                marker = 'd',
                color = 'pink') 
    
    axs[1,0].scatter(x = df_hadgem_pt_genlog.IPE, 
                y = df_hadgem_pt_genlog.disag_factor, 
                s= df_hadgem_pt_genlog.return_period,
                marker = 'd',
                color = 'pink') 
    
    axs[2,0].scatter(x = df_hadgem_pt_gev.IPE, 
                y = df_hadgem_pt_gev.disag_factor, 
                s= df_hadgem_pt_gev.return_period,
                marker = 'd',
                color = 'pink') 

    axs[3,0].scatter(x = df_hadgem_pt_norm.IPE, 
                y = df_hadgem_pt_norm.disag_factor, 
                s= df_hadgem_pt_norm.return_period,
                marker = 'd',
                color = 'pink')    

    axs[4,0].scatter(x = df_hadgem_pt_lognorm.IPE, 
                y = df_hadgem_pt_lognorm.disag_factor, 
                s= df_hadgem_pt_lognorm.return_period,
                marker = 'd',
                color = 'pink') 
 
    scatter5 = axs[0,0].scatter(x = df_hadgem_qm_gumbel.IPE, 
                y = df_hadgem_qm_gumbel.disag_factor, 
                s= df_hadgem_qm_gumbel.return_period,
                marker = 'P',
                color = 'lightgreen') 
    
    axs[1,0].scatter(x = df_hadgem_qm_genlog.IPE, 
                y = df_hadgem_qm_genlog.disag_factor, 
                s= df_hadgem_qm_genlog.return_period,
                marker = 'P',
                color = 'lightgreen') 
    
    axs[2,0].scatter(x = df_hadgem_qm_gev.IPE, 
                y = df_hadgem_qm_gev.disag_factor, 
                s= df_hadgem_qm_gev.return_period,
                marker = 'P',
                color = 'lightgreen') 

    axs[3,0].scatter(x = df_hadgem_qm_norm.IPE, 
                y = df_hadgem_qm_norm.disag_factor, 
                s= df_hadgem_qm_norm.return_period,
                marker = 'P',
                color = 'lightgreen')    

    axs[4,0].scatter(x = df_hadgem_qm_lognorm.IPE, 
                y = df_hadgem_qm_lognorm.disag_factor, 
                s= df_hadgem_qm_lognorm.return_period,
                marker = 'P',
                color = 'lightgreen')
    
    ##MIROC
    scatter0 = axs[0,1].scatter(x = df_miroc_dbc_gumbel.IPE, 
                y = df_miroc_dbc_gumbel.disag_factor, 
                s= df_miroc_dbc_gumbel.return_period,
                marker = '^',
                color = 'peachpuff') 
    
    axs[1,1].scatter(x = df_miroc_dbc_genlog.IPE, 
                y = df_miroc_dbc_genlog.disag_factor, 
                s= df_miroc_dbc_genlog.return_period,
                marker = '^',
                color = 'peachpuff') 
    
    axs[2,1].scatter(x = df_miroc_dbc_gev.IPE, 
                y = df_miroc_dbc_gev.disag_factor, 
                s= df_miroc_dbc_gev.return_period,
                marker = '^',
                color = 'peachpuff') 

    axs[3,1].scatter(x = df_miroc_dbc_norm.IPE, 
                y = df_miroc_dbc_norm.disag_factor, 
                s= df_miroc_dbc_norm.return_period,
                marker = '^',
                color = 'peachpuff')    

    axs[4,1].scatter(x = df_miroc_dbc_lognorm.IPE, 
                y = df_miroc_dbc_lognorm.disag_factor, 
                s= df_miroc_dbc_lognorm.return_period,
                marker = '^',
                color = 'peachpuff')

    scatter2 = axs[0,1].scatter(x = df_miroc_eqm_gumbel.IPE, 
                y = df_miroc_eqm_gumbel.disag_factor, 
                s= df_miroc_eqm_gumbel.return_period,
                marker = 'o',
                color = 'thistle') 
    
    axs[1,1].scatter(x = df_miroc_eqm_genlog.IPE, 
                y = df_miroc_eqm_genlog.disag_factor, 
                s= df_miroc_eqm_genlog.return_period,
                marker = 'o',
                color = 'thistle') 
    
    axs[2,1].scatter(x = df_miroc_eqm_gev.IPE, 
                y = df_miroc_eqm_gev.disag_factor, 
                s= df_miroc_eqm_gev.return_period,
                marker = 'o',
                color = 'thistle') 

    axs[3,1].scatter(x = df_miroc_eqm_norm.IPE, 
                y = df_miroc_eqm_norm.disag_factor, 
                s= df_miroc_eqm_norm.return_period,
                marker = 'o',
                color = 'thistle')    

    axs[4,1].scatter(x = df_miroc_eqm_lognorm.IPE, 
                y = df_miroc_eqm_lognorm.disag_factor, 
                s= df_miroc_eqm_lognorm.return_period,
                marker = 'o',
                color = 'thistle') 

    scatter3 = axs[0,1].scatter(x = df_miroc_md_gumbel.IPE, 
                y = df_miroc_md_gumbel.disag_factor, 
                s= df_miroc_md_gumbel.return_period,
                marker = 's',
                color = 'powderblue') 
    
    axs[1,1].scatter(x = df_miroc_md_genlog.IPE, 
                y = df_miroc_md_genlog.disag_factor, 
                s= df_miroc_md_genlog.return_period,
                marker = 's',
                color = 'powderblue') 
    
    axs[2,1].scatter(x = df_miroc_md_gev.IPE, 
                y = df_miroc_md_gev.disag_factor, 
                s= df_miroc_md_gev.return_period,
                marker = 's',
                color = 'powderblue') 

    axs[3,1].scatter(x = df_miroc_md_norm.IPE, 
                y = df_miroc_md_norm.disag_factor, 
                s= df_miroc_md_norm.return_period,
                marker = 's',
                color = 'powderblue')    

    axs[4,1].scatter(x = df_miroc_md_lognorm.IPE, 
                y = df_miroc_md_lognorm.disag_factor, 
                s= df_miroc_md_lognorm.return_period,
                marker = 's',
                color = 'powderblue') 

    scatter4 = axs[0,1].scatter(x = df_miroc_pt_gumbel.IPE, 
                y = df_miroc_pt_gumbel.disag_factor, 
                s= df_miroc_pt_gumbel.return_period,
                marker = 'd',
                color = 'pink') 
    
    axs[1,1].scatter(x = df_miroc_pt_genlog.IPE, 
                y = df_miroc_pt_genlog.disag_factor, 
                s= df_miroc_pt_genlog.return_period,
                marker = 'd',
                color = 'pink') 
    
    axs[2,1].scatter(x = df_miroc_pt_gev.IPE, 
                y = df_miroc_pt_gev.disag_factor, 
                s= df_miroc_pt_gev.return_period,
                marker = 'd',
                color = 'pink') 

    axs[3,1].scatter(x = df_miroc_pt_norm.IPE, 
                y = df_miroc_pt_norm.disag_factor, 
                s= df_miroc_pt_norm.return_period,
                marker = 'd',
                color = 'pink')    

    axs[4,1].scatter(x = df_miroc_pt_lognorm.IPE, 
                y = df_miroc_pt_lognorm.disag_factor, 
                s= df_miroc_pt_lognorm.return_period,
                marker = 'd',
                color = 'pink') 
 
    scatter5 = axs[0,1].scatter(x = df_miroc_qm_gumbel.IPE, 
                y = df_miroc_qm_gumbel.disag_factor, 
                s= df_miroc_qm_gumbel.return_period,
                marker = 'P',
                color = 'lightgreen') 
    
    axs[1,1].scatter(x = df_miroc_qm_genlog.IPE, 
                y = df_miroc_qm_genlog.disag_factor, 
                s= df_miroc_qm_genlog.return_period,
                marker = 'P',
                color = 'lightgreen') 
    
    axs[2,1].scatter(x = df_miroc_qm_gev.IPE, 
                y = df_miroc_qm_gev.disag_factor, 
                s= df_miroc_qm_gev.return_period,
                marker = 'P',
                color = 'lightgreen') 

    axs[3,1].scatter(x = df_miroc_qm_norm.IPE, 
                y = df_miroc_qm_norm.disag_factor, 
                s= df_miroc_qm_norm.return_period,
                marker = 'P',
                color = 'lightgreen')    

    axs[4,1].scatter(x = df_miroc_qm_lognorm.IPE, 
                y = df_miroc_qm_lognorm.disag_factor, 
                s= df_miroc_qm_lognorm.return_period,
                marker = 'P',
                color = 'lightgreen')
        
    axs[4,0].set_xlabel('IPE')
    axs[4,1].set_xlabel('IPE')
    
    axs[0,0].set_ylabel('Gumbel')
    axs[1,0].set_ylabel('GenLogistic')
    axs[2,0].set_ylabel('GEV')
    axs[3,0].set_ylabel('Normal')           
    axs[4,0].set_ylabel('Lognormal')           
    
    min = 0
    max = 0.008
    axs[0,0].set_xlim([min, max])
    axs[1,0].set_xlim([min, max])
    axs[2,0].set_xlim([min, max])
    axs[3,0].set_xlim([min, max])
    axs[4,0].set_xlim([min, max])
    axs[0,1].set_xlim([min, max])
    axs[1,1].set_xlim([min, max])
    axs[2,1].set_xlim([min, max])
    axs[3,1].set_xlim([min, max])
    axs[4,1].set_xlim([min, max])
    
    
    handles, labels = scatter.legend_elements(prop="sizes", num = [2, 5, 10, 25, 50, 100], alpha=0.6)
    legend = fig.legend(handles, labels, 
                        loc="upper right", 
                        bbox_to_anchor = (1, 0.98), 
                        title="RP")    
    


if __name__ == '__main__':
    #df = pd.read_csv('Graphs/errors/error_IDF_historical_obs_base.csv')
    #df = pd.read_csv('Graphs/errors/error_IDF_historical_obs_average.csv')
    #df = pd.read_csv('Graphs/errors/error_IDF_historical_obs_inmet_aut_nan.csv')
    #print(df)
    #plot_error_INMET(df)
    
    #df = pd.read_csv('Graphs/errors/error_IDF_historical_proj_base.csv')
    df = pd.read_csv('Graphs/errors/error_IDF_historical_proj_average.csv')
    #df = pd.read_csv('Graphs/errors/error_IDF_historical_proj_inmet_aut_nan.csv')
    plot_error_GCM(df)

    plt.show()
