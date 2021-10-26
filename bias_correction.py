import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from functions_get_distribution import *
from xlwt import Workbook
import xlrd
import statistics


def fit_distribution(data):
    MY_DISTRIBUTIONS = [st.norm, st.lognorm, st.genextreme, st.gumbel_r, st.genlogistic]
    results = fit_data(data, MY_DISTRIBUTIONS)
    #plot_histogram(data, results, 5)
    mean = data.mean()
    goodness_of_fit(data, results, 5, mean, plot = False)
    df_parameters = get_parameters(data, results, 5)
    #print(df_parameters)
    dist_name = df_parameters['distribution'][0]
    if dist_name == 'Generalized Logistic':
        dist = st.genlogistic
    elif dist_name == 'Normal':
        dist = st.norm
    elif dist_name == 'Lognormal':
        dist = st.lognorm
    elif dist_name == 'Gumbell':
        dist = st.gumbel_r
    elif dist_name == 'GEV':
        dist = st.genextreme
    else:
        print(dist_name)
        print('Insert dist name from MY_DISTRIBUTIONS')
        dist = input()
    
    print(dist)
    #input()
    return dist

def get_distribution(data, dist):
    MY_DISTRIBUTIONS = [dist]
    results = fit_data(data, MY_DISTRIBUTIONS)
    df_parameters = get_parameters(data, results, 5)
    dist = MY_DISTRIBUTIONS[0]
    c = df_parameters['c'][0]
    loc = df_parameters['loc'][0]
    scale = df_parameters['scale'][0]
    #print(c, loc, scale)
    
    if math.isnan(c) == True:
        prob_function_obj = dist(loc, scale)
    else:
        prob_function_obj = dist(c, loc, scale)
    
    return prob_function_obj

def quantile_mapping_baseline(data_obs, data_gcm_baseline):
    #Step 1 - fit distrubution in GCM and in obseved data
    dist = fit_distribution(data_gcm_baseline)
    dist_gcm = get_distribution(data_gcm_baseline, dist)
    
    dist = fit_distribution(data_obs)
    dist_obs = get_distribution(data_obs, dist) #5min
    
    #Step 2 - spatial downscale
    #y_stn_max = CDF(invCDF(X_GCM_max/teta_GCM)/teta_stn,j)
    #y_stn_max = statiscally downscaled sub-daily maximum series for jth duration ##aqui sera chamado de data_spatdown_5 (for 5min...)
    #X_GCM_max = maximum daily precipitation from GCM model ## aqui eh data_gcm_baseline
    #teta_GCM = the parameter of the fitted distribution for the maximum daily precipitation for GCM model ## aqui eh dist_gcm_baseline
    #teta_stn,j = the parameter of the fitted distribution for the maximum sub-daily precipitation at station STN for jth duration
    
    #invCDF(X_GCM_max/teta_GCM): the inverse of the CDF is the quantile function. In scipy it is called ppf function (percent point function)
    invCDF_gcm = dist_gcm.cdf(data_gcm_baseline)
    data_spatdown = dist_obs.ppf(invCDF_gcm)
    data_spatdown[data_spatdown<0] = 0
    #print(type(data_spatdown))
    #input()

    return data_spatdown

def quantile_mapping_future(data_obs, data_gcm_baseline, data_gcm_future):
    data_spatdown = quantile_mapping_baseline(data_obs, data_gcm_baseline)
    #Step 3 - Fit an equation to relate (y)data_spatdown and (x)data_gcm_baseline
    #print('Step 3 running... getting spatial downscale regression coefs')
    x = np.log(data_gcm_baseline)
    y = data_spatdown

    model = np.polyfit(x,y,1)
    a1 = model[0]
    b1 = model[1]
    predict = np.poly1d(model)
    r_square = r2_score(y, predict(x))
    print('Coefs - 5min')
    print('a: ', a1, ', b: ', b1, ', r-square: ', r_square)

    plt.scatter(x,y)
    plt.plot(x,predict(x))

    #Step 4 - Use equation to obtain future  data spatial downscaled
    data_spatdown_future = a1*(data_gcm_future) + b1
    
    return data_spatdown_future

def correct_baseline_by_quantile_mapping(name_gcm, name_obs):
    bias_correction_method = 'QM_adj'

    if name_gcm == 'HADGEM':
        df_baseline = pd.read_csv('GCM_data/{n}_baseline_complete.csv'.format(n = name_gcm))
    else:
        df_baseline = pd.read_csv('GCM_data/{n}_baseline.csv'.format(n = name_gcm))
    
    df_obs_data = pd.read_csv('{n_obs}'.format(n_obs = name_obs))
    
    data_obs = df_obs_data[['Precipitation']].dropna().values.ravel()    
    data_gcm_baseline = df_baseline[['Precipitation']].dropna().values.ravel()  
  
    baseline_corrected = quantile_mapping_baseline(data_obs, data_gcm_baseline)
    print(baseline_corrected)
    #print(future_corrected)
    
    df_baseline['Precipitation_corrected'] = baseline_corrected
    if name_gcm == 'HADGEM':
        df_baseline.columns = ['Date', 'Precipitation_original', 'Year', 'Month', 'Day', 'Precipitation']
    else:
        df_baseline.columns = ['Year', 'Month', 'Day', 'Precipitation_original', 'Precipitation']
    #print(df_future_model)
    df_baseline.to_csv('GCM_data/bias_correction/{g}_baseline_{bc}_daily.csv'.format(g = name_gcm, bc = bias_correction_method), index = False)

def correct_future_by_quantile_mapping(name_gcm, scenario, name_obs):
    bias_correction_method = 'QM_adj'

    if scenario == 'rcp 4.5':
        scenario_2 = 'rcp45'
    elif scenario == 'rcp 8.5':
        scenario_2 = 'rcp85'
    else:
        scenario_2 = scenario

    if name_gcm == 'HADGEM':
        df_baseline = pd.read_csv('GCM_data/{n}_baseline_complete.csv'.format(n = name_gcm))
        df_gcm_future = pd.read_csv('GCM_data/{n}_{sc}_complete.csv'.format(n = name_gcm, sc = scenario_2))
    else:
        df_baseline = pd.read_csv('GCM_data/{n}_baseline.csv'.format(n = name_gcm))
        df_gcm_future = pd.read_csv('GCM_data/{n}_{sc}.csv'.format(n = name_gcm, sc = scenario_2))
    
    df_obs_data = pd.read_csv(name_obs)
    
    data_obs = df_obs_data[['Precipitation']].dropna().values.ravel()    
    data_gcm_baseline = df_baseline[['Precipitation']].dropna().values.ravel()  
    data_gcm_future = df_gcm_future[['Precipitation']].dropna().values.ravel()  
  
    future_corrected = quantile_mapping_future(data_obs, data_gcm_baseline, data_gcm_future)
    print(future_corrected)
    
    df_gcm_future['Precipitation_corrected'] = future_corrected
    if name_gcm == 'HADGEM':
        df_gcm_future.columns = ['Date', 'Precipitation_original', 'Year', 'Month', 'Day', 'Precipitation']
    else:
        df_gcm_future.columns = ['Year', 'Month', 'Day', 'Precipitation_original', 'Precipitation']
    #print(df_future_model)
    df_gcm_future.to_csv('GCM_data/bias_correction/{g}_{sc}_{bc}_daily.csv'.format(g = name_gcm, sc = scenario_2, bc = bias_correction_method), index = False)

    
## DBC bias correction
def dbc_calib_valid(name_gcm, nyears=20):
    #nyears = number of years to calibrate and validate
    df = pd.read_csv('GCM_data\dbc_bias_correction\{n}_baseline_to_dbc.csv'.format(n = name_gcm))
    df = df.dropna()
    #print(df)
    df.to_excel('GCM_data\dbc_bias_correction\{n}_baseline_to_dbc.xls'.format(n = name_gcm), header = False, index = False)
    #input()
    
    wb = Workbook()
    wb1=Workbook()
    wb2=Workbook()
    th=1
    op=xlrd.open_workbook('GCM_data\dbc_bias_correction\{n}_baseline_to_dbc.xls'.format(n = name_gcm))
    sheet=op.sheet_by_index(0)
    vobs=[]
    v=[]
    y=0
    x1=0
    year=0
    
    np=1000 #number of percentiles
    percfact=[[[] for e1 in range (0,np)]for e in range(0,sheet.ncols-4)]
    while y < nyears+1: #create list with precipitation values
        prec = sheet.cell_value(x1, 0)
        #print('prec: ', prec)
        #input()
        if prec > 0.001:
            vobs.append(prec)
        x1 +=1
        
        if year< sheet.cell_value(x1, 1):
            y+=1
        year = sheet.cell_value(x1, 1)
        
    vobs.sort()
    sheet1 = wb1.add_sheet('percentfactors')
    sheet2=wb.add_sheet('correctedfinal')
    sheet3=wb2.add_sheet('th')
    for col in range (4,sheet.ncols):#calibrate and correct remaining years
        v = []
        Loci=[]
        print('Number of GCMs corrected:')
        print(col-3,'out of',sheet.ncols-4)
        for x in range(0,x1):
            prec=sheet.cell_value(x,col)
            v.append(prec)
        v.sort()
        th = v[len(v) - len(vobs)]
    
        for x in range(0,len(v)):
            if v[x]>th:
                Loci.append(v[x])
    
    
        for p in range (0,np):
    
            percfact[col-4][p]=vobs[int((p+1)/np*(len(vobs)-1))]/Loci[int((p+1)/np*(len(Loci)-1))]
    
        for e in range (0,np):
            sheet1.write(e, col-4, percfact[col-4][e])
        sheet3.write(0, col - 4, th)
    
        ## Inseri isso aqui para calcular pra toda a serie
        x1 = 0
        for x in range(x1,sheet.nrows):
            #print(x1, sheet.nrows)
            #input()
            prec=sheet.cell_value(x,col)
            #print('prec: ', prec)
            #input()
            if prec > th:
                ptl = 0
    
                while float(prec) > float(Loci[int((ptl + 1) / np * (len(Loci) - 1))]) and ptl < np-1:
                    ptl+=1
                sheet2.write(x-x1, col-4, percfact[col-4][ptl]*prec)
    
            else:
                sheet2.write(x-x1, col-4, 0)
            
            #.write(linha, coluna, texto)
            year = sheet.cell_value(x, 1)
            month = sheet.cell_value(x, 2)
            day = sheet.cell_value(x, 3)
            #print(x, x1, x-x1)
            #print(year, month, day)
            #input()
            
            sheet2.write(x-x1, col - 3, year)
            sheet2.write(x-x1, col - 2, month)
            sheet2.write(x-x1, col - 1, day)
    
    wb1.save('GCM_data\dbc_bias_correction\{n}_percentfactors.xls'.format(n = name_gcm))
    wb.save('GCM_data\dbc_bias_correction\{n}_baseline_validation.xls'.format(n = name_gcm))
    wb2.save('GCM_data\dbc_bias_correction\{n}_tresholds.xls'.format(n = name_gcm))
    

def correct_future_by_dbc(name_gcm, scenario):
    read=Workbook()
    wb=Workbook()
    op=xlrd.open_workbook('GCM_data\dbc_bias_correction\{n}_{s}_to_dbc.xlsx'.format(n = name_gcm, s = scenario))
    tresh=xlrd.open_workbook('GCM_data\dbc_bias_correction\{n}_tresholds.xls'.format(n = name_gcm))
    perc=xlrd.open_workbook('GCM_data\dbc_bias_correction\{n}_percentfactors.xls'.format(n = name_gcm))
    per=perc.sheet_by_index(0)
    all45=op.sheet_by_index(0)
    thold=tresh.sheet_by_index(0)
    sheet1=wb.add_sheet('2025-2050')
    sheet2=wb.add_sheet('2050-2075')
    sheet3=wb.add_sheet('2075-2100')
    #all85=op.sheet_by_index(1)
    v=[]
    np=per.nrows
    ptl=0
    nofyears=95
    for c in range (3,all45.ncols):
        print(c-3,all45.ncols)
        y=0
        t=0
        sr=-1
        percfact = 0
        th=thold.cell_value(0,c-3)
        year=all45.cell_value(0,0)
        year0=all45.cell_value(0,0)
        for r in range(0,all45.nrows):
            year=all45.cell_value(r,0)
    
            if all45.cell_value(r,c)> th:
                v.append(all45.cell_value(r,c))
            sr += 1
            if year>year0 or r==all45.nrows-1:
                y+=1
            year0=all45.cell_value(r,0)
            if y==nofyears:
                y=0
                cont = 0
                t+=1
                v.sort()
                Loci = [i for i in v]
                sr1=sr
                sr=0
                for r1 in range(r-sr1, r):
    
    
                    ptl=0
                    prec=0
    
                    if all45.cell_value(r1,c)>=th:
                        prec=all45.cell_value(r1,c)
    
                    while float(prec) > float(Loci[int((ptl) / np * (len(Loci) - 1))]) and int((ptl + 1) / np * (len(Loci) - 1))< len(Loci):
                        percfact = per.cell_value(ptl, c-3)
                        ptl+=1
    
    
                    if t==1:
                        sheet1.write(cont, c-3, percfact * prec)
                    if t==2:
                        sheet2.write(cont, c-3, percfact * prec)
                    if t==3:
                        sheet3.write(cont, c-3, percfact * prec)
                    cont+=1
    
                Loci=[]
                v=[]
    
    wb.save('GCM_data\dbc_bias_correction\{n}_{s}_DBC_daily.xls'.format(n = name_gcm, s = scenario))

   
## Bias correction pelo CMhyd
def get_CMhyd_file(name_gcm, scenario, bias_correction_method):
    if name_gcm == 'HADGEM':
        if scenario == 'baseline':
            if bias_correction_method == 'MD':
                df = pd.read_csv('CMhyd\projeta 8.5 hadgem 5km\Mapping distribution\PCP\MOD\DistributionMapping\historical\historico_dm_hist.txt', header = None)
            elif bias_correction_method == 'PT':
                df = pd.read_csv('CMhyd\projeta 8.5 hadgem 5km\Power transformation\PCP\MOD\PowerTransformation\historical\historico_pt_hist.txt', header = None)
            else:
                print('Bias correction not performed')
        elif scenario == 'rcp 4.5':
            if bias_correction_method == 'MD':
                df = pd.read_csv('CMhyd\projeta 4.5 hadgem 5km\saida mapping\PCP\MOD\DistributionMapping\EXP\\futuro_dm_sce.txt', header = None)
            elif bias_correction_method == 'PT':
                df = pd.read_csv('CMhyd\projeta 4.5 hadgem 5km\saida power transformation\PCP\MOD\PowerTransformation\EXP\\futuro_pt_sce.txt', header = None)
            else:
                print('Bias correction not performed')
        elif scenario == 'rcp 8.5':
            if bias_correction_method == 'MD':
                df = pd.read_csv('CMhyd\projeta 8.5 hadgem 5km\Mapping distribution\PCP\MOD\DistributionMapping\EXP\\futuro_dm_sce.txt', header = None)
            elif bias_correction_method == 'PT':
                df = pd.read_csv('CMhyd\projeta 8.5 hadgem 5km\Power transformation\PCP\MOD\PowerTransformation\EXP\\futuro_pt_sce.txt', header = None)
            else:
                print('Bias correction not performed')
        else:
            print('Scenario not bias corrected through CMhyd')
    
    elif name_gcm == 'MIROC5':
        if scenario == 'baseline':
            if bias_correction_method == 'MD':
                df = pd.read_csv('CMhyd\projeta 8.5 miroc5\saida mapping\PCP\MOD\DistributionMapping\historical\historico_dm_hist.txt', header = None)
            elif bias_correction_method == 'PT':
                df = pd.read_csv('CMhyd\projeta 8.5 miroc5\saida power transformation\PCP\MOD\PowerTransformation\historical\historico_pt_hist.txt', header = None)
            else:
                print('Bias correction not performed')
        elif scenario == 'rcp 4.5':
            if bias_correction_method == 'MD':
                df = pd.read_csv('CMhyd\projeta 4.5 miroc5\saida mapping\PCP\MOD\DistributionMapping\EXP\\futuro_dm_sce.txt', header = None)
            elif bias_correction_method == 'PT':
                df = pd.read_csv('CMhyd\projeta 4.5 miroc5\saida power transformation\PCP\MOD\PowerTransformation\EXP\\futuro_pt_sce.txt', header = None)
            else:
                print('Bias correction not performed')
        elif scenario == 'rcp 8.5':
            if bias_correction_method == 'MD':
                df = pd.read_csv('CMhyd\projeta 8.5 miroc5\saida mapping\PCP\MOD\DistributionMapping\EXP\\futuro_dm_sce.txt', header = None)
            elif bias_correction_method == 'PT':
                df = pd.read_csv('CMhyd\projeta 8.5 miroc5\saida power transformation\PCP\MOD\PowerTransformation\EXP\\futuro_pt_sce.txt', header = None)
            else:
                print('Bias correction not performed')
        else:
            print('Scenario not bias corrected through CMhyd')        
                        
            

    list_complete = df[0].to_list()
    prec_list = []
    for i in range(1,len(list_complete)):
        if list_complete[i] == -99.0:
            prec = 0
        else:
            prec = list_complete[i]
        prec_list.append(prec)
    #print(prec_list)

    date_begin_raw = list_complete[0]    
    #print(date_begin_raw)
    year = int(str(date_begin_raw)[:4])
    month = int(str(date_begin_raw)[4:6])
    day = int(str(date_begin_raw)[6:8])
    #print(year, month, day)
    date_begin = date(year, month, day)
    #print(date_begin)
    numdays = len(prec_list)
    date_list = [date_begin + timedelta(days=x) for x in range(numdays)]
    #print(date_list)
    dict_ = {'Date': date_list,
             'Precipitation': prec_list}
    df_new = pd.DataFrame(dict_)
    df_new['Year'] = pd.DatetimeIndex(df_new['Date']).year
    df_new['Month'] = pd.DatetimeIndex(df_new['Date']).month
    df_new['Day'] = pd.DatetimeIndex(df_new['Date']).day
    
    if scenario == 'rcp 4.5':
        scenario_2 = 'rcp45'
    elif scenario == 'rcp 8.5':
        scenario_2 = 'rcp85'
    else:
        scenario_2 = scenario

    df_new.to_csv('GCM_data/bias_correction/{g}_{s}_{bc}_daily.csv'.format(g = name_gcm, s = scenario_2, bc = bias_correction_method), index = False)
    

if __name__ == '__main__':
    print('Starting..')
    
    print('')
    print('DBC bias correction..')
     
    dbc_calib_valid('MIROC5', 43)
    #correct_future_by_dbc('HADGEM', 'baseline2')
    
#     print('')
#     print('Quantile mapping bias correction..')
#     
#     name_obs = 'Results/INMET_conv_daily_2.csv'
#     scenario = 'rcp 4.5'
#   
#     correct_baseline_by_quantile_mapping('HADGEM', name_obs)
#     correct_baseline_by_quantile_mapping('MIROC5', name_obs)
#     correct_future_by_quantile_mapping('HADGEM', scenario, name_obs)
#     correct_future_by_quantile_mapping('MIROC5', scenario, name_obs)
#      
#     print('')
#     print('Getting PT and MD files..')
#     #name_gcm = 'HADGEM'
#     #scenario = 'rcp 4.5'
#     #bias_correction_method = 'MD'
#      
#     get_CMhyd_file('HADGEM', 'baseline', 'MD')
#     get_CMhyd_file('HADGEM', 'rcp 4.5', 'MD')
#     get_CMhyd_file('HADGEM', 'rcp 8.5', 'MD')
#  
#     get_CMhyd_file('HADGEM', 'baseline', 'PT')
#     get_CMhyd_file('HADGEM', 'rcp 4.5', 'PT')
#     get_CMhyd_file('HADGEM', 'rcp 8.5', 'PT')
#  
#     get_CMhyd_file('MIROC5', 'baseline', 'MD')
#     get_CMhyd_file('MIROC5', 'rcp 4.5', 'MD')
#     get_CMhyd_file('MIROC5', 'rcp 8.5', 'MD')
#  
#     get_CMhyd_file('MIROC5', 'baseline', 'PT')
#     get_CMhyd_file('MIROC5', 'rcp 4.5', 'PT')
#     get_CMhyd_file('MIROC5', 'rcp 8.5', 'PT')    
    
    print('')
    print('Done!')
    

    
    
