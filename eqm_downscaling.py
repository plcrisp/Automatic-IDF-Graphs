from functions_get_distribution import *
from sklearn.metrics import r2_score
from datetime import date

## Definitions of the functions developed to treat the PROJETA data

def get_data_columns_for_hadgem(df):
#This function is used to get date values from HADGEM files, because they do not have a pattern

    list_i = range(len(df))
    #print(list_i)
    year_list = []
    month_list = []
    day_list = []
    
    for i in list_i:
        #print(i)
        if df['Data'][i].find('-') == -1: 
            data = df['Data'][i].split("/")
            #print(data)
            #input()
            day = data[0]
            month = data[1]
            year = data[2]
        else:
            data = df['Data'][i].split("-")
            year = data[0]
            month = data[1]
            day = data[2]
            
        year_list.append(year)
        month_list.append(month)
        day_list.append(day)
        
    df['Year'] = year_list
    df['Month'] = month_list
    df['Day'] = day_list
    
    return df
    
def process_gcm_data(name_gcm, scenario):
#This function will format the dataframe obtained by the PROJETA files in the format that we need to proceed with the EQM downscaling and IDF construction

    df = pd.read_csv('PROJETA/{n}_{s}_orig.csv'.format(n = name_gcm, s = scenario), sep = ';')
    #print(df)
    #input()
    if name_gcm == 'MIROC5':
        df[['Year', 'Month', 'Day']] = df.Data.str.split("-", expand=True)
    if name_gcm == 'HADGEM':
        df = get_data_columns_for_hadgem(df)
    
    df_new = df[['Year', 'Month', 'Day', 'PREC']]
    df_new.columns = ['Year', 'Month', 'Day', 'Precipitation']
    #print(df_new)
    df_new.to_csv('GCM_data/{n}_{s}.csv'.format(n = name_gcm, s = scenario), index = False)
    
def set_date(name):
#This is an intermediate function to set the date in the corrected dataframe

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
    df['Date'] = pd.to_datetime(df.Date)
    df = df.sort_values(by = 'Date')
    df = df.reset_index()
    df = df.set_index('Date')
    del df['index']
    #print(df)
    df['Date'] = df.index
    df = df[~df.index.duplicated()]
    df = df[['Date', 'Precipitation']]
    return df
    
def complete_date_series(name_gcm, scenario):
# This function complete the date if it is missing (usually it will be used for HADGEM)

    df_original = pd.read_csv('GCM_data/{n}_{s}.csv'.format(n = name_gcm, s = scenario))
    df = set_date(df_original)
    #print(df)
    #input()
     
    d0 = df['Date'].min()
    di = df['Date'].max()
    #print(d0, di)
    idx = pd.date_range(d0, di)
      
    df.index = pd.DatetimeIndex(df.index)
    df = df.reindex(idx)
    df['Date'] = df.index
    #print(df)
    df['Year'] = pd.DatetimeIndex(df['Date']).year
    df['Month'] = pd.DatetimeIndex(df['Date']).month
    df['Day'] = pd.DatetimeIndex(df['Date']).day
    df['Precipitation'] = df[['Precipitation']].fillna(0)

    #print(df)
    #input()
    df.to_csv('GCM_data/{n}_{s}_complete.csv'.format(n = name_gcm, s = scenario), index = False)
    #return df

def get_gcm_daily_max(name_gcm, scenario):
# This function calculate the daily max of the projected series

    df = pd.read_csv('GCM_data/{n}_{s}.csv'.format(n = name_gcm, s = scenario))
    df = df.dropna()
    df_new = df.groupby(['Year'])['Precipitation'].max().reset_index()
    #print(df_new)
    df_new.to_csv('GCM_data/{n}_{s}_max_daily.csv'.format(n = name_gcm, s = scenario), index = False)   

    return df_new   


## Definition of the functions used for the EQM

def get_gcm_daily_max_data_to_fit(name_gcm, scenario):
    df_gcm = pd.read_csv('GCM_data/{n}_{s}_max_daily.csv'.format(n = name_gcm, s = scenario))
    
    data_df = df_gcm[['Precipitation']]
    data_gcm = data_df.values.ravel()
    
    return data_gcm

def get_hist_subdaily_max_data_to_fit(name_file, disag_factor = 'nan'):
    if disag_factor == 'nan':
        df_hist = pd.read_csv('Results/max_subdaily_{n}.csv'.format(n = name_file))
    elif disag_factor == 'bl':
        df_hist = pd.read_csv('bartlet_lewis/max_subdaily_complete_{n}_bl.csv'.format(n = name_file, d = disag_factor))
    else:
        df_hist = pd.read_csv('Results/max_subdaily_{n}_{d}.csv'.format(n = name_file, d = disag_factor))
        
    #duration_list = ['5min', '10min', '20min', '30min', '1', '6', '8', '10', '12', '24']
    #pensei em fazer um for pro duration_list, mas eu nao sei como retornaria as listas depois no return.
    
    data_df_5 = df_hist[['Max_5min']]
    data_5 = data_df_5.values.ravel()

    data_df_10 = df_hist[['Max_10min']]
    data_10 = data_df_10.values.ravel()
    
    data_df_20 = df_hist[['Max_20min']]
    data_20 = data_df_20.values.ravel()

    data_df_30 = df_hist[['Max_30min']]
    data_30 = data_df_30.values.ravel()
    
    data_df_60 = df_hist[['Max_1']]
    data_60 = data_df_60.values.ravel()
    
    data_df_360 = df_hist[['Max_6']]
    data_360 = data_df_360.values.ravel()
    
    data_df_480 = df_hist[['Max_8']]
    data_480 = data_df_480.values.ravel()
    
    data_df_600 = df_hist[['Max_10']]
    data_600 = data_df_600.values.ravel()
    
    data_df_720 = df_hist[['Max_12']]
    data_720 = data_df_720.values.ravel()

    data_df_1440 = df_hist[['Max_24']]
    data_1440 = data_df_1440.values.ravel()
    
    return data_5, data_10, data_20, data_30, data_60, data_360, data_480, data_600, data_720, data_1440

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

def plot_cdf_theoretical(dist_function):        
    x =  np.linspace(dist_function.ppf(0.01), dist_function.ppf(0.99), 100)
    #print('')
    #print('x to plot cdf')
    #print(x)
    cdf = dist_function.cdf(x)
    #print('cdf')
    #print(cdf)
    plt.plot(x,cdf)
    plt.ylabel('Probability')
    plt.xlabel('Max_precipitation (mm)')
    plt.title('CDF_theoretical')
    #plt.show()
    
def plot_cdf_observed(data):
    x = np.sort(data)
    y = np.arange(len(x))/float(len(x))
    plt.plot(x, y)
    plt.ylabel('Probability')
    plt.xlabel('Max_precipitation (mm)')
    plt.title('CDF_empirical')    

def EQM_main(name_gcm, scenario, name_hist_file, disag_factor):
    ##Step 1 - get daily max for GCMs in baseline and future, and subdaily max for observed baseline
    print('Step 1 running...')
    #baseline
    df_gcm = get_gcm_daily_max(name_gcm, 'baseline')
    data_gcm_baseline = get_gcm_daily_max_data_to_fit(name_gcm, 'baseline')
    #future
    df_gcm_fut = get_gcm_daily_max(name_gcm, scenario)
    data_gcm_future = get_gcm_daily_max_data_to_fit(name_gcm, scenario)
    #historical
    data_5, data_10, data_20, data_30, data_60, data_360, data_480, data_600, data_720, data_1440 = get_hist_subdaily_max_data_to_fit(name_hist_file, disag_factor)
    
    ##Step 2 - get probability distribution
    print('')
    print('Step 2 running...')
    #baseline
    #print('')
    #print('---> Baseline')
    dist = fit_distribution(data_gcm_baseline)
    dist_gcm_baseline = get_distribution(data_gcm_baseline, st.genlogistic)
    #future
    #print('')
    #print('---> Future')
    dist = fit_distribution(data_gcm_future)
    dist_gcm_future = get_distribution(data_gcm_future, dist)
    #historical
    #print('')
    #print('---> Historical')
    dist = fit_distribution(data_5)
    dist_hist_5 = get_distribution(data_5, dist) #5min
    dist = fit_distribution(data_10)
    dist_hist_10 = get_distribution(data_10, dist) #10min
    dist = fit_distribution(data_20)
    dist_hist_20 = get_distribution(data_20, dist) #20min
    dist = fit_distribution(data_30)
    dist_hist_30 = get_distribution(data_30, dist) #30min
    dist = fit_distribution(data_60)
    dist_hist_60 = get_distribution(data_60, dist) #1h
    dist = fit_distribution(data_360)
    dist_hist_360 = get_distribution(data_360, dist) #6h
    dist = fit_distribution(data_480)
    dist_hist_480 = get_distribution(data_480, dist) #8h
    dist = fit_distribution(data_600)
    dist_hist_600 = get_distribution(data_600, dist) #10h
    dist = fit_distribution(data_720)
    dist_hist_720 = get_distribution(data_720, dist) #12h
    dist = fit_distribution(data_1440)
    dist_hist_1440 = get_distribution(data_1440, dist) #24h
    
#     #plot cdf to each subdaily (5min just to examplify)
#     plot_cdf_theoretical(dist_gcm_baseline)
#     plot_cdf_theoretical(dist_gcm_future)
#     plot_cdf_theoretical(dist_hist_5)
#     plot_cdf_theoretical(dist_hist_1440)
#     #plt.show()
#  
#     plot_cdf_observed(data_gcm_baseline)
#     plot_cdf_observed(data_gcm_future)
#     plot_cdf_observed(data_5)
#     plot_cdf_observed(data_1440)
#     #plt.show()
    
    ##Step 3 - Spatially downscale the data from the GCM daily maximuns to sub-daily maximuns
    print('')
    print('Step 3 running...')
    #y_stn_max = CDF(invCDF(X_GCM_max/teta_GCM)/teta_stn,j)
    #y_stn_max = statiscally downscaled sub-daily maximum series for jth duration ##aqui sera chamado de data_spatdown_5 (for 5min...)
    #X_GCM_max = maximum daily precipitation from GCM model ## aqui eh data_gcm_baseline
    #teta_GCM = the parameter of the fitted distribution for the maximum daily precipitation for GCM model ## aqui eh dist_gcm_baseline
    #teta_stn,j = the parameter of the fitted distribution for the maximum sub-daily precipitation at station STN for jth duration
    
    #invCDF(X_GCM_max/teta_GCM): the inverse of the CDF is the quantile function. In scipy it is called ppf function (percent point function)
    invCDF_gcm = dist_gcm_baseline.cdf(data_gcm_baseline)
    
    data_spatdown_5 = dist_hist_5.ppf(invCDF_gcm)
    data_spatdown_10 = dist_hist_10.ppf(invCDF_gcm)
    data_spatdown_20 = dist_hist_20.ppf(invCDF_gcm)
    data_spatdown_30 = dist_hist_30.ppf(invCDF_gcm)
    data_spatdown_60 = dist_hist_60.ppf(invCDF_gcm)
    data_spatdown_360 = dist_hist_360.ppf(invCDF_gcm)
    data_spatdown_480 = dist_hist_480.ppf(invCDF_gcm)
    data_spatdown_600 = dist_hist_600.ppf(invCDF_gcm)
    data_spatdown_720 = dist_hist_720.ppf(invCDF_gcm)
    data_spatdown_1440 = dist_hist_1440.ppf(invCDF_gcm)

    
    #print(data_gcm_baseline)
    #print(invCDF_gcm)
#     print('------5min------')
#     print(data_spatdown_5)
#     print('------10min------')
#     print(data_spatdown_10)
#     print('------20min------')
#     print(data_spatdown_20)
#     print('------30min------')
#     print(data_spatdown_30)
#     print('------1h------')
#     print(data_spatdown_60)
#     print('------6h------')
#     print(data_spatdown_360)
#     print('------8h------')
#     print(data_spatdown_480)
#     print('------10h------')
#     print(data_spatdown_600)
#     print('------12h------')
#     print(data_spatdown_720)
#     print('------24h------')
#     print(data_spatdown_1440)
    
    ##Step 4 - Temporal downscale the data from the projected GCM simulations of daily maximum to baseline GCM daily maximuns
    print('')
    print('Step 4 running...')
    #y_gcm_fut_max = CDF(invCDF(X_GCM_max/teta_gcm)/teta_gcm_fut)
    #y_gcm_fut_max = temporal downscaled daily maximum series for future scenario with baseline scenario ##aqui sera chamado de data_tempdown
    #X_GCM_max = maximum daily precipitation from GCM model
    #teta_GCM = the parameter of the fitted distribution for the maximum daily precipitation for GCM model
    #teta_gcm_fut = the parameter of the fitted distribution for the maximum daily precipitation for GCM future scenario ##aqui eh dist_gcm_fut
    
    data_tempdown = dist_gcm_future.ppf(invCDF_gcm)
    #print(len(data_gcm_baseline), len(invCDF_gcm), len(data_spatdown_10), len(data_tempdown))
    #print(data_tempdown)
    #input()

    ##Step 5 - Fit an equation to relate (y)data_spatdown and (x)data_gcm_baseline
    print('')
    print('Step 5 running... getting spatial downscale regression coefs')
    x = np.log(data_gcm_baseline)
    #5min
    y = data_spatdown_5

    model = np.polyfit(x,y,1)
    a1_5 = model[0]
    b1_5 = model[1]
    predict = np.poly1d(model)
    r_square = r2_score(y, predict(x))
    print('Coefs - 5min')
    print('a: ', a1_5, ', b: ', b1_5, ', r-square: ', r_square)

    plt.scatter(x,y)
    plt.plot(x,predict(x))
    plt.title('5min')
    #plt.show()
    
    #10min
    y = data_spatdown_10

    model = np.polyfit(x,y,1)
    a1_10 = model[0]
    b1_10 = model[1]
    predict = np.poly1d(model)
    r_square = r2_score(y, predict(x))
    print('Coefs - 10min')
    print('a: ', a1_10, ', b: ', b1_10, ', r-square: ', r_square)

    plt.scatter(x,y)
    plt.plot(x,predict(x))
    plt.scatter(x,y)
    plt.title('10min')
    #plt.show()

    #20min
    y = data_spatdown_20
    model = np.polyfit(x,y,1)
    a1_20 = model[0]
    b1_20 = model[1]
    predict = np.poly1d(model)
    r_square = r2_score(y, predict(x))
    print('Coefs - 20min')
    print('a: ', a1_20, ', b: ', b1_20, ', r-square: ', r_square)

    plt.scatter(x,y)
    plt.plot(x,predict(x))
    plt.title('20min')
    #plt.show()

    #30min
    y = data_spatdown_30
    model = np.polyfit(x,y,1)
    a1_30 = model[0]
    b1_30 = model[1]
    predict = np.poly1d(model)
    r_square = r2_score(y, predict(x))
    print('Coefs - 30min')
    print('a: ', a1_30, ', b: ', b1_30, ', r-square: ', r_square)

    plt.scatter(x,y)
    plt.plot(x,predict(x))
    plt.title('30min')
    #plt.show()
    
    #60min
    y = data_spatdown_60
    model = np.polyfit(x,y,1)
    a1_60 = model[0]
    b1_60 = model[1]
    predict = np.poly1d(model)
    r_square = r2_score(y, predict(x))
    print('Coefs - 60min')
    print('a: ', a1_60, ', b: ', b1_60, ', r-square: ', r_square)

    plt.scatter(x,y)
    plt.plot(x,predict(x))
    plt.title('60min')
    #plt.show()
    
    #360min
    y = data_spatdown_360
    model = np.polyfit(x,y,1)
    a1_360 = model[0]
    b1_360 = model[1]
    predict = np.poly1d(model)
    r_square = r2_score(y, predict(x))
    print('Coefs - 360min')
    print('a: ', a1_360, ', b: ', b1_360, ', r-square: ', r_square)

    plt.scatter(x,y)
    plt.plot(x,predict(x))
    plt.title('360min')
    #plt.show()
    
    #480min
    y = data_spatdown_480
    model = np.polyfit(x,y,1)
    a1_480 = model[0]
    b1_480 = model[1]
    predict = np.poly1d(model)
    r_square = r2_score(y, predict(x))
    print('Coefs - 480min')
    print('a: ', a1_480, ', b: ', b1_480, ', r-square: ', r_square)

    plt.scatter(x,y)
    plt.plot(x,predict(x))
    plt.title('480min')
    #plt.show()
    
    #600min
    y = data_spatdown_600
    model = np.polyfit(x,y,1)
    a1_600 = model[0]
    b1_600 = model[1]
    predict = np.poly1d(model)
    r_square = r2_score(y, predict(x))
    print('Coefs - 600min')
    print('a: ', a1_600, ', b: ', b1_600, ', r-square: ', r_square)

    plt.scatter(x,y)
    plt.plot(x,predict(x))
    plt.title('600min')
    #plt.show()
    
    #720min
    y = data_spatdown_720
    model = np.polyfit(x,y,1)
    a1_720 = model[0]
    b1_720 = model[1]
    predict = np.poly1d(model)
    r_square = r2_score(y, predict(x))
    print('Coefs - 720min')
    print('a: ', a1_720, ', b: ', b1_720, ', r-square: ', r_square)

    plt.scatter(x,y)
    plt.plot(x,predict(x))
    plt.title('720min')
    #plt.show()

    #1440min
    y = data_spatdown_720
    model = np.polyfit(x,y,1)
    a1_1440 = model[0]
    b1_1440 = model[1]
    predict = np.poly1d(model)
    r_square = r2_score(y, predict(x))
    print('Coefs - 1440min')
    print('a: ', a1_1440, ', b: ', b1_1440, ', r-square: ', r_square)

    plt.scatter(x,y)
    plt.plot(x,predict(x))
    plt.title('1440min')
    #plt.show()
      
    ##Step 6 - Fit an equation to relate data_tempdown and data_gcm_baseline
    print('')
    print('Step 6 running... getting temporal downscale regression coefs')
    y = data_tempdown
    model = np.polyfit(x,y,1)
    a2 = model[0]
    b2 = model[1]
    predict = np.poly1d(model)
    r_square = r2_score(y, predict(x))
    print('Coefs - Temporal downscaling')
    print('a2: ', a2, ', b2: ', b2, ', r-square: ', r_square)
    plt.scatter(x,y)
    plt.plot(x, predict(x))   
    plt.title('Temporal downscale corr')
    #plt.show()


    ##Step 7 - Combine equations to obtain future generated data spatial and temporal downscaled
    #5min
    data_spattemp_future_5 = a1_5*((data_gcm_future - b2)/a2) + b1_5

    #10min
    data_spattemp_future_10 = a1_10*((data_gcm_future - b2)/a2) + b1_10
    
    #20min
    data_spattemp_future_20 = a1_20*((data_gcm_future - b2)/a2) + b1_20

    #30min
    data_spattemp_future_30 = a1_30*((data_gcm_future - b2)/a2) + b1_30

    #60min
    data_spattemp_future_60 = a1_60*((data_gcm_future - b2)/a2) + b1_60
    
    #360min
    data_spattemp_future_360 = a1_360*((data_gcm_future - b2)/a2) + b1_360

    #480min
    data_spattemp_future_480 = a1_480*((data_gcm_future - b2)/a2) + b1_480

    #600min
    data_spattemp_future_600 = a1_600*((data_gcm_future - b2)/a2) + b1_600

    #720min
    data_spattemp_future_720 = a1_720*((data_gcm_future - b2)/a2) + b1_720

    #480min
    data_spattemp_future_1440 = a1_1440*((data_gcm_future - b2)/a2) + b1_1440

    ##Create the dataframes
    #max_subdaily for baseline   
    year_list = df_gcm['Year'].to_list()
    dict_ = {'Year': year_list,
            'baseline': data_gcm_baseline,
            'Max_5min': data_spatdown_5,
            'Max_10min': data_spatdown_10,
            'Max_20min': data_spatdown_20,
            'Max_30min': data_spatdown_30,
            'Max_1': data_spatdown_60,
            'Max_6': data_spatdown_360,
            'Max_8': data_spatdown_480,
            'Max_10': data_spatdown_600,
            'Max_12': data_spatdown_720,
            'Max_24': data_spatdown_720
        }
     
    df = pd.DataFrame(dict_)
    df.to_csv('GCM_data/max_subdaily_{n}_{s}_{nh}_{d}_baseline.csv'.format(n = name_gcm, s = scenario, nh = name_hist_file, d = disag_factor), index = False)    

    #max_subdaily for future   
    year_list = df_gcm_fut['Year'].to_list()
    dict_ = {'Year': year_list,
            'baseline': data_gcm_future,
            'Max_5min': data_spattemp_future_5,
            'Max_10min': data_spattemp_future_10,
            'Max_20min': data_spattemp_future_20,
            'Max_30min': data_spattemp_future_30,
            'Max_1': data_spattemp_future_60,
            'Max_6': data_spattemp_future_360,
            'Max_8': data_spattemp_future_480,
            'Max_10': data_spattemp_future_600,
            'Max_12': data_spattemp_future_720,
            'Max_24': data_spattemp_future_720
        }
     
    df = pd.DataFrame(dict_)
    df.to_csv('GCM_data/max_subdaily_{n}_{s}_{nh}_{d}_future.csv'.format(n = name_gcm, s = scenario, nh = name_hist_file, d = disag_factor), index = False)       


## Main code for running

if __name__ == '__main__':
    print('Starting...')
    
    #duration_list should contain the durations used to construct the IDF
    duration_list = ['5min', '10min', '20min', '30min', '1', '6', '8', '10', '12', '24']
    duration_list_min = [5, 10, 20, 30, 60, 360, 480, 600, 720, 1440]

    MY_DISTRIBUTIONS = [st.genextreme, st.gumbel_r, st.genlogistic]
    
    #name_gcm can be HADGEM or MIROC5
    name_gcm = 'HADGEM'
    #scenario can be rcp45, rcp85 or baseline
    scenario = 'rcp45'
    #name_hist_file should be the name of the file containing the historical observed data
    name_hist_file = 'INMET_conv'
    #disag_factor can be nan (dont use disaggregatino factor), ger (original), m0.2 (minus 20%), p0.2 (plus 20%), bl (bartlett lewis)
    disag_factor = 'ger'
    
    #process_gcm_data(name_gcm, scenario)
    #process_gcm_data(name_gcm, 'baseline')
    
    #complete_date_series(name_gcm, 'baseline')
    complete_date_series(name_gcm, scenario)


    #EQM_main(name_gcm, scenario, name_hist_file, disag_factor)
    
    
    print('Done!')

    

