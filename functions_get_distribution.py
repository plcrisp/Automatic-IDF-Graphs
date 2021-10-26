import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
import random
from cmath import nan

#mpl.style.use("ggplot")

def danoes_formula(data):
    """
    DANOE'S FORMULA
    https://en.wikipedia.org/wiki/Histogram#Doane's_formula
    """
    N = len(data)
    skewness = st.skew(data)
    sigma_g1 = math.sqrt((6*(N-2))/((N+1)*(N+3)))
    num_bins = 1 + math.log(N,2) + math.log(1+abs(skewness)/sigma_g1,2)
    num_bins = round(num_bins)
    return num_bins

def plot_histogram(data, results, n):
    ## n first distribution of the ranking
    N_DISTRIBUTIONS = {k: results[k] for k in list(results)[:n]}
    #print(N_DISTRIBUTIONS)
    
    ## Histogram of data
    plt.figure(figsize=(10, 5))
    plt.hist(data, density=True, ec='white', color=(63/235, 149/235, 170/235))
    plt.title('HISTOGRAM')
    plt.xlabel('Values')
    plt.ylabel('Frequencies')

    ## Plot n distributions
    for distribution, result in N_DISTRIBUTIONS.items():
        # print(i, distribution)
        sse = result[0]
        arg = result[1]
        loc = result[2]
        scale = result[3]
        #print('Distribution: ', distribution, 'SSE: ', sse, 'c: ', arg, 'loc: ', loc, 'scale: ', scale)
        x_plot = np.linspace(min(data), max(data), 1000)
        y_plot = distribution.pdf(x_plot, loc=loc, scale=scale, *arg)
        plt.plot(x_plot, y_plot, label=str(distribution)[32:-34] + ": " + str(sse)[0:6], color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
    
    plt.legend(title='DISTRIBUTIONS', bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.show()
    
def goodness_of_fit(data, results, n, mean, plot = True):
    ## n first distribution of the ranking
    N_DISTRIBUTIONS = {k: results[k] for k in list(results)[:n]}
    #print(N_DISTRIBUTIONS)
    
    ## Histogram of data
#     plt.hist(data, density=True, ec='white', color=(63/235, 149/235, 170/235))
#     plt.title('HISTOGRAM')
#     plt.xlabel('Values')
#     plt.ylabel('Frequencies')

    if plot == True:
        plt.figure(figsize=(10, 5))
        ## Plot n distributions
        for distribution, result in N_DISTRIBUTIONS.items():
            # print(i, distribution)
            sse = result[0]
            arg = result[1]
            loc = result[2]
            scale = result[3]
            rvs = distribution.rvs(loc = loc, scale = scale, *arg, size = (50,2))
            #print(rvs)
            #print('Distribution: ', distribution, 'SSE: ', sse, 'c: ', arg, 'loc: ', loc, 'scale: ', scale)
            x_plot = np.linspace(min(data), max(data), 1000)
            y_plot = distribution.cdf(x_plot, loc=loc, scale=scale, *arg)
            plt.plot(x_plot, y_plot, label=str(distribution)[32:-34] + ": " + str(sse)[0:6], color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
            #result_ks = st.kstest(rvs, distribution.cdf(rvs, loc=loc, scale = scale, *arg))
            #result_ks = st.anderson(data, 'norm')
            #result_ks = st.ttest_1samp(rvs, mean)
            #result_ks = st.cramervonmises(data, 'norm')
            #print(result_ks) 
            
        plt.legend(title='DISTRIBUTIONS', bbox_to_anchor=(1.05, 1), loc='upper left')
        #plt.show()    

def fit_data(data, MY_DISTRIBUTIONS):
    ## st.frechet_r,st.frechet_l: are disbled in current SciPy version
    ## st.levy_stable: a lot of time of estimation parameters
    ALL_DISTRIBUTIONS = [        
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm, st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]
    

    ## Calculate Histogram
    num_bins = danoes_formula(data)
    frequencies, bin_edges = np.histogram(data, num_bins, density=True)
    central_values = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]

    results = {}
    for distribution in MY_DISTRIBUTIONS:
        ## Get parameters of distribution
        params = distribution.fit(data)
        
        ## Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
    
        ## Calculate fitted PDF and error with fit in distribution
        pdf_values = [distribution.pdf(c, loc=loc, scale=scale, *arg) for c in central_values]
        
        ## Calculate SSE (sum of squared estimate of errors)
        sse = np.sum(np.power(frequencies - pdf_values, 2.0))
        
        ## Build results and sort by sse
        results[distribution] = [sse, arg, loc, scale]
        
    results = {k: results[k] for k in sorted(results, key=results.get)}
    return results

def get_parameters(data, results, n):
    ## n first distribution of the ranking
    N_DISTRIBUTIONS = {k: results[k] for k in list(results)[:n]}
    #print(N_DISTRIBUTIONS)
    
    Distributions_list = []
    SSE_list = []
    c_list = []
    loc_list = []
    scale_list = []

    for distribution, result in N_DISTRIBUTIONS.items():
        # print(i, distribution)
        if distribution == st.gumbel_r:
            dist = 'Gumbell'
        elif distribution == st.lognorm:
            dist = 'Lognormal'
        elif distribution == st.genextreme:
            dist = 'GEV'
        elif distribution == st.norm:
            dist = 'Normal'
        elif distribution == st.genlogistic:
            dist = 'Generalized Logistic'
        else:
            dist = distribution
            
        sse = result[0]
        arg = result[1]
        if len(arg) == 0:
            c = nan
        else:
            c = arg[0]
        loc = result[2]
        scale = result[3]
        #print('Distribution: ', dist)
        Distributions_list.append(dist)
        #print('SSE: ', sse)
        SSE_list.append(sse)
        #print('arg ', arg)
        #print('c: ', c)
        c_list.append(c)
        #print('loc: ', loc)
        loc_list.append(loc)
        #print('scale: ', scale)
        scale_list.append(scale)
        
    dict = {'distribution' : Distributions_list,
            'sse' : SSE_list,
            'c' : c_list,
            'loc' : loc_list,
            'scale': scale_list}
    
    df_result = pd.DataFrame(dict)
    
    return df_result
    
def main_daily():
    ## Import data
    #data = pd.Series(sm.datasets.elnino.load_pandas().data.set_index('YEAR').values.ravel())
    #data.to_csv('teste.csv')
    #data = pd.read_csv('teste.csv')
    name_file = 'INMET_conv'
    
    
    data_df_original = pd.read_csv('Results/max_daily_{n}_2.csv'.format(n = name_file))
    data_df = data_df_original[['Precipitation']]
    mean = data_df.iloc[:,0].mean()
    data = data_df.values.ravel()
    
    #print(data)
    #input()
    MY_DISTRIBUTIONS = [st.norm, st.lognorm, st.genextreme, st.gumbel_r, st.genlogistic]
    #MY_DISTRIBUTIONS = [st.genextreme, st.gumbel_r, st.genlogistic]
    #MY_DISTRIBUTIONS = [st.genextreme]
    results = fit_data(data, MY_DISTRIBUTIONS)
    plot_histogram(data, results, 5)
    goodness_of_fit(data, results, 5, mean)
    df_parameters = get_parameters(data, results, 5)
    #df_parameters.to_csv('Results/{n}_dist_params.csv'.format(n = name_file), index = False)

def main_subdaily(name_file, disag_factor, duration, directory = 'Results'):
    ## Import data
    #data = pd.Series(sm.datasets.elnino.load_pandas().data.set_index('YEAR').values.ravel())
    #data.to_csv('teste.csv')
    #data = pd.read_csv('teste.csv')
    
    data_df_original = pd.read_csv('{d}/max_subdaily_{n}{disag}.csv'.format(n = name_file, disag = disag_factor, d = directory))
    data_df = data_df_original[['Max_{dur}'.format(dur = duration)]]
    mean = data_df.iloc[:,0].mean()
    data = data_df.values.ravel()
    
    #print(data)
    #input()
    MY_DISTRIBUTIONS = [st.norm, st.lognorm, st.genextreme, st.gumbel_r, st.genlogistic]
    #MY_DISTRIBUTIONS = [st.norm, st.genextreme, st.gumbel_r, st.genlogistic]
    #MY_DISTRIBUTIONS = [st.genextreme]
    results = fit_data(data, MY_DISTRIBUTIONS)
    plot_histogram(data, results, 5)
    goodness_of_fit(data, results, 5, mean)
    df_parameters = get_parameters(data, results, 5)
    print(df_parameters)
    #df_parameters.to_csv('{d}/{n}_dist_params.csv'.format(n = name_file, d = directory), index = False)

if __name__ == "__main__":
#     name_file = 'complete_INMET_conv_bl'
#     directory = 'bartlet_lewis'
#     disag_factor = ''
#     duration = '20min'

    # EQM
    name_file = 'complete_HADGEM_baseline_MD'
    directory = 'GCM_data//bias_correction//gcm'
    disag_factor = ''
    duration = '24'  
        
    main_subdaily(name_file, disag_factor, duration, directory)
    
    
