import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm



def corr_plot(X_, Y_, xtitle, ytitle, yerr=None, auto_limits=True, logscale=False):

    pcxv_AME = "../Data//AME.txt"

    ### Potential fractioanl contribution from an UCHII region...
    f_uch     = np.genfromtxt(pcxv_AME,usecols = (19), dtype = 'float', delimiter=',',skip_header=1)
    ### AME Significance
    sig_ame = np.genfromtxt(pcxv_AME,usecols = (16), dtype = 'float', delimiter=',',skip_header=1)

    ### Subsets separating "Highly Significant" AME from "Low/Non-significant" AME
    subset_hs = (

    (sig_ame     > 5.000)   &
    (f_uch       <= 0.25)    #&

    )

    subset_ls = (subset_hs == False)


    # Log scaling of the input data:

    if logscale == True:

        if yerr != None:

            yerr = np.abs(Y_-yerr)/Y_

            X_ = np.log10(X_)
            Y_ = np.log10(Y_)

            yerr *= Y_
            yerr = np.abs(yerr)

        else:

            X_ = np.log10(X_)
            Y_ = np.log10(Y_)



    #Color sorter:

    colors = np.empty(np.shape(sig_ame), dtype=str)
    colors[subset_hs] = 'r'
    colors[subset_ls] = 'b'

    # Some basic settings
    def_fontsize = 16


    ## Put the data into a Pandas dataframe:
    ws = pd.DataFrame({
        'x': X_ ,
        'y': Y_ ,
        })

    ws_hs = pd.DataFrame({
        'x': X_[subset_hs],
        'y': Y_[subset_hs],
        })

    ws_ls = pd.DataFrame({
        'x': X_[subset_ls] ,
        'y': Y_[subset_ls] ,
        })

    ## Get the linear fitting results:
    ols_fit = sm.ols('x ~ y', data=ws).fit()
    ols_fit_hs = sm.ols('x ~ y', data=ws_hs).fit()
    ols_fit_ls = sm.ols('x ~ y', data=ws_ls).fit()
    #linfit = np.arange(0,200)

    ## Do the plotting
    #plt.scatter(X_,Y_, alpha =0.5, s=(30*(30/sig_ame))) # Scales the point size to AME sigma
    plt.scatter(X_, Y_, c= colors, alpha =0.5, s=200)

    if yerr != None:


        plt.errorbar(X_,Y_,yerr=np.abs(yerr), linestyle="None")

    #dc = plt.plot(linfit,linfit)
    #plt.title(title, fontsize= def_fontsize)
    plt.xlabel(xtitle, fontsize= def_fontsize)
    plt.ylabel(ytitle, fontsize= def_fontsize)

    if auto_limits==True:
        plt.ylim([min(Y_),max(Y_)])
        plt.xlim([min(X_),max(X_)])
    else:
        plt.xlim([0,100])
        plt.ylim([0,100])

    print round(ols_fit.rsquared,3), round(ols_fit_hs.rsquared,3), round(ols_fit_ls.rsquared,3)

print "Hi"
