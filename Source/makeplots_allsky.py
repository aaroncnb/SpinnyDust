
# coding: utf-8

# ### All-sky AME vs. IR Scatter Plots

# In[1]:

#from IPython.external import mathjax; mathjax.install_mathjax()
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import healpy.projector as pro
import astropy.io.fits as fits
from scipy.stats import gaussian_kde
import scipy
import pandas as pd
import pickle
matplotlib.style.use('seaborn-bright')
get_ipython().magic(u'matplotlib inline')


# # 0.1) Load data and masks:

# In[2]:

with open('../Data/maps_nest.pickle') as f:  # Python 3: open(..., 'rb')
    coords, planck_bb, planck_mw, phot, phot_modesub = pickle.load(f)
    



# In[191]:

print phot.head()


print planck_mw.head()

print planck_bb.head()


# In[5]:

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler


### Setup the standard pipeline to apply to all the data:
allsky_pipeline = Pipeline([
    ('imputer', Imputer(strategy="median")),
    ('std_scaler', StandardScaler(with_mean=False)),
])
# allsky_pipeline = Pipeline([
#     ('imputer', Imputer(strategy="median"))
# ])

phot_tr      = pd.DataFrame(allsky_pipeline.fit_transform(phot),columns=phot.columns)
planck_bb_tr = pd.DataFrame(allsky_pipeline.fit_transform(planck_bb),columns=planck_bb.columns)
planck_mw_tr = pd.DataFrame(allsky_pipeline.fit_transform(planck_mw),columns=planck_mw.columns)




# In[6]:

planck_bb.head()


# In[7]:

phot_corr = phot_tr.corr(method='spearman')
planck_bb_corr = planck_bb_tr.corr(method='spearman')
planck_mw_corr = planck_mw_tr.corr(method='spearman')


# ### 1.1) Cross-correlation among all IR photometric bands and AME map
# ##### Split by Galactic Latitude

# In[8]:

glatrange     = 10.0
glatrange_mid = 2.5
elatrange     = 10


gcut_l = np.where((abs(coords['glat']) < glatrange) & (abs(coords['elat']) > elatrange))
gcut_h = np.where((abs(coords['glat']) > glatrange) & (abs(coords['elat']) > elatrange))



# In[9]:

import seaborn as sb


# In[66]:

def plotCorrMatrix():
    
    phot_corr     = phot_tr.join(planck_mw_tr['AME']).join(planck_bb_tr['$R_{PR1}$']).corr(method='spearman')
    phot_corr_lgl = phot_tr.join(planck_mw_tr['AME']).join(planck_bb_tr['$R_{PR1}$']).iloc[gcut_l].corr(method='spearman')
    phot_corr_hgl = phot_tr.join(planck_mw_tr['AME']).join(planck_bb_tr['$R_{PR1}$']).iloc[gcut_h].corr(method='spearman')
    
    mask = np.zeros_like(phot_corr.values)
    mask[np.triu_indices_from(mask,k=1)] = True

    with sb.axes_style("white"):


        fig, ax = plt.subplots(1,3,figsize=(21,7))
        cbar_ax = fig.add_axes([.91, .2, .03, .7])

        sb.heatmap(
            phot_corr,
            #linewidths=.5,
            annot=True,
            mask=mask,
            cbar=False,
            yticklabels=True,
            xticklabels=True,
            ax = ax[0],
            vmin=0,
            vmax=1)

        ax[0].set_title("All-sky", fontsize=20)


        sb.heatmap(
            phot_corr_hgl,
            #linewidths=.5,
            annot=True,
            mask=mask,
            cbar=False,
            yticklabels=True,
            xticklabels=True,
            ax=ax[1],
            vmin=0,
            vmax=1)

        ax[1].set_title("$|b| > 10^{\circ}$", fontsize=20)



        sb.heatmap(
            phot_corr_lgl,
            #linewidths=.5,
            annot=True,
            mask=mask,
            cbar=True,
            cbar_ax=cbar_ax,
            yticklabels=True,
            xticklabels=True,
            ax=ax[2],
            vmin=0,
            vmax=1)

        ax[2].set_title("$|b| < 10^{\circ}$", fontsize=20)


        fig.tight_layout(rect=[0, 0, .9, 1])

        plt.show()

        fig.savefig("../Plots/all_bands_corr_matrix_wAME_spearman.pdf", bbox_inches='tight')
        


# In[113]:

plotCorrMatrix()


# ### 1.2) Cross-correlation among all IR photometric bands and AME map
# ##### Split by AKARI 9 micron detection limit (2 MJy/sr)

# In[283]:

def plotCorrMatrixwCutoff(mapframe, cutoff_map, lower_cutoff=5.0, upper_cutoff = 5e2):
    
    cutoff_map_cp = cutoff_map.copy()

    lim = np.where(
        (cutoff_map > lower_cutoff ) & 
        (cutoff_map < upper_cutoff)
        )
    
    phot_corr     = phot_tr.join(planck_mw_tr['AME']).join(planck_bb_tr['$R_{PR1}$']).corr(method='spearman')
    phot_corr_irc9_lim = mapframe.iloc[lim].corr(method='spearman')
    
    #bb_corr_drop = bb_corr.drop('AME',axis=0).drop('A9',axis=1)
    mask = np.zeros_like(phot_corr.values)
    mask[np.triu_indices_from(mask,k=1)] = True

    with sb.axes_style("white"):


        fig, ax = plt.subplots(1,2,figsize=(15,7.5))
        cbar_ax = fig.add_axes([.91, .2, .03, .7])

        sb.heatmap(
            phot_corr,
            #linewidths=.5,
            annot=True,
            mask=mask,
            cbar=False,
            yticklabels=True,
            xticklabels=True,
            ax = ax[0],
            vmin=0,
            vmax=1)

        ax[0].set_title("All-sky", fontsize=20)


        sb.heatmap(
            phot_corr_irc9_lim,
            #linewidths=.5,
            annot=True,
            mask=mask,
            cbar=True,
            cbar_ax = cbar_ax,
            yticklabels=True,
            xticklabels=True,
            ax=ax[1],
            vmin=0,
            vmax=1)

        ax[1].set_title("{} > $I_um$ > {} MJy/sr".format(upper_cutoff,lower_cutoff), fontsize=20)


        fig.tight_layout(rect=[0, 0, .9, 1])

        plt.show()

        fig.savefig("../Plots/all_bands_corr_matrix_wAME__IRC9lim{}_{}_MJysr_spearman.pdf".format(upper_cutoff,lower_cutoff), bbox_inches='tight')
        print lim
        cutoff_map_cp.iloc[:] = hp.UNSEEN
        cutoff_map_cp.iloc[lim] = cutoff_map.iloc[lim].copy()
        
        hp.mollview(cutoff_map_cp, nest=True, norm='hist', cmap="rainbow")


# In[291]:

plotCorrMatrixwCutoff(
                        phot_tr.join(planck_mw_tr['AME']).join(planck_bb_tr['$R_{PR1}$']), 
                        phot_modesub['A9'], 
                        lower_cutoff=0.5, 
                        upper_cutoff = 25
    
                        )


# Why is IRAS 25 $\mu$m the most unique map? Somewhat odd that it correlates miraculously well with IRAS 12 $\mu$m - despite its weak correlation with  the IRC $\mu$m map. Perhaps this is due to correlated noise, systematic effects in the IRAS MIR bands. Correlated Zodiacal light subtraction residuals, perhaps

# # All-sky AME vs. IR plots:

# In[71]:



def plotBandsCloud(nside=256):
    
    sb.set_style("whitegrid")

    ncols=4
    nrows=3
    aspect=1.0

    fig, axs = plt.subplots(ncols=ncols, 
                            nrows=nrows, 
                            sharey=True, 
                            sharex=True)
    #fig.subplots_adjust(hspace=0.1, left=0.1, right=0.7)
    plt.setp(axs.flat, aspect=1.0, adjustable='box-forced')

    k=0

    hsize = hp.nside2npix(nside)
    
    randsub = np.random.randint(low=0, high=hsize, size=hsize//10)


    for i in range(0,nrows):
        for j in range(0,ncols):

                if k > 11:

                    pass

                else:

                    x = phot_modesub.values[randsub,k].copy()


                    y = planck_mw['AME'].values[randsub].copy()

                    x_ = x[(x>0) & (y>0)].copy()
                    y_ = y[(x>0) & (y>0)].copy()

                    x_ = np.log10(x_)
                    y_ = np.log10(y_)


                    xmin = x_.min()
                    xmax = x_.max()
                    ymin = y_.min()
                    ymax = y_.max()

                    #print xmin
                    #print xmax
                    #print ymin
                    #print ymax

                    ax = axs[i,j]
                    #ax.set_aspect(aspect, adjustable='box')

                    #ax.set_xscale('log')
                    #ax.set_yscale('log')


                    sb.kdeplot(
                           x_,
                           y_,
                           shade=True,
                           shade_lowest=False,
                           gridsize=50,
                            ax = ax)



                    #ax.axis([xmin, xmax, ymin, ymax])
                    ax.axis([-1.5,2,1,3.5])

                    ax.text(0.2, 0.9,phot_modesub.columns[k], horizontalalignment='center',
                      verticalalignment='center',
                      transform=ax.transAxes, 
                      fontsize=15)

                    ax.grid(True)

                    ax.set_frame_on(True)

                    k += 1
                    
        ax = axs[1,0]
        ax.set_ylabel('AME [$\mu{}K_{CMB}$]', fontsize=15)
        ax = axs[-1,2]
        ax.set_xlabel('log $I_{\lambda}$ [MJy/sr]', fontsize=15)

        plt.show()

        fig.savefig("../Plots/AMEvsDust_allsky_allbands_kde.pdf", bbox_inches='tight')

                    
    return axs
    


# In[114]:

allbands_kde = plotBandsCloud()


# In[ ]:

## Check correlations, adding 10% noise after removal of first principal component


# ### Cross correlation between AME and Planck Mod-BB fits 
# (+the smoothed PR1 Radiance map, as used by Hensley+ 2016)

# ## Confirmation of all-sky correlation results:
#  Instead of downgrading the pixel sizes, just copy the same correaltion value of the 64 NSIDE 256 pixels
#  in a batch to all of those pixel positions in an output map (also size NSIDE 256).

# In[20]:

def testSpatialCorr(df, 
                    nside_in, 
                    nside_out,
                    method='spearman'):
    
    npix_in    = 12*nside_in**2
    npix_out   = 12*nside_out**2
    pix_interv = (nside_in/nside_out)**2
    
    ## First, do it the "normal way"-
    patches_corr = [df.iloc[i*pix_interv:(i+1)*pix_interv].corr(method=method) for i in range(0,npix_out)]
    corr_patches_pn = pd.Panel({i: patches_corr[i] for i in range(0,npix_out)})

    
    return corr_patches_pn

def displaySpatialCorr(corr_patches_pn,labels, ref_col=0):

    nside = len(corr_patches_pn.values[:,0,0])
    #fig = plt.figure(figsize=(8,4))

    for j in range(0,len(labels)):
        #plt.subplot(2,5,(j*2)+1)
        hp.cartview(corr_patches_pn.values[:,j,ref_col],
                         #sub=(1,4,j+1), 
                         #fig=fig,
                         cmap = "rainbow", 
                         cbar = False, 
                         min  = -1, 
                         max  = 1, 
                         nest = True, 
                         title="$S$({}:{}) NSIDE".format(labels[ref_col],labels[j],nside_out))
        
        plt.savefig("../Plots/Spearman_Map_nside"+str(nside_out)+"_{}to{}.pdf".format(labels[ref_col],labels[j]))


# In[24]:

nside_in = 256
nside_out = 16
test_frame = phot_modesub.join(planck_mw)
corr_patches_pn = testSpatialCorr(test_frame,
                                  nside_in, 
                                  nside_out)


# In[26]:

displaySpatialCorr(corr_patches_pn, test_frame.columns, ref_col = -1)


# In[27]:

nside_in = 256
nside_out = 8
test_frame = phot_modesub.join(planck_mw)
corr_patches_pn = testSpatialCorr(test_frame,
                                  nside_in, 
                                  nside_out)


# In[30]:

displaySpatialCorr(corr_patches_pn, test_frame.columns, ref_col = -4)


# In[ ]:

# The check appears successful, so make a plot grid of all the nsides:


# In[21]:

import matplotlib as mpl

def SpatialCorrAll(df, nside_in):
    
    nside_list = [32,16,8,4,2]
    
    fig = plt.figure()
    
    #fig.subplots_adjust(hspace=0, left=0.1, right=0.7)
    
    for k in range(0,5):
    
        npix_in    = 12*nside_in**2
        npix_out   = 12*nside_list[k]**2
        pix_interv = (nside_in/nside_list[k])**2

        ## First, do it the "normal way"-
        patches_corr = [df.iloc[i*pix_interv:(i+1)*pix_interv].corr(method='spearman') for i in range(0,npix_out)]
        
        corr_patches_pn = pd.Panel({i: patches_corr[i] for i in range(0,npix_out)})


        for j in range(0,4):
            
            sub_index = (j*5)+1+k
            
            ax = fig.add_subplot(4,5,sub_index)

    #             if j == 0:
    #                 ax_again = fig.add_subplot(4,5,sub_index)
    #                 ax_again.set_ylabel("$S$(AME:"+bba.columns[j])
    #             if k == 0:
    #                 ax_again = fig.add_subplot(4,5,sub_index)
    #                 ax_again.set_xlabel("NSIDE"+str(nside_list[k]))

            hp.cartview(
                corr_patches_pn.values[:,j,4],
                cmap="rainbow", 
                cbar=False,
                fig = ax,
                min=-1, 
                max=1, 
                nest=True, 
                title="",
                hold=False)
        
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    
    
    #cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = mpl.colorbar.ColorbarBase(cbar_ax, 
                                    cmap="rainbow",
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('$S$')

    plt.show()
    

    plt.savefig("../Plots/Spearman_Map_nsideALL_AMEtoMBBandA9.pdf")



# In[60]:

def dispAllbandsKDE(allbands_kde):
    
    ax = axs[1,0]
    ax.set_ylabel('AME [$\mu{}K_{CMB}$]', fontsize=15)
    ax = axs[-1,2]
    ax.set_xlabel('log $\nuI_nu$ [MJy/sr]', fontsize=15)

    plt.show()

    fig.savefig("../Plots/AMEvsDust_allsky_allbands_kde.pdf", bbox_inches='tight')
    
dispAllbandsKDE(allbands_kde)


# ## Compare with Planck PR2 Dust Intensity Map
# (With ref. frequency at 545 GHz)

# In[ ]:

ncols=6
nrows=3
aspect=1.0

fig, axs = plt.subplots(ncols=ncols, 
                        nrows=nrows, 
                        sharey=True, 
                        sharex=True)
#fig.subplots_adjust(hspace=0.1, left=0.1, right=0.7)
plt.setp(axs.flat, aspect=1.0, adjustable='box-forced')

k=0


for i in range(0,nrows):
    for j in range(0,ncols):
            
            if k > 17:
                
                pass
            
            else:
           
                x = phot_modesub.values[:,k]


                y = planck_mw['AME'].values[:]
                
                R = planck_bb['$R$'].values[:]
                
                y = y/R
                x = x/R

                x_ = x[(x>0) & (y>0)]
                y_ = y[(x>0) & (y>0)]


                xmin = 5e-5#x_.min()
                xmax = 0.1 #x_.max()
                ymin = 0.01#y_.min()
                ymax = y_.max()

                ax = axs[i,j]
                #ax.set_aspect(aspect, adjustable='box')

                hb = ax.hexbin(
                       x_,
                       y_, 
                       mincnt=1,
                       gridsize=300,
                       bins='log', 
                       cmap='inferno_r',
                       xscale='log',
                       yscale='log')


                ax.axis([xmin, xmax, ymin, ymax])

                ax.text(0.2, 0.9,phot_modesub.columns[k], horizontalalignment='center',
                  verticalalignment='center',
                  transform=ax.transAxes, 
                  fontsize=15)
                
                ax.grid(True)
                
                ax.set_frame_on(True)

                k += 1
            mAdd


# ## All-sky Noise Estimation:

# In[ ]:

hmap_hists =  pd.DataFrame.hist(phot_modesub, 
                                range=(-10, 25), 
                                bins=100, 
                                alpha=0.4, 
                                grid=True,
                                sharex=True, 
                                xlabelsize=15,
                                sharey=False,
                                ylabelsize=12,
                                figsize=(11,8.5),
                                **{'normed':True})
hmap_hists


# In[22]:

def plot_hdists(df):
    
    import seaborn as sns
    import scipy.stats as stats
    sns.distplot(df, bins=1000, kde=False, fit=stats.gamma )
    print 
    
plot_hdists(phot_modesub[(phot_modesub>-5) & (phot_modesub<25)].A9.dropna())


# In[70]:




# In[189]:

data = phot_modesub.dropna().values[:,0]
# phot_modesub[(phot_modesub>-5) & (phot_modesub<25)].A9.dropna()
# data.std()


# In[399]:

from astropy.modeling import models, fitting

def fitAndPlot(data, ymax=2.0, nbins=1000, amplitude =1, stddev = 1, mean=0, zero_mean=False, left_wing=False, left_mean=False, xrange=(-10,10)):
    

    # Get distribution
    y,x, patches = plt.hist(data, range=xrange, bins=nbins, normed=True,alpha=0.8)
    #print y
    # Fit the data using a Gaussian
    g_init = models.Gaussian1D(amplitude=amplitude, mean=mean, stddev=stddev)
    g_init.mean.fixed = zero_mean
    g_init.stddev.bounds = (0,None)
    fit_g = fitting.LevMarLSQFitter()
    
    if left_wing == True:
        g = fit_g(g_init, x[:-1][x[:-1]<0], y[x[:-1]<0])
    elif left_mean == True:
        g = fit_g(g_init, x[:-1][x[:-1]<np.mean(data)], y[x[:-1]<np.mean(data)])
    else:
        g = fit_g(g_init, x[:-1], y)



    plt.plot(x,g(x),label='Gaussian', color='black',alpha=0.8)



    # Plot the data with the best-fit model
    # plt.figure(figsize=(8,5))
    # plt.plot(x, y[:-1], 'ko')
    # plt.plot(x, g(x), label='Gaussian')
    plt.ylim(0,ymax)
    plt.ylabel('Norm. Pixel Count', fontsize=22)
    plt.xlabel('Intensity [MJy/sr]',fontsize=22)
    plt.legend(loc=2)
    plt.text((xrange[1]-abs(xrange[0])*3)/8,ymax/4,"Stddev: "+str(round(g.stddev.value,3)),fontsize=22)
    plt.text((xrange[1]-abs(xrange[0])*3)/8,(ymax*5)/8,"Data mean: "+str(round(np.mean(data),3)),fontsize=22)
    
    return g.stddev.value

# fitAndPlot(data)
# plt.show()
# plt.close()
# fitAndPlot(data, zero_mean=True)
# plt.show()
# plt.close()
# fitAndPlot(data, zero_mean=False, left_wing=True)
# plt.show()
# plt.close()
fitAndPlot(data, left_mean=True)
plt.show()
plt.close()
    
# Select data


# In[230]:

# Get distribution
y,x, patches = plt.hist(planck_mw.AME.dropna().values-0, range =(-30,150),bins=1000, normed=True,alpha=0.8)
#print y
# Fit the data using a Gaussian
g_init = models.Gaussian1D(amplitude=0.1, mean=0, stddev=1.)
g_init.mean.fixed = False
fit_g = fitting.LevMarLSQFitter()
g = fit_g(g_init, x[:-1][x[:-1]<0], y[x[:-1]<0])

plt.plot(x,g(x),label='Gaussian', color='black')

plt.ylabel('Norm. Pixel Count', fontsize=22)
plt.xlabel('Intensity [uKCMB]',fontsize=22)
plt.legend(loc=2)


print np.size(planck_mw.AME.dropna()==0)


# In[252]:

# Get distribution
y,x, patches = plt.hist(planck_bb['$T$'].dropna().values, range =(0,50),bins=100, normed=True,alpha=0.8)
#print y
# Fit the data using a Gaussian
g_init = models.Gaussian1D(amplitude=0.1, mean=20, stddev=1.)
g_init.mean.fixed = False
fit_g = fitting.LevMarLSQFitter()
g = fit_g(g_init, x[:-1], y)

plt.plot(x,g(x),label='Gaussian', color='black')



# Plot the data with the best-fit model
# plt.figure(figsize=(8,5))
# plt.plot(x, y[:-1], 'ko')
# plt.plot(x, g(x), label='Gaussian')
#plt.ylim(0,2.0)
plt.ylabel('Norm. Pixel Count', fontsize=22)
plt.xlabel('Intensity [uKCMB]',fontsize=22)
plt.legend(loc=2)
#plt.text(60,0.008,"Stddev: "+str(round(g.stddev.value,3)),fontsize=22)
#plt.text(60,0.006,"Data mean: "+str(round(np.mean(data),3)),fontsize=22)

#print np.size(planck_mw.T.dropna()==0)


# In[ ]:

#### Read Planck low-res Modified blackbody fitting results:
planck_bb_path    = filepath+"/COM_CompMap_dust-commander_0256_R2.00.fits.gz" #HEALPix FITS table containing Planck low-res modBB results
fields            = [4,7,1] #The field number in the HEALPix file
labels            = ["$T$","$B$","$R$"]

planck_bb = pd.DataFrame()
for i in range(0,3):
    planck_bb[labels[i]] = hp.read_map(planck_bb_path,field = fields[i], nest=nest)

planck_bb.replace(
    to_replace =hp.UNSEEN,
    value=np.nan,
    inplace=True
    )


#### Read Planck low-res microwave component fitting results:
planck_mw = pd.DataFrame()
labels = ['AME','CO','ff','Sync']

paths = ['COM_CompMap_AME-commander_0256_R2.00.fits.gz',


# In[359]:

## Whole sky without mean-fixing or wing-selection


# In[356]:

stddevs = []
for i in range(0,len(phot_modesub.columns)):
    data = phot_modesub.dropna().values[:,i]
    stddev = fitAndPlot(data)
    stddevs.append(stddev)
    plt.title(phot_modesub.columns[i], fontsize=22)
    plt.savefig("../Plots/allsky_pixdist_"+phot_modesub.columns[i]+".pdf", bbox_inches='tight', dpi=100)
    plt.show()
 


# In[360]:

## Whole sky with mean fixed to zero, using only the left wing


# In[358]:

stddevs = []

for i in range(0,len(phot_modesub.columns)):
    data = phot_modesub.dropna().values[:,i]
    stddev = fitAndPlot(data, zero_mean=True, left_wing=True)
    stddevs.append(stddev)
    plt.title(phot_modesub.columns[i], fontsize=22)
    plt.savefig("../Plots/allsky_pixdist_"+phot_modesub.columns[i]+"_leftWing_zeroMean.pdf", bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()
 


# ## Noise estimated for limited patches:

# #### Patch 1: (l:130, b:60) [25x25 degree Gal] , npix = 20164

# In[400]:

stddevs = []

noise_patches = [(coords.glon > 117.5) & (coords.glon < 142.5) & (coords.glat > 47.5) & (coords.glat < 72.5),
                 (coords.glon > 217.5) & (coords.glon < 242.5) & (coords.glat < -47.5) & (coords.glat > -72.5),
                 (coords.glon > 217.5) & (coords.glon < 242.5) & (coords.glat > 47.5) & (coords.glat < 72.5)]

for i in range(0,len(phot_modesub.columns)):
    data = phot_modesub[noise_patches[0]].dropna().values[:,i]
    # Using an initial stddev of 1 seems to lead to underfitting, here- using 0,5 instead
    stddev = fitAndPlot(data,
                        amplitude=2, 
                        mean=np.mean(data), 
                        ymax=7, 
                        stddev = 0.1, 
                        zero_mean=False, 
                        left_wing=False,
                        left_mean=False,                        
                        nbins=400, 
                        xrange=(-4,4))
    
    
    stddevs.append(stddev)
    plt.title(phot_modesub.columns[i], fontsize=22)
    plt.savefig("../Plots/allsky_pixdist_"+phot_modesub.columns[i]+"_noisePatch1.pdf", bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()
 


# #### Patch 2: (l:230, b:-60) [25x25 degree Gal] , npix = 20164

# In[395]:

for i in range(0,len(phot_modesub.columns)):
    data = phot_modesub[noise_patches[1]].dropna().values[:,i]
    # Using an initial stddev of 1 seems to lead to underfitting, here- using 0,5 instead
    stddev = fitAndPlot(data,
                        amplitude=2, 
                        mean=np.mean(data), 
                        ymax=7, 
                        stddev = 0.1, 
                        zero_mean=False, 
                        left_wing=False, 
                        nbins=1000, 
                        xrange=(data.min(),data.max()))
    
    stddevs.append(stddev)
    plt.title(phot_modesub.columns[i], fontsize=22)
    plt.savefig("../Plots/allsky_pixdist_"+phot_modesub.columns[i]+"_leftWing_zeroMean_noisePatch2.pdf", bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()


# #### Patch 3: (l:230, b:60) [25x25 degree Gal] , npix = 20164

# In[396]:

for i in range(0,len(phot_modesub.columns)):
    data = phot_modesub[noise_patches[2]].dropna().values[:,i]
    # Using an initial stddev of 1 seems to lead to underfitting, here- using 0,5 instead
    stddev = fitAndPlot(data,
                        amplitude=2, 
                        mean=np.mean(data), 
                        ymax=7, 
                        stddev = 0.1, 
                        zero_mean=False, 
                        left_wing=False, 
                        nbins=1000, 
                        xrange=(data.min(),data.max()))
    
    stddevs.append(stddev)
    plt.title(phot_modesub.columns[i], fontsize=22)
    plt.savefig("../Plots/allsky_pixdist_"+phot_modesub.columns[i]+"_leftWing_zeroMean_noisePatch3.pdf", bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()


# In[397]:

#### All 3 patches merged together:

noise_patches_merged = ((coords.glon > 117.5) & (coords.glon < 142.5) & (coords.glat > 47.5) & (coords.glat < 72.5)) |                  ((coords.glon > 217.5) & (coords.glon < 242.5) & (coords.glat < -47.5) & (coords.glat > -72.5)) |                  ((coords.glon > 217.5) & (coords.glon < 242.5) & (coords.glat > 47.5) & (coords.glat < 72.5) )


for i in range(0,len(phot_modesub.columns)):
    data = phot_modesub[noise_patches_merged].dropna().values[:,i]
    # Using an initial stddev of 1 seems to lead to underfitting, here- using 0,5 instead
    stddev = fitAndPlot(np.random.choice(data, size=len(noise_patches_merged)//1),
                        amplitude=2, 
                        mean=np.mean(data), 
                        ymax=7, 
                        stddev = 0.1, 
                        zero_mean=False, 
                        left_mean=False,                        
                        nbins=1000, 
                        xrange=(data.min(),5))
    
    stddevs.append(stddev)
    plt.title(phot_modesub.columns[i], fontsize=22)
    plt.savefig("../Plots/allsky_pixdist_"+phot_modesub.columns[i]+"_leftWing_zeroMean_noisePatchMerged.pdf", bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()


# ## Estimate noise for a limited part of the sky: Planck CMB Mask

# #### Masking test

# In[273]:

data =  phot_modesub.A9.values.copy
hp.mollview(data)
print len(data[np.isnan(data)==True])
hmask = hp.read_map('/work1/users/aaronb/Codebrary/Python/Projects/LOrionis/data/raw/healpix/referenceMaps/COM_Mask_CMB-IQU-common-field-MaskInt_0256.fits')
data[hmask==hp.UNSEEN] = np.nan
print len(data[np.isnan(data)==True])
print len(hmask[hmask==hp.UNSEEN])


# In[ ]:




# In[291]:

from astropy.modeling import models, fitting

def fitAndPlotMaskedTest(data,zero_mean=False, left_wing=False, left_mean=False):
    
    hp.mollview(data.values)
    plt.show()
    plt.close()
    
    data_unmask = data.copy()
    data_mask   = data.copy()
    
    data_unmask = data_unmask.dropna().values

    # Get distribution

    y,x, patches = plt.hist(data_unmask, range=(-10, 10), bins=1000, normed=True,alpha=0.8)
    #print y
    # Fit the data using a Gaussian
    g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
    g_init.mean.fixed = zero_mean
    fit_g = fitting.LevMarLSQFitter()
    
    if left_wing == True:
        g = fit_g(g_init, x[:-1][x[:-1]<0], y[x[:-1]<0])
    elif left_mean == True:
        g = fit_g(g_init, x[:-1][x[:-1]<np.median(data_unmask)], y[x[:-1]<np.median(data_unmask)])
    else:
        g = fit_g(g_init, x[:-1], y)



    plt.plot(x,g(x),label='Gaussian', color='black')



    # Plot the data with the best-fit model
    # plt.figure(figsize=(8,5))
    # plt.plot(x, y[:-1], 'ko')
    # plt.plot(x, g(x), label='Gaussian')
    plt.ylim(0,2.0)
    plt.ylabel('Norm. Pixel Count', fontsize=22)
    plt.xlabel('Intensity [MJy/sr]',fontsize=22)
    plt.legend(loc=2)
    plt.text(0.75,0.5,"Stddev: "+str(round(g.stddev.value,5)),fontsize=22)
    plt.text(0.75,0.75,"Data mean: "+str(round(np.mean(data_unmask),5)),fontsize=22)
    plt.show()
    plt.close()
    
    # Mask pixels from the degraded Planck Foreground Mask Map:
    hmask = hp.read_map('/work1/users/aaronb/Codebrary/Python/Projects/LOrionis/data/raw/healpix/referenceMaps/COM_Mask_CMB-IQU-common-field-MaskInt_0256.fits')


    data_mask[hmask==hp.UNSEEN] = np.nan
    hp.mollview(data_mask)
    plt.show()
    plt.close()
    
    y,x, patches = plt.hist(data_mask, range=(-10, 10), bins=1000, normed=True,alpha=0.8)
    #print y
    # Fit the data using a Gaussian
    g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
    g_init.mean.fixed = zero_mean
    fit_g = fitting.LevMarLSQFitter()
    
    if left_wing == True:
        g = fit_g(g_init, x[:-1][x[:-1]<0], y[x[:-1]<0])
    elif left_mean == True:
        g = fit_g(g_init, x[:-1][x[:-1]<np.median(data_mask)], y[x[:-1]<np.median(data_mask)])
    else:
        g = fit_g(g_init, x[:-1], y)



    plt.plot(x,g(x),label='Gaussian', color='black')



    # Plot the data with the best-fit model
    # plt.figure(figsize=(8,5))
    # plt.plot(x, y[:-1], 'ko')
    # plt.plot(x, g(x), label='Gaussian')
    plt.ylim(0,2.0)
    plt.ylabel('Norm. Pixel Count', fontsize=22)
    plt.xlabel('Intensity [MJy/sr]',fontsize=22)
    plt.legend(loc=2)
    plt.text(0.75,0.5,"Stddev: "+str(round(g.stddev.value,5)),fontsize=22)
    plt.text(0.75,0.75,"Data mean: "+str(round(np.mean(data_mask),5)),fontsize=22)
        
    
    
    
    
    return g.stddev.value

  
fitAndPlotMaskedTest(phot_modesub.A9.copy(), zero_mean=True, left_wing=True)
plt.show()
plt.close()

    
# Select data


# In[ ]:

from astropy.modeling import models, fitting

def fitAndPlotMasked(data,zero_mean=False, left_wing=False, left_mean=False):
    

    data_mask   = data.copy()
    
    # Mask pixels from the degraded Planck Foreground Mask Map:
    hmask = hp.read_map('/work1/users/aaronb/Codebrary/Python/Projects/LOrionis/data/raw/healpix/referenceMaps/COM_Mask_CMB-IQU-common-field-MaskInt_0256.fits')


    data_mask[hmask==hp.UNSEEN] = np.nan

    y,x, patches = plt.hist(data_mask, range=(-10, 10), bins=1000, normed=True,alpha=0.8)
    #print y
    # Fit the data using a Gaussian
    g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
    g_init.mean.fixed = zero_mean
    fit_g = fitting.LevMarLSQFitter()
    
    if left_wing == True:
        g = fit_g(g_init, x[:-1][x[:-1]<0], y[x[:-1]<0])
    elif left_mean == True:
        g = fit_g(g_init, x[:-1][x[:-1]<np.median(data_mask)], y[x[:-1]<np.median(data_mask)])
    else:
        g = fit_g(g_init, x[:-1], y)



    plt.plot(x,g(x),label='Gaussian', color='black')



    # Plot the data with the best-fit model
    # plt.figure(figsize=(8,5))
    # plt.plot(x, y[:-1], 'ko')
    # plt.plot(x, g(x), label='Gaussian')
    plt.ylim(0,2.0)
    plt.ylabel('Norm. Pixel Count', fontsize=22)
    plt.xlabel('Intensity [MJy/sr]',fontsize=22)
    plt.legend(loc=2)
    plt.text(0.75,0.5,"Stddev: "+str(round(g.stddev.value,5)),fontsize=22)
    plt.text(0.75,0.75,"Data mean: "+str(round(np.mean(data_mask),5)),fontsize=22)
        
    
    
    
    
    return g.stddev.value


stddevs = []

for i in range(0,len(phot_modesub.columns)):
    data = phot_modesub[phot_modesub.columns[i]].copy()
    stddev = fitAndPlotMasked(data, zero_mean=True, left_wing=True)
    plt.title(phot_modesub.columns[i], fontsize=22)
    plt.savefig("../Plots/allsky_pixdist_"+phot_modesub.columns[i]+"_leftWing_zeroMean_masked.pdf", bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()
    
# Select data


# ## Offset Uncorrected Maps:

# In[241]:

stddevs = []

for i in range(0,len(phot.columns)):
    data = phot.dropna().values[:,i]
    stddev = fitAndPlot(data, zero_mean=False, left_wing=False, left_mean=True)
    stddevs.append(stddev)
    plt.title(phot.columns[i], fontsize=22)
    plt.savefig("../Plots/allsky_pixdist_"+phot.columns[i]+"_nonOffsetCorr_leftWing_zeroMean.pdf", bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()
 


# # Angular Power Spectra:

# In[ ]:

plt.loglog(hp.anafast(planck_mw['AME'].values))


# In[ ]:

phot_unseens = phot.replace(
    to_replace =np.nan,
    value=hp.UNSEEN
    )

fig = plt.figure(figsize=(8,8))

plt.loglog(hp.anafast(phot_unseens['A9'].values), label="A9")
plt.loglog(hp.anafast(phot_unseens['I12'].values), label="I12")
plt.loglog(hp.anafast(phot_unseens['I60'].values), label="I100")
plt.loglog(hp.anafast(phot_unseens['A140'].values), label="A140")
plt.loglog(hp.anafast(phot_unseens['P545'].values), label="P857")

plt.loglog(hp.anafast(planck_mw['AME'].values), label="AME")

plt.title("Angular Power Spectra of AME and various IR bands", fontsize=20)
plt.xlabel("$l$", fontsize=20)
plt.ylabel("$Cl$",fontsize=20)




plt.legend()

fig.savefig("../Plots/AngPowerSpec_AMEandIR.pdf", bbox_inches='tight')


# In[ ]:

a140 = phot['A140'].replace(
    to_replace =np.nan,
    value=hp.UNSEEN
    ).values

hp.anafast(a140)


# In[ ]:

# For each pixel in map, query_disc a 5 degree ring of pixels
    # Take the spearman correlation coefficient of that ring of pixels
    # Set that pixel value to the spearman correlation coefficient


# In[19]:

import time

from sklearn.manifold import TSNE
np.random.seed(seed=42)

rndperm = np.random.permutation(phot.shape[0])



n_sne = 7000

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(phot_tr.loc[rndperm[:n_sne],:].values)

print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)


# In[75]:

phot_tr_tsne = phot_tr.loc[rndperm[:n_sne],:].copy()
phot_tr_tsne['x-tsne'] = tsne_results[:,0]
phot_tr_tsne['y-tsne'] = tsne_results[:,1]


# In[78]:

from ggplot import *


# In[93]:

chart = ggplot( phot_tr_tsne, aes(x='x-tsne', y='y-tsne'))         + geom_point(size=70, alpha = 0.1)         + ggtitle("tSNE dimensions colored by digit")


# In[94]:

chart


# In[ ]:

do_tsne(perp):
    
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=perp, n_iter=100)
    tsne_results = tsne.fit_transform(phot_tr.loc[rndperm[:n_sne],:].values)

    print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

    phot_tr_tsne = phot_tr.loc[rndperm[:n_sne],:].copy()
    phot_tr_tsne['x-tsne'] = tsne_results[:,0]
    phot_tr_tsne['y-tsne'] = tsne_results[:,1]

    chart = ggplot( phot_tr_tsne, aes(x='x-tsne', y='y-tsne'))             + geom_point(size=70, alpha = 0.1)             + ggtitle("tSNE dimensions colored by digit")

    return chart


# In[ ]:

do_tsne(10)


# In[105]:

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca_result = pca.fit_transform(phot_tr.values)
phot_tr_pca = phot_tr.copy()

phot_tr_pca['pca-one'] = pca_result[:,0]
phot_tr_pca['pca-two'] = pca_result[:,1]
#phot_tr_pca['pca-three'] = pca_result[:,2]


print "Explained variation per principal component:{}".format(pca.explained_variance_ratio_)


# In[106]:

chart_pca = ggplot(phot_tr_pca.loc[:,:], aes(x='pca-one',y='pca-two') )         + geom_point(size=75, alpha=0.2)         + ggtitle("First and Second Principal Components colored by digit")


# In[107]:

chart_pca


# In[113]:

from sklearn.decomposition import NMF

nmf = NMF(n_components=2)

nmf_result = nmf.fit_transform(phot_tr.values)
phot_tr_nmf = phot_tr.copy()

phot_tr_nmf['nmf-one'] = nmf_result[:,0]
phot_tr_nmf['nmf-two'] = nmf_result[:,1]
#phot_tr_pca['pca-three'] = pca_result[:,2]


print "Explained variation per principal component:{}".format(nmf.explained_variance_ratio_)


# In[55]:

planck_mw.head()


# In[56]:

planck_bb.head()


# In[69]:

hp.mollview(planck_mw['AME']/planck_bb['$R$'], norm='hist',min=1, max=10, nest=True, cmap='rainbow')


# In[70]:

hp.mollview(planck_mw_tr['AME']/planck_bb_tr['$R$'], norm='hist', nest=True, cmap='rainbow')


# In[72]:

hp.mollview(planck_mw_tr['Sync']/planck_bb_tr['$R$'], norm='hist', nest=True, cmap='rainbow')


# In[73]:

hp.mollview(planck_mw_tr['ff']/planck_bb_tr['$R$'], norm='hist', nest=True, cmap='rainbow')


# In[10]:

planck_mw.columns


# In[12]:

from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.preprocessing import Imputer

imp = Imputer()

#imp.fit(phot_modesub.values)
#X = imp.transform(phot_modesub.values)

imp.fit(planck_mw[['AME','ff','Sync']].values)
X = imp.transform(planck_mw[['AME','ff','Sync']].values)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

# Don't cheat - fit only on training data
scaler.fit(X)  
X = scaler.transform(X)  



#X = X[:,[0,1,-2]]


# In[13]:

#X = planck_mw_tr.drop(['CO'])

nmf = NMF(n_components=3)
S_nmf_ = nmf.fit(X).transform(X)

pca = PCA(n_components=3)
S_pca_ = pca.fit(X).transform(X)

rng = np.random.RandomState(42)
ica = FastICA(random_state = rng, n_components=3)
S_ica_ = ica.fit(X).transform(X)

S_ica_ /= S_ica_.std(axis=0)


# ## Apply PCA to the Planck-COMMANDER Maps:

# In[152]:

#AME Regions coords:


glat_regs    = np.genfromtxt('../Data/AME.txt',usecols = (2), dtype = 'float', delimiter=',')[1:]
glon_regs    = np.genfromtxt('../Data/AME.txt',usecols = (1), dtype = 'float', delimiter=',')[1:]

print glon_regs


# In[31]:




from matplotlib.colors import SymLogNorm
    
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.preprocessing import Imputer


def getCommanderPCs(commander_parmaps):
    
    imp = Imputer()

    #imp.fit(phot_modesub.values)
    #X = imp.transform(phot_modesub.values)

    imp.fit(commander_parmaps)
    X = imp.transform(commander_parmaps)

    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Don't cheat - fit only on training data
    scaler.fit(X)  
    X = scaler.transform(X)  
    
    pca = PCA(n_components=3)
    S_pca_ = pca.fit(X).transform(X)
    
    
    for i in range(0,np.size(S_pca_,axis=1)):
        #plt.subplot(5,4,i+1)
        #plt.figure(figsize=(20,20))
        p = hp.cartview(S_pca_[:,i], title="PC_"+str(i),
                   cmap = "rainbow", 
                    norm=SymLogNorm(linthresh=0.01,
                                    linscale=1,vmin=0), 
                    nest=True,
                    xsize = 1000
                       )
        hp.visufunc.projscatter(glon_regs, 
                                glat_regs,
                                lonlat=True, 
                                coord='G',
                                alpha = 1,
                                facecolors = 'none', 
                                edgecolors = 'red'
                               )
        
        plt.savefig("../Plots/pca_commander_ncomp3_{}.pdf".format(i), bbox_inches='tight')

    labels = ['AME','ff','Sync']

    for i in range(0,np.size(S_pca_,axis=1)):
    #for i in range(1,2):

        x_ = range(0,np.size(pca.components_,axis=1))
        y_ = pca.components_[i]

        fig, ax = plt.subplots()
        ax.scatter(x_,y_)
        for i, txt in enumerate(labels):
            #print x_[i], y_[i], labels[i]
            ax.annotate(labels[i], (x_[i],y_[i]))

        #plt.xscale('log')
        #plt.yscale('log')
        #plt.xlim(8,1000)
        #plt.xlabel("Wavelength (microns)       [EV_"+str(i)+"] "+str(round(eig_values[i]/sum(eig_values)*100,2))+"%")
        plt.ylabel("Relative Contribution")
        plt.show()
        plt.close()
        
        return S_pca, pca


# In[32]:

S_pca, pca = getCommanderPCs(planck_mw[['AME','ff','Sync']])


# In[112]:

def checkIonRatio(ion_r, ame):
    
    
    fontsize = 18
    
    hsize = len(ion_r)

    randsub = np.random.randint(low=0, 
                            high=hsize, 
                            size=hsize//20)

    p = sb.jointplot(ion_r.iloc[randsub], 
                     ame.iloc[randsub],
                     kind = "hex",
                    gridsize=20,
                    extent = (-1,20,-1, 215),
                    xlim = (-1,20),
                    ylim = (-1,150),
                    bins = 'log')
                    


    #plt.xlim(0,)
    #cb = plt.colorbar()
    #cb.set_label('log(N pixels)', fontsize=fontsize-3)
    #plt.title('PAH Ionization-tracing Band Ratio vs. AME Intensity', fontsize = fontsize)
    #plt.xlabel('$I_{9\mu{}m}$ / $I_{12\mu{}m}$', fontsize=fontsize-3)
    #plt.ylabel('$I_{AME}$ [$\mu{}K_{RJ}$]', fontsize=fontsize-3)
    #rint np.corrcoef(LOri_df.dropna().akari_9/LOri_df.dropna().iras_12, LOri_df.dropna().AME1)
    scipy.stats.spearmanr(ion_r, 
                          ame)
    
ion_r = phot_modesub.join(planck_mw.AME).dropna().A9 / phot_modesub.join(planck_mw.AME).dropna().I12

ame   = planck_mw.join(phot_modesub).dropna().AME

p = checkIonRatio(ion_r, ame)


# In[ ]:

# Make an NSIDE 64 version of the data to check for remaining issues caused by beam-size discrepancies


# In[211]:

phot_n64 = pd.DataFrame()

for map in phot.columns:

    phot_n64[map] = hp.ud_grade(phot_tr[map], nside_out = 64, order_in = 'NESTED', order_out = 'NESTED', pess = True)
    
    hp.mollview(phot_n64[map], nest=True, title=map, norm='hist')
    
planck_mw_n64 = pd.DataFrame()

print planck_mw.columns

for map in planck_mw.columns:
    
    planck_mw_n64[map] = hp.ud_grade(planck_mw_tr[map], 
                                     nside_out = 64, 
                                     order_in = 'NESTED', 
                                     order_out = 'NESTED', 
                                     pess = True)
    
    hp.mollview(planck_mw_n64[map], nest=True, title=map, norm='hist')


# In[212]:

plt.close()

s_corr = phot_n64.join(planck_mw_n64['AME']).dropna().corr(method='spearman')

p_corr = phot_n64.join(planck_mw_n64['AME']).dropna().corr(method='pearson')
plt.figure(figsize=(10,10))
sb.heatmap(s_corr, square=True, annot=True)
plt.show()
plt.close()
plt.figure(figsize=(10,10))
sb.heatmap(p_corr, square=True, annot=True)
plt.show()
plt.close()
plt.figure(figsize=(10,10))
sb.heatmap((s_corr - p_corr), square=True, annot=True)
plt.show()
plt.close()


# In[215]:

# Experiment with the Planck PR2 Tau353Ghz map:
dust_gnilc = pd.DataFrame()
dust_gnilc['tau'] = hp.read_map('/work1/users/aaronb/Databrary/HEALPix/COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits', nest=True)
dust_gnilc['rad'] = hp.read_map('/work1/users/aaronb/Databrary/HEALPix/COM_CompMap_Dust-GNILC-Radiance_2048_R2.00.fits', nest=True)


# In[216]:

dust_gnilc_n64 = pd.DataFrame()

dust_gnilc_n64['tau'] = hp.ud_grade(
    dust_gnilc['tau'], 
    nside_out = 64, 
    order_in = 'NESTED', 
    order_out = 'NESTED', 
    pess = True
    )


dust_gnilc_n64['rad'] = hp.ud_grade(
    dust_gnilc['rad'], 
    nside_out = 64, 
    order_in = 'NESTED', 
    order_out = 'NESTED', 
    pess = True
    )


# In[230]:

dust_gnilc_n64['isrf']              = dust_gnilc_n64['rad'] / dust_gnilc_n64['tau']
planck_mw_n64['ame_g0_scaled']      = planck_mw_n64['AME'] / dust_gnilc_n64['isrf']
planck_mw_n64['ame_tau_scaled']      = planck_mw_n64['AME'] / dust_gnilc_n64['tau']
planck_mw_n64['ame_rad_scaled']      = planck_mw_n64['AME'] / dust_gnilc_n64['rad']


# In[236]:

phot_n64['A9_I12'] = phot_n64['A9'] / phot_n64['I12']


# In[244]:

phot_tr['A9_I12'] = phot_modesub['A9'] / phot_modesub['I12']


# In[237]:

plt.close()

s_corr = phot_n64.join(
    planck_mw_n64[['AME','ame_g0_scaled','ame_tau_scaled','ame_rad_scaled']], 
        ).dropna().corr(method='spearman')

p_corr = phot_n64.join(
    planck_mw_n64[['AME','ame_g0_scaled','ame_tau_scaled','ame_rad_scaled']]).dropna().corr(method='pearson')


plt.figure(figsize=(10,10))
sb.heatmap(s_corr, square=True, annot=True)
plt.show()
plt.close()
plt.figure(figsize=(10,10))
sb.heatmap(p_corr, square=True, annot=True)
plt.show()
plt.close()
plt.figure(figsize=(10,10))
sb.heatmap((s_corr - p_corr), square=True, annot=True)
plt.show()
plt.close()


# In[ ]:



