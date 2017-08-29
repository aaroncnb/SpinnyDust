
# coding: utf-8

# ### All-sky AME vs. IR Scatter Plots

# In[ ]:




# In[6]:

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


# In[403]:

with open('../Data/maps_nest.pickle') as f:  # Python 3: open(..., 'rb')
    coords, planck_bb, planck_mw, phot, phot_modesub = pickle.load(f)
    
phot.head()


rad_pr1 = hp.read_map('/work1/users/aaronb/Databrary/HEALPix/AKARI_HEALPix_orig/256_nside/radiance_PR1_256_smooth.fits')


rad_pr1 = pd.DataFrame(rad_pr1, columns=['$R$'])



# In[404]:

rad_pr1.replace(to_replace=hp.UNSEEN, value=np.nan, inplace=True)


# In[9]:

glatrange     = 10.0
glatrange_mid = 2.5
elatrange     = 10


gcut_l = np.where((abs(coords['glat']) < glatrange) & (abs(coords['elat']) > elatrange))
gcut_h = np.where((abs(coords['glat']) > glatrange) & (abs(coords['elat']) > elatrange))







# In[307]:

print hp.fit_monopole(phot['A9'])
print hp.fit_monopole(phot['A9'].values[hmask!=hp.UNSEEN])
print phot['A9'].mode()
print hp.fit_monopole(phot['A18'])
print phot['A18'].mode()
print hp.fit_dipole(phot['A9'])
print hp.fit_dipole(phot['A18'])


# In[10]:

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

phot_tr = pd.DataFrame(allsky_pipeline.fit_transform(phot),columns=phot.columns)
planck_bb_tr = pd.DataFrame(allsky_pipeline.fit_transform(planck_bb),columns=planck_bb.columns)
planck_mw_tr = pd.DataFrame(allsky_pipeline.fit_transform(planck_mw),columns=planck_mw.columns)





# In[11]:

phot_corr = phot_tr.corr(method='spearman')
planck_bb_corr = planck_bb_tr.corr(method='spearman')
planck_mw_corr = planck_mw_tr.corr(method='spearman')


# In[12]:

import seaborn as sb
phot_corr     = phot_tr.join(planck_mw_tr['AME']).corr(method='spearman')
phot_corr_lgl = phot_tr.join(planck_mw_tr['AME']).iloc[gcut_l].corr(method='spearman')
phot_corr_hgl = phot_tr.join(planck_mw_tr['AME']).iloc[gcut_h].corr(method='spearman')


# In[13]:

#bb_corr_drop = bb_corr.drop('AME',axis=0).drop('A9',axis=1)
mask = np.zeros_like(phot_corr.values)
mask[np.triu_indices_from(mask,k=1)] = True

with sb.axes_style("white"):

    
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    cbar_ax = fig.add_axes([.91, .2, .03, .7])
    
    sb.heatmap(
        phot_corr,
        #linewidths=.5,
        annot=False,
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
        annot=False,
        mask=mask,
        cbar=False,
        yticklabels=False,
        xticklabels=False,
        ax=ax[1],
        vmin=0,
        vmax=1)
    
    ax[1].set_title("$|b| > 10^{\circ}$", fontsize=20)

    
    
    sb.heatmap(
        phot_corr_lgl,
        #linewidths=.5,
        annot=False,
        mask=mask,
        cbar=True,
        cbar_ax=cbar_ax,
        yticklabels=False,
        xticklabels=False,
        ax=ax[2],
        vmin=0,
        vmax=1)
    
    ax[2].set_title("$|b| < 10^{\circ}$", fontsize=20)


    fig.tight_layout(rect=[0, 0, .9, 1])
    
    plt.show()

    fig.savefig("../Plots/all_bands_corr_matrix_wAME_spearman.pdf", bbox_inches='tight')


# In[14]:

planck_bb_corr = planck_bb_tr.join(phot_tr['A9']).join(planck_mw_tr['AME']).corr(method='spearman')
#bb_corr_drop = bb_corr.drop('AME',axis=0).drop('A9',axis=1)
mask = np.zeros_like(planck_bb_corr.values)
mask[np.triu_indices_from(mask,k=1)] = True



with sb.axes_style("white"):
    
    fig = plt.figure(figsize=(6,5))
    sb.heatmap(
        planck_bb_corr,
        linewidths=.5,
        annot=True,
        mask=mask)
    
    
    fig.show()
    
    plt.title("Planck Mod-BB vs. AME and AKARI 9 $\mu$m emission",fontsize=10)
    
    fig.tight_layout(rect=[0, 0, .9, 1])

    fig.savefig("../Plots/PlanckModBBvsAMEandA9.pdf", bbox_inches='tight')
    


# In[15]:

## Force background color to be white:
### Note that seaborn plotting functions my override these settings.
#plt.rcParams['axes.facecolor']='white'
#plt.rcParams['figure.facecolor']='white'
#plt.rcParams['savefig.facecolor']='white'


# In[16]:

bba = planck_bb_tr.join(phot_tr['A9']).join(planck_mw_tr['AME'])


# ## Confirmation of all-sky correlation results:
# # Instead of downgrading the pixel sizes, just copy the same correaltion value of the 64 NSIDE 256 pixels
# # in a batch to all of those pixel positions in an output map (also size NSIDE 256).

# In[20]:

#import pandas as pd
#import numpy as np
#import healpy as hp

def testSpatialCorr(df, nside_in, nside_out, test=False):
    
    npix_in    = 12*nside_in**2
    npix_out   = 12*nside_out**2
    pix_interv = (nside_in/nside_out)**2
    
    ## First, do it the "normal way"-
    patches_corr = [df.iloc[i*pix_interv:(i+1)*pix_interv].corr(method='spearman') for i in range(0,npix_out)]
    corr_patches_pn = pd.Panel({i: patches_corr[i] for i in range(0,npix_out)})

#     ## Confirm it:
#     if test==True:
        
#         patches_corr_conf = []

#         for i in range(0,npix_out):
#             for j in range(0,pix_interv):
#                 patches_corr_conf.append(patches_corr[i])

#         corr_patches_pn_conf = pd.Panel({i: patches_corr_conf[i] for i in range(0,npix_in)})

    fig = plt.figure(figsize=(8,4))

    for j in range(0,4):
        #plt.subplot(2,5,(j*2)+1)
        im = hp.cartview(corr_patches_pn.values[:,j,4],
                         sub=(1,4,j+1), 
                         fig=fig,
                         cmap="rainbow", 
                         cbar=True, 
                         min=-1, 
                         max=1, 
                         nest=True, 
                         title="$S$(AME:"+bba.columns[j]+") NSIDE"+str(nside_out))
        
        #fig.subplots_adjust(right=0.8)
        #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        #fig.colorbar(plt.gca(),cax=cbar_ax)
        plt.savefig("../Plots/Spearman_Map_nside"+str(nside_out)+"_AMEtoMBBandA9.pdf")
    
    return corr_patches_pn


# In[ ]:

nside_in = 256
nside_out = 32
corr_patches_pn = testSpatialCorr(bba, nside_in, nside_out)


# In[ ]:

nside_in = 256
nside_out = 16
corr_patches_pn = testSpatialCorr(bba, nside_in, nside_out,test=False)


# In[ ]:

nside_in = 256
nside_out = 8
corr_patches_pn = testSpatialCorr(bba, nside_in, nside_out, test=False)


# In[ ]:

nside_in = 256
nside_out = 4
corr_patches_pn, corr_patches_pn_conf = testSpatialCorr(bba, nside_in, nside_out)


# In[ ]:

nside_in = 256
nside_out = 2

corr_patches_pn, corr_patches_pn_conf = testSpatialCorr(bba, nside_in, nside_out)


# In[ ]:

# The check appears successful, so make a plot grid of all the nsides:


# In[51]:

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



# In[52]:

SpatialCorrAll(bba, 256)


# In[26]:

PlotSpatialCorrAll(corr_patches_pn)


# In[ ]:

fig.subplots_adjust(hspace=0.1, left=0.1, right=0.7)
fig.show()


# # Correlation tests along Galactic meridians and parallels:

# In[53]:

start = -90
stop = 90
step = 1

glat_intervs = np.arange(start,stop,step)

start = 0
stop = 360
step = 1

glon_intervs = np.arange(start,stop,step)

glats = [np.where(np.logical_and(coords['glat']>i, coords['glat']< i+1))  for i in glat_intervs]

glons = [np.where(np.logical_and(coords['glon']>i, coords['glon']< i+1))  for i in glon_intervs]


# In[54]:


bb_corr_glats = [bba.iloc[i].corr(method='spearman') for i in glats]

bb_corr_glons = [bba.iloc[i].corr(method='spearman') for i in glons]

#bb_corr_glats_A9 = [bb_corr_glats[i]['A9']]

bb_corr_glats_pn = pd.Panel({i: bb_corr_glats[i] for i in glat_intervs})
bb_corr_glons_pn = pd.Panel({i: bb_corr_glons[i] for i in glon_intervs})


# In[55]:

planck_mw_corr_glats = [planck_mw.iloc[i].corr(method='spearman') for i in glats]

planck_mw_corr_glons = [planck_mw.iloc[i].corr(method='spearman') for i in glons]

#bb_corr_glats_A9 = [bb_corr_glats[i]['A9']]

planck_mw_corr_glats_pn = pd.Panel({i: planck_mw_corr_glats[i] for i in glat_intervs})
planck_mw_corr_glons_pn = pd.Panel({i: planck_mw_corr_glons[i] for i in glon_intervs})


# In[56]:

bba.columns


# In[57]:

import matplotlib as mpl
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['figure.edgecolor'] = 'black'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['figure.facecolor'] = 'white'


# In[58]:

#plt.rcParams['axes.facecolor']='white'
#matplotlib.style.use('seaborn-bright')


X = glat_intervs

Y = bb_corr_glats_pn.values[:,:,4]

fig, ax = plt.subplots(edgecolor='k')

for i in range(0,4):
    ax.set_frame_on(True)
    ax.scatter(X,Y[:,i], alpha=0.7, label=bba.columns[i])
    ax.patch.set_visible(True) 
    ax.grid(True) 
    ax.set_frame_on(True)
    
    
legend = ax.legend(loc=0, shadow=True)
# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

plt.xlim(-90,90)
plt.ylabel("$S$ (rel. to AME)", fontsize=20)
plt.xlabel("$GLAT [1^{\circ}$  bins]", fontsize=20)
fig.show()

fig.savefig("../Plots/PlanckModBBvsAMEandA9_byGLAT.pdf", 
            bbox_inches ='tight')



# In[59]:

X = glon_intervs

Y = bb_corr_glons_pn.values[:,:,4]

fig, ax  = plt.subplots()

for i in range(0,4):
    ax.scatter(X,Y[:,i], alpha=0.7, label=bba.columns[i])
    
legend = ax.legend(loc=0, shadow=True)
# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')
#plt.legend(loc=1,prop={'size':14},fancybox=True, framealpha=1)
#leg.get_frame().set_alpha(0.5)
plt.xlim(0,360)
plt.ylabel("$S$ (rel. to AME)", fontsize=20)
plt.xlabel(" $GLON [1^{\circ}$  bins]", fontsize=20)
fig.show()

fig.savefig("../Plots/PlanckModBBvsAMEandA9_byGLON.pdf", 
            bbox_inches ='tight',
            facecolor = fig.get_facecolor())
#plt.close()
# plt.hist(bb_corr_glats_pn.dropna().values[:,2,0], bins=10, alpha=0.4, label='Beta')
# plt.hist(bb_corr_glats_pn.dropna().values[:,3,0], bins=10, alpha=0.4, label='FIR')
# plt.hist(bb_corr_glats_pn.dropna().values[:,4,0], bins=10, alpha=0.4, label='A9')



# In[60]:

fig, ax = plt.subplots()

for i in range(0,4):
    ax.hist(bb_corr_glats_pn.dropna().values[:,i,4], alpha=0.7, label=bba.columns[i], bins=10)
ax.legend(loc=0)
plt.xlabel("$S$ (rel. to AME)", fontsize=20)
fig.show()

fig.savefig("../Plots/PlanckModBBvsAMEandA9_GLAT_hist.pdf", bbox_inches='tight')

#plt.plot[bb_corr_glats_pn.values[]])


# In[61]:

fig, ax = plt.subplots()

for i in range(0,4):
    ax.hist(bb_corr_glons_pn.dropna().values[:,i,4], alpha=0.7, label=bba.columns[i], bins=10)
ax.legend(loc=0)
plt.xlabel("$S$ (rel. to AME)", fontsize=20)
fig.show()

fig.savefig("../Plots/PlanckModBBvsAMEandA9_GLON_hist.pdf", bbox_inches='tight')
#plt.close()

#plt.plot[bb_corr_glats_pn.values[]])


# ## AME to IR Ratio Averages:

# In[62]:

## Just got the median intensities along GLON:
#### Using mode-subtracted maps


# In[63]:


planck_mw_glons_med  = [planck_mw.iloc[i].dropna().median() for i in glons]
planck_mw_glats_med  = [planck_mw.iloc[i].dropna().median() for i in glats]
planck_mw_glons_mean = [planck_mw.iloc[i].dropna().mean() for i in glons]
planck_mw_glats_mean = [planck_mw.iloc[i].dropna().mean() for i in glats]


phot_glons_med = [phot.iloc[i].dropna().median() for i in glons]
phot_glats_med = [phot.iloc[i].dropna().median() for i in glats]
phot_glons_mean= [phot.iloc[i].dropna().mean() for i in glons]
phot_glats_mean = [phot.iloc[i].dropna().mean() for i in glats]


# In[64]:

pd.DataFrame(phot_glons_med)[['A9','I12','A18','I25','I100','A140','P857']].join(pd.DataFrame(planck_mw_glons_med)['AME']) .plot(subplots=True,title="Med. Intensity Variation Along GLON")
plt.xlabel("$l$")
#plt.show()
plt.savefig("../Plots/IntensityByGLON_median.pdf", bbox_inches='tight')
plt.close()

pd.DataFrame(phot_glats_med)[['A9','I12','A18','I25','I100','A140','P857']].join(pd.DataFrame(planck_mw_glats_med)['AME']).set_index(glat_intervs) .plot(subplots=True,title="Med. Intensity Variation Along GLAT ")
plt.xlabel("$b$")
#plt.show()
plt.savefig("../Plots/IntensityByGLAT_median.pdf", bbox_inches='tight')
plt.close()

#Plot by Means:
pd.DataFrame(phot_glons_mean)[['A9','I12','A18','I25','I100','A140','P857']].join(pd.DataFrame(planck_mw_glons_mean)['AME']). plot(subplots=True,title="Mean Intensity Variation Along GLON")
plt.xlabel("$l$")
#plt.show()
plt.savefig("../Plots/IntensitybyGLON_mean.pdf", bbox_inches='tight')
plt.close()

pd.DataFrame(phot_glats_mean)[['A9','I12','A18','I25','I100','A140','P857']].join(pd.DataFrame(planck_mw_glats_mean)['AME']).set_index(glat_intervs) .plot(subplots=True,title="Mean Intensity Variation Along GLAT ")
plt.xlabel("$b$")
#plt.show()
plt.savefig("../Plots/IntensityByGLAT_mean.pdf", bbox_inches='tight')
plt.close()


# In[65]:

# Get the COMMANDER Components along GLON and GLAT:

pd.DataFrame(planck_mw_glons_med).plot(subplots=True,title="Med. Intensity Variation Along GLON")
plt.xlabel("$l$")
#plt.show()
plt.savefig("../Plots/CommanderByGLON_median.pdf", bbox_inches='tight')
plt.close()

pd.DataFrame(planck_mw_glats_med).set_index(glat_intervs) .plot(subplots=True,title="Med. Intensity Variation Along GLAT ")
plt.xlabel("$b$")
#plt.show()
plt.savefig("../Plots/CommanderByGLAT_median.pdf", bbox_inches='tight')
plt.close()
######################
## By Mean:
pd.DataFrame(planck_mw_glons_mean).plot(subplots=True,title="Mean Intensity Variation Along GLON")
plt.xlabel("$l$")
#plt.show()
plt.savefig("../Plots/CommanderByGLON_mean.pdf", bbox_inches='tight')
plt.close()

pd.DataFrame(planck_mw_glats_mean).set_index(glat_intervs) .plot(subplots=True,title="Mean Intensity Variation Along GLAT ")
plt.xlabel("$b$")
#plt.show()
plt.savefig("../Plots/CommanderByGLAT_mean.pdf", bbox_inches='tight')
plt.close()


# In[66]:

phot_AME_ratio = pd.DataFrame(
    
    (phot_tr.values.T/planck_mw_tr['AME'].values).T,
    columns = phot_tr.columns)

phot_AME_ratio.head()

phot_AME_ratio_glons_med = [phot_AME_ratio.iloc[i].median() for i in glons]
phot_AME_ratio_glats_med = [phot_AME_ratio.iloc[i].median() for i in glats]

phot_AME_ratio_glons_mean = [phot_AME_ratio.iloc[i].dropna().mean() for i in glons]
phot_AME_ratio_glats_mean = [phot_AME_ratio.iloc[i].dropna().mean() for i in glats]

planck_mw_glons_med_scaled  = [planck_mw_tr.iloc[i].dropna().median() for i in glons]
planck_mw_glats_med_scaled  = [planck_mw_tr.iloc[i].dropna().median() for i in glats]
planck_mw_glons_mean_scaled = [planck_mw_tr.iloc[i].dropna().mean() for i in glons]
planck_mw_glats_mean_scaled = [planck_mw_tr.iloc[i].dropna().mean() for i in glats]


# In[67]:

pd.DataFrame(phot_AME_ratio_glons_med)[['A9','I12','A18','I25','I100','A140','P857']].join(pd.DataFrame(planck_mw_glons_med_scaled)['AME']) .plot(subplots=True,title="Med. IR:AME Ratio Along GLON")
plt.xlabel("$l$")
plt.show()
plt.savefig("../Plots/Intensity_AME_ratio_ByGLON_median.pdf", bbox_inches='tight')
plt.close()

pd.DataFrame(phot_AME_ratio_glats_med)[['A9','I12','A18','I25','I100','A140','P857']].join(pd.DataFrame(planck_mw_glats_med_scaled)['AME']).set_index(glat_intervs) .plot(subplots=True,title="Med. IR:AME Ratio Along GLAT ")
plt.xlabel("$b$")
plt.show()
plt.savefig("../Plots/Intensity_AME_ratio_ByGLAT_median.pdf", bbox_inches='tight')
plt.close()

#Plot by Means:
pd.DataFrame(phot_AME_ratio_glons_mean)[['A9','I12','A18','I25','I100','A140','P857']].join(pd.DataFrame(planck_mw_glons_mean_scaled)['AME']). plot(subplots=True,title="Mean IR:AME Ratio Along GLON")
plt.xlabel("$l$")
plt.show()
plt.savefig("../Plots/Intensity_AME_ratio_byGLON_mean.pdf", bbox_inches='tight')
plt.close()

pd.DataFrame(phot_AME_ratio_glats_mean)[['A9','I12','A18','I25','I100','A140','P857']].join(pd.DataFrame(planck_mw_glats_mean_scaled)['AME']).set_index(glat_intervs) .plot(subplots=True,title="Mean IR:AME Ratio Along GLAT ")
plt.xlabel("$b$")
plt.show()
plt.savefig("../Plots/Intensity_AME_ratio_ByGLAT_mean.pdf", bbox_inches='tight')
plt.close()


# # All-sky AME vs. IR plots:

# In[18]:

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

                x_ = x[(x>0) & (y>0)]
                y_ = y[(x>0) & (y>0)]


                xmin = 5e-5#x_.min()
                xmax = 400 #x_.max()
                ymin = 0.01#y_.min()
                ymax = y_.max()

                ax = axs[i,j]
                #ax.set_aspect(aspect, adjustable='box')

                hb = ax.hexbin(
                       x_,
                       y_, 
                       mincnt=1,
                       gridsize=50,
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
            
ax = axs[1,0]
ax.set_ylabel('AME [$\mu{}K_{CMB}$]', fontsize=15)
ax = axs[-1,2]
ax.set_xlabel('IR [MJy/sr]', fontsize=15)

plt.show()

fig.savefig("../Plots/AMEvsDust_allsky_allbands.pdf", bbox_inches='tight', dpi=100)  


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


# ## Compare with Planck PR1 Radiance Map 
# (degraded and smoothed as in Hensley+ 2016)

# In[21]:

plt.close()

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
                
                R = rad_pr1['$R$'].values[:]
                
                y = y/R
                x = x/R

                x_ = x[(x>0) & (y>0)]
                y_ = y[(x>0) & (y>0)]


                xmin = x_.min()
                xmax = x_.max()
                ymin = y_.min()
                ymax = y_.max()

                ax = axs[i,j]
                #ax.set_aspect(aspect, adjustable='box')

                hb = ax.hexbin(
                       x_,
                       y_, 
                       mincnt=1,
                       gridsize=10,
                       bins='log', 
                       cmap='inferno_r',
                       xscale='log',
                       yscale='log')


                ax.axis([xmin, xmax, ymin, ymax])

                ax.text(0.4, 0.4,
                        phot_modesub.columns[k], 
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes, 
                        fontsize=15)
                
                ax.grid(True)
                
                ax.set_frame_on(True)

                k += 1
            
ax = axs[1,0]
ax.set_ylabel('$I_{AME} / R$', fontsize=15)
ax = axs[-1,3]
ax.set_xlabel('$I_{IR} / R$', fontsize=15)

plt.show()

fig.savefig("../Plots/AMEtoRvsDusttoR_allsky_allbands.pdf", bbox_inches='tight')  




# In[19]:

plt.close()

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


                y = rad_pr1['$R$'].values[:]
                
                #y = y/R
                #x = x/R

                x_ = x[(x>0)]
                y_ = y[(x>0)]


                xmin = x_.min()
                xmax = x_.max()
                ymin = y_.min()
                ymax = y_.max()

                ax = axs[i,j]
                #ax.set_aspect(aspect, adjustable='box')

                hb = ax.hexbin(
                       x_,
                       y_, 
                       mincnt=1,
                       gridsize=50,
                       bins='log', 
                       cmap='inferno_r',
                       xscale='log',
                       yscale='log')


                ax.axis([xmin, xmax, ymin, ymax])

                ax.text(0.8, 0.5,phot_modesub.columns[k], horizontalalignment='center',
                  verticalalignment='center',
                  transform=ax.transAxes, 
                  fontsize=15)
                
                ax.grid(True)
                
                ax.set_frame_on(True)

                k += 1
            
ax = axs[1,0]
ax.set_ylabel('$R$', fontsize=15)
ax = axs[-1,3]
ax.set_xlabel('$I_{IR}$', fontsize=15)

plt.show()

fig.savefig("../Plots/IRvsR_allsky_allbands.pdf", bbox_inches='tight', dpi=100)  


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


# In[ ]:



