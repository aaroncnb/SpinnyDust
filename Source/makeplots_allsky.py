
# coding: utf-8

# ### All-sky AME vs. IR Scatter Plots

# In[ ]:




# In[5]:

get_ipython().magic(u'matplotlib inline')
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


# In[6]:




# In[7]:

with open('../Data/maps.pickle') as f:  # Python 3: open(..., 'rb')
    coords, planck_bb, planck_mw, phot, phot_modesub = pickle.load(f)
    
phot.head()
#planck_bb.head()
#planck_mw.head()
#coords.head()


# In[9]:

glatrange     = 10.0
glatrange_mid = 2.5
elatrange     = 10


gcut_l = np.where((abs(coords['glat']) < glatrange) & (abs(coords['elat']) > elatrange))
gcut_h = np.where((abs(coords['glat']) > glatrange) & (abs(coords['elat']) > elatrange))







# In[10]:

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler


### Setup the standard pipeline to apply to all the data:
allsky_pipeline = Pipeline([
    ('imputer', Imputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

phot_tr = pd.DataFrame(allsky_pipeline.fit_transform(phot),columns=phot.columns)
planck_bb_tr = pd.DataFrame(allsky_pipeline.fit_transform(planck_bb),columns=planck_bb.columns)
planck_mw_tr = pd.DataFrame(allsky_pipeline.fit_transform(planck_mw),columns=planck_mw.columns)


phot_corr = phot_tr.corr(method='spearman')
planck_bb_corr = planck_bb_tr.corr(method='spearman')
planck_mw_corr = planck_mw_tr.corr(method='spearman')


# In[13]:

import seaborn as sb
phot_corr     = phot_tr.join(planck_mw_tr['AME']).corr(method='spearman')
phot_corr_lgl = phot_tr.join(planck_mw_tr['AME']).iloc[gcut_l].corr(method='spearman')
phot_corr_hgl = phot_tr.join(planck_mw_tr['AME']).iloc[gcut_h].corr(method='spearman')


# In[14]:

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


# In[15]:

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
    


# In[16]:

## Force background color to be white:
### Note that seaborn plotting functions my override these settings.
plt.rcParams['axes.facecolor']='white'
plt.rcParams['figure.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'


# # Correlation tests along Galactic meridians and parallels:

# In[17]:

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


# In[18]:

bba = planck_bb_tr.join(phot_tr['A9']).join(planck_mw_tr['AME'])

bb_corr_glats = [bba.iloc[i].corr(method='spearman') for i in glats]

bb_corr_glons = [bba.iloc[i].corr(method='spearman') for i in glons]

#bb_corr_glats_A9 = [bb_corr_glats[i]['A9']]

bb_corr_glats_pn = pd.Panel({i: bb_corr_glats[i] for i in glat_intervs})
bb_corr_glons_pn = pd.Panel({i: bb_corr_glons[i] for i in glon_intervs})


# In[19]:

bba.columns


# In[20]:

X = glat_intervs

Y = bb_corr_glats_pn.values[:,:,4]

fig, ax = plt.subplots()

for i in range(0,4):
    ax.scatter(X,Y[:,i], alpha=0.7, label=bba.columns[i])
    
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



# In[80]:

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



# In[82]:

fig, ax = plt.subplots()

for i in range(0,4):
    ax.hist(bb_corr_glats_pn.dropna().values[:,i,4], alpha=0.7, label=bba.columns[i], bins=10)
ax.legend(loc=0)
plt.xlabel("$S$ (rel. to AME)", fontsize=20)
fig.show()

fig.savefig("../Plots/PlanckModBBvsAMEandA9_GLAT_hist.pdf", bbox_inches='tight')

#plt.plot[bb_corr_glats_pn.values[]])


# In[83]:

fig, ax = plt.subplots()

for i in range(0,4):
    ax.hist(bb_corr_glons_pn.dropna().values[:,i,4], alpha=0.7, label=bba.columns[i], bins=10)
ax.legend(loc=0)
plt.xlabel("$S$ (rel. to AME)", fontsize=20)
fig.show()

fig.savefig("../Plots/PlanckModBBvsAMEandA9_GLON_hist.pdf", bbox_inches='tight')

#plt.plot[bb_corr_glats_pn.values[]])


# ## AME to IR Ratio Averages:

# In[21]:

(phot.values.T/planck_mw['AME'].values).T


# In[22]:

## Just got the median intensities along GLON:
#### Using mode-subtracted maps


# In[32]:

glats


# In[23]:


phot_glons = [phot.iloc[i].dropna().median() for i in glons]
phot_glats = [phot.iloc[i].dropna().median() for i in glats]
np.shape(phot_glons)
pd.DataFrame(phot_glons).head()


# In[34]:

pd.DataFrame(phot_glons)[['A9','I12','A18','I25','I100','A140','P857']].join(planck_mw['AME']).plot(subplots=True,title="Med. Intensity Variation Along GLON")
plt.xlabel("$l$")
plt.show()
plt.savefig("../Plots/IRPhotvsAMEbyGLON.pdf", bbox_inches='tight')
plt.close()

pd.DataFrame(phot_glats)[['A9','I12','A18','I25','I100','A140','P857']].join(planck_mw['AME']).set_index(glat_intervs
).plot(subplots=True,title="Med. Intensity Variation Along GLAT ")
plt.xlabel("$b$")
plt.show()
plt.savefig("../Plots/IRPhotvsAMEbyGLAT.pdf", bbox_inches='tight')
plt.close()


# In[ ]:

## Get the IR:AME ratios along GLON:


# In[44]:

phot_AME_ratio = pd.DataFrame(
    
    (phot.values.T/planck_mw['AME'].values).T,
    columns = phot.columns)
phot_AME_ratio.head()


# In[52]:

phot_AME_ratio_glons = [phot_AME_ratio.iloc[i].dropna().median() for i in glons]
phot_AME_ratio_glats = [phot_AME_ratio.iloc[i].dropna().median() for i in glats]
np.shape(phot_AME_ratio_glons)
pd.DataFrame(phot_AME_ratio_glons).head()


# In[78]:

plt.rcParams['axes.facecolor']='white'
plt.rcParams['figure.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'

pd.DataFrame(phot_AME_ratio_glons).plot(logy=True)

import sklearn


# In[85]:

pd.DataFrame(phot_AME_ratio_glons)[['A9','I12','A18','I25','I100','A140','P857']].join(planck_mw['AME']).plot(subplots=True, title="IR:AME Ratio Along GLON")
#plt.title("Variation of IR:AME Ratio Along Glon")
plt.xlabel("$l$")
plt.show()
fig.savefig("../Plots/IR_AME_Ratio_byGLON.pdf", bbox_inches='tight')
plt.close()
pd.DataFrame(phot_AME_ratio_glats)[['A9','I12','A18','I25','I100','A140','P857']].join(planck_mw['AME']).plot(subplots=True,logy=True, title="IR:AME Ratio Along GLAT")
plt.xlabel("$b$")
fig.savefig("../Plots/IR_AME_Ratio_byGLAT.pdf", bbox_inches='tight')
plt.show()
plt.close()


# In[20]:


#IR_AME_ratios_glats = [phot.iloc[i]/pla) for i in glats]

phot_AME_ratio_glons = [phot_AME_ratio.iloc[i].dropna().mean() for i in glons]
phot_AME_ratio_glats = [phot_AME_ratio.iloc[i].dropna().mean() for i in glats]
#bb_corr_glats_A9 = [bb_corr_glats[i]['A9']]

phot_AME_ratio_glons_pn = pd.Panel({i: phot_AME_ratio_glons[i] for i in glon_intervs})
phot_AME_ratio_glats_pn = pd.Panel({i: phot_AME_ratio_glats[i] for i in glat_intervs})


# In[ ]:

phot_AME_ratio_glats


# # All-sky AME vs. IR plots:

# In[ ]:

ncols=3
nrows=6

fig, axs = plt.subplots(ncols=ncols, 
                        nrows=nrows, 
                        sharey=True, 
                        sharex=True,
                       figsize=(20,20))
fig.subplots_adjust(hspace=0.1, left=0.1, right=0.7)

k=0


for i in range(0,nrows):
    for j in range(0,ncols):
        
            k += 1
            x = phot_modesub.values[:,k]

            y = planck_mw['AME'].values[:]
            
            x_ = x[(x>0) & (y>0)]
            y_ = y[(x>0) & (y>0)]
            
            
            xmin = 5e-5#x_.min()
            xmax = 400 #x_.max()
            ymin = 0.01#y_.min()
            ymax = y_.max()
            
            ax = axs[i,j]

            hb = ax.hexbin(x_, y_, 
                   mincnt=0,
                   gridsize=100,
                   bins='log', 
                   cmap='inferno_r',
                   xscale='log',
                   yscale='log')
            
            
            ax.axis([xmin, xmax, ymin, ymax])
            
            ax.text(0.2, 0.9,phot_modesub.columns[k], horizontalalignment='center',
              verticalalignment='center',
              transform=ax.transAxes, 
              fontsize=18)
            
            
ax = axs[0,0]
ax.set_ylabel('AME Intensity [MJy/sr]', fontsize=20)
ax = axs[-1,0]
ax.set_xlabel('IR Intensity [MJy/sr]', fontsize=20)

plt.show()

fig.savefig("../Plots/AMEvsDust_allsky_allbands.pdf", bbox_inches='tight')


# In[ ]:

phot_modesub.columns[k]


# # Angular Power Spectra:

# In[16]:

plt.loglog(hp.anafast(planck_mw['AME'].values))


# In[23]:

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


# In[32]:

a140 = phot['A140'].replace(
    to_replace =np.nan,
    value=hp.UNSEEN
    ).values

hp.anafast(a140)


# In[ ]:



