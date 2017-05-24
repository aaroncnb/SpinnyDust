
# coding: utf-8

# ### All-sky AME vs. IR Scatter Plots

# In[ ]:




# In[3]:

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


# In[4]:

with open('../Data/maps.pickle') as f:  # Python 3: open(..., 'rb')
    coords, planck_bb, planck_mw, phot, phot_modesub = pickle.load(f)
    
phot.head()
#planck_bb.head()
#planck_mw.head()
#coords.head()


# In[49]:

glatrange     = 10.0
glatrange_mid = 2.5
elatrange     = 10


gcut_l = np.where((abs(coords['glat']) < glatrange) & (abs(coords['elat']) > elatrange))
gcut_h = np.where((abs(coords['glat']) > glatrange) & (abs(coords['elat']) > elatrange))







# In[9]:

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


# In[50]:

import seaborn as sb
phot_corr     = phot_tr.join(planck_mw_tr['AME']).corr(method='spearman')
phot_corr_lgl = phot_tr.join(planck_mw_tr['AME']).iloc[gcut_l].corr(method='spearman')
phot_corr_hgl = phot_tr.join(planck_mw_tr['AME']).iloc[gcut_h].corr(method='spearman')


# In[60]:

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
    
    ax[1].set_title("|GLAT| > 10", fontsize=20)

    
    
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
    
    ax[2].set_title("|GLAT| < 10", fontsize=20)


    fig.tight_layout(rect=[0, 0, .9, 1])
    
    plt.show()

    fig.savefig("../Plots/all_bands_corr_matrix_wAME_spearman.pdf", bbox_inches='tight')


# ## Applying Independent Component Analysis (ICA) from 'skitlearn':
# 
# (From: 
# http://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_vs_pca.html#sphx-glr-auto-examples-decomposition-plot-ica-vs-pca-py)
# 

# In[ ]:

from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import TSNE



# In[ ]:

from sklearn.preprocessing import StandardScaler, MinMaxScaler



nmf = NMF()
S_nmf_ = nmf.fit(X).transform(X)

pca = PCA(n_components=5)
S_pca_ = pca.fit(X).transform(X)

rng = np.random.RandomState(42)
ica = FastICA(random_state = rng, n_components=np.size(X,axis=1))
S_ica_ = ica.fit(X).transform(X)


tsne = TSNE

S_ica_ /= S_ica_.std(axis=0)


# In[ ]:

np.size(nmf.components_[1])


# In[ ]:

print np.shape(S_ica_)
print np.shape(S_pca_)
print np.shape(S_nmf_)


# In[ ]:

from matplotlib.colors import SymLogNorm

plt.figure(figsize=(20,20))
for i in range(1,np.size(S_pca_,axis=1)):
    #plt.subplot(5,4,i+1)
    #plt.figure(figsize=(20,20))
    hp.cartview(
    #hp.gnomview(
        S_pca_[:,i-1], title="PCA_"+str(i-1),
        cmap = "rainbow", 
        norm = SymLogNorm(linthresh=0.01,linscale=1,vmin=0),
        sub  = (6,4,i)
    )
plt.show()
plt.close()


# In[3]:

from matplotlib.colors import SymLogNorm

plt.figure(figsize=(20,20))
for i in range(1,np.size(S_ica_,axis=1)):
    #plt.subplot(5,4,i+1)
    #plt.figure(figsize=(20,20))
    hp.cartview(
    #hp.gnomview(
        S_ica_[:,i-1], title="IC_"+str(i-1),
        cmap = "rainbow", 
        norm = SymLogNorm(linthresh=0.01,linscale=1,vmin=0),
        sub  = (7,4,i)
    )
plt.show()
plt.close()


# In[4]:

from matplotlib.colors import SymLogNorm

plt.figure(figsize=(20,20))
for i in range(1,np.size(S_nmf_,axis=1)):
    #plt.subplot(5,4,i+1)
    #plt.figure(figsize=(20,20))
    hp.cartview(
    #hp.gnomview(
        S_nmf_[:,i-1], title="NMF_"+str(i-1),
        cmap = "rainbow", 
        norm = SymLogNorm(linthresh=0.01,linscale=1,vmin=0),
        sub  = (7,4,i)
    )
plt.show()
plt.close()


# In[5]:

labels = [str(phot.columns[i]) for i in range(0,26)]
plt.figure(figsize=(30,30))

for i in range(0,np.size(S_nmf_,axis=1)):
#for i in range(1,2):
    plt.subplot(7,4,i+1)
    
    x_ = range(0,np.size(nmf.components_,axis=1))
    y_ = nmf.components_[i]
    
    #fig, ax = plt.subplots()
    plt.scatter(x_,y_)
    for i, txt in enumerate(labels):
        #print x_[i], y_[i], labels[i]
        plt.annotate(labels[i], (x_[i],y_[i]))
        
    
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.xlim(8,1000)
    #plt.xlabel("Wavelength (microns)       [EV_"+str(i)+"] "+str(round(eig_values[i]/sum(eig_values)*100,2))+"%")
    plt.ylabel("Relative Contribution")
    #plt.show()
    #plt.close()


# ## Applying decomposition of AME with Blackbody Parameters (Planck):

# In[6]:

from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import TSNE


# In[ ]:

nmf = NMF(n_components=5)
S_nmf_ = nmf.fit(X).transform(X)

pca = PCA(n_components=5)
S_pca_ = pca.fit(X).transform(X)

rng = np.random.RandomState(42)
ica = FastICA(random_state = rng, n_components=5)
S_ica_ = ica.fit(X).transform(X)


#tsne = TSNE

S_ica_ /= S_ica_.std(axis=0)


# In[ ]:

from matplotlib.colors import SymLogNorm

plt.figure(figsize=(20,20))
for i in range(0,np.size(S_pca_,axis=1)):
    #plt.subplot(5,4,i+1)
    #plt.figure(figsize=(20,20))
    hp.cartview(
    #hp.gnomview(
        S_pca_[:,i], title="PCA_"+str(i),
        cmap = "rainbow", 
        norm = SymLogNorm(linthresh=0.01,linscale=1,vmin=0),
        sub  = (3,2,i+1)
    )
plt.show()
plt.close()


# In[ ]:

from matplotlib.colors import SymLogNorm

plt.figure(figsize=(20,20))
for i in range(0,np.size(S_ica_,axis=1)):
    #plt.subplot(5,4,i+1)
    #plt.figure(figsize=(20,20))
    hp.cartview(
    #hp.gnomview(
        S_ica_[:,i], title="IC_"+str(i),
        cmap = "rainbow", 
        norm = SymLogNorm(linthresh=0.01,linscale=1,vmin=0),
        sub  = (5,3,i+1)
    )
plt.show()
plt.close()


# In[ ]:




# In[ ]:

labels = [str(bb.columns[i]) for i in range(0,5)]
plt.figure(figsize=(8,8))

from matplotlib.colors import SymLogNorm

plt.figure(figsize=(15,20))

for i in range(0,np.size(S_nmf_,axis=1)):
    #plt.subplot(5,4,i+1)
    #plt.figure(figsize=(20,20))
    hp.cartview(
    #hp.gnomview(
        S_nmf_[:,i], title="NMF_"+str(i),
        cmap = "rainbow", 
        norm = SymLogNorm(linthresh=0.01,linscale=1,vmin=0),
        sub  = (5,2,(i*2)+1)
    )
    
    plt.subplot(5,2,(i+1)*2)
    
    x_ = range(0,np.size(nmf.components_,axis=1))
    y_ = nmf.components_[i]
    
    #fig, ax = plt.subplots()
    plt.bar(x_,y_)
    
    for j, txt in enumerate(labels):
        #print x_[i], y_[i], labels[i]
        plt.annotate(labels[j], (x_[j],y_[j]))
        
    
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.xlim(8,1000)
    #plt.xlabel("Wavelength (microns)       [EV_"+str(i)+"] "+str(round(eig_values[i]/sum(eig_values)*100,2))+"%")
    plt.ylabel("Rel. NMF Mag")
    plt.tight_layout()
    
    plt.tick_params(
         axis='x',          # changes apply to the x-axis
         which='both',      # both major and minor ticks are affected
         bottom='off',      # ticks along the bottom edge are off
         top='off',         # ticks along the top edge are off
         labelbottom='off') # labels along the bottom edge are off
    


#plt.tight_layout()
plt.show()
plt.close()


#for i in range(0,np.size(S_nmf_,axis=1)):
#for i in range(1,2):
    
    
    #plt.show()
    #plt.close()


# In[ ]:

start = -90
stop = 90
step = 1

glat_intervs = np.arange(start,stop,step)

glats = [np.where(np.logical_and(glat>i, glat< i+1))  for i in glat_intervs]



# In[ ]:

import seaborn as sb
bb_corr = bb.join(phot['A9']).corr(method='spearman')
bb_corr_drop = bb_corr.drop('AME',axis=0).drop('A9',axis=1)
mask = np.zeros_like(bb_corr_drop.values)
mask[np.triu_indices_from(mask,k=1)] = True

with sb.axes_style("white"):
    sb.heatmap(
        bb_corr_drop,
        linewidths=.5,
        annot=True,
        mask=mask)
    
    plt.title("Planck Mod-BB vs. AME and AKARI 9 micron emission")



# In[ ]:

import seaborn as sb
bb_corr_normd = bb_normd.corr(method='spearman')
bb_corr_normd_drop = bb_corr_normd.drop('AME',axis=0).drop('A9',axis=1)
mask = np.zeros_like(bb_corr_normd_drop.values)
mask[np.triu_indices_from(mask,k=1)] = True

with sb.axes_style("white"):
    sb.heatmap(
        bb_corr_norm_drop,
        linewidths=.5,
        annot=True,
        mask=mask)
    
    plt.title("Planck Mod-BB vs. AME and AKARI 9 micron emission")



# In[ ]:

from sklearn import preprocessing

bba = bb.join(phot['A9'])

from sklearn.preprocessing import Imputer

imp = Imputer()

imp.fit(bba.values)
X = imp.transform(bba.values)

min_max_scaler = preprocessing.MinMaxScaler()
bb_normd =  min_max_scaler.fit_transform(X)

bb_normd = pd.DataFrame(bb_normd, columns=bba.columns)

bb_normd


# In[ ]:

try:
    p = sb.PairGrid(bb_normd)
    p.map_upper(plt.scatter,alpha=0.1)
    p.map_lower(sb.kdeplot,cmap = "Blues_d")
    p.map_diag(sb.kdeplot, lw=3, legend=False)
except ValueError:  #raised if `y` is empty.
    pass



# In[ ]:

try:
    p = sb.PairGrid(bb_normd)
    p.map_upper(plt.scatter)
    p.map_lower(sb.kdeplot,cmap = "Blues_d")
    g.map_diag(sb.kdeplot, lw=3, legend=False)
except ValueError:  #raised if `y` is empty.
    pass



# In[ ]:

bba.iloc[glats[40]].corr(method='spearman')
#glats


# In[ ]:

bb_corr_glats = [bba.iloc[i].corr(method='spearman') for i in glats]

#     bb_corr_drop = bb_corr.drop('AME',axis=0).drop('A9',axis=1)
#     mask = np.zeros_like(bb_corr_drop.values)
#     mask[np.triu_indices_from(mask,k=1)] = True

#     with sb.axes_style("white"):
#         sb.heatmap(
#             bb_corr_drop,
#             linewidths=.5,
#             annot=True,
#             mask=mask)

#         plt.title("Planck Mod-BB vs. AME and AKARI 9 micron emission")

bb_corr_glats_A9 = [bb_corr_glats[i]['A9']]


# In[ ]:

bb_corr_glats_pn = pd.Panel({i: bb_corr_glats[i] for i in glat_intervs})


# In[ ]:

X = glat_intervs

Y = bb_corr_glats_pn.values[:,:,0]

plt.figure(figsize=(12,12))

for i in range(1,5):
    plt.scatter(X,Y[:,i], alpha=0.7, label=bba.columns[i])
plt.legend(loc=1,prop={'size':15},fancybox=True, framealpha=0.9)
#leg.get_frame().set_alpha(0.5)
plt.xlim(-90,90)
plt.ylabel("Spearman Rank Coefficient (rel. to AME)")
plt.xlabel("Galactic Latitude [1-deg. bins]")
plt.xkcd()
plt.show()
plt.close()
# plt.hist(bb_corr_glats_pn.dropna().values[:,2,0], bins=10, alpha=0.4, label='Beta')
# plt.hist(bb_corr_glats_pn.dropna().values[:,3,0], bins=10, alpha=0.4, label='FIR')
# plt.hist(bb_corr_glats_pn.dropna().values[:,4,0], bins=10, alpha=0.4, label='A9')


# In[ ]:

plt.figure(figsize=(10,10))
plt.hist(bb_corr_glats_pn.dropna().values[:,1,0], bins=10, alpha=0.4, label='T')
plt.hist(bb_corr_glats_pn.dropna().values[:,2,0], bins=10, alpha=0.4, label='Beta')
plt.hist(bb_corr_glats_pn.dropna().values[:,3,0], bins=10, alpha=0.4, label='FIR')
plt.hist(bb_corr_glats_pn.dropna().values[:,4,0], bins=10, alpha=0.4, label='A9')
plt.legend()
#plt.xkcd()
#plt.plot[bb_corr_glats_pn.values[]])


# In[ ]:

labels = [str(bb.columns[i]) for i in range(0,5)]
plt.figure(figsize=(5,5))

for i in range(0,np.size(S_ica_,axis=1)):
#for i in range(1,2):
    plt.subplot(3,2,i+1)
    
    x_ = range(0,np.size(ica.components_,axis=1))
    y_ = ica.components_[i]
    
    #fig, ax = plt.subplots()
    plt.scatter(x_,y_)
    for i, txt in enumerate(labels):
        #print x_[i], y_[i], labels[i]
        plt.annotate(labels[i], (x_[i],y_[i]))
        
    
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.xlim(8,1000)
    #plt.xlabel("Wavelength (microns)       [EV_"+str(i)+"] "+str(round(eig_values[i]/sum(eig_values)*100,2))+"%")
    plt.ylabel("Relative Contribution")
    #plt.show()
    #plt.close()
    
    


# In[ ]:

labels = [str(bb.columns[i]) for i in range(0,5)]
plt.figure(figsize=(20,20))

for i in range(0,np.size(S_pca_,axis=1)):
#for i in range(1,2):
    plt.subplot(3,2,i+1)
    
    x_ = range(0,np.size(pca.components_,axis=1))
    y_ = pca.components_[i]
    
    #fig, ax = plt.subplots()
    plt.scatter(x_,y_)
    for i, txt in enumerate(labels):
        #print x_[i], y_[i], labels[i]
        plt.annotate(labels[i], (x_[i],y_[i]))
        
    
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.xlim(8,1000)
    #plt.xlabel("Wavelength (microns)       [EV_"+str(i)+"] "+str(round(eig_values[i]/sum(eig_values)*100,2))+"%")
    plt.ylabel("Relative Contribution")
    #plt.show()
    #plt.close()


# In[ ]:

plt.figure(figsize=(20,20))
plt.scatter(S_pca_[:,0], S_pca_[:,1], alpha= 0.1)


# In[ ]:

plt.figure(figsize=(20,20))
plt.scatter(S_ica_[:,0], S_ica_[:,1], alpha= 0.1)


# In[ ]:

plt.figure(figsize=(20,20))
plt.scatter(S_nmf_[:,0], S_nmf_[:,1], alpha= 0.1)


# In[ ]:

model = TSNE(
    n_components=2,
    random_state=42,
    perplexity=50,
    verbose=2
        )
np.set_printoptions(suppress=True)
# for i in range(0,100):
#     try:
#         model.fit_transform(X[::i]) 
#     except:
#         print "Memory Error"


# In[ ]:

S_tsne =model.fit_transform(X[::100]) 


# In[ ]:

plt.figure(figsize=(20,20))
plt.scatter(S_tsne[:,0],S_tsne[:,1], alpha=0.2)


# In[ ]:

model = TSNE(
    n_components=2,
    random_state=42,
    perplexity=25,
    verbose=2
        )
#np.set_printoptions(suppress=True)
# for i in range(0,100):
#     try:
#         model.fit_transform(X[::i]) 
#     except:
#         print "Memory Error"

S_tsne =model.fit_transform(X[::100]) 

plt.figure(figsize=(20,20))
plt.scatter(S_tsne[:,0],S_tsne[:,1], alpha=0.2)


# In[ ]:

model = TSNE(
    n_components=2,
    random_state=42,
    perplexity=25,
    verbose=2
        )
#np.set_printoptions(suppress=True)
# for i in range(0,100):
#     try:
#         model.fit_transform(X[::i]) 
#     except:
#         print "Memory Error"

X 

S_tsne =model.fit_transform(X[gcut_1]) 

plt.figure(figsize=(20,20))
plt.scatter(S_tsne[:,0],S_tsne[:,1], alpha=0.2)


# In[ ]:

model = TSNE(
    n_components=2,
    random_state=42,
    perplexity=50,
    verbose=2
        )
#np.set_printoptions(suppress=True)
# for i in range(0,100):
#     try:
#         model.fit_transform(X[::i]) 
#     except:
#         print "Memory Error"

S_tsne =model.fit_transform(X[gcut_2]) 

plt.figure(figsize=(20,20))
plt.scatter(S_tsne[:,0],S_tsne[:,1], alpha=0.2)


# In[ ]:



