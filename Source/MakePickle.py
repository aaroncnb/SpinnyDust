### Make a gargantuan cube. The "layers" are the FIR data:
import numpy as np
import pandas as pd
import healpy as hp
import sys

filepath      =  "../Data/raw/"

nside         = 256
npix          = 12*nside**2

nest = bool(sys.argv[1])


band_names =   [ "akari_9",\
                    "iras_12", \
                    "akari_18", \
                    "iras_25", \
                    "iras_60","akari_65",\
                    "akari_90",\
                    "iras_100",\
                    "akari_140",\
                    "akari_160", \
                    "planck_857", "planck_545"]


band_abbr =   [ "A9",\
                    "I12", \
                    "A18", \
                    "I25", \
                    "I60","A65",\
                    "A90",\
                    "I100",\
                    "A140",\
                    "A160", \
                    "P857", "P545"]



band_labels  = ["AKARI 9 $\mu{m}$",\
                "IRAS 12 $\mu{m}$", \
                "AKARI 18 $\mu{m}$",\
                "IRAS 25 $\mu{m}$", \
                "IRAS 60 $\mu{m}$","AKARI 65 $\mu{m}$", \
                "AKARI 90 $\mu{m}$", \
                "IRAS 100 $\mu{m}$",\
                "AKARI 140 $\mu{m}$",\
                "AKARI 160 $\mu{m}$",\
                "PLANCK 350 $\mu{m}$","PLANCK 550 $\mu{m}$" ]

waves_micron  = [ 9,12,18,25,60,65,90,100,140,160,350,550]

nbands_all        = len(band_names)



### Import the Galactic coordinate reference columns:
### These are just "maps" of glat and glon. That way you can easily get the center pixel coordinates from a given pixel index

coords = pd.DataFrame()
coords['glon'] = hp.read_map(filepath+"pixel_coords_map_ring_galactic_res8.fits", field = 0, nest=nest)
coords['glat'] = hp.read_map(filepath+"pixel_coords_map_ring_galactic_res8.fits", field = 1, nest=nest)
coords['elon'] = hp.read_map(filepath+"pixel_coords_map_ring_ecliptic_res8.fits", field = 0, nest=nest)
coords['elat'] = hp.read_map(filepath+"pixel_coords_map_ring_ecliptic_res8.fits", field = 1, nest=nest)



#### Read Planck low-res Modified blackbody fitting results:
planck_bb_path    = filepath+"/COM_CompMap_dust-commander_0256_R2.00.fits.gz" #HEALPix FITS table containing Planck low-res modBB results
fields            = [4,7,1] #The field number in the HEALPix file
labels            = ["$T$","$B$","$I_{dust}(545)$"]

planck_bb = pd.DataFrame()
for i in range(0,3):
    planck_bb[labels[i]] = hp.read_map(planck_bb_path,field = fields[i], nest=nest)
    
### After adding the PR2 BB-fit results, also add the PR1 radiance map:
### But first we have to load it and smooth it, as in Hensley+ 2016)
### PR1 Dust Parameter Maps (NSIDE 2048) available here:
###
# 
#http://irsa.ipac.caltech.edu/data/Planck/release_1/all-sky-maps/previews/HFI_CompMap_ThermalDustModel_2048_R1.20/index.html
#
#
#planck_bb['$R_{PR1}$'] = 
#hp.read_map('/work1/users/aaronb/Databrary/HEALPix/AKARI_HEALPix_orig/256_nside/radiance_PR1_256_smooth.fits',
#                      nest = nest)

planck_bb['$R_{PR1}$'] = hp.ud_grade(		
		hp.read_map(
			filepath+"/HFI_CompMap_ThermalDustModel_2048_R1.20.fits",
			nest=True,
			field=3 
		),
		nside_out = 256,
		order_in = 'NESTED',
		order_out = 'NESTED')

planck_bb['$\tau_{353,PR1}$'] = hp.ud_grade(
                hp.read_map(
                        filepath+"/HFI_CompMap_ThermalDustModel_2048_R1.20.fits",
                        nest=True,
                        field=0),
                nside_out = 256,
                order_in = 'NESTED',
                order_out = 'NESTED')

planck_bb['$U$'] =  planck_bb['$R_{PR1}$'].divide(planck_bb['$\tau_{353,PR1}$'], axis=0)



planck_bb.replace(
    to_replace =hp.UNSEEN,
    value=np.nan,
    inplace=True
    )


#### Read Planck low-res microwave component fitting results:
planck_mw = pd.DataFrame()
labels = ['AME','CO','ff','Sync']

paths = ['COM_CompMap_AME-commander_0256_R2.00.fits.gz',
         'COM_CompMap_CO-commander_0256_R2.00.fits.gz',
         'COM_CompMap_freefree-commander_0256_R2.00.fits.gz',
         'COM_CompMap_Synchrotron-commander_0256_R2.00.fits.gz']

for label, path in zip(labels, paths):
    planck_mw[label] = hp.read_map(filepath+path,field = 0, nest=nest)

planck_mw.replace(
    to_replace =hp.UNSEEN,
    value=np.nan,
    inplace=True
    )
print "COMMANDER MW Maps Read"



#### Read in the MIR to FIR photometry data:
phot = pd.DataFrame()

for i in range(0,len(band_names)):
    phot[band_abbr[i]] = hp.read_map(filepath+band_names[i]+"_"+str(nside)+"_1dres.fits.gz", nest=nest)

print "IR Maps Read"

phot.replace(
    to_replace =hp.UNSEEN,
    value=np.nan,
    inplace=True
    )

    ## Calculate the mode of each HEALPix map.
## Round to 3 decimal places, to consolidate multipleunique modes

allsky_modes = phot.round(3).mode(axis=0)

## Subtract the all-sky mode from each map:
## Trying a vectorized way now, using the Pandas ".subtract" method
phot_modesub = pd.DataFrame(phot.values-allsky_modes.values,columns=phot.columns)

## ALternate background correction: Subtract a monopole using healpy

phot_mpsub = pd.DataFrame()

for map in phot.columns:
	phot_mpsub[map] =  hp.remove_monopole(phot[map], nest=True, gal_cut=20)


import pickle

# obj0, obj1, obj2 are created here...

# Saving the objects:
if nest == True:
    pname = "maps_nest.pickle"
else:
    pname = "maps.pickle"

with open("../Data/"+pname, 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([coords, planck_bb, planck_mw, phot, phot_modesub, phot_mpsub], f)
