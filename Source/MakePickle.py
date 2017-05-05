### Make a gargantuan cube. The "layers" are the FIR data:
import numpy as np
import pandas as pd
import healpy as hp

filepath      =  "../Data/raw"

nside         = 256

bands         = ["akari_90",\
                    "dirbe_100", "iras_100",\
                    "dirbe_140", "akari_140",\
                    "akari_160", \
                    "dirbe_240", \
                    "planck_857", "planck_545"]


band_names =   [ "akari_9",\
                    "dirbe_12","iras_12", "wise_12", \
                    "akari_18", \
                    "dirbe_25","iras_25", \
                    "dirbe_60","iras_60","akari_65",\
                    "akari_90",\
                    "dirbe_100", "iras_100",\
                    "dirbe_140", "akari_140",\
                    "akari_160", \
                    "dirbe_240", \
                    "planck_857", "planck_545"]


band_abbr =   [ "A9",\
                    "D12","I12", "W12", \
                    "A18", \
                    "D25","I25", \
                    "D60","I60","A65",\
                    "A90",\
                    "D100", "I100",\
                    "D140", "A140",\
                    "A160", \
                    "D240", \
                    "P857", "P545"]



band_labels  = ["AKARI 9 $\mu{m}$",\
                "DIRBE 12 $\mu{m}$","IRAS 12 $\mu{m}$","WISE 12 $\mu{m}$", \
                "AKARI 18 $\mu{m}$",\
                "DIRBE 25 $\mu{m}$", "IRAS 25 $\mu{m}$", \
                "DIRBE 60 $\mu{m}$","IRAS 60 $\mu{m}$","AKARI 65 $\mu{m}$", \
                "AKARI 90 $\mu{m}$", \
                "DIRBE 100 $\mu{m}$","IRAS 100 $\mu{m}$",\
                "DIRBE 140 $\mu{m}$","AKARI 140 $\mu{m}$",\
                "AKARI 160 $\mu{m}$",\
                "DIRBE 240 $\mu{m}$",\
                "PLANCK 350 $\mu{m}$","PLANCK 550 $\mu{m}$" ]

waves_micron  = [ 9,12,12,12,18,25,25,60,60,65,90,100,100,140,140,160,240,350,550]

nbands            = len(bands)
nbands_all        = len(band_names)

planck_bb_path    = "../Data/raw/COM_CompMap_dust-commander_0256_R2.00.fits" #HEALPix FITS table containing Planck low-res modBB results
fields  = [4,7,1] #The field number in the HEALPix file
labels  = ["Temperature","Beta","Radiance"]

planck_bb = pd.DataFrame()
for i in range (0,2):
    planck_bb[labels[i]] = hp.read_map(planck_bb_path,field = fields[i])







def mode(ndarray,axis=0):
    if ndarray.size == 1:
        return (ndarray[0],1)
    elif ndarray.size == 0:
        raise Exception('Attempted to find mode on an empty array!')
    try:
        axis = [i for i in range(ndarray.ndim)][axis]
    except IndexError:
        raise Exception('Axis %i out of range for array with %i dimension(s)' % (axis,ndarray.ndim))
    srt = np.sort(ndarray,axis=axis)
    dif = np.diff(srt,axis=axis)
    shape = [i for i in dif.shape]
    shape[axis] += 2
    indices = np.indices(shape)[axis]
    index = tuple([slice(None) if i != axis else slice(1,-1) for i in range(dif.ndim)])
    indices[index][dif == 0] = 0
    indices.sort(axis=axis)
    bins = np.diff(indices,axis=axis)
    location = np.argmax(bins,axis=axis)
    mesh = np.indices(bins.shape)
    index = tuple([slice(None) if i != axis else 0 for i in range(dif.ndim)])
    index = [mesh[i][index].ravel() if i != axis else location.ravel() for i in range(bins.ndim)]
    counts = bins[tuple(index)].reshape(location.shape)
    index[axis] = indices[tuple(index)]
    modals = srt[tuple(index)].reshape(location.shape)
    return (modals, counts)


glatrange = 10.0
glatrange_mid = 2.5
elatrange = 15.0

### Import the Galactic coordinate reference columns:
### These are just "maps" of glat and glon. That way you can easily get the center pixel coordinates from a given pixel index

glon = hp.read_map(filepath+str(nside)+"_nside/pixel_coords_map_ring_galactic_res8.fits", field = 0, memmap=False)
glat = hp.read_map(filepath+str(nside)+"_nside/pixel_coords_map_ring_galactic_res8.fits", field = 1, memmap=False)
elon = hp.read_map(filepath+str(nside)+"_nside/pixel_coords_map_ring_ecliptic_res8.fits", field = 0, memmap=False)
elat = hp.read_map(filepath+str(nside)+"_nside/pixel_coords_map_ring_ecliptic_res8.fits", field = 1, memmap=False)


gcut_1 = np.where((abs(glat > glatrange)) & (abs(elat) > elatrange))
gcut_2 = np.where((abs(glat < glatrange)) & (abs(elat) > elatrange))

glon = pd.DataFrame(glon, columns=['GLON'])
glat = pd.DataFrame(glat, columns=['GLAT'])

### Same for the ecliptic coordinates:


##### Now, we have a cube of the FIR data saved as "fir"
##### We want to compare the individual maps in a way that makes some physical sense
##### How about we start by assuming an SED? Next: Modified blackbody fitting
layer = 0
nside = 256
npix  = 12*nside**2
phot  = np.ones([npix, nbands_all])


for band in band_names:

    phot[:,layer] = hp.read_map(filepath+str(nside)+"_nside/"+band+"_"+str(nside)+"_1dres.fits",memmap=False);
    layer += 1

print "IR Maps Read"
phot = pd.DataFrame(phot, columns = band_abbr)

AME, AME_hdr = hp.read_map('/work1/users/aaronb/Databrary/HEALPix/COM_CompMap_AME-commander_0256_R2.00.fits',field = 0, memmap=False, h=True);
print "AME Map read"
CO, CO_hdr   = hp.read_map('/work1/users/aaronb/Databrary/HEALPix/COM_CompMap_CO-commander_0256_R2.00.fits',field = 0, memmap=False, h=True);
print "CO Map Read"
ff, ff_hdr   = hp.read_map('/work1/users/aaronb/Databrary/HEALPix/COM_CompMap_freefree-commander_0256_R2.00.fits', field = 0, memmap=False, h=True)
print "Free-free Map Read"
Sync, Sync_hdr   = hp.read_map('/work1/users/aaronb/Databrary/HEALPix/COM_CompMap_Synchrotron-commander_0256_R2.00.fits', field = 0, memmap=False, h=True)
print "Synchrotron Map Read"





## Replace the HEALPix "UNSEEN" pixels with NaN, in a Pandas Dataframe:



phot = phot.join(pd.DataFrame(AME,  columns= ['AME']))
phot = phot.join(pd.DataFrame(ff,   columns= ['FF']))
phot = phot.join(pd.DataFrame(CO,   columns= ['CO']))
phot = phot.join(pd.DataFrame(Sync, columns= ['Syn']))

bb = pd.DataFrame(AME,  columns= ['AME'])
bb = bb.join(pd.DataFrame(Planck_T, columns= ['T']))
bb = bb.join(pd.DataFrame(Planck_B, columns= ['Beta']))
bb = bb.join(pd.DataFrame(Planck_FIR, columns= ['FIR']))

phot.replace(
    to_replace =hp.UNSEEN,
    value=np.nan,
    inplace=True
    )

bb.replace(
    to_replace =hp.UNSEEN,
    value=np.nan,
    inplace=True
    )

# bb = bb.join(
#       pd.DataFrame(
#             (bb['T'].values / 17.5)**(4+2),
#             columns=['G0']
#                 )
#                 )

# print "G0 Map Calculated"




## Calculate the mode of each HEALPix map.
## Round to 3 decimal places, to consolidate multipleunique modes

allsky_modes = phot.round(3).mode(axis=0)

## Subtract the all-sky mode from each map:
## Trying a vectorized way now, using the Pandas ".subtract" method
phot_modesub = pd.DataFrame(phot.values-allsky_modes.values,columns=phot.columns)


#allsky_corrcoeff = pd.DataFrame(pd.DataFrame.corr(phot, method='pearson'))
#allsky_corrcoeff_gcut1 = pd.DataFrame(pd.DataFrame.corr(phot.iloc[gcut_1], method='pearson'))
#allsky_corrcoeff_gcut2 = pd.DataFrame(pd.DataFrame.corr(phot.iloc[gcut_2], method='pearson'))
