import numpy as np
from pyraf import iraf
from astropy.io import fits,ascii
from astropy.modeling.models import Legendre2D
import os

def mask_image(hdulist, bpm='badpix.fits', objmask=0, handmask=0):
    '''
    Returns a version of the input image where masked pixels are set to -999
    Requires:
        --Image HDUList, output from e.g. fits.open()
        --Name of bad pixel mask file
        --Name of other mask file; if you want to use only the bpm, don't give this a value
        --Name of hand mask file; if you don't have it, don't give this a value
    Returns:
        --HDUList with data now masked (masked pixels replaced with -999)
    '''
    bpmdat = fits.getdata(bpm)
    
    if objmask != 0:
        objmaskdat = fits.getdata(objmask)
    else:
        objmaskdat = bpmdat * 0.0
        
    if handmask != 0:
        handmaskdat = fits.getdata(handmask)
    else:
        handmaskdat = bpmdat * 0.0
    
    # Note: use of this on the raw frames results in these becoming values of 
    # 64537.  Probably this is because the raw frames are 16 bit, so use only
    # with processed frames.
    hdulist[0].data[bpmdat != 0] = -999
    hdulist[0].data[objmaskdat != 0] = -999
    hdulist[0].data[handmaskdat != 0] = -999
    
    return hdulist

#-----------------------------------------------------------------------------
def skysub(dire,Ntrial,Nloops,degree=2,overwrite=True):
    '''
    Subtracts sky models from images.
    '''
    pvals = ascii.read(dire+'/Trial'+str(Ntrial)+'/pvals'+str(Nloops-1)+'.dat')
    for i in range(len(pvals)):
        im=fits.open(dire+'/f'+pvals['fname'][i])
        dim1 = np.arange(1, im[0].data.shape[1]+1)
        dim2 = np.arange(1, im[0].data.shape[0]+1)
        X, Y = np.meshgrid(dim1, dim2)
        m = Legendre2D(degree, degree,
                       c0_0 = pvals['c0_0'][i],
                       c1_0 = pvals['c1_0'][i],
                       c2_0 = pvals['c2_0'][i],
                       c0_1 = pvals['c0_1'][i],
                       c1_1 = pvals['c1_1'][i],
                       c2_1 = pvals['c2_1'][i],
                       c0_2 = pvals['c0_2'][i],
                       c1_2 = pvals['c1_2'][i],
                       c2_2 = pvals['c2_2'][i])
        skyplane = m(X,Y)
        im[0].data -= skyplane
        im.writeto(dire+'/sf'+pvals['fname'][i], overwrite=overwrite)
        im.close()
        print('created sf'+pvals['fname'][i])

        
#-----------------------------------------------------------------------------
def mask(dire,lis,ipref,mpref,bpm, overwrite=True):
    '''
    Masks images with object masks + bad pixel map.
    '''
    tab=ascii.read(dire+'/'+lis,format='no_header')
    for ii in tab:
        ima=ipref+ii[0]
        msk=0
        if os.path.exists(dire+'/'+mpref+ii[0]):
            msk=dire+'/'+mpref+ii[0]
        hdul=fits.open(dire+'/'+ima)
        hdul=mask_image(hdul,dire+'/'+bpm,objmask=msk)
        hdul.writeto(dire+'/m'+ima,overwrite=overwrite)
        hdul.close()
        print('created m'+ima)


#-----------------------------------------------------------------------------
def register(dire,lis,ipref,ref,prefix='W',overwrite=True):
    '''
    Registers images to a reference image. Requires:
        --lis, file name of list of images (no header)
        --ipref, prefix for the image names
        --ref, a reference image
    '''
    if overwrite:
        imlist = ascii.read(dire+'/'+lis, format='no_header')
        for i in imlist['col1']:
            if os.path.exists(dire+'/'+prefix+ipref+i):
                os.remove(dire+'/'+prefix+ipref+i)

    ims=ipref+'@'+dire+'/'+lis
    iraf.wregister(input=dire+'/'+ims,reference=ref,output=dire+'/'+prefix+ims, fitgeometry='rxyscale',function='legendre',xxterms='full',yxterms='full',boundary='constant',constant=-999.0,fluxconserve='no')
    
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    # Requires 'target.lis' lists of raw (tz) target (not sky) frames and 'vmap_on/off.fits' vignetting bad pixel maps on both on and off directories
    ondir='Flats/on'
    offdir='Flats/off'
    mosdir='Mosaic'
    
    tlis='target.lis'
    
    onbpm='vmap_on.fits'
    offbpm='vmap_off.fits'
    refim='refimage.fits'

    Ntrial=5
    Nloops=5

    print('Subtracting skies....')
    skysub(ondir,Ntrial,Nloops)
    skysub(offdir,Ntrial,Nloops)
    
    print('Applying masks....')
    mask(ondir,tlis,'sf','hm',onbpm)
    mask(offdir,tlis,'sf','hm',offbpm)

    
    if os.path.isfile(refim):
        print('Reference image already exists, skipping creation.')
    else:
        print('Creating reference image....')
        iraf.artdata()
        iraf.artdata.mkpattern(refim, ncols=2000, nlines=2000, v1=0)
        iraf.imcoords()
        iraf.imcoords.ccsetwcs(refim, "", xref=1000, yref=1000, xmag=-0.2138004, ymag=0.2138004, lngref='02:12:54.6', latref='-19:19:06.0')

    print('Registering images to reference....')
    register(ondir,tlis,'msf',refim)
    register(offdir,tlis,'msf',refim)

    print('Moving registered images to new '+mosdir+' directory')
    if not os.path.isdir(mosdir):
        os.system('mkdir '+mosdir)
    os.system('cp '+ondir+'/'+'Wmsf*fits '+mosdir)
    os.system('cp '+offdir+'/'+'Wmsf*fits '+mosdir)
