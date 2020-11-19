import numpy as np
from pyraf import iraf
from astropy.io import fits,ascii
import os
from astropy.table import Table

#-----------------------------------------------------------------------------
def shift(dire,lis,ipref,star,mosaic=False,prefix='sh',overwrite=True):
    '''
    Uses phot to align images. Requires:
        --lis, file name of list of images (no header)
        --ipref, prefix for the image names
        --star, file name with the coordinates of the reference star
        --mosaic as a reference image
    '''
    ims=ipref+'@'+dire+'/'+lis
    iraf.noao.digiphot()
    iraf.noao.digiphot.apphot()
    iraf.unlearn('centerpars')
    iraf.unlearn('phot')
    iraf.noao.digiphot.apphot.datapars.fwhmpsf='4.54'
    iraf.noao.digiphot.apphot.centerpars.cbox='70.'
    iraf.noao.digiphot.apphot.centerpars.calgorithm='gauss'
    iraf.digiphot.apphot.phot(image=dire+'/'+ims, output='tmp.mag', coords=star, wcsin='world', wcsout='logical', verify='No',interactive='No')
    
    phottab=ascii.read('tmp.mag')
    os.remove('tmp.mag')

    if mosaic == False:
        tab = ascii.read(dire+'/'+lis, format='no_header')
        mosaic = dire + '/' + ipref + tab['col1'][0]

    iraf.phot(image=mosaic, output='tmp2.mag', coords=star, wcsin='world', wcsout='logical', verify='No',interactive='No')
    reftab=ascii.read('tmp2.mag')
    os.remove('tmp2.mag')
    refx=reftab['XCENTER'][0]
    refy=reftab['YCENTER'][0]
    
    imlist = ascii.read(dire+'/'+lis, format='no_header')
    ims2 = [ipref + i for i in imlist['col1']]

    for i in range(len(phottab)):
        cenx = phottab['XCENTER'][i]
        ceny = phottab['YCENTER'][i]
        shiftx = -(cenx-refx)
        shifty = -(ceny-refy)
        if overwrite and os.path.exists(dire+'/'+prefix+ims2[i]):
            os.remove(dire+'/'+prefix+ims2[i])
        iraf.imshift(input=dire+'/'+ims2[i], output=dire+'/'+prefix+ims2[i], xshift=shiftx, yshift=shifty, interp_type='linear', boundary_type='nearest', constant=0.0)
        print('created '+prefix+ims2[i])
    


#-----------------------------------------------------------------------------
def phot_standard(dire,lis,ipref,region,overwrite=True):
    '''
    Calculates magnitude in an elliptical aperture. Requires:
        --lis, file name of list of images (no header)
        --ipref, prefix for the image names
        --region, ds9 region file for the aperture
    '''
    imlist = ascii.read(dire+'/'+lis, format='no_header')
    frames= [ipref + i for i in imlist['col1']]

    tab=ascii.read(region)
    xcen=tab['col3'][0]
    ycen=tab['col4'][0]
    a=np.max([tab['col5'][0],tab['col6'][0]])
    b=np.min([tab['col5'][0],tab['col6'][0]])
    pa=tab['col7'][0]

    mags=[]
    for frame in frames:
        hdul=fits.open(dire+'/'+frame)
        im=hdul[0].data
        hdr=hdul[0].header
        hdul.close()
        x = np.arange(1, len(im)+1)
        y = x.reshape(-1, 1)
        xEll = (x - xcen)*np.cos(np.radians(pa+90)) + (y - ycen)*np.sin(np.radians(pa+90))
        yEll = -(x - xcen)*np.sin(np.radians(pa+90)) + (y - ycen)*np.cos(np.radians(pa+90))
        ell=1-(b/a)
        ellRad = np.sqrt(xEll**2 + (yEll/(1-ell))**2)
        want = ellRad <= 100
        flux=sum(im[want])
        exp=hdr['EXPTIME']
        mags.append(-2.5*np.log10(flux) + 2.5*np.log10(exp))

    mag_standard=np.median(mags)

    tab=Table()
    tab['xcen']=[xcen]
    tab['ycen']=[ycen]
    tab['pa']=[pa]
    tab['a']=[a]
    tab['b']=[b]
    tab['mag']=[mag_standard]
    ascii.write(tab,lis[:-4]+'_es.cat',overwrite=overwrite)
    print('wrote '+lis[:-4]+'_es.cat')
    
#-----------------------------------------------------------------------------
def phot_scale(dire,lis,ipref,std,prefix='P',overwrite=True):
    '''
    Scales images photometrically. Requires:
        --lis, file name of list of images (no header)
        --ipref, prefix for the image names
        --std, standard magnitude file (given by phot_standard)
    '''
    photstd = ascii.read(std)
    xcen=photstd['xcen'][0]
    ycen=photstd['ycen'][0]
    pa=photstd['pa'][0]
    a=photstd['a'][0]
    b=photstd['b'][0]
    smag=photstd['mag'][0]

    imlist = ascii.read(dire+'/'+lis, format='no_header')
    frames= [ipref + i for i in imlist['col1']]

    for frame in frames:
        hdul=fits.open(dire+'/'+frame)
        im=hdul[0].data
        hdr=hdul[0].header
        hdul.close()
        x = np.arange(1, len(im)+1)
        y = x.reshape(-1, 1)
        xEll = (x - xcen)*np.cos(np.radians(pa+90)) + (y - ycen)*np.sin(np.radians(pa+90))
        yEll = -(x - xcen)*np.sin(np.radians(pa+90)) + (y - ycen)*np.cos(np.radians(pa+90))
        ell=1-(b/a)
        ellRad = np.sqrt(xEll**2 + (yEll/(1-ell))**2)
        want = ellRad <= 100
        flux=sum(im[want])
        exp=hdr['EXPTIME']
        fmag= -2.5*np.log10(flux) + 2.5*np.log10(exp)
        diff=str(fmag-smag)

        if overwrite and os.path.exists(dire+'/'+prefix+frame):
            os.remove(dire+'/'+prefix+frame)

        iraf.imexpr(expr='a*10**(-0.4*('+diff+'))', output='temp'+frame, a=dire+'/'+frame)
        iraf.imexpr(expr='(a<(-700)) ? -999 : a', output=dire+'/'+prefix+frame, a='temp'+frame)
        if os.path.exists('temp'+frame):
            os.remove('temp'+frame)
        
#-----------------------------------------------------------------------------
def combine(dire,lis,ipref,mosaic,overwrite=True):
    '''
    Combines images. Requires:
        --lis, file name of list of images (no header)
        --ipref, prefix for the image names
        --mosaic, file name for the output mosaic
    '''
    if overwrite and os.path.exists(mosaic):
        os.remove(mosaic)

    ims=dire+'/'+ipref+'@'+dire+'/'+lis
    iraf.imcombine(input=ims, output=mosaic, masktype='none', scale='exposure', expname='EXPTIME', lthreshold='-200.0', combine='median', reject='sigclip',blank=-999.0)

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    # Requires all.lis, on.lis and off.lis in Mosaic, containing names of all images, on-band images and off-band images, respectively
    # Requires star.coo file that contains the coordinates of the reference star
    # Requires ellphot.reg file that contains the aperture for photometric scaling
    mosdir='Mosaic'

    onlis='on.lis'
    offlis='off.lis'

    star='star.coo'
    region='ellphot.reg'
    
    onmos='ESO544_027_ha.fits'
    offmos='ESO544_027_r.fits'

    print('Aligning images....')
    shift(mosdir,'all.lis','',star)

    print('Calculating the photometric standard....')
    phot_standard(mosdir,onlis,'sh',region)
    phot_standard(mosdir,offlis,'sh',region)

    print('Photometric scaling....')
    phot_scale(mosdir,onlis,'sh','on_es.cat')
    phot_scale(mosdir,offlis,'sh','off_es.cat')

    print('Combining to final mosaics....')
    combine(mosdir,onlis,'Psh',onmos)
    combine(mosdir,offlis,'Psh',offmos)
