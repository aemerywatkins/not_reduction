import glob
import numpy as np
from pyraf import iraf
from astropy.io import fits
from astropy.io import ascii as asc
from astropy.table import Table
import os
from shutil import copyfile
from astropy.wcs import WCS

# Precursor values for convenience
sfstr = 'sf' # prefix for sky-subtracted and flat-fielded images
mstr = 'm' # prefix for masked images
Wstr = 'W' # prefix for registered images
shstr = 'sh' # prefix for shifted images
Pstr = 'P' # prefix for photometrically scaled images 
hmstr = 'hm' # prefix for handmasks
if os.path.exists('Flats/on_mos'):
    ondir = 'Flats/on_mos'
    offdir = 'Flats/off_mos'
else:
    ondir = 'Flats/on'
    offdir = 'Flats/off'
mosdir = 'Mosaic'
tlis = 'target.lis'
onbpm = 'vmap_on.fits'
offbpm = 'vmap_off.fits'
refim = 'refimage.fits'
onlis = 'on.lis'
offlis = 'off.lis'
star = 'star.coo'
region = 'ellphot.reg'
onmos = 'ESO544_027_ha.fits'
offmos = 'ESO544_027_r.fits'
Ntrial = 5
Nloops = 5


def mask(dire, lis, ipref, mpref, bpm, overwrite=True):
    '''
    Masks images with object masks + bad pixel map
    Masked pixels are set to -999
    '''
    tab=asc.read(dire+'/'+lis, format='no_header')
    for ii in tab:
        ima=ipref+ii[0]
        msk=0
        if os.path.exists(dire+'/'+mpref+ii[0]):
            msk=dire+'/'+mpref+ii[0]
        hdulist=fits.open(dire+'/'+ima)
        bpmdat = fits.getdata(dire+'/'+bpm)
        
        if msk != 0:
            handmaskdat = fits.getdata(msk)
        else:
            handmaskdat = bpmdat * 0.0
    
        # Note: use of this on the raw frames results in these becoming values of 
        # 64537.  Probably this is because the raw frames are 16 bit, so use only
        # with processed frames.
        hdulist[0].data[bpmdat != 0] = -999
        hdulist[0].data[handmaskdat != 0] = -999
        hdulist.writeto(dire+'/'+mstr+ima, overwrite=overwrite)
        hdulist.close()
        
        print('created '+mstr+ima)

def skysub_legendre(dire, Ntrial, Nloops, degree=2, overwrite=True):
    '''
    Subtracts poynomial sky models from images.
    Requires:
      -- Directory name to work in (e.g., 'on' while making flats)
      -- Final trial number, to descend in proper flats directory (see skyflats.py)
      -- Final loop number (see skyflats.py)
      -- Degree of polynomial fit used to make the parameter file pvals*.dat
    Return:
      -- Nothing, but write sky subtracted images to the disk
    '''
    pvals = asc.read(dire+'/Trial'+str(Ntrial)+'/pvals'+str(Nloops-1)+'.dat')
    
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


def register(dire, lis, ipref, ref, prefix=Wstr, overwrite=True):
    '''
    Registers images to a reference image. Requires:
        --lis, file name of list of images (no header)
        --ipref, prefix for the image names
        --ref, a reference image
    '''
    if overwrite:
        imlist = asc.read(dire+'/'+lis, format='no_header')
        for i in imlist['col1']:
            if os.path.exists(dire+'/'+prefix+ipref+i):
                os.remove(dire+'/'+prefix+ipref+i)

    ims=ipref+'@'+dire+'/'+lis
    iraf.wregister(input=dire+'/'+ims,
                   reference=ref,
                   output=dire+'/'+prefix+ims,
                   fitgeometry='rxyscale',
                   function='legendre',
                   xxterms='full',
                   yxterms='full',
                   boundary='constant',
                   constant=-999.0,
                   fluxconserve='no')


def mk_target_list(dirnm):
    '''
    Creates files 'target.lis', lists of image names of only galaxy images
    '''
    fnms = glob.glob(dirnm+'/tz*.fits')
    keep = []
    for nm in fnms:
        header = fits.getheader(nm)
        if header['OBJECT'] == 'target':
            keep.append(nm[nm.find('tz') : ])

    asc.write([keep], dirnm+'/target.lis', format='no_header', overwrite=True)


def shift(dire, lis, ipref, star, mosaic=False, prefix=shstr, overwrite=True):
    '''
    Uses phot to align images. Requires:
        --lis, file name of list of images (no header)
        --ipref, prefix for the image names
        --star, file name with the coordinates of the reference star
        --mosaic as a reference image
    '''
    ims = ipref+'@'+dire+'/'+lis
    iraf.noao.digiphot()
    iraf.noao.digiphot.apphot()
    iraf.unlearn('centerpars')
    iraf.unlearn('phot')
    iraf.noao.digiphot.apphot.datapars.fwhmpsf='4.54'
    iraf.noao.digiphot.apphot.centerpars.cbox='75.'
    iraf.noao.digiphot.apphot.centerpars.calgorithm='gauss'
    iraf.digiphot.apphot.phot(image=dire+'/'+ims,
                              output='tmp.mag',
                              coords=star,
                              wcsin='world',
                              wcsout='logical',
                              verify='No',
                              interactive='No')
    
    phottab = asc.read('tmp.mag')
    os.remove('tmp.mag')

    if mosaic == False:
        tab = asc.read(dire+'/'+lis, format='no_header')
        mosaic = dire + '/' + ipref + tab['col1'][0]

    iraf.phot(image=mosaic,
              output='tmp2.mag',
              coords=star,
              wcsin='world',
              wcsout='logical',
              verify='No',
              interactive='No')
    reftab = asc.read('tmp2.mag')
    os.remove('tmp2.mag')
    refx = reftab['XCENTER'][0]
    refy = reftab['YCENTER'][0]
    
    imlist = asc.read(dire+'/'+lis, format='no_header')
    ims2 = [ipref + i for i in imlist['col1']]

    for i in range(len(phottab)):
        cenx = phottab['XCENTER'][i]
        ceny = phottab['YCENTER'][i]
        shiftx = -(cenx-refx)
        shifty = -(ceny-refy)
        if overwrite and os.path.exists(dire+'/'+prefix+ims2[i]):
            os.remove(dire+'/'+prefix+ims2[i])
        iraf.imshift(input=dire+'/'+ims2[i],
                     output=dire+'/'+prefix+ims2[i],
                     xshift=shiftx,
                     yshift=shifty,
                     interp_type='linear',
                     boundary_type='nearest',
                     constant=0.0)
        print('Created '+prefix+ims2[i])


def phot_standard(dire, lis, ipref, region, overwrite=True):
    '''
    Calculates magnitude in an elliptical aperture. Requires:
        --lis, file name of list of images (no header)
        --ipref, prefix for the image names
        --region, ds9 region file for the aperture
    '''
    imlist = asc.read(dire+'/'+lis, format='no_header')
    frames = [ipref + i for i in imlist['col1']]

    tab = asc.read(region)
    xcen = tab['col3'][0]
    ycen = tab['col4'][0]
    a = np.max([tab['col5'][0],tab['col6'][0]])
    b = np.min([tab['col5'][0],tab['col6'][0]])
    pa = tab['col7'][0]

    mags = []
    for frame in frames:
        hdul = fits.open(dire+'/'+frame)
        im = hdul[0].data
        hdr = hdul[0].header
        hdul.close()
        x = np.arange(1, len(im)+1)
        y = x.reshape(-1, 1)
        xEll = (x - xcen)*np.cos(np.radians(pa+90)) + (y - ycen)*np.sin(np.radians(pa+90))
        yEll = -(x - xcen)*np.sin(np.radians(pa+90)) + (y - ycen)*np.cos(np.radians(pa+90))
        ell = 1-(b/a)
        ellRad = np.sqrt(xEll**2 + (yEll/(1-ell))**2)
        want = ellRad <= 100
        flux = sum(im[want])
        exp = hdr['EXPTIME']
        mags.append(-2.5*np.log10(flux) + 2.5*np.log10(exp))

    mag_standard = np.median(mags)

    tab = Table()
    tab['xcen'] = [xcen]
    tab['ycen'] = [ycen]
    tab['pa'] = [pa]
    tab['a'] = [a]
    tab['b'] = [b]
    tab['mag'] = [mag_standard]
    asc.write(tab,lis[:-4]+'_es.cat',overwrite=overwrite)
    
    print('wrote '+lis[:-4]+'_es.cat')


def phot_scale(dire, lis, ipref, std, prefix=Pstr, overwrite=True):
    '''
    Scales images photometrically. Requires:
        --lis, file name of list of images (no header)
        --ipref, prefix for the image names
        --std, standard magnitude file (given by phot_standard)
    '''
    photstd = asc.read(std)
    xcen = photstd['xcen'][0]
    ycen = photstd['ycen'][0]
    pa = photstd['pa'][0]
    a = photstd['a'][0]
    b = photstd['b'][0]
    smag = photstd['mag'][0]

    imlist = asc.read(dire+'/'+lis, format='no_header')
    frames = [ipref + i for i in imlist['col1']]

    for frame in frames:
        hdul = fits.open(dire+'/'+frame)
        im = hdul[0].data
        hdr = hdul[0].header
        hdul.close()
        x = np.arange(1, len(im)+1)
        y = x.reshape(-1, 1)
        xEll = (x - xcen)*np.cos(np.radians(pa+90)) + (y - ycen)*np.sin(np.radians(pa+90))
        yEll = -(x - xcen)*np.sin(np.radians(pa+90)) + (y - ycen)*np.cos(np.radians(pa+90))
        ell = 1-(b/a)
        ellRad = np.sqrt(xEll**2 + (yEll/(1-ell))**2)
        want = ellRad <= 100
        flux = sum(im[want])
        exp = hdr['EXPTIME']
        fmag = -2.5*np.log10(flux) + 2.5*np.log10(exp)
        diff = str(fmag-smag)

        if overwrite and os.path.exists(dire+'/'+prefix+frame):
            os.remove(dire+'/'+prefix+frame)

        iraf.imexpr(expr='a*10**(-0.4*('+diff+'))',
                    output='temp'+frame,
                    a=dire+'/'+frame)
        iraf.imexpr(expr='(a<(-700)) ? -999 : a',
                    output=dire+'/'+prefix+frame,
                    a='temp'+frame)
        if os.path.exists('temp'+frame):
            os.remove('temp'+frame)


def combine(dire, lis, ipref, mosaic, overwrite=True):
    '''
    Combines images. Requires:
        --lis, file name of list of images (no header)
        --ipref, prefix for the image names
        --mosaic, file name for the output mosaic
    '''
    if overwrite and os.path.exists(mosaic):
        os.remove(mosaic)

    ims=dire+'/'+ipref+'@'+dire+'/'+lis
    iraf.imcombine(input=ims,
                   output=mosaic,
                   masktype='none',
                   scale='none', # ?? Is this appropriate?  None, or median?
                   lthreshold='-500.0', # Scatter in BG is actually ~200 ADU
                   combine='median',
                   reject='sigclip',
                   blank=-999.0)


def adjust_lists(lisnm, pref):
    lis = asc.read(mosdir+'/'+lisnm, format='no_header')
    new_lis = [i for i in lis['col1']]
    for i in range(len(new_lis)):
        new_lis[i] = pref+new_lis[i]
    asc.write([new_lis], mosdir+'/'+lisnm, format='no_header', overwrite=True)


if __name__ == '__main__':
    # Assumes vmap_on.fits, vmap_off.fits in on/ and off/ flats directories already
    # (Should be from previous reduction steps)
    # Also requires star.coo and ellphot.reg files in working directory

    #First clear previous outputs
    if os.path.exists('Mosaic'):
        os.system('/bin/rm -r Mosaic')
        os.system('/bin/rm '+ondir+'/'+Wstr+'*.fits')
        os.system('/bin/rm '+offdir+'/'+Wstr+'*.fits')
        os.system('/bin/rm '+ondir+'/'+mstr+sfstr+'tz*.fits')
        os.system('/bin/rm '+offdir+'/'+mstr+sfstr+'tz*.fits')
    if os.path.exists(onmos) or os.path.exists(offmos):
        os.system('/bin/rm '+onmos)
        os.system('/bin/rm '+offmos)
    
    # Creates list of object exposures in each directory
    mk_target_list(ondir)
    mk_target_list(offdir)
    
    print('Applying masks....')
    mask(ondir, tlis, sfstr, hmstr, onbpm)
    mask(offdir, tlis, sfstr, hmstr, offbpm)

    if os.path.isfile(refim):
        print('Reference image already exists, skipping creation.')
        
    else:
        print('Creating reference image....')
        iraf.artdata()
        iraf.artdata.mkpattern(refim,
                               ncols = 2000,
                               nlines = 2000,
                               v1 = 0)
        iraf.imcoords()
        iraf.imcoords.ccsetwcs(refim,
                               "",
                               xref = 1000,
                               yref = 1000,
                               xmag = -0.2138004,
                               ymag = 0.2138004,
                               lngref = '02:12:54.6',
                               latref = '-19:19:06.0',
                               lngunits = 'hours', # For safety
                               latunits = 'hours') # For safety
        
    print('Registering images to reference....')
    register(ondir, tlis, mstr+sfstr, refim)
    register(offdir, tlis, mstr+sfstr, refim)

    print('Moving registered images to new '+mosdir+' directory')
    if not os.path.isdir(mosdir):
        os.system('mkdir '+mosdir)
    os.system('cp '+ondir+'/'+Wstr+mstr+sfstr+'*fits '+mosdir)
    os.system('cp '+offdir+'/'+Wstr+mstr+sfstr+'*fits '+mosdir)

    copyfile(ondir + '/target.lis', mosdir + '/on.lis')
    copyfile(offdir + '/target.lis', mosdir + '/off.lis')
    # Makes the lists local to work with IRAF
    adjust_lists(onlis, Wstr+mstr+sfstr)
    adjust_lists(offlis, Wstr+mstr+sfstr)
    # Combines both bands into one list
    os.system('cat '+mosdir+'/on.lis '+mosdir+'/off.lis > '+mosdir+'/all.lis')

    # Now full mosaic creation
    # Functions from here need to just add the proper prefix
    print('Aligning images....')
    shift(mosdir, 'all.lis', '', star)

    print('Calculating the photometric standard....')
    phot_standard(mosdir, onlis, shstr, region)
    phot_standard(mosdir, offlis, shstr, region)

    print('Photometric scaling....')
    phot_scale(mosdir, onlis, shstr, 'on_es.cat')
    phot_scale(mosdir, offlis, shstr, 'off_es.cat')

    print('Combining to final mosaics....')
    combine(mosdir, onlis, Pstr+shstr, onmos)
    combine(mosdir, offlis, Pstr+shstr, offmos)

    print('Wrote '+onmos+' & '+offmos)
