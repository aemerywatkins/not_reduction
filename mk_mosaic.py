import glob
import numpy as np
from pyraf import iraf
from astropy.io import fits
from astropy.io import ascii as asc
from astropy.modeling.models import custom_model, Legendre2D
from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter
from astropy.table import Table
import os
from shutil import copyfile


# Precursor values for convenience
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
# Use only full-image Trial
Ntrial = 1
# Auto-generates
loopfiles = glob.glob('Flats/on/Trial1/Flat*')
loopnums = [int(i[-6]) for i in loopfiles]
Nloops = np.max(loopnums)
fstr = 'f'
onfiles = np.array(glob.glob('Flats/on/tz*.fits'))
offfiles = np.array(glob.glob('Flats/off/tz*.fits'))


def write_im_head(imarr, head, fname):
    '''
    Write out an image array, with the specified header, to the hard disk.
    Requires:
      - Image array to be written as FITS image
      - Header object (hdu[i].header)
      - File name to write to the disk

    Automatically overwrites files of the same name, so use caution!
    '''
    hdu = fits.PrimaryHDU(imarr, header=head)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(fname, overwrite=True)
    

def flatten(flist, flat):
    '''
    Flattens all images in a list of images.
    Requires:
     - List of image names to be flattened
     - Name of the flat to be used to flatten the images (string)
    '''
    for f in flist:
        if os.path.exists(f[:f.find('tz')]+fstr+f[f.find('tz'):]):
            os.remove(f[:f.find('tz')]+fstr+f[f.find('tz'):])
        iraf.imarith(operand1=f,
                     op='/',
                     operand2=flat,
                     result=f[:f.find('tz')]+fstr+f[f.find('tz'):])
        

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


def skysub_legendre(dire, Ntrial, Nloops, degree=2, overwrite=True):
    '''
    Subtracts poynomial sky models from images.
    Requires:
      -- Directory name to work in (e.g., 'on' while making flats)
      -- Final trial number, to descend in proper flats directory (see skyflats.py)
      -- Desired loop number (see skyflats.py)
      -- Degree of polynomial fit used to make the parameter file pvals*.dat
    Return:
      -- Nothing, but write sky subtracted images to the disk
    '''
    pvals = asc.read(dire+'/Trial'+str(Ntrial)+'/pvals'+str(Nloops)+'.dat')
    
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


def mask(dire, lis, ipref, mpref, bpm, overwrite=True):
    '''
    Masks images with object masks + bad pixel map.
    '''
    tab=asc.read(dire+'/'+lis, format='no_header')
    for ii in tab:
        ima=ipref+ii[0]
        msk=0
        if os.path.exists(dire+'/'+mpref+ii[0]):
            msk=dire+'/'+mpref+ii[0]
        hdul=fits.open(dire+'/'+ima)
        hdul=mask_image(hdul,dire+'/'+bpm, objmask=msk)
        hdul.writeto(dire+'/m'+ima, overwrite=overwrite)
        hdul.close()
        
        print('created m'+ima)


def register(dire, lis, ipref, ref, prefix='W', overwrite=True):
    '''
    Registers images to a reference image. Requires:
        --lis, file name of list of images (no header)
        --ipref, prefix for the image names
        --ref, a reference image

    OUTDATED: use only if headers not updated using Astrometry.net!
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


def register_swarp(dire, lis, ipref, ref, prefix='W', overwrite=True):
    '''
    Registers images to a reference image, using SWarp
    First runs SCamp to output necessary files
    '''
    imlist = asc.read(dire+'/'+lis, format='no_header')
    if overwrite:
        for i in imlist['col1']:
            if os.path.exists(dire+'/'+prefix+ipref+i):
                os.remove(dire+'/'+prefix+ipref+i)

    ims = [dire+'/'+ipref+i for i in imlist['col1']]
    for i in range(len(ims)):
        if not os.path.exists(ims[i][:-5]+'.head'):
            os.system('/usr/bin/source-extractor '+ims[i])
            os.system('/usr/local/bin/scamp output.cat') # Ensure scamp.conf points to GAIA-EDR3
            os.system('/bin/mv output.head '+ims[i][:-5]+'.head')
        os.system('/usr/bin/SWarp '+ref+' '+ims[i]) # Using apt install swarp version (2014)
        wim = fits.open('coadd.fits')
        refim = fits.getdata(ref)
        # Expanding and fixing masks
        wim[0].data[wim[0].data < -400] = -999
        wim[0].data[wim[0].data == 0] = -999
        write_im_head(wim[0].data, wim[0].header, dire+'/'+prefix+ipref+imlist['col1'][i])
    os.system('/bin/rm coadd.fits coadd.weight.fits')
    

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


def shift(dire, lis, ipref, star, mosaic=False, prefix='sh', overwrite=True):
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
    iraf.noao.digiphot.apphot.centerpars.cbox='50.'
    iraf.noao.digiphot.apphot.centerpars.calgorithm='gauss'
    iraf.noao.digiphot.apphot.centerpars.maxshift=100
    iraf.noao.digiphot.apphot.photpars.apertures=10
    iraf.noao.digiphot.apphot.fitskypars.annulus=20
    iraf.noao.digiphot.apphot.fitskypars.dannulus=10
    iraf.digiphot.apphot.phot(image=dire+'/'+ims,
                              output='tmp.mag',
                              coords=star,
                              wcsin='world',
                              wcsout='logical',
                              verify='No',
                              interactive='No')
    
    phottab = asc.read('tmp.mag')
    #os.remove('tmp.mag')

    # Picks the first image in the list to register the others
    if mosaic == False:
        tab = asc.read(dire+'/'+lis, format='no_header')
        mosaic = dire + '/' + ipref + tab['col1'][0]

    iraf.phot(image=mosaic,
              output='ref.mag',
              coords=star,
              wcsin='world',
              wcsout='logical',
              verify='No',
              interactive='No')
    reftab = asc.read('ref.mag')
    #os.remove('tmp2.mag')
    
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


def adjust_shifts(dire, lis, ipref, prefix='sh2'):
    iraf.noao.digiphot()
    iraf.noao.digiphot.apphot()
    iraf.unlearn('centerpars')
    iraf.unlearn('phot')
    iraf.noao.digiphot.apphot.datapars.fwhmpsf='4.54'
    iraf.noao.digiphot.apphot.centerpars.cbox='70.'
    iraf.noao.digiphot.apphot.centerpars.calgorithm='gauss'
    
    reftab = asc.read('ref.mag')
    


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


def phot_scale(dire, lis, ipref, std, prefix='P', overwrite=True):
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
                   scale='exposure', # ?? Is this appropriate?  None, or median?
                   expname='EXPTIME',
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


def fix_mos_header(mosnm):
    mosim = fits.open(mosnm)
    want = mosim[0].data <= -400
    write_im_head(np.array(want, dtype=float), mosim[0].header, 'flag.fits')
    iraf.chpixtype('flag.fits', 'flag.fits', newpixtype='int')
    os.system('/usr/bin/source-extractor '+mosnm+' -FLAG_IMAGE flag.fits')
    os.system('/usr/local/bin/scamp output.cat')
    os.system('/bin/mv output.head '+mosnm[:-5]+'.head')
    os.system('/usr/bin/SWarp refimage.fits '+mosnm)
    # Ovewriting old mosaic with the new image header version
    os.system('/bin/mv coadd.fits '+mosnm)
    os.system('/bin/rm flag.fits coadd.weight.fits output.cat *.png')


if __name__ == '__main__':
    # Assumes vmap_on.fits, vmap_off.fits in on/ and off/ flats directories already
    # (Should be from previous reduction steps)
    # Also requires star.coo and ellphot.reg files in working directory

    #First clear previous outputs
    if os.path.exists('Mosaic'):
        os.system('/bin/rm -r Mosaic')
        os.system('/bin/rm '+ondir+'/W*.fits')
        os.system('/bin/rm '+offdir+'/W*.fits')
        os.system('/bin/rm '+ondir+'/msftz*.fits')
        os.system('/bin/rm '+offdir+'/msftz*.fits')
    if os.path.exists(onmos) or os.path.exists(offmos):
        os.system('/bin/rm '+onmos)
        os.system('/bin/rm '+offmos)

    # Do the polynomial subtraction on the first run
    if not os.path.exists('Flats/on_mos'):
        print('Flattening images...')
        onflat = 'Flats/on/Trial1/Flat'+str(Nloops)+'.fits'
        offflat = 'Flats/off/Trial1/Flat'+str(Nloops)+'.fits'
        flatten(onfiles, onflat)
        flatten(offfiles, offflat)

        print('Subtracting skies....')
        skysub_legendre(ondir, Ntrial, Nloops)
        skysub_legendre(offdir, Ntrial, Nloops)

        # Making cosmic ray masks for next step
        print('Doing cosmic ray rejection...')
        os.system('/home/aew/anaconda3/bin/python3 clean_cr.py sf 0')

    else:
        print('Using mosaic-subtracted sky models....')
        
    # Creates list of object exposures in each directory
    print(ondir)
    mk_target_list(ondir)
    mk_target_list(offdir)
    
    print('Applying masks....')
    mask(ondir, tlis, 'sf', 'hm', onbpm)
    mask(offdir, tlis, 'sf', 'hm', offbpm)

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
                               latref = '-19:18:59.5',
                               lngunits = 'hours', # For safety
                               latunits = 'hours') # For safety

    print('Cleaning cosmic ray hits...')
    os.system('/home/aew/anaconda3/bin/python3 clean_cr.py msf 1')
        
    print('Registering images to reference....')
    register(ondir, tlis, 'msf', refim)
    register(offdir, tlis, 'msf', refim)
    #register_swarp(ondir, tlis, 'msf', refim)
    #register_swarp(offdir, tlis, 'msf', refim)

    print('Moving registered images to new '+mosdir+' directory')
    if not os.path.isdir(mosdir):
        os.system('mkdir '+mosdir)
    os.system('/bin/mv '+ondir+'/'+'Wmsf*fits '+mosdir)
    os.system('/bin/mv '+offdir+'/'+'Wmsf*fits '+mosdir)

    copyfile(ondir + '/target.lis', mosdir + '/on.lis')
    copyfile(offdir + '/target.lis', mosdir + '/off.lis')
    # Makes the lists local to work with IRAF
    adjust_lists(onlis, 'Wmsf')
    adjust_lists(offlis, 'Wmsf')
    # Combines both bands into one list
    os.system('cat '+mosdir+'/on.lis '+mosdir+'/off.lis > '+mosdir+'/all.lis')

    # Now full mosaic creation
    # Functions from here need to just add the proper prefix
    print('Aligning images....')
    shift(mosdir, 'all.lis', '', star)

    print('Calculating the photometric standard....')
    phot_standard(mosdir, onlis, 'sh', region)
    phot_standard(mosdir, offlis, 'sh', region)

    print('Photometric scaling....')
    phot_scale(mosdir, onlis, 'sh', 'on_es.cat')
    phot_scale(mosdir, offlis, 'sh', 'off_es.cat')

    print('Combining to final mosaics....')
    combine(mosdir, onlis, 'Psh', onmos)
    combine(mosdir, offlis, 'Psh', offmos)

    print('Wrote '+onmos+' & '+offmos)


