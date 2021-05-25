import os
from shutil import copyfile
import glob
import sys
from astropy.io import fits
from astropy.io import ascii as asc
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from pyraf import iraf
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy import ndimage


# Default to using the full image flat directory
loopfiles = glob.glob('on/Trial1/Flat*')
loopnums = [int(i[-6]) for i in loopfiles]
Nloops = np.max(loopnums) + 1
block = 64 # MUST BE LARGE!  Change at your own risk.
ondir = 'on_mos/'
offdir = 'off_mos/'
fstr = 'f'
mstr = 'M'
bnstr = 'bn'
pstr = 'p'
sstr = 's'+fstr
substr = 'ms_'
skystr = 'sky_'
onfiles = np.array(glob.glob('on/tz*.fits'))
offfiles = np.array(glob.glob('off/tz*.fits'))
f_onfiles = np.array([ondir+fstr+f[f.find('tz'):] for f in onfiles]) # flattened
f_offfiles = np.array([offdir+fstr+f[f.find('tz'):] for f in offfiles])
m_onfiles = np.array([ondir+mstr+f[f.find('tz'):] for f in onfiles]) # mask
m_offfiles = np.array([offdir+mstr+f[f.find('tz'):] for f in offfiles])
bn_onfiles = np.array([ondir+bnstr+f[f.find('tz'):] for f in onfiles]) # binned
bn_offfiles = np.array([offdir+bnstr+f[f.find('tz'):] for f in offfiles])
p_onfiles = np.array([ondir+pstr+f[f.find('tz'):] for f in onfiles]) # de-planed
p_offfiles = np.array([offdir+pstr+f[f.find('tz'):] for f in offfiles])
pf_onfiles = np.array([ondir+pstr+fstr+f[f.find('tz'):] for f in onfiles]) # de-planed, flattened
pf_offfiles = np.array([offdir+pstr+fstr+f[f.find('tz'):] for f in offfiles])
s_onfiles = np.array([ondir+sstr+f[f.find('tz'):] for f in onfiles]) # sky-subtracted
s_offfiles = np.array([offdir+sstr+f[f.find('tz'):] for f in offfiles])
sbn_onfiles = np.array([ondir+sstr+bnstr+f[f.find('tz'):] for f in onfiles]) # binned and ss-ed
sbn_offfiles = np.array([offdir+sstr+bnstr+f[f.find('tz'):] for f in offfiles])
ms_onfiles = np.array([ondir+substr+f[f.find('tz'):] for f in onfiles]) # mosaic-subtracted
ms_offfiles = np.array([offdir+substr+f[f.find('tz'):] for f in offfiles])
sky_onfiles = np.array([ondir+skystr+f[f.find('tz'):] for f in onfiles]) # sky image
sky_offfiles = np.array([offdir+skystr+f[f.find('tz'):] for f in offfiles])
onfiles = np.array([ondir+f[f.find('tz'):] for f in onfiles]) # tz*
offfiles = np.array([offdir+f[f.find('tz'):] for f in offfiles])


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


def bin_image(hdulist, block=9):
    '''
    Median bins image into block x block size.
    Makes for more accurate plane fit.
    Requires:
        --Image HDUList object, so the output of fits.open()
        --Block factor (or if none given, defaults to 9px x 9px)
    Returns:
        --Binned image array, with masks == -999
        --Binned image array header, with WCS transformed to binned coordinates
    '''
    if block > 1:     
        # Shaving off excess pixels given bin size
        xedge = np.shape(hdulist[0].data)[0] % block
        yedge = np.shape(hdulist[0].data)[1] % block
        imtrim = hdulist[0].data[xedge:, yedge:]
        
        # Reshape image array into arrays of block x block
        binim = np.reshape(imtrim,
                           (np.shape(imtrim)[0]//block,
                            block,
                            np.shape(imtrim)[1]//block,
                            block)
                           )

        # Take median of all block x block boxes and ignore -999 values
        binim[binim == -999.] = np.nan
        binned = np.nanmedian(binim, axis=(1,3))
    
        binned[np.isnan(binned)] = -999.0
        
        # Transform WCS to the binned coordinate system
        bnhead = hdulist[0].header.copy()
        bnhead['CRPIX1'] = bnhead['CRPIX1']/block
        bnhead['CRPIX2'] = bnhead['CRPIX2']/block
        try:
            bnhead['CD1_1'] = bnhead['CD1_1']*block
        except:
            pass
        try:
            bnhead['CD2_2'] = bnhead['CD2_2']*block
        except:
            pass
        try:
            bnhead['CD1_2'] = bnhead['CD1_2']*block
        except:
            pass
        try:
            bnhead['CD2_1'] = bnhead['CD2_1']*block
        except:
            pass
        return binned, bnhead
    
    else:
        return hdulist[0].data, hdulist[0].header


def bin_and_interp(hdulist, block=9):
    '''
    Median bins image into block x block size, then interpolates flux using
    across masked pixels using nearest neighbors approach.
    Requires:
        --Image HDUList object, so the output of fits.open()
        --Block factor (or if none given, defaults to 9px x 9px)
    Returns:
        --Binned image resized to standard image size (no header)
    ''' 
    # Shaving off excess pixels given bin size
    xedge = np.shape(hdulist[0].data)[0] % block
    yedge = np.shape(hdulist[0].data)[1] % block
    imtrim = hdulist[0].data.copy()
    imtrim = imtrim[xedge:, yedge:]
        
    # Reshape image array into arrays of block x block
    binim = np.reshape(imtrim,
                       (np.shape(imtrim)[0]//block,
                        block,
                        np.shape(imtrim)[1]//block,
                        block)
                       )
    binim[binim == -999.] = np.nan
        
    # Have to keep bins with very few masked pixels from skewing the results
    binned = np.zeros((np.shape(imtrim)[0]//block, np.shape(imtrim)[1]//block))
    for i in range(binim.shape[0]):
        for j in range(binim.shape[2]):
            box = binim[i, : , j , :]
            msk = np.isfinite(box)
            if len(msk[msk]) <= 30:
                binned[i,j] = np.nan
            else:
                binned[i,j] = np.nanmedian(box)
    binned[np.isnan(binned)] = -999.0
        
    # Interpolate flux across masks using nearest neighbors
    bnmsk = binned!=-999
    X, Y = np.meshgrid(np.arange(binned.shape[1]), np.arange(binned.shape[0]))
    xym = np.vstack( (np.ravel(X[bnmsk]), np.ravel(Y[bnmsk])) ).T
    data = np.ravel(binned[bnmsk])
    interp = NearestNDInterpolator(xym, data)
    binned = interp(np.ravel(X), np.ravel(Y)).reshape(X.shape)
        
    # Big version of binned image
    for i in range(binim.shape[0]):
        for j in range(binim.shape[2]):
            imtrim[i*block : (i+1)*block, j*block : (j+1)*block] = binned[i, j]
                
    # Creates a small edge mask of clipped pixels if size % block != 0
    bigbin = np.zeros(hdulist[0].data.shape) - 999 
    bigbin[xedge:, yedge:] = imtrim
    
    return bigbin


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


def mk_flat(flist, mlist, indx, vmap, pflag=1):
    '''
    Creates a flat using imcombine.  Rejects masked pixels by creating
    .pl versions of the masks and adding these to the header of the
    input images.
    Requires:
     - List of images that go into making the flat (tz*.fits)
     - List of masks associated with the above images (M*.fits)
     - Index of current loop in final flat-making process (int)
     - Name of the vignetting mask (e.g., vmap_on.fits)
     - Flag indicated whether to use the raw images (tz*) or
     plane-corrected images (ptz*)

    Set pflag to 0 for first iteration (before any planes are
    measured).
    Currently this uses ALL exposures, object and sky.
    '''
    dir_nm = mlist[0][:mlist[0].find(mstr)]
    m_nm = mlist[0][mlist[0].find(mstr):]
    if os.path.exists('on/'+m_nm[:m_nm.find('.fits')]+'.pl'):
        print('Killing previous masks....')
        os.system('/bin/rm '+dir_nm+'*.pl')
    for i in range(len(flist)):
        m_nm = mlist[i][mlist[i].find(mstr):]
        iraf.imcopy(input=mlist[i],
                  output=dir_nm+m_nm[:m_nm.find('.fits')]+'.pl')
        iraf.hedit(images=flist[i],
                   fields='BPM',
                   value=m_nm[:m_nm.find('.fits')]+'.pl',
                   add='yes',
                   verify='no')
    if pflag:
        instr =dir_nm+pstr+'tz*.fits' # This only works if previous loop's gunk is deleted
        outstr = dir_nm+'Flat'+str(indx)+'.fits' # Need to move these
        # after all iterations
    else:
        instr = dir_nm+'tz*.fits'
        outstr = dir_nm+'CrudeFlat.fits'
    iraf.unlearn('imcombine')
    iraf.imcombine(input=instr,
                   output=outstr,
                   combine='median',
                   reject='avsigclip',
                   masktype='goodvalue',
                   maskvalue=0,
                   scale='median',
                   lthreshold=-50, # Should take care of the masking....
                   hthreshold='INDEF',
                   mclip='yes',
                   lsigma=3.0,
                   hsigma=3.0)
    # Normalizing the output flat
    flat = fits.open(outstr)
    vmap = fits.getdata(dir_nm + vmap)
    flat_data = flat[0].data.copy()
    flat_data[vmap==1] = np.nan
    mn = np.nanmean(flat_data[np.isfinite(flat_data)])
    flat[0].data /= mn
    
    write_im_head(flat[0].data, flat[0].header, outstr)
    print('Flat normalized.')


def flux_scale_gal(mosTrimNm, imNm, galRa, galDec, galRad):
    '''
    Uses a circular aperture around the galaxy to scale the trimmed
    mosaic in flux to the image it's being subtracted from.  Results in
    less systematic over- and under-subtraction due to this method.

    mosTrimNm: file name of the trimmed mosaic to be subtracted
    imNm : file name of image you want to derive the sky on
    galRa : galaxy right ascension in degrees
    galDec : galaxy declination in degrees
    galRad : maximum photometry radius, in pixels

    Returns a scale factor, doing a linear fit within certain flux bounds.
    If the scale factors seem nonsensical, try adjusting the flux bounds.
    '''
    mos = fits.open(mosTrimNm)
    im = fits.open(imNm)
    medsky = np.nanmedian(im[0].data[im[0].data > -400])
    
    wcs = WCS(im[0].header)
    coo = wcs.wcs_world2pix(galRa, galDec, 1)

    x = np.arange(1, im[0].data.shape[0]+1)
    y = x.reshape(-1, 1)
    rad = np.sqrt((x-coo[0])**2 + (y-coo[1])**2)
    wantRad = rad <= galRad
    cutoutIm = im[0].data[wantRad]
    cutoutMos = mos[0].data[wantRad]
    want = (cutoutIm < medsky+4000) & (cutoutMos < 4000)
    wantmsk = (cutoutIm > 200) & (cutoutMos > 200)

    fit = np.polyfit(cutoutMos[want & wantmsk], cutoutIm[want & wantmsk], 1)

    return fit[0]


def mos_sub(inlist, outlist, mosaic, star_coo, overwrite=True):
    '''
    Subtracts a mosaic from images. Requires:
     - List of image names (e.g., onfiles)
     - List of output image names (e.g. ms_*.fits)
     - file name of the mosaic to be subtracted
     - file containing the coordinates of a reference star
    '''
    mos = fits.open(mosaic)
    iraf.images()
    iraf.immatch()
    iraf.noao.digiphot()
    iraf.noao.digiphot.apphot()
    iraf.unlearn('centerpars')
    iraf.unlearn('phot')
    iraf.noao.digiphot.apphot.datapars.fwhmpsf = '5.08'
    iraf.noao.digiphot.apphot.centerpars.cbox = '70.'
    iraf.noao.digiphot.apphot.centerpars.calgorithm = 'gauss'
    for i in range(len(inlist)):
        if overwrite and os.path.exists(outlist[i]):
            os.remove(outlist[i])
        im = fits.open(inlist[i])
        dirnm = inlist[i][:inlist[i].find('ftz')]
        crnm = dirnm+'cr_s'+inlist[i][inlist[i].find('ftz'):]
        bandnm = inlist[i][:inlist[i].find('_mos')]
        if im[0].header['OBJECT'] == 'target':
            # ---------------------------------------------------
            # First, rotate and flip mosaic to appropriate position angle
            posang = 90. - im[0].header['FIELD']
            iraf.rotate(input=mosaic, output=inlist[i]+'.mos.fits', rotation=posang)
            rtmos = fits.open(inlist[i]+'.mos.fits')

            # Next, use a chosen star to shift mosaic to appropriate coordinates
            # Use the CR masked version of the image for this, to avoid centroid issues
            # File name is star.coo, contains one row: RA(deg) Dec(deg)
            iraf.digiphot.apphot.phot(image=crnm,
                                      output=inlist[i]+'.mag.1',
                                      coords=star_coo,
                                      wcsin='world',
                                      wcsout='logical',
                                      verify='No',
                                      interactive='No')
            iraf.phot(image=inlist[i]+'.mos.fits',
                      output=inlist[i]+'.mag.2',
                      coords=star_coo,
                      wcsin='world',
                      wcsout='logical',
                      verify='No',
                      interactive='No')
            phottab = asc.read(inlist[i]+'.mag.1')
            mosphottab = asc.read(inlist[i]+'.mag.2')
            censx = phottab['XCENTER']
            censy = phottab['YCENTER']
            moscensx = mosphottab['XCENTER']
            moscensy = mosphottab['YCENTER']
            shiftsx = -(moscensx-censx)
            shiftsy = -(moscensy-censy)
            print(shiftsx, shiftsy)
            shiftx = np.median(shiftsx)
            shifty = np.median(shiftsy)
            iraf.imshift(input=inlist[i]+'.mos.fits',
                         output=inlist[i]+'.mos.fits',
                         xshift=shiftx,
                         yshift=shifty)
            # ---------------------------------------------------
            # Next, trim shifted, rotated mosaic to image size
            iraf.imcopy(input=inlist[i]+'.mos.fits'+'[1:1731, 1:1731]',
                        output=inlist[i]+'.mos.fits')
            # ---------------------------------------------------
            # Now scale the mosaic flux using the target galaxy
            galRa=33.2275
            galDec=-19.3165278
            galRad=250
            scaleFac = flux_scale_gal(inlist[i]+'.mos.fits', inlist[i], galRa, galDec, galRad)
            rtmos = fits.open(inlist[i]+'.mos.fits')
            rtmos[0].data *= scaleFac
            write_im_head(rtmos[0].data, rtmos[0].header, inlist[i]+'.mos.fits')
            
            # Now subtract mosaic from image
            iraf.imarith(operand1=inlist[i],
                         op='-',
                         operand2=inlist[i]+'.mos.fits',
                         result=outlist[i])
            os.remove(inlist[i]+'.mag.1')
            os.remove(inlist[i]+'.mag.2')
            os.remove(inlist[i]+'.mos.fits')
        else:
            os.system('cp '+inlist[i]+' '+outlist[i])


def sky_sub(flist):
    '''
    Subtracts off the smoothed sky maps from the flattened images.
    Requires:
      --List of nearly unprocessed images (tz*.fits)
    '''
    for i in range(len(flist)):
        dirnm = flist[i][ : flist[i].find('tz')]
        fnm = flist[i][flist[i].find('tz') : ]
        fim = fits.open(dirnm+fstr+fnm)
        sim = fits.getdata(dirnm+skystr+fnm)
        fim[0].data -= sim
        write_im_head(fim[0].data, fim[0].header, dirnm+sstr+fnm)
    

def desky(flist, vmsk, block=block):
    '''
    Derives and subtracts the sky by eroding the original masks, binning
    the mosaic-subtracted images, interpolating flux across the eroded 
    masks, resizing the binned, interpolated image to standard size, smoothing
    it with a Gaussian kernel half the bin size in width, and then dividing this
    pattern from the nearly raw images (tz*.fits).
    Requires:
      --tz*.fits file list (to append prefixes in each step)
      --Filename of vignetting mask (e.g., vmap_on.fits)
      --Block factor.  Use a large one!  64 seems to work (of order galaxy width)
    Returns:
      --Writes de-skied images to the hard disk
      --Writes the actual sky map as well
    '''
    vmsk = fits.getdata(vmsk)
    for i in range(len(flist)):
        dirnm = flist[i][ : flist[i].find('tz')]
        fnm = flist[i][flist[i].find('tz') : ]
        ssub = fits.open(dirnm+substr+fnm) # Sky image
        msk = fits.getdata(dirnm+mstr+fnm) # NoiseChisel mask
        if ssub[0].header['OBJECT'] == 'target':
            # For erosion, unless on sky frame; then don't erode
            struct = np.array([[0,0,1,0,0],
                               [0,1,1,1,0],
                               [1,1,1,1,1],
                               [0,1,1,1,0],
                               [0,0,1,0,0]])
            # Using 10 iterations; too large of masks results in interpolation issues
            er_msk = ndimage.binary_erosion(msk, struct, 10)
        else:
            er_msk = msk.copy()
        ssub[0].data[er_msk==1] = -999
        ssub[0].data[vmsk==1] = -999
        bigbin = bin_and_interp(ssub, block)
        # Gaussian smoothing sky map
        smoothed = ndimage.gaussian_filter(bigbin, block//2)
        write_im_head(smoothed, ssub[0].header, dirnm+skystr+fnm)
        # Then normalizing and removing it from the nearly raw image
        md = np.nanmedian(smoothed[er_msk == 0])
        smoothed /= md
        
        fim = fits.open(dirnm+fnm)
        fim[0].data /= smoothed
        write_im_head(fim[0].data, fim[0].header, dirnm+pstr+fnm)
        

if __name__ == '__main__':
    if not os.path.exists('on_mos') or not os.path.exists('off_mos'):
        os.mkdir('on_mos')
        os.mkdir('off_mos')
        copyfile('on/vmap_on.fits', 'on_mos/vmap_on.fits')
        copyfile('off/vmap_off.fits', 'off_mos/vmap_off.fits')
        # Uses last iteration of flats for full-image trial
        copyfile('on/Trial1/Flat'+str(Nloops)+'.fits', 'on_mos/flat.fits')
        copyfile('off/Trial1/Flat'+str(Nloops)+'.fits', 'off_mos/flat.fits')
        copyfile('../ESO544_027_ha.fits', './on_mos.fits')
        copyfile('../ESO544_027_r.fits', './off_mos.fits')
        copyfile('../star.coo', './star.coo')
        os.system('/bin/cp on/M*.fits on_mos')
        os.system('/bin/cp off/M*.fits off_mos')
        os.system('/bin/cp on/tz*.fits on_mos')
        os.system('/bin/cp off/tz*.fits off_mos')
        os.system('/bin/cp on/cr_*.fits on_mos')
        os.system('/bin/cp off/cr_*.fits off_mos')

    # Clearing previous outputs
    for direc in ['on_mos/', 'off_mos/']:
        os.system('/bin/rm '+direc+'Flat*.fits')
        os.system('/bin/rm '+direc+pstr+'*.fits')
        os.system('/bin/rm '+direc+fstr+'tz*.fits')
        os.system('/bin/rm '+direc+substr+'*.fits')
        os.system('/bin/rm '+direc+skystr+'*.fits')
        os.system('/bin/rm '+direc+sstr+'*.fits')
        os.system('/bin/rm '+direc+'cr_*.fits')


    for n in range(Nloops):
        print('Doing loop ',n+1,' of ',Nloops)

        # Flattening images
        print('Flattening images....')
        if n==0:
            flat = 'flat.fits'
        else:
            flat = 'Flat'+str(n-1)+'.fits'
        flatten(onfiles, ondir+flat)
        flatten(offfiles, offdir+flat)

        # Deriving backgrounds through mosaic subtraction
        print('Removing astrophysical objects....')
        mos_sub(f_onfiles, ms_onfiles, 'on_mos.fits', 'star.coo')
        mos_sub(f_offfiles, ms_offfiles, 'off_mos.fits', 'star.coo')

        print('=====================\n')
        print('Now modeling and removing skies from tz* images...')
        desky(onfiles, 'on_mos/vmap_on.fits')
        desky(offfiles, 'off_mos/vmap_off.fits')

        # No re-masking... assuming the last iteration's masks are fine
        print('Making new flat....')
        mk_flat(p_onfiles, m_onfiles, n, 'vmap_on.fits')
        mk_flat(p_offfiles, m_offfiles, n, 'vmap_off.fits')

    print('Making final generation flattened images...')
    for direc in ['on_mos/', 'off_mos/']:
        os.system('/bin/rm '+direc+'ftz*.fits')
    flatten(onfiles, ondir+flat)
    flatten(offfiles, offdir+flat)

    print('Making sky subtracted images....')
    sky_sub(onfiles)
    sky_sub(offfiles)
