# Changes:
# Added sky modelling with Legendre2D (fit_sky_L2D, desky)
# Optimized NoiseChisel commands
# Changed mk_bnf_im to also use a bad pixel map

import os
from shutil import copyfile
import glob
import sys
from astropy.io import fits
from astropy.io import ascii as asc
from astropy.modeling.models import custom_model, Legendre2D
from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter
from pyraf import iraf
import numpy as np


# Precursor stuff for later convenience
# Making these global so importing functions works in other codes
Nloops = 2 #Iterations
NTrials = 1 #Half splits
block = 16
pfile = 'pvals.dat'
fstr = 'f'
mstr = 'M'
bnstr = 'bn'
pstr = 'p'
pfstr = 'pf'
sstr = 's'
onfiles = np.array(glob.glob('on/tz*.fits'))
offfiles = np.array(glob.glob('off/tz*.fits'))
f_onfiles = np.array([f[:f.find('tz')]+fstr+f[f.find('tz'):] for f in onfiles])
f_offfiles = np.array([f[:f.find('tz')]+fstr+f[f.find('tz'):] for f in offfiles])
m_onfiles = np.array([f[:f.find('tz')]+mstr+f[f.find('tz'):] for f in onfiles])
m_offfiles = np.array([f[:f.find('tz')]+mstr+f[f.find('tz'):] for f in offfiles])
bn_onfiles = np.array([f[:f.find('tz')]+bnstr+f[f.find('tz'):] for f in onfiles])
bn_offfiles = np.array([f[:f.find('tz')]+bnstr+f[f.find('tz'):] for f in offfiles])
p_onfiles = np.array([f[:f.find('tz')]+pstr+f[f.find('tz'):] for f in onfiles])
p_offfiles = np.array([f[:f.find('tz')]+pstr+f[f.find('tz'):] for f in offfiles])
pf_onfiles = np.array([f[:f.find('tz')]+pfstr+f[f.find('tz'):] for f in onfiles])
pf_offfiles = np.array([f[:f.find('tz')]+pfstr+f[f.find('tz'):] for f in offfiles])
#bnpf_onfiles = np.array([f[:f.find('tz')]+bnstr+pfstr+f[f.find('tz'):] for f in onfiles])
#bnpf_offfiles = np.array([f[:f.find('tz')]+bnstr+pfstr+f[f.find('tz'):] for f in offfiles])
s_onfiles = np.array([f[:f.find('tz')]+sstr+f[f.find('tz'):] for f in onfiles])
s_offfiles = np.array([f[:f.find('tz')]+sstr+f[f.find('tz'):] for f in offfiles])
sbn_onfiles = np.array([f[:f.find('tz')]+sstr+bnstr+f[f.find('tz'):] for f in onfiles])
sbn_offfiles = np.array([f[:f.find('tz')]+sstr+bnstr+f[f.find('tz'):] for f in offfiles])


@custom_model
def plane(x, y, dx=0., dy=0., const=0.):
    '''
    DEPRECATED
    =============
    For LevMarLSQ Fitter
    '''
    return (dx*x + dy*y + const)


def mk_plane(x, y, dx, dy, const):
    '''
    DEPRECATED
    =============
    For creating plane image from table
    '''
    return dx*x + dy*y + const


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
    

def mk_mask(f, mstr, commands='', diagnostic=False):
    '''
    Creates a mask file by running GNUAstro package NoiseChisel on the
    image provided.
    Requires:
      - Filename of image to be masked (requires .fits extension)
      - String of additional astnoisechisel commands (optional)

    Requires that NoiseChisel is installed.
    See installation instructions on the GNU webpage:
      www.gnu.org/software/gnuastro 

    NOTE: assumes you are using flattened images (either ftz*.fits or 
    pftz*.fits)
    '''
    if f.find(pfstr+'tz') == -1:
        pre = fstr
    else:
        pre = pfstr
    os.system('astnoisechisel '+f+' -h0 --rawoutput '+commands)
    ncIm = fits.open(f[f.find(pre+'tz'):f.find('.fits')]+'_detected.fits')
    detections = ncIm[1].data
    fnm = f[:f.find(pre+'tz')]+mstr+f[f.find('tz'):]
    write_im_head(detections, ncIm[0].header, fnm)
    if diagnostic:
        dfnm = f[:f.find(pre+'tz')]+mstr+diagnostic+f[f.find('tz'):]
        write_im_head(detections, ncIm[0].header, dfnm)
    os.remove(f[f.find(pre+'tz'):f.find('.fits')]+'_detected.fits')
    
    print('Removed NC output cube.')
    print('Wrote binary mask file to disk as '+fnm)


def mk_bn_im(f, mf, fnm, block, bpmf='vmap.fits'):
    '''
    Creates a masked, binned image using previously generated masks.
    Requires:
     - Image filename to be binned and masked (ftz*.fits)
     - Filename of associated mask image (Mftz*.fits)
    Write masked, binned images to the hard disk.
    '''
    hdulist = fits.open(f)
    msk = fits.getdata(mf)
    bpm = fits.getdata(f[:f.find('/')]+'/'+bpmf)
    # Masking pixels in image
    hdulist[0].data[msk == 1] = -999
    hdulist[0].data[bpm == 1] = -999
    # Binning image
    bn_data, bn_header = bin_image(hdulist, block)
    # Writing to hard disk
    write_im_head(bn_data, bn_header, fnm)
    
    print('Wrote binned image as ',fnm)


def mk_init_masks():
    '''
    Run this first to produce preliminary masks.  Actual flat
    construction will proceed by making CrudeFlat, a combination of
    all masked object+sky exposures.

    Fails if flattened images or mask images already exist in the /on
    or /off directories.
    '''
    if os.path.exists(m_onfiles[-1]) | os.path.exists(m_offfiles[-1]): 
        print('Masks already made.  Skipping....')
    else:
        print('Flattening all frames...')
        flatten(onfiles, 'onFlat.fits')
        flatten(offfiles, 'rFlat.fits')
        
        print('Masking images....')
        for f in f_onfiles:
            mk_mask(f, mstr, commands='--outliersigma=30 --detgrowquant=0.75 --tilesize=60,60')
        for f in f_offfiles:
            mk_mask(f, mstr, commands='--outliersigma=30 --detgrowquant=0.75 --tilesize=60,60')

        print('Now deleting preliminary flattened images....')
        os.system('/bin/rm on/ftz*.fits')
        os.system('/bin/rm off/ftz*.fits')


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
    

def mk_flat_alt(flist, mlist, indx, vmap):
    '''
    CURRENTLY NOT IN USE
    ===============
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
        ima = fits.open(flist[i])
        msk = fits.getdata(mlist[i])
#        bpm = fits.getdata(dir_nm+vmap)
#        ima[0].data[bpm == 1] = -999.
        ima[0].data[msk == 1] = -999.
        write_im_head(ima[0].data, ima[0].header, dir_nm+'m_'+flist[i][flist[i].find(pstr):])
        ima.close()

    instr =dir_nm+'m_'+pstr+'tz*.fits' # This only works if previous loop's gunk is deleted
    outstr = dir_nm+'Flat'+str(indx)+'.fits' # Need to move these
    # after all iterations
    iraf.unlearn('imcombine')
    iraf.imcombine(input=instr,
                   output=outstr,
                   combine='median',
                   reject='avsigclip',
                   masktype='none',
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
    

def fit_sky(bnlist, block, pfile):
    '''
    Does plane fits to each binned, masked image and records the
    parameter values to a table.
    Requires:
     - List of binned images (e.g. bn_onfiles)
     - Block factor used to bin the images
     - Name of planes table to be written to hard drive
    
    Writes plane fit parameters to a table called pvals.dat in the
    appropriate directory.
    '''
    dir_nm = bnlist[0][:bnlist[0].find(bnstr)]
    nrej, nsig = 3, 3
    m_init = plane()
    fit = LevMarLSQFitter()
    
    dx = np.zeros(len(bnlist))
    dy = np.zeros(len(bnlist))
    const = np.zeros(len(bnlist))
    for i in range(len(bnlist)):            
        bnim = fits.open(bnlist[i])
        dim1 = np.arange(block//2+1,bnim[0].data.shape[1]*block,block)
        dim2 = np.arange(block//2+1,bnim[0].data.shape[0]*block,block)
        Xb, Yb = np.meshgrid(dim1, dim2)
        _sky = np.nanmedian(bnim[0].data[bnim[0].data!=-999])
        _dsky = np.nanstd(bnim[0].data[bnim[0].data!=-999])
        #Accepting only pixels within 1 standard deviation of background
        good = (bnim[0].data > _sky-_dsky) & (bnim[0].data < _sky+_dsky)
        Xbf, Ybf, binnedf = Xb[good], Yb[good], bnim[0].data[good]
        good = np.ones_like(binnedf, dtype=bool)
        
        # Rejecting outliers; 3 rounds 3 sigma rejection
        for it in range(nrej):
            m = fit(m_init, Xbf, Ybf, binnedf, weights=good)
            scatter = np.nanstd(m(Xbf[good], Ybf[good]) - binnedf[good])
            diff = m(Xbf, Ybf) - binnedf
            good = good & (np.abs(diff) < (nsig*_dsky))
            m_init = m
        # Doing final fit    
        m = fit(m_init, Xbf, Ybf, binnedf, weights=good)

        # Recording values for later use
        dx[i] = m.dx.value
        dy[i] = m.dy.value
        const[i] = m.const.value

    fnames = [i.strip(dir_nm+bnstr) for i in bnlist]
    asc.write([fnames, dx, dy, const],
              dir_nm+pfile,
              names=['fname', 'dx', 'dy', 'const'],
              overwrite=True)
    print('Wrote plane fits params to '+dir_nm+pfile)

    
def fit_sky_L2D(bnlist, block, pfile, degree):
    '''
    Does Legendre2D fits to each binned, masked image and records the
    parameter values to a table.
    Requires:
     - List of binned images (e.g. bn_onfiles)
     - Block factor used to bin the images
     - Name of planes table to be written to hard drive
    
    Writes plane fit parameters to a table called pvals.dat in the
    appropriate directory.
    '''
    dir_nm = bnlist[0][:bnlist[0].find(bnstr)]
    nrej, nsig = 3, 3
    m_init = Legendre2D(degree, degree)
    fit = LevMarLSQFitter()
    #fit = LinearLSQFitter()
    
    c0_0 = np.zeros(len(bnlist))
    c1_0 = np.zeros(len(bnlist))
    c2_0 = np.zeros(len(bnlist))
    c0_1 = np.zeros(len(bnlist))
    c1_1 = np.zeros(len(bnlist))
    c2_1 = np.zeros(len(bnlist))
    c0_2 = np.zeros(len(bnlist))
    c1_2 = np.zeros(len(bnlist))
    c2_2 = np.zeros(len(bnlist))
    for i in range(len(bnlist)):            
        bnim = fits.open(bnlist[i])
        dim1 = np.arange(block//2+1, bnim[0].data.shape[1]*block, block)
        dim2 = np.arange(block//2+1, bnim[0].data.shape[0]*block, block)
        Xb, Yb = np.meshgrid(dim1, dim2)
        _sky = np.nanmedian(bnim[0].data[bnim[0].data!=-999])
        _dsky = np.nanstd(bnim[0].data[bnim[0].data!=-999])
        #Accepting only pixels within 3 standard deviations of background
        good = (bnim[0].data > _sky-3*_dsky) & (bnim[0].data < _sky+3*_dsky)
        
        # Rejecting outliers; 3 rounds 3 sigma rejection
        for it in range(nrej):
            m = fit(m_init, Xb, Yb, bnim[0].data, weights=good)
            #m = fit(m_init, Xb[good], Yb[good], bnim[0].data[good])
            scatter = np.nanstd(m(Xb[good], Yb[good]) - bnim[0].data[good])
            diff = bnim[0].data - m(Xb, Yb)
            good = good & (np.abs(diff) < (nsig*_dsky))
            m_init = m
        # Doing final fit    
        m = fit(m_init, Xb, Yb, bnim[0].data, weights=good)
        #m = fit(m_init, Xb[good], Yb[good], bnim[0].data[good])

        # Recording values for later use
        c0_0[i] = m.c0_0.value
        c1_0[i] = m.c1_0.value
        c2_0[i] = m.c2_0.value
        c0_1[i] = m.c0_1.value
        c1_1[i] = m.c1_1.value
        c2_1[i] = m.c2_1.value
        c0_2[i] = m.c0_2.value
        c1_2[i] = m.c1_2.value
        c2_2[i] = m.c2_2.value

    fnames = [i.strip(dir_nm+bnstr) for i in bnlist]
    asc.write([fnames, c0_0, c1_0, c2_0, c0_1, c1_1, c2_1, c0_2, c1_2, c2_2],
              dir_nm+pfile,
              names=['fname', 'c0_0', 'c1_0', 'c2_0', 'c0_1', 'c1_1', 'c2_1', 'c0_2', 'c1_2', 'c2_2'],
              overwrite=True)
    print('Wrote plane fits params to '+dir_nm+pfile)


def calc_reduced_chi_square(fit, x, y, yerr, N, n_free):
    '''
    https://learn.astropy.org/rst-tutorials/Models-Quick-Fit.html
    Checking output of sky fits quantitatively
    Requires:
      - fit (array) values for the fit
      - x,y,yerr (arrays) data
      - N total number of points
      - n_free number of parameters we are fitting
    '''
    return 1.0/(N-n_free)*sum(((fit - y)/yerr)**2)
    

def deplane(flist, plist, pfile, diagnostic=False):
    '''
    DEPRECATED
    -----------------------------------
    Removes plane fits from unflattened images
    Requires:
     - List of image names (e.g., onfiles)
     - List of output image names (e.g. ptz*.fits)
     - Name of plane fits data table (output from fit_sky())

    On first run, give it tz*.fits.  Every other run, give it ptz*.fits.
    '''
    if flist[0].find(pfstr+'tz') != -1:
        dir_nm = flist[0][:flist[0].find(pfstr+'tz')]
    elif (flist[0].find(pstr+'tz') != -1):
        dir_nm = flist[0][:flist[0].find(pstr+'tz')]
    elif (flist[0].find(pfstr+'tz') == -1) & (flist[0].find(fstr+'tz') != -1):
        dir_nm = flist[0][:flist[0].find(fstr+'tz')]
    else:
        dir_nm = flist[0][:flist[0].find('tz')]
    pvals = asc.read(dir_nm+pfile)
    for i in range(len(flist)):
        im = fits.open(flist[i])
        dim1 = np.arange(1, im[0].data.shape[1]+1)
        dim2 = np.arange(1, im[0].data.shape[0]+1)
        X, Y = np.meshgrid(dim1, dim2)
        skyplane = mk_plane(X, Y,
                         pvals['dx'][i],
                         pvals['dy'][i],
                         pvals['const'][i])
        skyplane /= np.mean(skyplane) # Normalize the plane
        im[0].data /= skyplane
        write_im_head(im[0].data, im[0].header, plist[i])
        if diagnostic:
            write_im_head(skyplane, im[0].header, plist[i]+'.sky')

            
def desky(flist, plist, pfile, degree, vmap, indx=0, diagnostic=False):
    '''
    Removes sky fits from unflattened images
    Requires:
     - List of image names (e.g., onfiles)
     - List of output image names (e.g. ptz*.fits)
     - Name of plane fits data table (output from fit_sky())

    On first run, give it tz*.fits.  Every other run, give it ptz*.fits.
    '''
    if flist[0].find(pfstr+'tz') != -1:
        dir_nm = flist[0][:flist[0].find(pfstr+'tz')]
    elif (flist[0].find(pstr+'tz') != -1):
        dir_nm = flist[0][:flist[0].find(pstr+'tz')]
    elif (flist[0].find(pfstr+'tz') == -1) & (flist[0].find(fstr+'tz') != -1):
        dir_nm = flist[0][:flist[0].find(fstr+'tz')]
    else:
        dir_nm = flist[0][:flist[0].find('tz')]
    pvals = asc.read(dir_nm+pfile)
    vmap = fits.getdata(dir_nm + vmap)
    for i in range(len(flist)):
        im = fits.open(flist[i])
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
        sky_good = skyplane.copy()
        sky_good[vmap==1] = np.nan
        mn = np.nanmean(sky_good[np.isfinite(sky_good)])
        skyplane /= mn # Normalize the sky
        im[0].data /= skyplane
        write_im_head(im[0].data, im[0].header, plist[i])
        if diagnostic:
            write_im_head(skyplane, im[0].header, plist[i]+'.'+str(indx)+'.sky')


def sky_sub(flist, slist, pfile, degree, vmap):
    '''
    Subtracts the sky fits from the images.
    Requires:
      - List of flattened images to undergo sky subtraction
      - Output filename list
      - Pfile.dat, sky fits parameters file to use
      - Degree of polynomial fit used to make pfile.dat
      - Mask for vignetted pixels (vmap_on, vmap_off)
    '''
    dir_nm = flist[0][:flist[0].find(fstr+'tz')]
    pvals = asc.read(dir_nm+pfile)
    vmap = fits.getdata(dir_nm + vmap)
    for i in range(len(flist)):
        im = fits.open(flist[i])
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
        write_im_head(im[0].data, im[0].header, slist[i])
            
    
if __name__ == '__main__':
    # Run this from the Flats main directory, not not_reduction folder
    # Couple preliminary checks to avoid crashes
    if not os.path.exists('on') or  not os.path.exists('off'):
        print('Requires directories ./on and ./off, with tz*.fits images.')
        print('Directories not found.  Quitting program....')
        sys.exit()
    if not os.path.exists('on/vmap_on.fits') or not os.path.exists('off/vmap_off.fits'):
        print('Vignetting maps vmap_on.fits and vmap_off.fits not found.')
        print('Must be in ./on and ./off directories, respectively.')
        print('Quitting program....')
        sys.exit()
    
    # Linking to master twilight flats for first round
    if os.path.exists('rFlat.fits'):
        pass
    else:
        os.system('ln -s ../rFlat.fits .')
        os.system('ln -s ../onFlat.fits .')

    # If not already done, making preliminary masks for initial flat creation
    mk_init_masks()

    # Now initial crude flat construction
    if os.path.exists('on/CrudeFlat.fits'):
        print('Initial on flat already made.  Skipping....')
    else:
        mk_flat(onfiles, m_onfiles, indx=0, vmap='vmap_on.fits', pflag=0)

    if os.path.exists('off/CrudeFlat.fits'):
        print('Initial off flat already made.  Skipping....')
    else:
        mk_flat(offfiles, m_offfiles, indx=0, vmap='vmap_off.fits', pflag=0)

    # Clearing previous output automatically
    for direc in ['on/', 'off/']:
        os.system('/bin/rm -r '+direc+'Trial*')
        os.system('/bin/rm '+direc+'Flat*.fits')
        os.system('/bin/rm '+direc+'pvals*.dat')
        os.system('/bin/rm '+direc+sstr+fstr+'tz*.fits')
        os.system('/bin/rm '+direc+sstr+bnstr+'tz*.fits')
        os.system('/bin/rm '+direc+bnstr+'tz*.fits')
        os.system('/bin/rm '+direc+pstr+fstr+'tz*.fits')
        os.system('/bin/rm '+direc+pstr+'tz*.fits')
        os.system('/bin/rm '+direc+'W*.fits')
        os.system('/bin/rm '+direc+'msftz*.fits')

    # Half-split trials to check consistency
    for m in range(NTrials):
        print('Doing Trial ',m+1,' of ',NTrials)
        if not os.path.exists('on/Trial'+str(m+1)):
            os.mkdir('on/Trial'+str(m+1))
        if not os.path.exists('off/Trial'+str(m+1)):
            os.mkdir('off/Trial'+str(m+1))

        # On first trial, use all images
        if (m == 0):
            oninds = np.arange(len(onfiles))
            offinds = np.arange(len(offfiles))
        # On remaining trials, use random half-splits of the images
        else:
            oninds = np.random.choice(np.arange(len(onfiles)),
                                    size=len(onfiles)//2,
                                    replace=False)
            oninds = np.array(oninds, dtype=int)
            offinds = np.random.choice(np.arange(len(offfiles)),
                                    size=len(offfiles)//2,
                                    replace=False)
            offinds = np.array(offinds, dtype=int)

        # Removing previous run's output to start fresh
        # Preserves existing masks and CrudeFlat.fits
        for direc in ['on/', 'off/']:
            os.system('/bin/rm '+direc+'p*.fits')
            os.system('/bin/rm '+direc+'bn*.fits')
            os.system('/bin/rm '+direc+'ftz*.fits')
            os.system('/bin/rm '+direc+'*.pl')
            
        for n in range(Nloops):
            print('Doing loop ',n+1,' of ',Nloops)
            
            # Flattening images
            print('Flattening images....')
            if n==0:
                flat = 'CrudeFlat.fits'
            else:
                flat = 'Flat'+str(n-1)+'.fits'
            flatten(onfiles[oninds], 'on/'+flat)
            flatten(offfiles[offinds], 'off/'+flat)
    
            # Binning images w/masks
            if os.path.exists(bn_onfiles[oninds][-1]):
                print('Killing previous binned images....')
                os.system('/bin/rm on/'+bnstr+'*.fits')
                os.system('/bin/rm off/'+bnstr+'*.fits')
            print('Binning masked images....')
            for i in range(len(onfiles[oninds])):
                mk_bn_im(f_onfiles[oninds][i], m_onfiles[oninds][i], bn_onfiles[oninds][i], block, bpmf='vmap_on.fits')
            for i in range(len(offfiles[offinds])):
                mk_bn_im(f_offfiles[offinds][i], m_offfiles[offinds][i], bn_offfiles[offinds][i], block, bpmf='vmap_off.fits')
        
            # Fitting sky to binned, masked images
            print('Fitting skies to images....')
            fit_sky_L2D(bn_onfiles[oninds], block, pfile, 2)
            fit_sky_L2D(bn_offfiles[offinds], block, pfile, 2)
    
            # First remove skies from flattened images to redo masks
            print('Removing skies from flattened images....')
            desky(f_onfiles[oninds], pf_onfiles[oninds], pfile, 2, 'vmap_on.fits', indx=n, diagnostic=False)
            desky(f_offfiles[offinds], pf_offfiles[offinds], pfile, 2, 'vmap_off.fits', indx=n, diagnostic=False)

            if (m, n) == (0, 0):
                # Re-masking flattened, de-skied images....
                print('Killing old masks....')
                for i in range(len(m_onfiles[oninds])):
                    os.system('/bin/rm '+m_onfiles[oninds][i])
                for i in range(len(m_offfiles[offinds])):
                    os.system('/bin/rm '+m_offfiles[offinds][i])
                print('Masking images....')
                for f in pf_onfiles[oninds]:
                    mk_mask(f, mstr, commands='--outliersigma=30 --detgrowquant=0.75 --tilesize=60,60', diagnostic=False)
                for f in pf_offfiles[offinds]:
                    mk_mask(f, mstr, commands='--outliersigma=30 --detgrowquant=0.75 --tilesize=60,60', diagnostic=False)
    
            # Then remove skies from unprocessed images to make new flat
            print('Removing sky planes from unprocessed images....')
            desky(onfiles[oninds], p_onfiles[oninds], pfile, 2, 'vmap_on.fits')
            desky(offfiles[offinds], p_offfiles[offinds], pfile, 2, 'vmap_off.fits')
    
            # Making new flat from de-skied raw images w/new masks
            os.system('rm on/Flat'+str(n)+'.fits')
            os.system('rm off/Flat'+str(n)+'.fits')
            print('Making new flat....')
            mk_flat(p_onfiles[oninds], m_onfiles[oninds], n, 'vmap_on.fits')
            mk_flat(p_offfiles[offinds], m_offfiles[offinds], n, 'vmap_off.fits')
    
            # Now saving the outputs with appropriate filenames
            print('Re-naming sky files....')
            new_pfile = pfile[:pfile.find('.dat')]+str(n)+'.dat'
            os.rename('on/'+pfile, 'on/'+new_pfile)
            os.rename('off/'+pfile, 'off/'+new_pfile)
            
            # Diagnostic binned and masked images
            # Only do this step on the last loop to save time
            if (n == Nloops-1):
                print('Making diagnostic binned and masked images....')
                sky_sub(f_onfiles[oninds], s_onfiles[oninds], new_pfile, 2, 'vmap_on.fits')
                sky_sub(f_offfiles[offinds], s_offfiles[offinds], new_pfile, 2, 'vmap_off.fits')
                for i in range(len(onfiles[oninds])):
                    mk_bn_im(s_onfiles[oninds][i], m_onfiles[oninds][i], sbn_onfiles[oninds][i], 9, bpmf='vmap_on.fits')
                for i in range(len(offfiles[offinds])):
                    mk_bn_im(s_offfiles[offinds][i], m_offfiles[offinds][i], sbn_offfiles[offinds][i], 9, bpmf='vmap_off.fits')
    
        # Moving the diagnostic images to the Trial directory
        print('Saving diagnostic files in Trial'+str(m+1)+'/')
        for i in range(len(sbn_onfiles[oninds])):
            fnm_on = s_onfiles[oninds][i][s_onfiles[oninds][i].find(sstr):]
            bn_on = sbn_onfiles[oninds][i][sbn_onfiles[oninds][i].find(sstr):]
            os.rename(sbn_onfiles[oninds][i], 'on/Trial'+str(m+1)+'/'+bn_on)
            os.rename(s_onfiles[oninds][i], 'on/Trial'+str(m+1)+'/'+fnm_on)
        for i in range(len(sbn_offfiles[offinds])):
            fnm_off = s_offfiles[offinds][i][s_offfiles[offinds][i].find(sstr):]
            bn_off = sbn_offfiles[offinds][i][sbn_offfiles[offinds][i].find(sstr):]
            os.rename(sbn_offfiles[offinds][i], 'off/Trial'+str(m+1)+'/'+bn_off)
            os.rename(s_offfiles[offinds][i], 'off/Trial'+str(m+1)+'/'+fnm_off)
        # Moving the Flats and pvals files to the Trial directory
        print('Saving FlatN and pvalsN files to Trial'+str(m+1)+'/')
        for i in range(Nloops):
            os.rename('on/Flat'+str(i)+'.fits', 'on/Trial'+str(m+1)+'/Flat'+str(i)+'.fits')
            os.rename('off/Flat'+str(i)+'.fits', 'off/Trial'+str(m+1)+'/Flat'+str(i)+'.fits')
            os.rename('on/pvals'+str(i)+'.dat', 'on/Trial'+str(m+1)+'/pvals'+str(i)+'.dat')
            os.rename('off/pvals'+str(i)+'.dat', 'off/Trial'+str(m+1)+'/pvals'+str(i)+'.dat')
