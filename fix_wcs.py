#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:50:55 2021

@author: aew

Run this after all images are flattened!
"""

from astropy.io import fits
from astropy.io import ascii as asc
from astropy.table import Table
import numpy as np
import os
import glob
import astroscrappy


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


if not os.path.exists('WCS'):
    os.mkdir('WCS')
    os.mkdir('WCS/on')
    os.mkdir('WCS/off')
    # Just in case, keep Flats directory as backups for this
    # Replace later with w*.fits files, when successful
    os.system('/bin/cp Flats/on/ftz*.fits WCS/on')
    os.system('/bin/cp Flats/off/ftz*.fits WCS/off')
    # Grabbing SEXtractor parameters from Github repo
    os.system('/bin/cp not_reduction/default.* .')
    os.system('/bin/cp not_reduction/gauss_3.0_7x7.conv .')
    cfgFile = open('WCS/astrometry.cfg', 'w')
    cfgFile.write('inparallel \n'
                  +'cpulimit 300\n'
                  +'add_path /usr/share/astrometry/\n' # CHANGE THIS IF NEEDED!
                  +'autoindex\n')
    cfgFile.close()
    

def do_band(band):
    '''
    band is 'on' or 'off'
    '''
    prev = 'WCS/'
    if band=='on':
        start = 7
    elif band=='off':
        start = 8
    imlist = glob.glob(prev+band+'/ftz*.fits')
    for im in imlist:
        if os.path.exists(prev+band+'/w'+im[start:]):
            print('WCS-registered image exists.  Skipping...')
            continue
        # First try to reject cosmic rays
        imHdu = fits.open(im)
        flag = fits.getdata('vmap_'+band+'.fits')
        gain = imHdu[0].header['GAIN']
        rdnoise = imHdu[0].header['RDNOISE']
        crmask, cleanarr = astroscrappy.detect_cosmics(imHdu[0].data, flag, gain=gain,
                                              readnoise=rdnoise, satlevel=600000)
        write_im_head(cleanarr, imHdu[0].header, prev+band+'/c'+im[start:])
    
        # Then re-derive the WCS
        # --overwrite command applies the new WCS to the working image, ftz*.fits
        os.system('solve-field '+prev+band+'/c'+im[start:]
                  +' --config /home/aew/Science/NOTHalpha/WCS/astrometry.cfg ' 
                  +'--scale-units arcsecperpix --scale-low 0.15 --scale-high 0.25 '
                  +'--odds-to-solve 1e8 --no-plots --no-verify --overwrite '
                  +'--axy tmp.axy  --use-sextractor '
                  +'--sextractor-path /usr/bin/source-extractor '
                  +'--new-fits '+prev+band+'/w'+im[start:])
        

if __name__ == '__main__':
    if not os.path.exists('WCS'):
        print('Doing on-band images...')
        do_band('on')
        print('Doing off-band images...')
        do_band('off')
        print('Done!')
    else:
        print('Using previous WCS solutions.  Delete directory to redo.')
    
    # Copies headers to the tz* images in Flats directory
    onims = glob.glob('WCS/on/w*.fits')
    for i in range(len(onims)):
        fname = onims[i][9:]
        #os.system('/bin/cp '+onims[i]+' Flats/on/'+fname)
        im = fits.open(onims[i])
        imFlats = fits.open('Flats/on/'+fname)
        write_im_head(imFlats[0].data, im[0].header, 'Flats/on/'+fname)
        
    offims = glob.glob('WCS/off/w*.fits')
    for i in range(len(offims)):
        fname = offims[i][10:]
        #os.system('/bin/cp '+offims[i]+' Flats/off/'+fname)
        im = fits.open(offims[i])
        imFlats = fits.open('Flats/off/'+fname)
        write_im_head(imFlats[0].data, im[0].header, 'Flats/off/'+fname)

    print('Copied headers from w*.fits to Flats directory tz*.fits.')

# Citation info on astroscrappy:
#If you use this code, please cite the Zendo DOI: https://zenodo.org/record/1482019
#Please cite the original paper which can be found at: http://www.astro.yale.edu/dokkum/lacosmic/
#van Dokkum 2001, PASP, 113, 789, 1420 (article : http://adsabs.harvard.edu/abs/2001PASP..113.1420V)

