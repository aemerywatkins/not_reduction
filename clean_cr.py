from astropy.io import fits
import glob
import astroscrappy
import os
import sys


pref = sys.argv[1]
ow_flag = sys.argv[2]

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


def do_cleaning(direc, pref, band, ow_flag):
    '''
    direc -- path to images (e.g. Flats/on_mos/)
    pref -- prefix before tzALDi*.fits (e.g., msf)
    band -- 'on' or 'off'
    ow_flag -- 0 means create new file, append 'cr', 1 means overwrite
    
    '''
    ow_flag = int(ow_flag)
    imlist = glob.glob(direc+pref+'*.fits')
    for im in imlist:
        imHdu = fits.open(im)
        flag = fits.getdata(direc+'vmap_'+band+'.fits')
        gain = imHdu[0].header['GAIN']
        rdnoise = imHdu[0].header['RDNOISE']
        crmask, cleanarr = astroscrappy.detect_cosmics(imHdu[0].data, flag, gain=gain,
                                              readnoise=rdnoise, satlevel=600000,
                                              sigclip=25, sigfrac=0.8)
        imHdu[0].data[crmask] = -999 #Mask out detected cosmic rays
        if ow_flag:
            write_im_head(imHdu[0].data, imHdu[0].header, direc+pref+im[im.find('tz'):])
        else:
            write_im_head(imHdu[0].data, imHdu[0].header, direc+'cr_'+pref+im[im.find('tz'):])


if __name__ == '__main__':
    maindir = '/home/aew/Science/NOTHalpha/'
    if not os.path.exists(maindir+'+Flats/on_mos'):
        direc_on = maindir+'Flats/on/'
        direc_off = maindir+'Flats/off/'
    else:
        direc_on = maindir+'Flats/on_mos/'
        direc_off = maindir+'Flats/off_mos/'  
        
    print('Cleaning on-band images...')
    do_cleaning(direc_on, pref, 'on', ow_flag)
    print('Cleaning off-band images...')
    do_cleaning(direc_off, pref, 'off', ow_flag)
    

    
