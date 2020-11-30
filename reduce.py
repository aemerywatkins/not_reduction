import os
from shutil import copyfile # May need to install this via pip
from pyraf import iraf
from astropy.io import ascii as asc
from astropy.io import fits
import numpy as np


def ccdproc(imlis):
    iraf.noao.imred.ccdred.ccdproc(images='@'+imlis,
                                   output='tz//@'+imlis,
                                   ccdtype='',
                                   fixpix='no',
                                   oversca='no',
                                   trim='yes',
                                   zerocor='yes',
                                   darkcor='no',
                                   flatcor='no',
                                   illumcor='no',
                                   fringecor='no',
                                   readcor='no',
                                   scancor='no',
                                   trimsec=trimsec,
                                   zero='Bias.fits',
                                   flat='')


def flatcombine(imlis, output):
    iraf.noao.imred.ccdred.flatcombine(input='tz//@'+imlis,
                                       output=output,
                                       combine='median',
                                       reject='avsigclip',
                                       ccdtype='',
                                       process='no',
                                       subsets='no')


if __name__ == '__main__':
    # Made from cl terminal due to outputting only to stdout
    # --> hsel *fits[0] $I,object,FAFLTNM,FBFLTNM yes > headers.lis
    imheaders = asc.read('Science/headers.lis', data_start=0, format='no_header')
    headers = asc.read('Calib/headers.lis', data_start=0, format='no_header')
    
    rims =  [nm for i,nm in enumerate(imheaders['col1']) if (imheaders['col4'][i].find('r_Gun')!=-1)]
    onims =  [nm for i,nm in enumerate(imheaders['col1']) if (imheaders['col4'][i].find('661_5')!=-1)]
    bias = [nm for i,nm in enumerate(headers['col1']) if (headers['col2'][i].find('bias')!=-1)]
    rflats = [nm for i,nm in enumerate(headers['col1']) if (headers['col2'][i].find('FLAT')!=-1)
                  & (headers['col4'][i].find('r_Gun')!=-1)]
    onflats = [nm for i,nm in enumerate(headers['col1']) if (headers['col2'][i].find('FLAT')!=-1)
                  & (headers['col4'][i].find('661_5')!=-1)]
    
    rims = ['Science/'+i[:-3]+'[1]' for i in rims]
    onims = ['Science/'+i[:-3]+'[1]' for i in onims]
    bias = ['Calib/'+i[:-3]+'[1]' for i in bias]
    rflats =  ['Calib/'+i[:-3]+'[1]' for i in rflats]
    onflats =  ['Calib/'+i[:-3]+'[1]' for i in onflats]
    
    for i in range(len(rims)):
        if i%5==0:
            iraf.flpr()
        iraf.imcopy(input=rims[i], output=rims[i][8:-3])
    
    for i in range(len(onims)):
        if i%5==0:
            iraf.flpr()
        iraf.imcopy(input=onims[i], output=onims[i][8:-3])
    
    for i in range(len(bias)):
        if i%5==0:
            iraf.flpr()
        iraf.imcopy(input=bias[i], output=bias[i][6:-3])
        
    for i in range(len(rflats)):
        if i%5==0:
            iraf.flpr()
        iraf.imcopy(input=rflats[i], output=rflats[i][6:-3])
        
    for i in range(len(onflats)):
        if i%5==0:
            iraf.flpr()
        iraf.imcopy(input=onflats[i], output=onflats[i][6:-3])
    
    asc.write([[i[8:-3] for i in rims]], 'rims.lis', format='no_header', overwrite=True)
    asc.write([[i[8:-3] for i in onims]], 'onims.lis', format='no_header', overwrite=True)
    asc.write([[i[6:-3] for i in bias]], 'bias.lis', format='no_header', overwrite=True)
    asc.write([[i[6:-3] for i in rflats]], 'rflats.lis', format='no_header', overwrite=True)
    asc.write([[i[6:-3] for i in onflats]], 'onflats.lis', format='no_header', overwrite=True)
    print('Wrote image name lists....')
    
    # Make the master bias from all biases
    print('Making master bias....')
    iraf.noao(_doprint=0)
    iraf.noao.imred(_doprint=0)
    iraf.noao.imred.ccdred(_doprint=0)
    iraf.noao.imred.ccdred.zerocombine(input='@bias.lis',
                                       output='Bias.fits',
                                       ccdtype='',
                                       combine='median',
                                       reject='avsigclip')
    
    # Trim and bias-subtract images
    print('Trimming and bias-subtracting images....')
    trimsec = '[220:1950,180:1910]'
    ccdproc('rflats.lis')
    ccdproc('onflats.lis')
    ccdproc('rims.lis')
    ccdproc('onims.lis')
    
    # Make master flats
    print('Making master flats....')
    flatcombine('rflats.lis', 'rFlat.fits')
    flatcombine('onflats.lis', 'onFlat.fits')
    
    # Normalize the master flats
    print('Normalizing master flats....')
    rFlat = fits.getdata('rFlat.fits')
    mn_r = np.mean(rFlat)
    onFlat = fits.getdata('onFlat.fits')
    mn_on = np.mean(onFlat)
    iraf.imarith(operand1='rFlat.fits', op='/', operand2=str(mn_r),  result='rFlat.fits')
    iraf.imarith(operand1='onFlat.fits', op='/', operand2=str(mn_on),  result='onFlat.fits')
    
    print('Done!')
    print('-----')
    print('tz = trimmed, zero-subtracted')
    print('Master on-band: onFlat.fits')
    print('Master off-band: rFlat.fits')
    
    # Some final organization of files
    if not os.path.exists('twilights'):
        os.mkdir('twilights')
        os.mkdir('twilights/on')
        os.mkdir('twilights/off')
    
    for f in [i[6:-3] for i in rflats]:
        os.rename('tz'+f, "twilights/off/tz"+f)
    
    for f in [i[6:-3] for i in onflats]:
        os.rename('tz'+f, "twilights/on/tz"+f)
    
    print('Moved processed flats to /twilights')
    
    
    if not os.path.exists('tzfiles'):
        os.mkdir('tzfiles')
    
    for f in [i[8:-3] for i in rims]:
        os.rename('tz'+f, "tzfiles/tz"+f)
    
    for f in [i[8:-3] for i in onims]:
        os.rename('tz'+f, "tzfiles/tz"+f)
    
    print('Moved processed sky+object images to /tzfiles')
    
    
    if not os.path.exists('Flats'):
        os.mkdir('Flats')
        os.mkdir('Flats/on')
        os.mkdir('Flats/off')
    
    for f in [i[8:-3] for i in rims]:
        copyfile('tzfiles/tz'+f, "Flats/off/tz"+f)
    
    for f in [i[8:-3] for i in onims]:
        copyfile('tzfiles/tz'+f, "Flats/on/tz"+f)
    
    print('Copied sky+object images to /Flats')
    print('Now proceed to making night sky flats in /Flats directory.')
