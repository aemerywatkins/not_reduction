from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
import os

ra, dec = (33.227750, -19.316889)
coord = SkyCoord(ra, dec, unit=(u.degree, u.degree), frame='icrs')
width = u.Quantity(0.6, u.deg)
height = u.Quantity(0.6, u.deg)

Gaia.ROW_LIMIT = -1 # No limit
r = Gaia.query_object_async(coordinate=coord, width=width, height=height)

rTab = Table([r['ra'], r['dec'], r['phot_g_mean_mag']])
rTab.rename_column('ra', 'RA')
rTab.rename_column('dec', 'Dec')
rTab.rename_column('phot_g_mean_mag', 'Gmag')
rTab.write('gaia.fits', format='fits', overwrite=True)
# Gaia citation info found here: https://www.cosmos.esa.int/web/gaia-users/credits

# Requires Astrometry.net installation
# To understand these commands, see http://astrometry.net/doc/readme.html
os.system('build-astrometry-index -i gaia.fits -o gaia0.index -P 0 -S Gmag -E -I 160420210')
os.system('build-astrometry-index -i gaia.fits -o gaia2.index -P 2 -S Gmag -E -I 160420212')
os.system('build-astrometry-index -i gaia.fits -o gaia4.index -P 4 -S Gmag -E -I 160420214')
os.system('build-astrometry-index -i gaia.fits -o gaia6.index -P 6 -S Gmag -E -I 160420214')

print('Gaia index files created.  Move these to the proper directory now e.g. /usr/share/astrometry')
