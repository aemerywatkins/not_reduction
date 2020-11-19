# not_reduction
NOT ALFOSC Halpha data reduction software

Dependencies:

- PyRAF
- `shutil` (`pip install shutils` in Python 2.7 IRAF environment)
- NoiseChisel (carefully follow instructions listed on webpage, starting with dependencies: http://www.gnu.org/software/gnuastro/)

Steps for processing:

1. Separate sky+object exposures into directory `/Science`
2. Separate bias, flats, and other calibration exposures into `/Calib`
3. Clear `/Calib` directories of all irrelevent data
4. From IRAF, run: `hsel *fits[0] $I,object,FAFLTNM,FBFLTNM yes > headers.lis` in both `/Calib` and `/Science`
5. Run `reduce.py` in global data directory
6. Move `skyflats.py` into new `/Flats` directory (produced by `reduce.py`)
7. Remove bad data (ALDi270136, ALDi270137, ALDi290072 & ALDi290073)
8. Run `skyflats.py`
9. From IRAF run: 'hsel tz*fits $I 'OBJECT == "target"' > target.lis' in both '/Flats/on' and 'Flats/off'
10. Move 'vmap_on/off.fits' into their respective directories in '/Flats' and 'hmtzALDi290079.fits' into '/Flats/on'
11. Run 'more_reduction.py' in the global data directory
12. From IRAF run in '/Mosaic':
    --> hsel *fits $I yes > all.lis
    --> hsel *fits $I 'FBFLTNM == "661_5"' > on.lis
    --> hsel *fits $I 'FBFLTNM == "r_Gun 680_102"' > off.lis
13. Run 'combine.py' in the global data directory


Lingering issues (in no particular order):

- Sky flats currently only do one loop using all images.  For best testing, will need to incorporate half-splits as well, to ensure that there are no serious systematics occurring during flat construction..
- Residual phantom light north of the galaxy remains in the mosaics, it is present in ~40% of individual frames.

    
