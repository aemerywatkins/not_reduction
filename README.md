# not_reduction
NOT ALFOSC Halpha data reduction software

Dependencies:

- PyRAF
- `shutil` (`pip install shutils` in Python 2.7 IRAF environment)
- NoiseChisel (carefully follow instructions listed on webpage, starting with dependencies: http://www.gnu.org/software/gnuastro/)
- Astrometry.net (Ubuntu apt install astrometry.net and associated libraries/Python packages)
- SWarp (can sudo install)
- SCamp (also can sudo install, but should get latest version from Github for GAIA-EDR3 catalogue access)

Steps for processing:

1. Separate sky+object exposures into directory `/Science`
2. Separate bias, flats, and other calibration exposures into `/Calib`
3. Clear `/Calib` directories of all irrelevent data
4. From IRAF, run: `hsel *fits[0] $I,object,FAFLTNM,FBFLTNM yes > headers.lis` in both `/Calib` and `/Science`
5. Run `reduce.py` in global data directory
6. Move `skyflats.py` into new `/Flats` directory (produced by `reduce.py`)
7. Move 'vmap_on/off.fits' into their respective directories in '/Flats' and 'hmtzALDi290079.fits' into '/Flats/on'
8. Remove bad data (ALDi270136, ALDi270137, ALDi290072, ALDi290085 & ALDi290073)
9. Run `skyflats.py` from Flats directory
10. Create Gaia index files for improved astrometry using mk_gaia_indexes.py
   - NOTE: need to move these to the proper directory, or else add command to solve-field to find them
11. Redo WCS on object exposures using fix_wcs.py from main directory (Python 3!)
    - Auto-creates astrometry.net config file: change install directory here if not finding application
12. From main directory, run 'mk_mosaic.py' (Back to Python 2, PyRAF environment)
13. After mosaic is made, run 'skyflats_mosaic.py' to remake flat using altered sky subtraction
14. Run 'mk_mosaic.py' again to remake mosaic using new sky subtraction


# DEFUNCT BELOW
================
10. From IRAF run: 'hsel tz*fits $I 'OBJECT == "target"' > target.lis' in both '/Flats/on' and 'Flats/off'
11. Run 'more_reduction.py' in the global data directory
12. From IRAF run in '/Mosaic':
    --> hsel *fits $I yes > all.lis
    --> hsel *fits $I 'FBFLTNM == "661_5"' > on.lis
    --> hsel *fits $I 'FBFLTNM == "r_Gun 680_102"' > off.lis
13. Run 'combine.py' in the global data directory


Lingering issues (in no particular order):

- Adjust mosaic rotation and shifting to use SWarp instead of IRAF
- Check if background in initial mosaic is offset from zero on average
- Model injection testing: make polynomial fit mosaic with/without models and check,
  make new sky-sub mosaic with/without models and check as well
- Fix the NTT pipeline too, applying new SS method to those galaxies and fixing WCS headers.

    
