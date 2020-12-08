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
7. Move 'vmap_on/off.fits' into their respective directories in '/Flats' and 'hmtzALDi290079.fits' into '/Flats/on'
8. Remove bad data (ALDi270136, ALDi270137, ALDi290072, ALDi290085 & ALDi290073)
9. Run `skyflats.py` from Flats directory
10. From main directory, run 'mk_mosaic.py'
11. After mosaic is made, run 'skyflats_mosaic.py' to remake flat using altered sky subtraction
12. Run 'mk_mosaic.py' again to remake mosaic using new sky subtraction


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

- Test if iterating mosaic sky subtraction method improves final product at all (do by hand for now)
- Need to test method with model galaxy to see if flux is being lost or gained using this method
  -- Inject model into tz* frames, then run through the whole pipeline and compare output w/model parameters

    
