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
7. Run `skyflats.py`

Lingering issues (in no particular order):

- Need to optimize NoiseChisel parameters for masking
- What to do about vignetted regions?  Exclude?
- Sky flats currently remove sky with plane fits; appears acceptable for on-band filter, but r-band seems to need a circular sky.  Consider adjusting sky subtraction to use IRAF `imsurfit`.
- Sky flats currently only do one loop using all images.  For best testing, will need to incorporate half-splits as well, to ensure that there are no serious systematics occurring during flat construction.
- Some examination of sky-subtracted images is needed to check how successful plane-fitting is doing.  After run, consider binning and masking pftz*.fits files.
- Check for any bad data that shouldn't be included.
