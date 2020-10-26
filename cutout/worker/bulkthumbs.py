#!/usr/bin/env python3

"""
author: Landon Gelman, 2018-2020
author: Francisco Paz-Chinchon, 2019
author: T. Andrew Manning, 2020
description: command line tools for making large numbers and multiple kinds of cutouts from the Dark Energy Survey catalogs
"""

import os, sys
import argparse
import datetime
import logging
import glob
import time
import easyaccess as ea
import numpy as np
import pandas as pd
import PIL
import uuid
import json
import yaml
import shlex
import subprocess
from astropy import units
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs import utils
from astropy.visualization import make_lupton_rgb as mlrgb
from mpi4py import MPI as mpi
from PIL import Image
import math
from io import StringIO
import re

Image.MAX_IMAGE_PIXELS = 144000000        # allows Pillow to not freak out at a large filesize
ARCMIN_TO_DEG = 0.0166667        # deg per arcmin
# TODO: Move the database and release names to environment variables or to a config file instead of hard-coding
VALID_DATA_SOURCES = {
    'DESDR': [
        'DR1',
        'DR2',
    ],
    'DESSCI': [
        'Y6A1',
        'Y3A2',
        'Y1A1',
        'SVA1',
    ]
}
# TODO: remove these unnecessary global variables
TILES_FOLDER = ''

comm = mpi.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

class MPILogHandler(logging.FileHandler):
    def __init__(self, filename, comm, amode=mpi.MODE_WRONLY|mpi.MODE_CREATE|mpi.MODE_APPEND):
        self.comm = comm
        self.filename = filename
        self.amode = amode
        self.encoding = 'utf-8'
        logging.StreamHandler.__init__(self, self._open())
    def _open(self):
        stream = mpi.File.Open(self.comm, self.filename, self.amode)
        stream.Set_atomicity(True)
        return stream
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.Write_shared((msg+self.terminator).encode(self.encoding))
        except Exception:
            self.handleError(record)
    def close(self):
        if self.stream:
            self.stream.Sync()
            self.stream.Close()
            self.stream = None

def getPathSize(path):
    dirsize = 0
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            dirsize += getPathSize(entry.path)
        else:
            try:
                dirsize += os.path.getsize(entry)
            except FileNotFoundError:
                continue

    return dirsize

def _DecConverter(ra, dec):
    ra1 = np.abs(ra/15)
    raHH = int(ra1)
    raMM = int((ra1 - raHH) * 60)
    raSS = (((ra1 - raHH) * 60) - raMM) * 60
    raSS = np.round(raSS, decimals=4)
    raOUT = '{0:02d}{1:02d}{2:07.4f}'.format(raHH, raMM, raSS) if ra > 0 else '-{0:02d}{1:02d}{2:07.4f}'.format(raHH, raMM, raSS)

    dec1 = np.abs(dec)
    decDD = int(dec1)
    decMM = int((dec1 - decDD) * 60)
    decSS = (((dec1 - decDD) * 60) - decMM) * 60
    decSS = np.round(decSS, decimals=4)
    decOUT = '-{0:02d}{1:02d}{2:07.4f}'.format(decDD, decMM, decSS) if dec < 0 else '+{0:02d}{1:02d}{2:07.4f}'.format(decDD, decMM, decSS)

    return raOUT + decOUT

def filter_colors(colorString):
    # Returns a comma-separated string of color characters ordered by wavelength
    if isinstance(colorString, str):
        # Discard all invalid characters and delete redundancies
        color_list_filtered_deduplicated = list(set(re.sub(r'([^grizy])', '', colorString.lower())))
        # Do not order the color sets because that order is how the user selects which bands are represented by Red/Green/Blue 
        #
        # ordered_colors = []
        # # Order the colors from long to short wavelength
        # for color in list('yzirg'):
        #     if color in color_list_filtered_deduplicated:
        #         ordered_colors.append(color)
        return ''.join(color_list_filtered_deduplicated)

def make_rgb(cutout, color_set, outdir, basename):
    logger = logging.getLogger(__name__)
    output_files = []
    if len(color_set) != 3:
        logger.error('Exactly three colors are required for RGB generation.')
        return output_files
    # Support color set specification as list of letters or a string
    if isinstance(color_set, list):
        color_set = ''.join(color_set)
    # Ensure that FITS source file has been generated for each required color. This
    # should be unnecessary unless this function is used independently
    fits_filepaths = {}
    for color in color_set:
        # TODO: Consolidate these naming conventions in a dedicated function
        fits_filepath = os.path.join(outdir, basename + '_{}.fits'.format(color))
        fits_filepaths[color] = fits_filepath
        if not os.path.exists(outdir) or not glob.glob(fits_filepath):
            files = make_fits_cut(cutout, color, outdir, basename)
            output_files.extend(files)
            # If the file remains absent, log the error
            if not os.path.exists(outdir) or not glob.glob(fits_filepath):
                logger.error('Error creating the required FITS file "{}".'.format(fits_filepath))
                return output_files
    
    # Output RGB file basepath
    filename_base = os.path.join(outdir, '{0}_{1}'.format(basename, color_set))
    fits_file_list = [fits_filepaths[color] for color in fits_filepaths]

    if cutout['MAKE_RGB_LUPTON']:
        # TODO: Verify that the comparison of generated size to the requested size is redundant since 
        # this is logged in the FITS cutout file generation
        r_data = fits.getdata(fits_file_list[0], 'SCI')
        g_data = fits.getdata(fits_file_list[1], 'SCI')
        b_data = fits.getdata(fits_file_list[2], 'SCI')
        # Generate RGB image from three FITS file data
        image = mlrgb(
            # FITS file data
            r_data, g_data, b_data, 
            # Lupton parameters
            minimum=cutout['RGB_MINIMUM'], 
            stretch=cutout['RGB_STRETCH'], 
            Q=cutout['RGB_ASINH']
        )
        image = Image.fromarray(image, mode='RGB')
        image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        luptonnm = filename_base + '_lupton'
        filename = luptonnm+'.png'
        image.save(filename, format='PNG')
        output_files.append(filename)

    if cutout['MAKE_RGB_STIFF']:
        stiffnm = filename_base + '_stiff'
        # Call STIFF using the 3 bands.
        cmd_stiff = 'stiff {}'.format(' '.join(fits_file_list))
        cmd_stiff += ' -OUTFILE_NAME {}'.format(stiffnm+'.tiff')
        cmd_stiff = shlex.split(cmd_stiff)
        try:
            subprocess.call(cmd_stiff)
        except OSError as e:
            logger.error(e)

        # Convert the STIFF output from tiff to png and remove the tiff file.
        filename = stiffnm+'.png'
        cmd_convert = 'convert {0} {1}'.format(stiffnm+'.tiff', filename)
        cmd_convert = shlex.split(cmd_convert)
        try:
            subprocess.call(cmd_convert)
            output_files.append(filename)
        except OSError as e:
            logger.error(e)
        try:
            os.remove(stiffnm+'.tiff')
        except OSError as e:
            logger.error(e)

    return output_files

def make_fits_cut(cutout, colors, outdir, basename):
    logger = logging.getLogger(__name__)
    # Array of generated files
    output_files = []
    # Create output subdirectory
    os.makedirs(outdir, exist_ok=True)

    # Iterate over individual colors (i.e. bands)
    for color in colors.lower():
        # Construct the output filename, with different naming scheme based on coord or coadd position type
        filename = basename + '_{}.fits'.format(color)
        filepath = os.path.join(outdir, filename)
        # If file exists, continue to the next color
        if glob.glob(filepath):
            continue

        try:
            # Y-band color must be uppercase; others lowercase
            source_file_color = color.upper() if color.lower() == 'y' else color.lower()
            hdu_list = fits.open(glob.glob(cutout['TILEDIR'] + '*_{}.fits.fz'.format(source_file_color))[0])
        except IndexError as e:
            print('No FITS file in {0} color band found. Will not create cutouts in this band.'.format(color))
            logger.error('MakeFitsCut - No FITS file in {0} color band found. Will not create cutouts in this band.'.format(color))
            continue        # Just go on to the next color in the list
        

        # Iterate over all HDUs in the tile
        new_hdu_list = fits.HDUList()
        pixelscale = None
        for hdu in hdu_list:
            if hdu.name == 'PRIMARY':
                continue
            data = hdu.data
            header = hdu.header.copy()
            wcs = WCS(header)
            cutout_2D = Cutout2D(data, cutout['POSITION'], cutout['SIZE'], wcs=wcs, mode='trim')
            crpix1, crpix2 = cutout_2D.position_cutout
            x, y = cutout_2D.position_original
            crval1, crval2 = wcs.wcs_pix2world(x, y, 1)
            header['CRPIX1'] = crpix1
            header['CRPIX2'] = crpix2
            header['CRVAL1'] = float(crval1)
            header['CRVAL2'] = float(crval2)
            header['HIERARCH RA_CUTOUT'] = cutout['RA']
            header['HIERARCH DEC_CUTOUT'] = cutout['DEC']
            if not new_hdu_list:
                new_hdu = fits.PrimaryHDU(data=cutout_2D.data, header=header)
                pixelscale = utils.proj_plane_pixel_scales(wcs)
            else:
                new_hdu = fits.ImageHDU(data=cutout_2D.data, header=header, name=header['EXTNAME'])
            new_hdu_list.append(new_hdu)
        # Check the size of the cutout compared to the requested size and warn if different
        if pixelscale is not None:
            dx = int(cutout['SIZE'][1] * ARCMIN_TO_DEG / pixelscale[0] / units.arcmin)        # pixelscale is in degrees (CUNIT)
            dy = int(cutout['SIZE'][0] * ARCMIN_TO_DEG / pixelscale[1] / units.arcmin)
            if (new_hdu_list[0].header['NAXIS1'], new_hdu_list[0].header['NAXIS2']) != (dx, dy):
                logger.info('MakeFitsCut - {} is smaller than user requested. This is likely because the object/coordinate was in close proximity to the edge of a tile.'.format(('/').join(filename.split('/')[-2:])))
        # Save the resulting cutout to disk
        new_hdu_list.writeto(filename, output_verify='exception', overwrite=True, checksum=False)
        new_hdu_list.close()
        # Add the output file to the return list
        output_files.append(filename)

    return output_files

def run(conf):

    os.makedirs(conf['outdir'], exist_ok=True)
    # Configure logging
    logtime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logname = os.path.join(conf['outdir'], 'BulkThumbs_' + logtime + '.log')         # use for local
    formatter = logging.Formatter('%(asctime)s - '+str(rank)+' - %(levelname)-8s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = MPILogHandler(logname, comm)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Validate the configuration and obtain the primary DataFrame object 
    user_df = validate_config(conf)

    # Initialize the database connection and basic job info
    username, jobid, outdir = None, None, None
    # Only attempt to establish a 
    if rank == 0:
        # Get username from config
        username = conf['username']
        # Get job ID from config
        jobid = conf['jobid']
        # Get database connection and cursor objects using easyaccess
        uu = conf['username']
        pp = conf['password']
        if conf['db'].lower() == 'desdr':
            # Use the Oracle service account to access the relevant tile path info table if provided in the config
            if conf['oracle_service_account_db'] and conf['oracle_service_account_user'] and conf['oracle_service_account_pass']:
                db = conf['oracle_service_account_db']
                uu = conf['oracle_service_account_user']
                pp = conf['oracle_service_account_pass']
            else:
                db = conf['db'].lower()
        elif conf['db'].lower() == 'dessci':
            db = conf['db'].lower()
        else:
            logger.error('Invalid database.')
            return
        conn = ea.connect(db, user=uu, passwd=pp)
        curs = conn.cursor()

        # Create the output directory. Fail if it already exists.
        outdir = os.path.join(conf['outdir'], '')
        try:
            os.makedirs(outdir, exist_ok=True)
        except OSError as e:
            print(e)
            print('Error creating output directory. Aborting job.')
            # print('Specified jobid already exists in output directory. Aborting job.')
            conn.close()
            sys.stdout.flush()
            comm.Abort()

    # Broadcast variable values to parallel processes
    username, jobid, outdir = comm.bcast([username, jobid, outdir], root=0)


    # xs = float(conf.xsize)
    # ys = float(conf.ysize)
    # colors = conf.colors_fits.split(',')

    complete_df = None
    if rank == 0:
        # Record the configuration of the cutout requests
        summary = {
            'options': conf,
            'cutouts': user_df,
        }
        # Start a timer for the database query
        start = time.time()

        logger.info('Requested cutouts and options:')
        logger.info(user_df)

        # Subset of DataFrame where position type is RA/DEC coordinates
        coord_df = user_df[user_df['POSITION_TYPE'] == 'coord']
        coord_df = coord_df[['RA', 'DEC', 'RA_ADJUSTED', 'XSIZE', 'YSIZE']]
        logger.info('coord_df: {}'.format(coord_df))

        # Subset of DataFrame where position type is Coadd ID
        coadd_df = user_df[user_df['POSITION_TYPE'] == 'coadd']
        coadd_df = coadd_df[['COADD_OBJECT_ID', 'XSIZE', 'YSIZE']]
        logger.info('coadd_df: {}'.format(coadd_df))

        # Define the temporary database tablename and output CSV filepath
        tablename = 'BTL_'+jobid.upper().replace("-","_")
        tablename_csv_filepath = os.path.join(outdir, tablename+'.csv')

        # Define the catalogs to query based on the chosen database
        if conf['db'].upper() == 'DESSCI':
            catalog_coord = 'Y3A2_COADDTILE_GEOM'
            catalog_coadd = 'Y3A2_COADD_OBJECT_SUMMARY'
        elif conf['db'].upper() == 'DESDR':
            catalog_coord = 'DR1_Tile_INFO'
            catalog_coadd = 'DR1_MAIN'
        else:
            logger.error('Invalid database.')
            sys.exit(1)

        #############################################################
        # Find tile names associated with each position by COORDINATE
        #
        unmatched_coords = {'RA':[], 'DEC':[]}

        # Create the temporary database table from a CSV dump of the DataFrame
        coord_df.to_csv(tablename_csv_filepath, index=False)
        conn.load_table(tablename_csv_filepath, name=tablename)
        logger.info('Created temporary table from CSV')
        # conn.pandas_to_db(coord_df, tablename=tablename)
        # logger.info('Created temporary table directly from DataFrame')

        query = '''
            select temp.RA, temp.DEC, temp.RA_ADJUSTED, temp.RA as ALPHAWIN_J2000, temp.DEC as DELTAWIN_J2000, m.TILENAME, temp.XSIZE, temp.YSIZE
            from {tablename} temp 
            left outer join {catalog} m on 
            (
                m.CROSSRA0='N' and 
                (temp.RA between m.URAMIN and m.URAMAX) and 
                (temp.DEC between m.UDECMIN and m.UDECMAX)
            ) or 
            (
                m.CROSSRA0='Y' and 
                (temp.RA_ADJUSTED between m.URAMIN-360 and m.URAMAX) and 
                (temp.DEC between m.UDECMIN and m.UDECMAX)
            )
        '''.format(tablename=tablename, catalog=catalog_coord)
        
        # Overwrite DataFrame with extended table that has tilenames
        coord_df = conn.query_to_pandas(query)
        # Drop the temporary table
        curs.execute('drop table {}'.format(tablename))
        os.remove(tablename_csv_filepath)

        # # Refine the values
        # coord_df = coord_df.replace('-9999',np.nan)
        # coord_df = coord_df.replace(-9999.000000,np.nan)

        # Record unmatched positions
        dftemp = coord_df[ (coord_df['TILENAME'].isnull()) ]
        unmatched_coords['RA'] = dftemp['RA'].tolist()
        unmatched_coords['DEC'] = dftemp['DEC'].tolist()
        # # Drop unmatched entries from the DataFrame
        # coord_df = coord_df.dropna(axis=0, how='any', subset=['TILENAME'])

        #############################################################
        # Find tile names associated with each position by COADD ID
        #
        unmatched_coadds = []

        # Create the temporary database table from a CSV dump of the DataFrame
        coadd_df.to_csv(tablename_csv_filepath, index=False)
        conn.load_table(tablename_csv_filepath, name=tablename)

        query = '''
            select temp.COADD_OBJECT_ID, m.ALPHAWIN_J2000, m.DELTAWIN_J2000, m.RA, m.DEC, m.TILENAME, temp.XSIZE, temp.YSIZE
            from {tablename} temp 
            left outer join {catalog} m on temp.COADD_OBJECT_ID=m.COADD_OBJECT_ID
        '''.format(tablename=tablename, catalog=catalog_coadd)
        
        # Overwrite DataFrame with extended table that has tilenames
        coadd_df = conn.query_to_pandas(query)
        # Drop the temporary table
        curs.execute('drop table {}'.format(tablename))
        os.remove(tablename_csv_filepath)

        # # Refine the values
        # coadd_df = coadd_df.replace('-9999',np.nan)
        # coadd_df = coadd_df.replace(-9999.000000,np.nan)
        
        # Record unmatched positions
        dftemp = coadd_df[ (coadd_df['TILENAME'].isnull()) | (coadd_df['ALPHAWIN_J2000'].isnull()) | (coadd_df['DELTAWIN_J2000'].isnull()) | (coadd_df['RA'].isnull()) | (coadd_df['DEC'].isnull()) ]
        unmatched_coadds = dftemp['COADD_OBJECT_ID'].tolist()
        # # Drop unmatched entries from the DataFrame
        # coadd_df = coadd_df.dropna(axis=0, how='any', subset=['TILENAME','ALPHAWIN_J2000','DELTAWIN_J2000','RA','DEC'])

        # Merge results with the original sub-df
        complete_df = pd.merge(left=user_df, right=coord_df, on=['RA', 'DEC'], how='left')
        complete_df = pd.merge(left=complete_df, right=coadd_df, on=['COADD_OBJECT_ID'], how='left')

        # Refine the values
        complete_df = complete_df.replace('-9999',np.nan)
        complete_df = complete_df.replace(-9999.000000,np.nan)
        
        # Drop unmatched entries from the DataFrame
        complete_df = complete_df.dropna(axis=0, how='any', subset=['TILENAME','ALPHAWIN_J2000','DELTAWIN_J2000','RA','DEC'])


        #############################################################
        # Recombine the subcomponent DataFrames
        #
        # complete_df = pd.concat([coord_df, coadd_df])
        complete_df = complete_df.sort_values(by=['TILENAME'])
        # Requesting multiple cutouts of the same position with different sizes is not allowed.
        complete_df = complete_df.drop_duplicates(['RA','DEC'], keep='first')

        end1 = time.time()
        query_elapsed = '{0:.2f}'.format(end1-start)
        print('Querying took (s): ' + query_elapsed)
        logger.info('Querying took (s): ' + query_elapsed)
        summary['query_time'] = query_elapsed

    # Split the table into equal parts and distribute to the parallel processes
    df = comm.scatter(np.array_split(complete_df, nprocs), root=0)

    qtemplate = "select FITS_IMAGES from {} where tilename = '{}' and band = 'i'"
    table_path = "MCARRAS2.{}_TILE_PATH_INFO".format(conf['release'])
    # Determine the file paths for each unique relevant tile  
    for tilename in df['TILENAME'].unique():
        try:
            if conf.tiledir != 'auto':
                tiledir = os.path.join(conf.tiledir, tilename)
            else:
                dftile = conn.query_to_pandas(qtemplate.format(table_path, tilename))
                tiledir = os.path.dirname(dftile.FITS_IMAGES.iloc[0])
                if conf['release'] in ('Y6A1', 'Y3A2', 'DR1'):
                    tiledir = tiledir.replace('https://desar2.cosmology.illinois.edu/DESFiles/desarchive/OPS/', '/des003/desarchive/') + '/'
                elif conf['release'] in ('SVA1', 'Y1A1'):
                    tiledir = tiledir.replace('https://desar2.cosmology.illinois.edu/DESFiles/desardata/OPS/coadd/', '/des004/coadd/') + '/'
                logger.info('Using DB and table {} to determine paths...'.format(table_path))
            # Clean up path formatting
            tiledir = os.path.join(tiledir, '')
            # Store tiledir in table 
            df.loc[df['TILENAME'] == tilename, 'TILEDIR'] = tiledir
        except Exception as e:
            logger.error(str(e).strip())

    #############################################################
    # Main iteration loop over all cutout requests
    #
    # Iterate over each row and validate parameter values
    for row_index, cutout in df.iterrows():
        # Collect all generated files associated with this position
        generated_files = []
        cutout['SIZE'] = units.Quantity((cutout['YSIZE'], cutout['XSIZE']), units.arcmin)
        cutout['POSITION'] = SkyCoord(cutout['ALPHAWIN_J2000'], cutout['DELTAWIN_J2000'], frame='icrs', unit='deg', equinox='J2000', representation_type='spherical')

        # Files are stored in subdirectories based on the unique position of the cutout request
        if cutout['POSITION_TYPE'] == 'coadd':
            cutout_dirname = cutout['COADD_OBJECT_ID']
        else:
            cutout_dirname = 'DESJ' + _DecConverter(cutout['RA'], cutout['DEC'])
        # Output directory stucture: [base outdir path]/[source tile name]/[position]
        cutout_outdir = os.path.join(outdir, cutout['TILENAME'], cutout_dirname)

        # Make all FITS cutout files necessary for requested FITS files and any RGB files
        all_colors = ''
        for rgb_type in [['MAKE_FITS', 'FITS_COLORS'], ['MAKE_RGB_STIFF', 'RGB_STIFF_COLORS'], ['MAKE_RGB_LUPTON', 'RGB_LUPTON_COLORS']]:
            if cutout[rgb_type[0]]:
                # Add the color if it is an acceptable letter do not duplicate
                for color in cutout[rgb_type[1]]:
                    if color in 'grizy' and color not in all_colors:
                        all_colors += color
        for color in all_colors:
            output_files = make_fits_cut(cutout, color, cutout_outdir, cutout_dirname)
            generated_files.extend(output_files)

        # Now that all required FITS files have been generated, create any requested RGB images
        for rgb_type in [['MAKE_RGB_STIFF', 'RGB_STIFF_COLORS'], ['MAKE_RGB_LUPTON', 'RGB_LUPTON_COLORS']]:
            if cutout[rgb_type[0]]:
                color_sets = cutout[rgb_type[1]].split(';')
                for color_set in color_sets:
                    output_files = make_rgb(cutout, color_set, cutout_outdir, cutout_dirname)
                    generated_files.extend(output_files)

        # Add new output files to the list of all files generated for this position
        file_list = json.loads(cutout['FILES'])
        df.at[row_index, 'FILES'] = json.dumps(file_list.extend(generated_files))

    # Close database connection
    conn.close()
    # Synchronize parallel processes at this line to ensure all processing is complete
    comm.Barrier()

    if rank == 0:
        logger.info('All processes finished.')
        end2 = time.time()
        processing_time = '{0:.2f}'.format(end2-end1)
        print('Processing took (s): ' + processing_time)
        logger.info('Processing took (s): ' + processing_time)
        summary['processing_time'] = processing_time

        # Calculate total size of generated files on disk
        dirsize = getPathSize(outdir)
        dirsize = dirsize * 1. / 1024
        if dirsize > 1024. * 1024:
            dirsize = '{0:.2f} GB'.format(1. * dirsize / 1024. / 1024)
        elif dirsize > 1024.:
            dirsize = '{0:.2f} MB'.format(1. * dirsize / 1024.)
        else:
            dirsize = '{0:.2f} KB'.format(dirsize)
        logger.info('Total file size on disk: {}'.format(dirsize))
        summary['size_on_disk'] = str(dirsize)

        # Recombine the slices of the complete DataFrame
        complete_df = comm.gather(df, root=0)

        all_generated_files = []
        for row_index, cutout in complete_df.iterrows():
            file_list = json.loads(cutout['FILES'])
            all_generated_files.extend(file_list)
            
        # files = glob.glob(os.path.join(outdir, '*/*'))
        logger.info('Total number of files: {}'.format(len(all_generated_files)))
        summary['number_of_files'] = len(all_generated_files)
        # Save the DataFrame in the job summary file
        summary['cutouts'] = json.loads(complete_df.to_json(orient="records"))

        # jsonfile = os.path.join(outdir, 'BulkThumbs_'+logtime+'_SUMMARY.json')
        jsonfile = os.path.join(outdir, 'summary.json')
        with open(jsonfile, 'w') as fp:
            json.dump(summary, fp)

def validate_config(conf):

    logger = logging.getLogger(__name__)
    # Load default values
    with open(os.path.join(os.path.dirname(__file__), 'config.default.yaml'), 'r') as configfile:
        defaults = yaml.load(configfile, Loader=yaml.FullLoader)
    logger.info('Defaults file loaded: {}'.format(json.dumps(defaults, indent=2)))
    logger.info('positions: {}'.format(conf['positions']))
    # Import CSV-formatted table of positions (and options) to a DataFrame object
    try:
        df = pd.DataFrame(pd.read_csv(StringIO(conf['positions']), skipinitialspace=True, dtype={
            'COADD_OBJECT_ID': str,
            'RA': np.float64,
            'DEC': np.float64,
            'XSIZE': np.float64,
            'YSIZE': np.float64,
            'FITS_COLORS': str,
            'RGB_STIFF_COLORS': str,
            'RGB_LUPTON_COLORS': str,
            'MAKE_FITS': np.float64,
            'MAKE_RGB_STIFF': np.float64,
            'MAKE_RGB_LUPTON': np.float64,
        },
        na_values={
            'COADD_OBJECT_ID': '',
            'RA': '',
            'DEC': '',
            'XSIZE': '',
            'YSIZE': '',
            'FITS_COLORS': '',
            'RGB_STIFF_COLORS': '',
            'RGB_LUPTON_COLORS': '',
        }
        ))
    except Exception as e:
        logger.info('Error importing positions CSV file: {}'.format(str(e).strip()))
        sys.exit(1)

    logger.info('df: {}'.format(df))
    # Ensure that each parameter column is populated in the DataFrame
    for param in ['XSIZE', 'YSIZE', 'FITS_COLORS', 'RGB_STIFF_COLORS', 'RGB_LUPTON_COLORS', 'MAKE_FITS', 'MAKE_RGB_LUPTON', 'MAKE_RGB_STIFF']:
        # If the parameter was not included in the CSV file
        if not param in df:
            # Check if a global default was provided by the user.
            if param in conf:
                default_val = conf[param]
            # If not, use the default nominal value
            else:
                default_val = defaults[param]
            df[param] = [default_val for c in range(len(df))]
    
    # Add additional columns required for processing:
    #
    # Add a column marking clearly whether the cutout is based on coordinates or coadd IDs
    df['POSITION_TYPE'] = ['' for c in range(len(df))]
    # Add a column for the adjusted RA value
    df['RA_ADJUSTED'] = [None for c in range(len(df))]
    # Add a column for the output file paths
    df['FILES'] = ['[]' for c in range(len(df))]
    # Add a column for the path to the associated data tile
    df['TILEDIR'] = ['' for c in range(len(df))]

    # Iterate over each row and validate parameter values
    for row_index, cutout in df.iterrows():
        # logger.info('\n{}:\n{}'.format(row_index, cutout))
        # If both COADD ID and RA/DEC coords are specified, fail
        if 'COADD_OBJECT_ID' in cutout and isinstance(cutout['COADD_OBJECT_ID'], str) and ('RA' in cutout and not math.isnan(cutout['RA']) or 'DEC' in cutout and not math.isnan(cutout['DEC'])):
            logger.info('Only COADD_OBJECT_ID or RA/DEC coordinates my be specified, not both.')
            sys.exit(1)
        # If both RA and DEC coords are not specified, fail
        if not all(k in cutout for k in ['RA', 'DEC']) or any(math.isnan(cutout[k]) for k in ['RA', 'DEC']):
            if not 'COADD_OBJECT_ID' in cutout or not isinstance(cutout['COADD_OBJECT_ID'], str):
                logger.info('RA and DEC must both be specified.')
                sys.exit(1)
        # Label the cutout with the type of position to simplify subsequent logic
        if 'COADD_OBJECT_ID' in cutout and isinstance(cutout['COADD_OBJECT_ID'], str):
            # The position is based on Coadd ID
            df.at[row_index, 'POSITION_TYPE'] = 'coadd'
        else:
            # The position is based on RA/DEC coordinate
            df.at[row_index, 'POSITION_TYPE'] = 'coord'
            # Set the adjusted RA value
            df.at[row_index, 'RA_ADJUSTED'] = 360-cutout['RA'] if cutout['RA'] > 180 else cutout['RA']
        # Ensure numerical values with limited ranges are respected by coercing invalid values
        for param in ['XSIZE', 'YSIZE', 'RGB_MINIMUM', 'RGB_STRETCH', 'RGB_ASINH']:
            param_min = '{}_MIN'.format(param)
            param_max = '{}_MAX'.format(param)
            # If the param is not specified or is not a number value
            if not param in cutout or math.isnan(cutout[param]):
                # Check if a global default was provided by the user.
                if param in conf:
                    default_val = conf[param]
                # If not, use the default nominal value
                else:
                    default_val = defaults[param]
                df.at[row_index, param] = default_val
            # If the size is too small, rail at minimum, if there is a minimum
            elif param_min in defaults and cutout[param] < defaults[param_min]:
                df.at[row_index, param] = defaults[param_min]
            # If the size is too large, rail at maximum, if there is a maximum
            elif param_max in defaults and cutout[param] > defaults[param_max]:
                df.at[row_index, param] = defaults[param_max]
        # Ensure that colors are set correctly if output boolean flags are set
        for param in ['MAKE_FITS', 'MAKE_RGB_STIFF', 'MAKE_RGB_LUPTON']:
            if not param in cutout or math.isnan(cutout[param]):
                # Check if a global default was provided by the user.
                if param in conf:
                    default_val = conf[param] # == True
                # If not, use the default nominal value
                else:
                    default_val = defaults[param] # == True
                df.at[row_index, param] = default_val
        for param in ['FITS_COLORS', 'RGB_STIFF_COLORS', 'RGB_LUPTON_COLORS']:
            if not param in cutout or not isinstance(cutout[param], str):
                # Check if a global default was provided by the user.
                if param in conf:
                    default_val = conf[param]
                # If not, use the default nominal value
                else:
                    default_val = defaults[param]
                df.at[row_index, param] = default_val
            else:
                color_sets = cutout[param].lower().split(';')
                color_sets_filtered_ordered = []
                for color_set in color_sets:
                    color_set_filtered = filter_colors(color_set)
                    valid_num_colors = True
                    if param in ['FITS_COLORS'] and len(color_set_filtered) == 0:
                        valid_num_colors = False
                    # Exactly three colors must be specified for RGB images
                    elif param in ['RGB_STIFF_COLORS', 'RGB_LUPTON_COLORS'] and len(color_set_filtered) != 3:
                        valid_num_colors = False
                    # Must have valid number of colors and not be a duplicate
                    if valid_num_colors and color_set_filtered not in color_sets_filtered_ordered:
                        color_sets_filtered_ordered.append(color_set_filtered)
                # Store in DataFrame as semi-colon-delimited list
                color_sets_filtered_ordered_str = ';'.join(color_sets_filtered_ordered)
                # If there are no valid color sets, fail if that output type is requested
                if not color_sets_filtered_ordered_str:
                    if (param == 'FITS_COLORS' and cutout['MAKE_FITS']) or (param == 'RGB_STIFF_COLORS' and cutout['MAKE_RGB_STIFF']) or (param == 'RGB_LUPTON_COLORS' and cutout['MAKE_RGB_LUPTON']):
                        logger.info('At least one valid "{}" color set is required.'.format(param))
                        sys.exit(1)
                df.at[row_index, param] = color_sets_filtered_ordered_str
    

    # Display processed DataFrame
    logger.info('\nProcessed DataFrame {}'.format(df))
    
    # Locally mounted directory containing tile data files
    if not conf['tiledir']:
        conf['tiledir'] = 'auto'
    elif conf['tiledir'] != 'auto' and not os.path.exists(conf['tiledir']):
        logger.info('tiledir path not found.')
        sys.exit(1)
    # Local directory where generated output files will be stored
    if not conf['outdir']:
        logger.info('outdir must be a path.')
        sys.exit(1)

    if conf['db'].upper() not in VALID_DATA_SOURCES:
        logger.info('Please select a valid database: {}.'.format(VALID_DATA_SOURCES))
        sys.exit(1)
    if conf['release'].upper() not in VALID_DATA_SOURCES[conf['db'].upper()]:
        logger.info('Please select a valid data release: {}.'.format(VALID_DATA_SOURCES[conf['db'].upper()]))
        sys.exit(1)
    if not conf['positions']: # or not os.path.exists(conf['positions']):
        logger.info('Please specify a valid CSV-formatted table file containing the cutout positions.')
        sys.exit(1)
    if not conf['username'] or not conf['password']:
        logger.info('A valid username and password must be provided.')
        sys.exit(1)
    if 'jobid' not in conf or not conf['jobid']:
        conf['jobid'] = str(uuid.uuid4())
    elif not isinstance(conf['jobid'], str):
        logger.info('jobid must be a string if specified')
        sys.exit(1)
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This program will make any number of cutouts, using the master tiles.")

    # Config file
    parser.add_argument('--config', type=str, required=True, help='Optional file to list all these arguments in and pass it along to bulkthumbs.')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    with open(args.config, 'r') as configfile:
        conf = yaml.load(configfile, Loader=yaml.FullLoader)

    run(conf)

    # # Object inputs
    # parser.add_argument('--csv', type=str, required=False, help='A CSV with columns \'COADD_OBJECT_ID \' or \'RA,DEC\'')
    # parser.add_argument('--ra', nargs='*', required=False, type=float, help='RA (decimal degrees)')
    # parser.add_argument('--dec', nargs='*', required=False, type=float, help='DEC (decimal degrees)')
    # parser.add_argument('--coadd', nargs='*', required=False, help='Coadd ID for exact object matching.')

    # # Jobs
    # parser.add_argument('--make_fits', action='store_true', help='Creates FITS files in the desired bands of the cutout region.')
    # parser.add_argument('--make_rgb_lupton', action='store_true', help='Creates 3-color images from the color bands you select, from reddest to bluest. This method uses the Lupton RGB combination method.')
    # parser.add_argument('--make_rgb_stiff', action='store_true', help='Creates 3-color images from the color bands you select, from reddest to bluest. This method uses the program STIFF to combine the images.')
    # parser.add_argument('--return_list', action='store_true', help='Saves list of inputted objects and their matched tiles to user directory.')

    # # Parameters
    # parser.add_argument('--xsize', default=1.0, help='Size in arcminutes of the cutout x-axis. Default: 1.0')
    # parser.add_argument('--ysize', default=1.0, help='Size in arcminutes of the cutout y-axis. Default: 1.0')
    # parser.add_argument('--colors_fits', default='I', type=str.upper, help='Color bands for the fits cutout. Default: i')
    # parser.add_argument('--colors_rgb', action='append', type=str.lower, metavar='R,G,B', help='Bands from which to combine the the RGB image, e.g.: z,r,g. Call multiple times for multiple colors combinations, e.g.: --colors_rgb z,r,g --colors_rgb z,i,r.')

    # # Lupton RGB Parameters
    # parser.add_argument('--rgb_minimum', default=1.0, help='The black point for the 3-color image. Default 1.0')
    # parser.add_argument('--rgb_stretch', default=50.0, help='The linear stretch of the image. Default 50.0.')
    # parser.add_argument('--rgb_asinh', default=10.0, help='The asinh softening parameter. Default 10.0')

    # # Database access and Bookkeeping
    # parser.add_argument('--db', default='DESSCI', type=str.upper, required=False, help='Which database to use. Default: DESSCI, Options: DESDR, DESSCI.')
    # parser.add_argument('--release', default='Y6A1', type=str.upper, required=False, help='Which data release to use. Default: Y6A1. Options: Y6A1, Y3A2, Y3A1, SVA1.')
    # parser.add_argument('--jobid', required=False, help='Option to manually specify a jobid for this job.')
    # parser.add_argument('--usernm', required=False, help='Username for database; otherwise uses values from desservices file.')
    # parser.add_argument('--passwd', required=False, help='Password for database; otherwise uses values from desservices file.')
    # parser.add_argument('--tiledir', required=False, help='Directory where tiles are stored.')
    # parser.add_argument('--outdir', required=False, help='Overwrite for output directory.')

    # parser.add_argument('--oracle_service_account_db', required=False, help='Oracle service account database name.')
    # parser.add_argument('--oracle_service_account_user', required=False, help='Oracle service account username.')
    # parser.add_argument('--oracle_service_account_pass', required=False, help='Oracle service account password.')
