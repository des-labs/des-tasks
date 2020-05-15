#!/usr/bin/env python3

"""
author: Landon Gelman, 2018-2020
author: Francisco Paz-Chinchon, 2019
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
from astropy import units as u
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs import utils
from astropy.visualization import make_lupton_rgb as mlrgb
from mpi4py import MPI as mpi
from PIL import Image

Image.MAX_IMAGE_PIXELS = 144000000        # allows Pillow to not freak out at a large filesize
ARCMIN_TO_DEG = 0.0166667        # deg per arcmin
VALID_DATABASES = ['DESDR','DESSCI']
VALID_RELEASES = ['Y6A1','Y3A2','Y1A1','SVA1']

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

def MakeRGBLupton(df, p, xs, ys, r, g, b, w, bp, s, q):
    pixelscale = utils.proj_plane_pixel_scales(w)
    dx = int(0.5 * xs * ARCMIN_TO_DEG / pixelscale[0])        # pixelscale is in degrees (CUNIT)
    dy = int(0.5 * ys * ARCMIN_TO_DEG / pixelscale[1])

    image = mlrgb(r, g, b, minimum=bp, stretch=s, Q=q)

    image = Image.fromarray(image, mode='RGB')
    image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)

    if 'XSIZE' in df and not np.isnan(df['XSIZE'][p]):
        udx = int(0.5 * df['XSIZE'][p] * ARCMIN_TO_DEG / pixelscale[0])
    else:
        udx = dx
    if 'YSIZE' in df and not np.isnan(df['YSIZE'][p]):
        udy = int(0.5 * df['YSIZE'][p] * ARCMIN_TO_DEG / pixelscale[0])
    else:
        udy = dy

    if image.size != (2*udx, 2*udy):
        issmaller = True
    else:
        issmaller = False

    return image, issmaller

def MakeRGB(tiledir, outdir, df, positions, xs, ys, colors, makestiff, makelupton, luptonargs):
    logger = logging.getLogger(__name__)

    for i in colors:
        c = i.split(',')

        if not os.path.exists(outdir):      # Nothing has been created. No color bands exist.
            # Call to MakeFitsCut with all colors
            size = u.Quantity((ys, xs), u.arcmin)
            MakeFitsCut(tiledir, outdir, size, positions, c, df)
        else:       # Outdir exists, now check if the right color bands exist.
            c2 = []
            if not glob.glob(os.path.join(outdir, '*_{}.fits'.format(c[0]))):        # Color band doesn't exist
                c2.append(c[0])            # append color to list to make
            if not glob.glob(os.path.join(outdir, '*_{}.fits'.format(c[1]))):        # Color band doesn't exist
                c2.append(c[1])            # append color to list to make
            if not glob.glob(os.path.join(outdir, '*_{}.fits'.format(c[2]))):        # Color band doesn't exist
                c2.append(c[2])            # append color to list to make

            if c2:        # Call to MakeFitsCut with necessary colors
                logger.info('MakeRGB - Some required color band fits files are missing so we will call MakeFitsCut.')
                size = u.Quantity((ys, xs), u.arcmin)
                MakeFitsCut(tiledir, outdir, size, positions, c2, df)

        # Iterate over each requested position
        for p in range(len(positions)):
            if 'COADD_OBJECT_ID' in df:
                nm = df['COADD_OBJECT_ID'][p]
                filenm = os.path.join(outdir, '{0}_{1}{2}{3}'.format(nm, c[0], c[1], c[2]))
            else:
                nm = 'DESJ' + _DecConverter(df['RA'][p], df['DEC'][p])
                filenm = os.path.join(outdir, '{0}_{1}{2}{3}'.format(nm, c[0], c[1], c[2]))

            try:
                file_r = glob.glob(os.path.join(outdir, '{0}_{1}.fits'.format(nm, c[0])))
            except IndexError:
                print('No FITS file in {0} band found for object {1}. Will not create RGB cutout.'.format(c[0], nm))
                logger.error('MakeLuptonRGB - No FITS file in {0} band found for {1}. Will not create RGB cutout.'.format(c[0], nm))
                continue
            else:
                r, header = fits.getdata(file_r[0], 'SCI', header=True)
                w = WCS(header)
            try:
                file_g = glob.glob(os.path.join(outdir, '{0}_{1}.fits'.format(nm, c[1])))
            except IndexError:
                print('No FITS file in {0} band found for object {1}. Will not create RGB cutout.'.format(c[1], nm))
                logger.error('MakeLuptonRGB - No FITS file in {0} band found for {1}. Will not create RGB cutout.'.format(c[1], nm))
                continue
            else:
                g = fits.getdata(file_g[0], 'SCI')
            try:
                file_b = glob.glob(os.path.join(outdir, '{0}_{1}.fits'.format(nm, c[2])))
            except IndexError:
                print('No FITS file in {0} band found for object {1}. Will not create RGB cutout.'.format(c[2], nm))
                logger.error('MakeLuptonRGB - No FITS file in {0} band found for {1}. Will not create RGB cutout.'.format(c[2], nm))
                continue
            else:
                b = fits.getdata(file_b[0], 'SCI')

            if makelupton:
                luptonnm = filenm + '_lupton'
                bp, s, q = luptonargs
                newimg, issmaller = MakeRGBLupton(df, p, xs, ys, r, g, b, w, bp, s, q)
                newimg.save(luptonnm+'.png', format='PNG')

                if issmaller:
                    logger.info('MakeLuptonRGB - {} is smaller than user requested. This is likely because the object/coordinate was in close proximity to the edge of a tile.'.format(('/').join(luptonnm.split('/')[-2:])))
                    print(('/').join(filenm.split('/')[-2:]))

            if makestiff:
                stiffnm = filenm + '_stiff'
                # Call STIFF using the 3 bands.
                fits_list = [file_r[0], file_g[0], file_b[0]]
                cmd_stiff = 'stiff {}'.format(' '.join(fits_list))
                cmd_stiff += ' -OUTFILE_NAME {}'.format(stiffnm+'.tiff')
                cmd_stiff = shlex.split(cmd_stiff)

                try:
                    subprocess.call(cmd_stiff)
                except OSError as e:
                    logger.error(e)

                # Convert the STIFF output from tiff to png and remove the tiff file.
                cmd_convert = 'convert {0} {1}'.format(stiffnm+'.tiff', stiffnm+'.png')
                cmd_convert = shlex.split(cmd_convert)
                try:
                    subprocess.call(cmd_convert)
                except OSError as e:
                    logger.error(e)
                try:
                    os.remove(stiffnm+'.tiff')
                except OSError as e:
                    logger.error(e)

    if makelupton:
        logger.info('MakeLuptonRGB - Tile {} complete.'.format(df['TILENAME'][0]))
    if makestiff:
        logger.info('MakeStiffRGB - Tile {} complete.'.format(df['TILENAME'][0]))

def MakeTiffCut(tiledir, outdir, positions, xs, ys, df, maketiff, makepngs):
    logger = logging.getLogger(__name__)
    os.makedirs(outdir, exist_ok=True)

    imgname = glob.glob(tiledir + '*.tiff')
    try:
        im = Image.open(imgname[0])
    except IndexError as e:
        print('No TIFF file found for tile ' + df['TILENAME'][0] + '. Will not create true-color cutout.')
        logger.error('MakeTiffCut - No TIFF file found for tile ' + df['TILENAME'][0] + '. Will not create true-color cutout.')
        return

    # try opening I band FITS (fallback on G, R bands)
    hdul = None
    for _i in ['i','g','r','z','Y']:
        tilename = glob.glob(tiledir+'*_{}.fits.fz'.format(_i))
        try:
            hdul = fits.open(tilename[0])
        except IOError as e:
            hdul = None
            logger.warning('MakeTiffCut - Could not find master FITS file: ' + tilename)
            continue
        else:
            break
    if not hdul:
        print('Cannot find a master fits file for this tile.')
        logger.error('MakeTiffCut - Cannot find a master fits file for this tile.')
        return

    w = WCS(hdul['SCI'].header)

    pixelscale = utils.proj_plane_pixel_scales(w)
    dx = int(0.5 * xs * ARCMIN_TO_DEG / pixelscale[0])        # pixelscale is in degrees (CUNIT)
    dy = int(0.5 * ys * ARCMIN_TO_DEG / pixelscale[1])

    pixcoords = utils.skycoord_to_pixel(positions, w, origin=0, mode='wcs')

    for i in range(len(positions)):
        if 'COADD_OBJECT_ID' in df:
            filenm = os.path.join(outdir, str(df['COADD_OBJECT_ID'][i]))
        else:
            #filenm = outdir + 'x{0}y{1}'.format(df['RA'][i], df['DEC'][i])     # for test purposes
            filenm = os.path.join(outdir, 'DESJ' + _DecConverter(df['RA'][i], df['DEC'][i]))

        if 'XSIZE' in df and not np.isnan(df['XSIZE'][i]):
            udx = int(0.5 * df['XSIZE'][i] * ARCMIN_TO_DEG / pixelscale[0])
        else:
            udx = dx
        if 'YSIZE' in df and not np.isnan(df['YSIZE'][i]):
            udy = int(0.5 * df['YSIZE'][i] * ARCMIN_TO_DEG / pixelscale[0])
        else:
            udy = dy

        left = int(max(0, pixcoords[0][i] - udx))
        upper = int(max(0, im.size[1] - pixcoords[1][i] - udy))
        right = int(min(pixcoords[0][i] + udx, 10000))
        lower = int(min(im.size[1] - pixcoords[1][i] + udy, 10000))
        newimg = im.crop((left, upper, right, lower))

        if maketiff:
            filenmtiff = filenm + '.tiff'
            newimg.save(filenmtiff, format='TIFF')
        if makepngs:
            filenmpng = filenm + '.png'
            newimg.save(filenmpng, format='PNG')
        if newimg.size != (2*udx, 2*udy):
            logger.info('MakeTiffCut - {} is smaller than user requested. This is likely because the object/coordinate was in close proximity to the edge of a tile.'.format(('/').join(filenm.split('/')[-2:])))
    logger.info('MakeTiffCut - Tile {} complete.'.format(df['TILENAME'][0]))

def MakeFitsCut(tiledir, outdir, size, positions, colors, df):
    logger = logging.getLogger(__name__)
    os.makedirs(outdir, exist_ok=True)            # Check if outdir exists

    for c in range(len(colors)):        # Iterate over all desired colors
        # Finish the tile's name and open the file. Camel-case check is required because Y band is always capitalized.
        if colors[c] == 'Y' or colors[c] == 'y':
            tilename = glob.glob(tiledir + '*_{}.fits.fz'.format(colors[c].upper()))
        else:
            tilename = glob.glob(tiledir + '*_{}.fits.fz'.format(colors[c].lower()))
        try:
            hdul = fits.open(tilename[0])
        except IndexError as e:
            print('No FITS file in {0} color band found. Will not create cutouts in this band.'.format(colors[c]))
            logger.error('MakeFitsCut - No FITS file in {0} color band found. Will not create cutouts in this band.'.format(colors[c]))
            continue        # Just go on to the next color in the list

        for p in range(len(positions)):            # Iterate over all inputted coordinates
            if 'COADD_OBJECT_ID' in df:
                filenm = os.path.join(outdir, '{0}_{1}.fits'.format(df['COADD_OBJECT_ID'][p], colors[c].lower()))
            else:
                #filenm = outdir + 'x{0}y{1}_{2}.fits'.format(df['RA'][p], df['DEC'][p], colors[c].lower())
                filenm = os.path.join(outdir, 'DESJ' + _DecConverter(df['RA'][p], df['DEC'][p]) + '_{}.fits'.format(colors[c].lower()))

            newhdul = fits.HDUList()
            pixelscale = None

            if 'XSIZE' in df or 'YSIZE' in df:
                if 'XSIZE' in df and not np.isnan(df['XSIZE'][p]):
                    uxsize = df['XSIZE'][p] * u.arcmin
                else:
                    uxsize = size[1]
                if 'YSIZE' in df and not np.isnan(df['YSIZE'][p]):
                    uysize = df['YSIZE'][p] * u.arcmin
                else:
                    uysize = size[0]
                usize = u.Quantity((uysize, uxsize))
            else:
                usize = size

            # Iterate over all HDUs in the tile
            for i in range(len(hdul)):
                if hdul[i].name == 'PRIMARY':
                    continue

                h = hdul[i].header
                data = hdul[i].data
                header = h.copy()
                w=WCS(header)

                cutout = Cutout2D(data, positions[p], usize, wcs=w, mode='trim')
                crpix1, crpix2 = cutout.position_cutout
                x, y = cutout.position_original
                crval1, crval2 = w.wcs_pix2world(x, y, 1)

                header['CRPIX1'] = crpix1
                header['CRPIX2'] = crpix2
                header['CRVAL1'] = float(crval1)
                header['CRVAL2'] = float(crval2)
                header['HIERARCH RA_CUTOUT'] = df['RA'][p]
                header['HIERARCH DEC_CUTOUT'] = df['DEC'][p]

                if not newhdul:
                    newhdu = fits.PrimaryHDU(data=cutout.data, header=header)
                    pixelscale = utils.proj_plane_pixel_scales(w)
                else:
                    newhdu = fits.ImageHDU(data=cutout.data, header=header, name=h['EXTNAME'])
                newhdul.append(newhdu)

            if pixelscale is not None:
                dx = int(usize[1] * ARCMIN_TO_DEG / pixelscale[0] / u.arcmin)        # pixelscale is in degrees (CUNIT)
                dy = int(usize[0] * ARCMIN_TO_DEG / pixelscale[1] / u.arcmin)
                if (newhdul[0].header['NAXIS1'], newhdul[0].header['NAXIS2']) != (dx, dy):
                    logger.info('MakeFitsCut - {} is smaller than user requested. This is likely because the object/coordinate was in close proximity to the edge of a tile.'.format(('/').join(filenm.split('/')[-2:])))

            newhdul.writeto(filenm, output_verify='exception', overwrite=True, checksum=False)
            newhdul.close()
    logger.info('MakeFitsCut - Tile {} complete.'.format(df['TILENAME'][0]))

def run(args):
    # If a config is provided, process the input arguments accordingly. Otherwise, return `args` as-is.
    args = import_arguments_from_config(args)
    # Validate the input arguments
    validate_arguments(args)
    logger = logging.getLogger(__name__)
    logger.info(args)
    if rank == 0:
        if args.db == 'DESDR':
            db = 'desdr'
            uu = args.usernm    #DR1_UU
            pp = args.passwd    #DR1_PP
            conn = ea.connect(db, user=uu, passwd=pp)
        elif args.db == 'DESSCI':
            db = 'dessci'
            uu = args.usernm    #DR1_UU
            pp = args.passwd    #DR1_PP
            conn = ea.connect(db, user=uu, passwd=pp)

        curs = conn.cursor()

        usernm = str(conn.user)
        if args.jobid:
            jobid = args.jobid
        else:
            jobid = str(uuid.uuid4())

        outdir = os.path.join(args.outdir, '') #, + usernm + '/' + jobid + '/'        # use for local
        #outdir = OUTDIR        # use for desaccess

        # Remove this try/except block for desaccess. Keep for local.
        try:
            os.makedirs(outdir, exist_ok=False)
        except OSError as e:
            print(e)
            print('Specified jobid already exists in output directory. Aborting job.')
            conn.close()
            sys.stdout.flush()
            comm.Abort()
    else:
        usernm, jobid, outdir = None, None, None

    usernm, jobid, outdir = comm.bcast([usernm, jobid, outdir], root=0)

    logtime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logname = os.path.join(outdir, 'BulkThumbs_' + logtime + '.log')         # use for local
    #logname = outdir + 'log.log'       # use for des-labs
    formatter = logging.Formatter('%(asctime)s - '+str(rank)+' - %(levelname)-8s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    fh = MPILogHandler(logname, comm)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    xs = float(args.xsize)
    ys = float(args.ysize)
    colors = args.colors_fits.split(',')

    if rank == 0:
        summary = {}
        start = time.time()

        logger.info('Selected Options:')

        # This puts any input type into a pandas dataframe
        if args.csv:
            userdf = pd.DataFrame(pd.read_csv(args.csv))
            logger.info('    CSV: '+args.csv)
            summary['csv'] = args.csv
        elif args.ra:
            coords = {}
            coords['RA'] = args.ra
            coords['DEC'] = args.dec
            userdf = pd.DataFrame.from_dict(coords, orient='columns')
            logger.info('    RA: '+str(args.ra))
            logger.info('    DEC: '+str(args.dec))
            summary['ra'] = str(args.ra)
            summary['dec'] = str(args.dec)
        elif args.coadd:
            coadds = {}
            coadds['COADD_OBJECT_ID'] = args.coadd
            userdf = pd.DataFrame.from_dict(coadds, orient='columns')
            logger.info('    CoaddID: '+str(args.coadd))
            summary['coadd'] = str(args.coadd)

        logger.info('    X size: '+str(args.xsize))
        logger.info('    Y size: '+str(args.ysize))
        logger.info('    Make TIFFs? '+str(args.make_tiffs))
        logger.info('    Make PNGs? '+str(args.make_pngs))
        logger.info('    Make FITS? '+str(args.make_fits))
        logger.info('    Make RGBs (Lupton)? {}'.format('True' if args.make_rgb_lupton else 'False'))
        logger.info('    Make RGBs (STIFF)? {}'.format('True' if args.make_rgb_stiff else 'False'))
        summary['xsize'] = str(args.xsize)
        summary['ysize'] = str(args.ysize)
        summary['make_tiffs'] = str(args.make_tiffs)
        summary['make_pngs'] = str(args.make_pngs)
        summary['make_fits'] = str(args.make_fits)
        summary['make_rgb_lupton'] = 'True' if args.make_rgb_lupton else 'False'
        summary['make_rgb_stiff'] = 'True' if args.make_rgb_stiff else 'False'
        if args.make_fits:
            logger.info('        Bands (fits): '+args.colors_fits)
            summary['bands_fits'] = args.colors_fits
        if args.make_rgb_lupton or args.make_rgb_stiff:
            logger.info('        Bands (rgb): '+str(args.colors_rgb))
            summary['bands_rgb'] = args.colors_rgb
        summary['db'] = args.db

        df = pd.DataFrame()
        unmatched_coords = {'RA':[], 'DEC':[]}
        unmatched_coadds = []

        logger.info('User: ' + usernm)
        logger.info('JobID: ' + str(jobid))
        summary['user'] = usernm
        summary['jobid'] = str(jobid)

        tablename = 'BTL_'+jobid.upper().replace("-","_")    # "BulkThumbs_List_<jobid>"
        tablename_csv_filepath = os.path.join(outdir, tablename+'.csv')
        if 'RA' in userdf:
            ra_adjust = [360-userdf['RA'][i] if userdf['RA'][i]>180 else userdf['RA'][i] for i in range(len(userdf['RA']))]
            userdf = userdf.assign(RA_ADJUSTED = ra_adjust)
            userdf.to_csv(tablename_csv_filepath, index=False)
            conn.load_table(tablename_csv_filepath, name=tablename)

            #query = "select temp.RA, temp.DEC, temp.RA_ADJUSTED, temp.RA as ALPHAWIN_J2000, temp.DEC as DELTAWIN_J2000, m.TILENAME from {} temp left outer join Y3A2_COADDTILE_GEOM m on (m.CROSSRA0='N' and (temp.RA between m.URAMIN and m.URAMAX) and (temp.DEC between m.UDECMIN and m.UDECMAX)) or (m.CROSSRA0='Y' and (temp.RA_ADJUSTED between m.URAMIN-360 and m.URAMAX) and (temp.DEC between m.UDECMIN and m.UDECMAX))".format(tablename)
            query = "select temp.RA, temp.DEC, temp.RA_ADJUSTED, temp.RA as ALPHAWIN_J2000, temp.DEC as DELTAWIN_J2000, m.TILENAME"
            if 'XSIZE' in userdf:
                query += ", temp.XSIZE"
            if 'YSIZE' in userdf:
                query += ", temp.YSIZE"
            if args.db == 'DESSCI':
                catalog = 'Y3A2_COADDTILE_GEOM'
            elif args.db == 'DESDR':
                catalog = 'DR1_Tile_INFO'
            query += " from {0} temp left outer join {1} m on (m.CROSSRA0='N' and (temp.RA between m.URAMIN and m.URAMAX) and (temp.DEC between m.UDECMIN and m.UDECMAX)) or (m.CROSSRA0='Y' and (temp.RA_ADJUSTED between m.URAMIN-360 and m.URAMAX) and (temp.DEC between m.UDECMIN and m.UDECMAX))".format(tablename, catalog)

            df = conn.query_to_pandas(query)
            curs.execute('drop table {}'.format(tablename))
            os.remove(tablename_csv_filepath)

            df = df.replace('-9999',np.nan)
            df = df.replace(-9999.000000,np.nan)
            #dftemp = df[df.isnull().any(axis=1)]
            dftemp = df[ (df['TILENAME'].isnull()) ]
            unmatched_coords['RA'] = dftemp['RA'].tolist()
            unmatched_coords['DEC'] = dftemp['DEC'].tolist()
            df = df.dropna(axis=0, how='any', subset=['TILENAME'])

            logger.info('Unmatched coordinates: \n{0}\n{1}'.format(unmatched_coords['RA'], unmatched_coords['DEC']))
            summary['Unmatched_Coords'] = unmatched_coords
            print(unmatched_coords)

        if 'COADD_OBJECT_ID' in userdf:
            userdf.to_csv(tablename_csv_filepath, index=False)
            conn.load_table(tablename_csv_filepath, name=tablename)

            #query = "select temp.COADD_OBJECT_ID, m.ALPHAWIN_J2000, m.DELTAWIN_J2000, m.RA, m.DEC, m.TILENAME from {} temp left outer join Y3A2_COADD_OBJECT_SUMMARY m on temp.COADD_OBJECT_ID=m.COADD_OBJECT_ID".format(tablename)
            query = "select temp.COADD_OBJECT_ID, m.ALPHAWIN_J2000, m.DELTAWIN_J2000, m.RA, m.DEC, m.TILENAME"
            if 'XSIZE' in userdf:
                query += ", temp.XSIZE"
            if 'YSIZE' in userdf:
                query += ", temp.YSIZE"
            if args.db == 'DESSCI':
                catalog = 'Y3A2_COADD_OBJECT_SUMMARY'
            elif args.db == 'DESDR':
                catalog = 'DR1_MAIN'
            query += " from {0} temp left outer join {1} m on temp.COADD_OBJECT_ID=m.COADD_OBJECT_ID".format(tablename, catalog)

            df = conn.query_to_pandas(query)
            curs.execute('drop table {}'.format(tablename))
            os.remove(tablename_csv_filepath)

            df = df.replace('-9999',np.nan)
            df = df.replace(-9999.000000,np.nan)
            #dftemp = df[df.isnull().any(axis=1)]
            dftemp = df[ (df['TILENAME'].isnull()) | (df['ALPHAWIN_J2000'].isnull()) | (df['DELTAWIN_J2000'].isnull()) | (df['RA'].isnull()) | (df['DEC'].isnull()) ]
            unmatched_coadds = dftemp['COADD_OBJECT_ID'].tolist()
            df = df.dropna(axis=0, how='any', subset=['TILENAME','ALPHAWIN_J2000','DELTAWIN_J2000','RA','DEC'])

            logger.info('Unmatched coadd ID\'s: \n{}'.format(unmatched_coadds))
            summary['Unmatched_Coadds'] = unmatched_coadds
            print(unmatched_coadds)

        conn.close()
        df = df.sort_values(by=['TILENAME'])
        df = df.drop_duplicates(['RA','DEC'], keep='first')

        # TODO: Move the check for `return_list` into the RA/DEC and Coadd if statements above to
        # avoid deleting and recreating the CSV file unnecessarily
        if args.return_list:
            os.makedirs(outdir, exist_ok=True)
            df.to_csv(tablename_csv_filepath, index=False)

        df = np.array_split(df, nprocs)

        end1 = time.time()
        query_elapsed = '{0:.2f}'.format(end1-start)
        print('Querying took (s): ' + query_elapsed)
        logger.info('Querying took (s): ' + query_elapsed)
        summary['query_time'] = query_elapsed

    else:
        df = None

    df = comm.scatter(df, root=0)

    tilenm = df['TILENAME'].unique()
    ###################### from bulkthumbs2.py ################################
    conn_temp = ea.connect('dessci', user=args.usernm, passwd=args.passwd)
    qtemplate = "select FITS_IMAGES from {} where tilename = '{}' and band = 'i'"
    table_path = "MCARRAS2.{}_TILE_PATH_INFO".format(args.release)
    ###################### from bulkthumbs2.py ################################
    for i in tilenm:
        if args.tiledir != 'auto':
            tiledir = os.path.join(args.tiledir, i)
        else:
            ###################### from bulkthumbs2.py ################################
            dftile = conn_temp.query_to_pandas(qtemplate.format(table_path, i))
            tiledir = os.path.dirname(dftile.FITS_IMAGES.iloc[0])
            if args.release in ('Y6A1', 'Y3A2'):
                tiledir = tiledir.replace('https://desar2.cosmology.illinois.edu/DESFiles/desarchive/OPS/', '/des003/desarchive/') + '/'
            if args.release in ('SVA1', 'Y1A1'):
                tiledir = tiledir.replace('https://desar2.cosmology.illinois.edu/DESFiles/desardata/OPS/coadd/', '/des004/coadd/') + '/'
            logger.info('Using DB and table {} to determine paths...'.format(table_path))
            ###################### from bulkthumbs2.py ################################
        tiledir = os.path.join(tiledir, '')
        udf = df[ df.TILENAME == i ]
        udf = udf.reset_index()

        size = u.Quantity((ys, xs), u.arcmin)
        positions = SkyCoord(udf['ALPHAWIN_J2000'], udf['DELTAWIN_J2000'], frame='icrs', unit='deg', equinox='J2000', representation_type='spherical')

        if args.make_tiffs or args.make_pngs:
            MakeTiffCut(tiledir, os.path.join(outdir, i), positions, xs, ys, udf, args.make_tiffs, args.make_pngs)

        if args.make_fits:
            MakeFitsCut(tiledir, os.path.join(outdir, i), size, positions, colors, udf)

        if args.make_rgb_lupton or args.make_rgb_stiff:
            luptonargs = [float(args.rgb_minimum), float(args.rgb_stretch), float(args.rgb_asinh)]
            MakeRGB(tiledir, os.path.join(outdir, i), udf, positions, xs, ys, args.colors_rgb, args.make_rgb_stiff, args.make_rgb_lupton, luptonargs)

    ###################### from bulkthumbs2.py ################################
    conn_temp.close()
    ###################### from bulkthumbs2.py ################################
    comm.Barrier()

    if rank == 0:
        end2 = time.time()
        processing_time = '{0:.2f}'.format(end2-end1)
        print('Processing took (s): ' + processing_time)
        logger.info('Processing took (s): ' + processing_time)
        summary['processing_time'] = processing_time

        dirsize = getPathSize(outdir)
        dirsize = dirsize * 1. / 1024
        if dirsize > 1024. * 1024:
            dirsize = '{0:.2f} GB'.format(1. * dirsize / 1024. / 1024)
        elif dirsize > 1024.:
            dirsize = '{0:.2f} MB'.format(1. * dirsize / 1024.)
        else:
            dirsize = '{0:.2f} KB'.format(dirsize)

        logger.info('All processes finished.')
        logger.info('Total file size on disk: {}'.format(dirsize))
        summary['size_on_disk'] = str(dirsize)

        files = glob.glob(os.path.join(outdir, '*/*'))
        logger.info('Total number of files: {}'.format(len(files)))
        summary['number_of_files'] = len(files)
        files = [i.split('/')[-2:] for i in files]
        files = [('/').join(i) for i in files]
        if 'COADD_OBJECT_ID' in userdf:
            files = [i.split('.')[-2] for i in files]
        else:
            files = [('.').join(i.split('.')[-4:-1]) for i in files]
        files = [i.split('_')[0] for i in files]
        files = list(set(files))
        summary['files'] = files

        jsonfile = os.path.join(outdir, 'BulkThumbs_'+logtime+'_SUMMARY.json')
        with open(jsonfile, 'w') as fp:
            json.dump(summary, fp)

def import_arguments_from_config(args):
    # If the --config argument is provided, import the rest of the arguments from
    # the specified YAML file
    if args.config:
        arg_list = []
        with open(args.config, 'r') as configfile:
            config = yaml.load(configfile)
            for key, value in config.items():
                if value is True:
                    arg_list.append('--{}'.format(key))
                elif value is False:
                    continue
                else:
                    arg_list.append('--{}'.format(key))
                    arg_list.append('{}'.format(value))
        args = parser.parse_args(args=arg_list)
    return args

def validate_arguments(args):
    if not args.tiledir or not args.outdir:
        try:
            with open('config/bulkthumbsconfig.yaml','r') as cfile:
                conf = yaml.load(cfile)
            # TODO: use os library to build path strings
            if not args.tiledir:
                args.tiledir = os.path.join(conf['directories']['tiles'], '')
            if not args.outdir:
                args.outdir = os.path.join(conf['directories']['outdir'], '')
        except FileNotFoundError:
            print('Bulkthumbs config file not found. Either no --tiledir or no --outdir set.')
            sys.exit(1)

    #DR1_UU = conf['dr1_user']['usernm']
    #DR1_PP = conf['dr1_user']['passwd']

    if args.db not in VALID_DATABASES:
        print('Please select a valid database: {}.'.format(VALID_DATABASES))
        sys.exit(1)
    if args.db == 'DR1' and not (args.usernm and args.passwd):
        print('Please include the --usernm and --passwd to use the DR1 database.')
        sys.exit(1)
    if args.release not in VALID_RELEASES:
        print('Please select a valid data release: {}.'.format(VALID_RELEASES))
        sys.exit(1)
    if not args.csv and not (args.ra and args.dec) and not args.coadd:
        print('Please include either RA/DEC coordinates or Coadd IDs.')
        sys.exit(1)
    if (args.ra and args.dec) and len(args.ra) != len(args.dec):
        print('Remember to have the same number of RA and DEC values when using coordinates.')
        sys.exit(1)
    if (args.ra and not args.dec) or (args.dec and not args.ra):
        print('Please include BOTH RA and DEC if not using Coadd IDs.')
        sys.exit(1)
    if not args.make_tiffs and not args.make_pngs and not args.make_fits and not args.make_rgb_lupton and not args.make_rgb_stiff and not args.return_list:
        print('Nothing to do. Please select either/all make_tiff, make_fits, make_rgb_lupton, or make_rgb_stiff.')
        sys.exit(1)
    if (args.make_rgb_stiff or args.make_rgb_lupton) and not args.colors_rgb:
        print('Please include --colors_rgb when calling --make_rgb_stiff or --make_rgb_lupton')
        sys.exit(1)

    # We don't need a catch for not including args.colors_fits because we included a default value, i band, in the argument declaration.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This program will make any number of cutouts, using the master tiles.")

    # Config file
    parser.add_argument('--config', type=str, required=False, help='Optional file to list all these arguments in and pass it along to bulkthumbs.')

    # Object inputs
    parser.add_argument('--csv', type=str, required=False, help='A CSV with columns \'COADD_OBJECT_ID \' or \'RA,DEC\'')
    parser.add_argument('--ra', nargs='*', required=False, type=float, help='RA (decimal degrees)')
    parser.add_argument('--dec', nargs='*', required=False, type=float, help='DEC (decimal degrees)')
    parser.add_argument('--coadd', nargs='*', required=False, help='Coadd ID for exact object matching.')

    # Jobs
    parser.add_argument('--make_tiffs', action='store_true', help='Creates a TIFF file of the cutout region.')
    parser.add_argument('--make_fits', action='store_true', help='Creates FITS files in the desired bands of the cutout region.')
    parser.add_argument('--make_pngs', action='store_true', help='Creates a PNG file of the cutout region.')
    parser.add_argument('--make_rgb_lupton', action='store_true', help='Creates 3-color images from the color bands you select, from reddest to bluest. This method uses the Lupton RGB combination method.')
    parser.add_argument('--make_rgb_stiff', action='store_true', help='Creates 3-color images from the color bands you select, from reddest to bluest. This method uses the program STIFF to combine the images.')
    parser.add_argument('--return_list', action='store_true', help='Saves list of inputted objects and their matched tiles to user directory.')

    # Parameters
    parser.add_argument('--xsize', default=1.0, help='Size in arcminutes of the cutout x-axis. Default: 1.0')
    parser.add_argument('--ysize', default=1.0, help='Size in arcminutes of the cutout y-axis. Default: 1.0')
    parser.add_argument('--colors_fits', default='I', type=str.upper, help='Color bands for the fits cutout. Default: i')
    parser.add_argument('--colors_rgb', action='append', type=str.lower, metavar='R,G,B', help='Bands from which to combine the the RGB image, e.g.: z,r,g. Call multiple times for multiple colors combinations, e.g.: --colors_rgb z,r,g --colors_rgb z,i,r.')

    # Lupton RGB Parameters
    parser.add_argument('--rgb_minimum', default=1.0, help='The black point for the 3-color image. Default 1.0')
    parser.add_argument('--rgb_stretch', default=50.0, help='The linear stretch of the image. Default 50.0.')
    parser.add_argument('--rgb_asinh', default=10.0, help='The asinh softening parameter. Default 10.0')

    # Database access and Bookkeeping
    parser.add_argument('--db', default='DESSCI', type=str.upper, required=False, help='Which database to use. Default: DESSCI, Options: DESDR, DESSCI.')
    parser.add_argument('--release', default='Y6A1', type=str.upper, required=False, help='Which data release to use. Default: Y6A1. Options: Y6A1, Y3A2, Y3A1, SVA1.')
    parser.add_argument('--jobid', required=False, help='Option to manually specify a jobid for this job.')
    parser.add_argument('--usernm', required=False, help='Username for database; otherwise uses values from desservices file.')
    parser.add_argument('--passwd', required=False, help='Password for database; otherwise uses values from desservices file.')
    parser.add_argument('--tiledir', required=False, help='Directory where tiles are stored.')
    parser.add_argument('--outdir', required=False, help='Overwrite for output directory.')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    run(args)
