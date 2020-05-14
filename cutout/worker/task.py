import logging
import sys
import yaml
import os
import requests
import rpdb
import bulkthumbs2
from astropy.io import fits

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def task_start(config):
    logging.info('Running  job:{} at {}'.format(
       config['metadata']['name'], os.path.basename(__file__)))

    logging.info("Reporting job start to jobhandler (apitoken: {})...".format(
       config['metadata']['apiToken']))
    requests.post(
        '{}/job/start'.format(config['metadata']['apiBaseUrl']),
        json={
            'apitoken': config['metadata']['apiToken']
        }
    )


def task_complete(config, response):
    # Report that work has completed
    logging.info("Reporting completion to jobhandler (apitoken: {})...".format(
        config['metadata']['apiToken']))
    requests.post(
        '{}/job/complete'.format(config['metadata']['apiBaseUrl']),
        json={
            'apitoken': config['metadata']['apiToken'],
            'response': response
        }
    )


def execute_task(config):
    logging.info('Executing bulkthumbs2.py:')
    args_dict = {
        # ra: array of floats
        'ra': config['spec']['inputs']['ra'],
        # dec: array of floats
        'dec': config['spec']['inputs']['dec'],
        # coad: array of integers
        'coadd': config['spec']['inputs']['coadd'],
        # release: atomic string
        'release': config['spec']['inputs']['release'],
        # make_fits: boolean
        'make_fits': config['spec']['inputs']['make_fits'],
        # 'make_tiffs': config['spec']['inputs']['make_tiffs'],
        # 'make_pngs': config['spec']['inputs']['make_pngs'],
        # 'make_rgbs': config['spec']['inputs']['make_rgbs'],
        # rgb_values:
        # 'rgb_values': config['spec']['inputs']['make_fits'],
        # xsize: float
        'xsize': config['spec']['inputs']['xsize'],
        # ysize: float
        'ysize': config['spec']['inputs']['ysize'],
        # return_list: boolean
        # 'return_list': config['spec']['inputs']['return_list'],
        # db: atomic string
        'db': config['spec']['inputs']['db'],
        # colors: CSV-formatted string
        'colors': config['spec']['inputs']['colors'],
        # colors_stiff: CSV-formatted string
        # 'colors_stiff': config['spec']['inputs']['colors_stiff'],
        'outdir': '/home/worker/output/{}'.format(config['metadata']['jobId']),
        'usernm': config['metadata']['username'],
        'passwd': config['metadata']['password'],
        'jobid': config['metadata']['jobId']
        }

    bulkthumbs2.run(dotdict(args_dict))

    # Verifying outputs
    path = '/home/worker/output/{}'.format(config['metadata']['jobId'])
    for file in os.listdir(path):
        if file.endswith(".fits"):
            try:
                fullpath = path + file
                hdus = fits.open(fullpath,checksum=True)
                hdus.verify()
            except:
                return({'status':'error','msg':'Execution complete'})

    return({'status':'ok','msg':'Execution complete'})

if __name__ == "__main__":

    # Import job configuration
    try:
       input_file = sys.argv[1]
    except:
        input_file = 'configjob.yaml'
    with open(input_file) as cfile:
        config = yaml.safe_load(cfile)

    # Initialize info and error logging
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(config['metadata']['log']),
            logging.StreamHandler()
        ]
    )

    # Report to the JobHandler that the job has begun
    task_start(config)

    # Trigger debugging if activated and pause execution
    debug_loop = config['metadata']['debug']
    if debug_loop:
        logging.info('Debugging is enabled. Invoking RPDB...')
        rpdb.set_trace()

    # Execute the primary task
    response = execute_task(config)
    # The `debug_loop` variable can be set to false using the interactive debugger to break the loop
    while debug_loop == True:
        response = execute_task(config)

    logging.info("Database query response:\n{}".format(response))

    # Report to the JobHandler that the job is complete
    task_complete(config, response)
