import logging
import sys
import yaml
import os
import requests
import rpdb
from astropy.io import fits
import subprocess
import glob
import shutil
import json
import time

STATUS_OK = 'ok'
STATUS_ERROR = 'error'

logger = logging.getLogger(__name__)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def task_start(config):
    logger.info('Starting job: {}'.format(config['metadata']['name']))
    requests.post(
        '{}/job/start'.format(config['metadata']['apiBaseUrl']),
        json={
            'apitoken': config['metadata']['apiToken']
        }
    )


def task_complete(config, response):
    # Report that work has completed
    # If no errors have occurred already, parse the job summary file for file info
    path = config['spec']['outdir']
    files = glob.glob(os.path.join(path, '*/*/*'))
    relpaths = []
    total_size = 0.0
    for file in files:
        relpaths.append(os.path.relpath(file, start=config['cutout_dir']))
        total_size += os.path.getsize(file)
    response['files'] = relpaths
    response['sizes'] = total_size
    response['summary'] = {}
    try:
        with open(os.path.join(path, 'summary.json'), 'r') as summary_file:
            response['summary'] = json.load(summary_file)
    except Exception as e:
        logger.error('Error loading "summary.json": {}'.format(str(e).strip()))

    # logger.info("Cutout response:\n{}".format(response))

    fail_count = 0
    while fail_count <= 5:
        completeResponse = requests.post(
            '{}/job/complete'.format(config['metadata']['apiBaseUrl']),
            json={
                'apitoken': config['metadata']['apiToken'],
                'response': response
            }
        )
        
        if completeResponse.status_code == 200:
            break
        else:
            fail_count += 1
            time.sleep(1)


def execute_task(config):
    response = {
        'status': STATUS_OK,
        'msg': ''
    }

    # Dump cutout config to YAML file in working directory
    cutout_config_file = 'cutout_config.yaml'
    with open(cutout_config_file, 'w') as file:
        yaml.dump(config['spec'], file)

    # Launch the cutout generation using the Message Passing Interface (MPI) system to parallelize execution
    num_cpus = config['spec']['num_cpus'] if 'num_cpus' in config['spec'] and isinstance(config['spec']['num_cpus'], int) else 1
    args = 'mpirun -n {} python3 bulkthumbs.py --config {}'.format(num_cpus, cutout_config_file)
    try:
        run_output = subprocess.check_output([args], shell=True)
    except subprocess.CalledProcessError as e:
        logger.error(e.output)
        response['status'] = STATUS_ERROR
        response['msg'] = e.output

    # Verifying outputs
    path = config['spec']['outdir']
    for file in os.listdir(path):
        if file.endswith(".fits"):
            try:
                fullpath = os.path.join(path, file)
                hdus = fits.open(fullpath,checksum=True)
                hdus.verify()
            except:
                response['status'] = STATUS_ERROR
                response['msg'] = 'FITS file not found'
                return response

    # Generate compressed archive file
    try:
        root_dir = config['cutout_dir']
        base_dir = '{}'.format(config['metadata']['jobId'])
        logger.info('Generating archive file for directory "{}" in "{}"'.format(base_dir, root_dir))
        shutil.make_archive(
            '{}/{}'.format(root_dir, config['metadata']['jobId']),
            'gztar',
            root_dir=root_dir, base_dir=base_dir,
            logger=logging
        )
    except Exception as e:
        response['status'] = STATUS_ERROR
        response['msg'] = str(e).strip()

    return response

def run(config, user_dir='/home/worker/output'):

    # Make the cutout subdirectory if it does not already exist.
    cutout_dir = os.path.join(user_dir, 'cutout')
    os.makedirs(cutout_dir, exist_ok=True)
    config['cutout_dir'] = cutout_dir

    # Configure logging
    formatter = logging.Formatter("%(asctime)s  %(name)8s  %(levelname)5s  %(message)s")
    fh = logging.FileHandler(config['metadata']['log'])
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    # Report to the JobHandler that the job has begun
    task_start(config)

    # Trigger debugging if activated and pause execution
    debug_loop = config['metadata']['debug']
    if debug_loop:
        logger.info('Debugging is enabled. Invoking RPDB...')
        rpdb.set_trace()

    # Execute the primary task
    response = execute_task(config)
    # The `debug_loop` variable can be set to false using the interactive debugger to break the loop
    while debug_loop == True:
        response = execute_task(config)

    # Report to the JobHandler that the job is complete
    task_complete(config, response)


if __name__ == "__main__":

    # Import job configuration
    try:
       input_file = sys.argv[1]
    except:
        input_file = 'configjob.yaml'
    with open(input_file) as cfile:
        config = yaml.safe_load(cfile)

    run(config)
