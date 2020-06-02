import logging
import sys
import yaml
import os
import requests
import rpdb
from astropy.io import fits
import subprocess
import glob

STATUS_OK = 'ok'
STATUS_ERROR = 'error'

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def task_start(config):
    logging.info('Starting job: {}'.format(config['metadata']['name']))
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
    files = glob.glob(os.path.join(path, '*/*'))
    relpaths = []
    total_size = 0.0
    for file in files:
        relpaths.append(os.path.relpath(file, start='/home/worker/output/cutout'))
        total_size += os.path.getsize(file)
    response['files'] = relpaths
    response['sizes'] = total_size

    logging.info("Cutout response:\n{}".format(response))

    requests.post(
        '{}/job/complete'.format(config['metadata']['apiBaseUrl']),
        json={
            'apitoken': config['metadata']['apiToken'],
            'response': response
        }
    )


def execute_task(config):
    response = {
        'status': STATUS_OK,
        'msg': ''
    }

    if 'positions' in config['spec']:
        # Dump CSV-formatted data to a CSV file in working directory
        position_csv_file = 'positions.csv'
        with open(position_csv_file, 'w') as file:
            file.write(config['spec']['positions'].encode('utf-8').decode('unicode-escape'))
        config['spec']['csv'] = position_csv_file
        config['spec'].pop('positions', None)

    # Dump cutout config to YAML file in working directory
    cutout_config_file = 'cutout_config.yaml'
    with open(cutout_config_file, 'w') as file:
        yaml.dump(config['spec'], file)

    # TODO: replace hard-coded value with number of CPUs allocated to the k8s Job
    num_cpus = 1
    args = 'mpirun -n {} python3 bulkthumbs.py --config {}'.format(num_cpus, cutout_config_file)
    try:
        run_output = subprocess.check_output([args], shell=True)
    except subprocess.CalledProcessError as e:
        logging.info(e.output)

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

    return response

    return({'status':'ok','msg':'Execution complete'})

if __name__ == "__main__":

    # Make the cutout subdirectory if it does not already exist.
    os.makedirs('/home/worker/output/cutout', exist_ok=True)

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

    # Report to the JobHandler that the job is complete
    task_complete(config, response)
