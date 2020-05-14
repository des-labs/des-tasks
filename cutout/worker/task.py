import logging
import sys
import yaml
import os
import requests
import rpdb
#import bulkthumbs2
from astropy.io import fits
import subprocess

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
