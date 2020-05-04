import logging
import sys
import yaml
import os
import requests
import rpdb
import bulkthumbs2

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
        'ra': 20.0035,
        'dec': -26.767,
        'make_fits': 'gri',
        'xsize': 5,
        'ysize': 3,
        'db': 'DESSCI',
        'colors': 'g,r,i',
        'release': 'Y6A1',
        'outdir': 'output',
        'usernm': config['metadata']['username'],
        'passwd': config['metadata']['password']
        }
    """
    args_dict = {
        'ra': config['inputs']['ra'],
        'dec': config['inputs']['dec'],
        'make_fits': config['inputs']['make_fits'],
        'xsize': config['inputs']['xsize'],
        'ysize': config['inputs']['ysize'],
        'return_list': config['inputs']['return_list'],
        'db': config['inputs']['db'],
        'usernm': config['metadata']['username'],
        'passwd': config['metadata']['password'],
    }
    """
        
    bulkthumbs2.run(dotdict(args_dict))
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
