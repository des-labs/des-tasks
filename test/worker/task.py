import time
import logging
import sys
import yaml
import os
import requests

try:
   input_file = sys.argv[1]
except:
    input_file = 'configjob.yaml'

with open(input_file) as cfile:
    config = yaml.safe_load(cfile)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(config['metadata']['log']),
        logging.StreamHandler()
    ]
)

t = config['spec']['inputs']['time']
logging.info('********')
logging.info('Running  job:{} at {}'.format(
   config['metadata']['name'], os.path.basename(__file__)))
logging.debug("This is a debug message")
logging.info("Working... for  {} seconds".format(t))
time.sleep(t)
logging.info("Reporting completion to jobhandler (apitoken: {})...".format(config['metadata']['apiToken']))

requests.post(
    '{}/job/complete'.format(config['metadata']['apiBaseUrl']),
    json={
        'apitoken': config['metadata']['apiToken'],
        'response': 'Done in {} seconds'.format(t)
    }
)

logging.info("Done!".format(t))
