import logging
import sys
import yaml
import os
import requests
import ea_tasks

try:
   input_file = sys.argv[1]
except:
    input_file = 'configjob.yaml'

with open(input_file) as cfile:
    config = yaml.safe_load(cfile)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(config['spec']['outputs']['log']),
        logging.StreamHandler()
    ]
)

# Report that work has started
logging.info("Reporting job start to jobhandler (apitoken: {})...".format(config['metadata']['apiToken']))
requests.post(
    '{}/job/start'.format(config['metadata']['apiBaseUrl']),
    json={
        'apitoken': config['metadata']['apiToken']
    }
)

query_string = config['spec']['inputs']['queryString']
logging.info('********')
logging.info('Running  job:{} at {}'.format(
   config['metadata']['name'], os.path.basename(__file__)))
logging.debug("This is a debug message")
logging.info('Querying database:\n"{}"'.format(query_string))

run = ea_tasks.check_query(
    query_string,
    db,
    config['metadata']['username'],
    lp.decode()
)
response = run

# Report that work has completed
logging.info("Reporting completion to jobhandler (apitoken: {})...".format(config['metadata']['apiToken']))
requests.post(
    '{}/job/complete'.format(config['metadata']['apiBaseUrl']),
    json={
        'apitoken': config['metadata']['apiToken']
    }
)

logging.info("Done!")
