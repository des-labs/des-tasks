import logging
import sys
import yaml
import os
import requests
import ea_tasks
import rpdb
# from Crypto.Cipher import AES
# import base64

SECRET = 'my_secret_key'

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

logging.info('Running  job:{} at {}'.format(
   config['metadata']['name'], os.path.basename(__file__)))

# Report that work has started
logging.info("Reporting job start to jobhandler (apitoken: {})...".format(
   config['metadata']['apiToken']))
requests.post(
    '{}/job/start'.format(config['metadata']['apiBaseUrl']),
    json={
        'apitoken': config['metadata']['apiToken']
    }
)

#############################
# Primary task activity
#############################
debug_loop = True
while debug_loop == True:
    # Loop if debugging is activated, and pause execution
    debug_loop = config['metadata']['debug']
    if debug_loop:
        logging.info('Debugging is enabled. Invoking RPDB...')
        rpdb.set_trace()
    # Query the Oracle database using easyaccess
    query_string = config['spec']['inputs']['queryString']
    logging.info('Querying database:\n"{}"'.format(query_string))
    response = ea_tasks.check_query(
        query_string,
        'dessci',
        config['metadata']['username'],
        config['spec']['inputs']['dbPassword']
    )
    logging.info("Database query check response:\n{}".format(response))


# Report that work has completed
logging.info("Reporting completion to jobhandler (apitoken: {})...".format(config['metadata']['apiToken']))
requests.post(
    '{}/job/complete'.format(config['metadata']['apiBaseUrl']),
    json={
        'apitoken': config['metadata']['apiToken'],
        'response': response
    }
)

logging.info("Done!")
