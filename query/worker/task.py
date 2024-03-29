import logging
import sys
import yaml
import os
import requests
import ea_tasks
import rpdb
import time


def task_start(config):
    logging.info('Starting job: {}'.format(config['metadata']['name']))
    requests.post(
        '{}/job/start'.format(config['metadata']['apiBaseUrl']),
        json={
            'apitoken': config['metadata']['apiToken']
        }
    )


def task_complete(config, response):
    if response['msg'] == 'preStop pod lifecycle hook triggered':
        logging.warning(f'''{response['msg']} for query job {config['metadata']['name']}''')
    ## Report that work has completed
    ## Try to post job completion data to API server multiple times in the
    ## event that something temporarily disrupts the HTTP request
    fail_count = 0
    while fail_count <= 5:
        try:
            completeResponse = requests.post(
                '{}/job/complete'.format(config['metadata']['apiBaseUrl']),
                json={
                    'apitoken': config['metadata']['apiToken'],
                    'response': response
                }
            )
            if completeResponse.status_code == 200:
                break
        except Exception as e:
            logging.error(f'''Error posting job completion to API server: {e}''')
        fail_count += 1
        time.sleep(1)


def execute_task(config):
    # Query the Oracle database using easyaccess
    query_string = config['spec']['inputs']['queryString']
    logging.info('Querying database:\n"{}"'.format(query_string))
    # Verify that the query is valid
    check_query = ea_tasks.check_query(
        query_string,
        config['metadata']['database'],
        config['metadata']['username'],
        config['metadata']['password']
    )
    if check_query['status'] == 'error':

        if config["spec"]["inputs"]["checkQuery"] == True:
            # If the task is simply to check the validity of the query, then the job did not fail
            check_query['status'] = 'ok'
            return check_query
        else:
            logging.error("Invalid query.")

    elif config["spec"]["inputs"]["checkQuery"] == True:
        # If the task is simply to check the validity of the query, then do not execute the query
        response = {
            'status': 'ok',
            'msg': ''
        }
        return response
    # Submit the query and obtain resulting data
    if config["spec"]["inputs"]["quickQuery"] == True:
        response = ea_tasks.run_quick(
            query_string,
            config['metadata']['database'],
            config['metadata']['username'],
            config['metadata']['password']
        )
    else:
        logging.info('{}'.format(config['spec']['inputs']))
        response = ea_tasks.run_query(
            query_string,
            config['metadata']['database'],
            config['metadata']['username'],
            config['metadata']['password'],
            '/home/worker/output/query/{}'.format(config['metadata']['jobId']),
            config["spec"]["inputs"]["fileName"],
            config["spec"]["inputs"]["compression"]
        )
    return response


if __name__ == "__main__":

    # Make the cutout subdirectory if it does not already exist.
    os.makedirs('/home/worker/output/query', exist_ok=True)

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
