# import logging
import sys
import yaml
import os
import shutil

try:
   input_file = sys.argv[1]
except:
    input_file = 'configjob.yaml'

with open(input_file) as cfile:
    config = yaml.safe_load(cfile)

# logging.basicConfig(
#     level=logging.DEBUG,
#     handlers=[
#         logging.FileHandler(config['metadata']['log']),
#         logging.StreamHandler()
#     ]
# )

action = config['spec']['action']
if action == 'delete_job_files':
    for delete_path in config['spec']['delete_paths']:
        if os.path.isdir(delete_path):
            # logging.info('Deleting path "{}"...'.format(delete_path))
            shutil.rmtree(delete_path)
