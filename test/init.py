import yaml
import requests

with open('configjob.yaml') as cfile:
    config = yaml.safe_load(cfile)

requests.post(
    '{}/job/start'.format(config['metadata']['apiBaseUrl']),
    json={
        'apitoken': config['metadata']['apiToken']
    }
)
