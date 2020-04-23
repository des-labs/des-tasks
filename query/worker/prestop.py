import yaml
import requests

STATUS_OK = 'ok'
STATUS_ERROR = 'error'
DEFAULT_RESPONSE = {
    'status': STATUS_OK,
    'msg': '',
    'elapsed': 0.0,
    'data': {},
    'files': [],
    'sizes': []
}
with open('configjob.yaml') as cfile:
    config = yaml.safe_load(cfile)

# This preStop pod lifecycle hook function should only execute if Kubernetes
# prematurely terminates it
response = DEFAULT_RESPONSE
response['status'] = STATUS_ERROR
response['msg'] = 'preStop pod lifecycle hook triggered'
requests.post(
    '{}/job/complete'.format(config['metadata']['apiBaseUrl']),
    json={
        'apitoken': config['metadata']['apiToken'],
        'response': response
    }
)
