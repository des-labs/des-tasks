import yaml
from task import task_complete

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
## Do not report the job status as an error, since evidence suggests that some
## jobs whose preStop hooks are triggered are actually successful and complete.
# response['status'] = STATUS_ERROR
response['msg'] = 'preStop pod lifecycle hook triggered'
task_complete(config, response)
