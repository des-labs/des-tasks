Example task definition for Kubernetes Jobs
-------------------------------------------

This repo contains definitions of various task types designed to run as independent Kubernetes Jobs managed by the [JobHandler](https://github.com/des-labs/des-jobhandler/) deployment. The primary components required by the [JobHandler](https://github.com/des-labs/des-jobhandler/) are the files (shown relative to this repo root path)

* `[task_type]/Dockerfile`
* `[task_type]/worker/init.py`
* `[task_type]/worker/task.py`
* `[task_type]/worker/prestop.py`
* `[task_type]/worker/jobconfig_spec.tpl.yaml`

The `Dockerfile` is used to build the image that will define the Pod container that the Job creates once launched by Kubernetes. By convention, the container will execute the command `python init.py` by the pod's lifecycle `postStart` hook, and then `python task.py` will be executed by the container to accomplish the desired task. If the Job is terminated by Kubernetes, triggering the `preStop` lifecycle hook, the `prestop.py` script should be used to report the task failure to the JobHandler.

Task containers are provided information by the JobHandler via a Kubernetes ConfigMap mounted as a file in the container's filesystem: `/home/worker/configjob.yaml`. The structure of this YAML file is [defined by the JobHandler using a template file](https://github.com/des-labs/des-jobhandler/blob/master/jobconfig_base.tpl.yaml). The `metadata` key in this data structure contains the information common to all task types. The `spec` key is defined individually by each task type via the `jobconfig_spec.tpl.yaml` file to satisfy its unique requirements.
