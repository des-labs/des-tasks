Example task definition for Kubernetes Jobs
-------------------------------------------

This folder contains the files needed to define a "task" that will be accomplished by spawning a Kubernetes Job. The primary components are the files

* `Dockerfile`
* `init.py`
* `task.py`

The `Dockerfile` is used to build the image that will define the Pod that the Job creates once launched by Kubernetes. By convention, the container will execute the command `python init.py` by the pod's lifecycle `postStart` hook, and then `python task.py` will be executed by the container to accomplish the desired task.

Sample query:

```
SELECT RA, DEC, MAG_AUTO_G, TILENAME from Y3_GOLD_2_2 sample(0.001)
```
