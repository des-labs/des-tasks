Base images
============================

The base images are required by several Docker images used by DESaccess that require the `easyaccess` package. They are automatically built via the `https://gitlab.com/des-labs/deployment/-/tree/master/apps/desaccess/common/build/scripts/prebuild.sh` hook script that is executed by the deployment script. Use the following sequence to build them manually.

First build the Oracle client base image:
```
docker build --tag oracleclient:latest -f Dockerfile-oracleclient .
```

Then build the easyaccess base image:
```
docker build --tag easyaccessclient:latest -f Dockerfile-easyaccessclient .
```
