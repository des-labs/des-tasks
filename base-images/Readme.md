Base images
============================

The base images are required by several Docker images used by DESaccess that require the `easyaccess` package. They are automatically built via the `https://gitlab.com/des-labs/deployment/-/tree/master/apps/desaccess/common/build/scripts/prebuild.sh` hook script that is executed by the deployment script. Use the following sequence to build them manually.

Build the easyaccess base image:
```
cd easyaccess
docker build --tag easyaccessclient:latest .
```
