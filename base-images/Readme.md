Base images
============================

The base images must be built **before running the deployment script** (at least until they are properly integrated into the reproducible build workflow, at which time this message will be removed).

First build the Oracle client base image:
```
docker build --tag oracleclient:latest -f Dockerfile-oracleclient .
```

Then build the easyaccess base image:
```
docker build --tag easyaccessclient:latest -f Dockerfile-easyaccessclient .
```
