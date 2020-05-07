## Build from a parent image
#FROM oraclelinux:7-slim as oracle
#RUN  curl -o /etc/yum.repos.d/public-yum-ol7.repo https://yum.oracle.com/public-yum-ol7.repo && \
#     yum-config-manager --enable ol7_oracle_instantclient && \
#     yum -y install oracle-instantclient18.3-basic
#
#FROM ubuntu:18.04
## Do STUFF
#ENV PATH=$PATH:/usr/lib/oracle/18.3/client64/bin
#ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/oracle/18.3/client64/lib:/usr/lib
#COPY --from=oracle /usr/lib/oracle/ /usr/lib/oracle
#COPY --from=oracle /lib64/libaio.so.1 /usr/lib

FROM ubuntu:18.04

RUN apt-get update --fix-missing
RUN apt-get install -y  \
    python3-pip         \
    libaio1             \
    wget                \
    unzip

RUN apt install -y \
    mpich \
    libopenmpi-dev \
    openssh-client

RUN rm -rf /var/lib/apt/lists/*

# ORACLE
# Instructions at https://oracle.github.io/odpi/doc/installation.html#oracle-instant-client-zip
RUN mkdir -p /opt/oracle
ADD https://download.oracle.com/otn_software/linux/instantclient/19600/instantclient-basic-linux.x64-19.6.0.0.0dbru.zip /opt/oracle/instantclient-basic-linux.x64-19.6.0.0.0dbru.zip
RUN \
  cd /opt/oracle && \
  unzip instantclient-basic-linux.x64-19.6.0.0.0dbru.zip && \
  echo /opt/oracle/instantclient_19_6 > /etc/ld.so.conf.d/oracle-instantclient.conf && \
  ldconfig

RUN useradd --create-home --shell /bin/bash worker --uid 1001

#RUN pip3 install --upgrade pip

# The order of operation is important for code iteration, because building Docker
# images takes time and we want to take advantage of caching.
# First copy easyaccess and compile:
WORKDIR /home/worker
COPY --chown=worker:worker ./easyaccess ./easyaccess
WORKDIR /home/worker/easyaccess
RUN python3 setup.py install

# Next, install the required Python modules:
USER worker
WORKDIR /home/worker
COPY --chown=worker:worker ./worker/requirements.txt requirements.txt
RUN pip3 install --user -r requirements.txt

# Finally, copy in the files under the most active development:
COPY --chown=worker:worker ./worker/* ./

ENV PATH="/home/worker/.local/bin:${PATH}"
ENV TMPDIR="/tmp"


CMD ["python3" "task.py"]
