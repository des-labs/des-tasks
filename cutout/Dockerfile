FROM ubuntu:18.04

RUN apt-get update --fix-missing
RUN apt-get install -y  \
    python3-pip         \
    libaio1             \
    wget                \
    imagemagick         \
    stiff               \
    unzip

RUN apt install -y \
    mpich \
    libopenmpi-dev \
    openssh-client

RUN rm -rf /var/lib/apt/lists/*

# ORACLE DB Client installation (https://oracle.github.io/odpi/doc/installation.html#oracle-instant-client-zip)
RUN mkdir -p /opt/oracle
ADD https://download.oracle.com/otn_software/linux/instantclient/19600/instantclient-basic-linux.x64-19.6.0.0.0dbru.zip /opt/oracle/instantclient-basic-linux.x64-19.6.0.0.0dbru.zip
RUN \
  cd /opt/oracle && \
  unzip instantclient-basic-linux.x64-19.6.0.0.0dbru.zip && \
  echo /opt/oracle/instantclient_19_6 > /etc/ld.so.conf.d/oracle-instantclient.conf && \
  ldconfig

RUN useradd --create-home --shell /bin/bash worker --uid 1001

# The order of operation is important for code development iteration, because
# building Docker images takes time and we want to take advantage of caching.

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
