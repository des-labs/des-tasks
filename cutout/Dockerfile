FROM registry.gitlab.com/des-labs/kubernetes/easyaccess:1.4.11

ARG UID=68586
ARG GID=2402
RUN groupmod -g ${GID} worker && usermod -u ${UID} -g ${GID} worker

# Next, install the required Python modules:
USER worker
WORKDIR /home/worker
COPY --chown=worker:worker ./worker/requirements.txt requirements.txt
RUN pip3 install --user -r requirements.txt

# Finally, copy in the files under the most active development:
COPY --chown=worker:worker ./worker/* ./

ENV PATH="/home/worker/.local/bin:${PATH}"
CMD ["python3" "task.py"]
