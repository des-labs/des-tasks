FROM hub.ncsa.illinois.edu/des/easyaccessclient:latest

# Next, install the required Python modules:
USER worker
WORKDIR /home/worker
COPY --chown=worker:worker ./worker/requirements.txt requirements.txt
RUN pip3 install --user -r requirements.txt

# Finally, copy in the files under the most active development:
COPY --chown=worker:worker ./worker/* ./

ENV PATH="/home/worker/.local/bin:${PATH}"
CMD ["python3" "task.py"]
