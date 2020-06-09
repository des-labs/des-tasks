FROM python:3

RUN useradd --create-home --shell /bin/bash worker --uid 1001

RUN pip3 install --upgrade pip

USER worker
WORKDIR /home/worker
COPY --chown=worker:worker ./worker/requirements.txt requirements.txt
RUN pip install --user -r requirements.txt
ENV PATH="/home/worker/.local/bin:${PATH}"
COPY --chown=worker:worker ./worker/* ./
CMD ["python" "task.py"]
