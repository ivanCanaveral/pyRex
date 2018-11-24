# ockerfile for kubeflow distributed training
# Download base image ubuntu 16.04
FROM python:3.6

# Only for local testing with docker
ENV TF_CONFIG '{"cluster":{"master":["localhost:2220"],"ps":["localhost:2221"],"worker":["localhost:2222", "localhost:2223"]},"task":{"type":"ps","index":"0"},"environment":"cloud"}'

# Change directory
WORKDIR /app

# Installing dependencies with pip
RUN pip3 install tensorflow numpy pandas

RUN pip3 --no-cache-dir install \
        google-cloud-storage

ENV GOOGLE_APPLICATION_CREDENTIALS "/app/key.json"

# Copy files
ADD *.csv train_kubeflow_save_gcs.py key.json /app/

CMD python3 train_kubeflow_save_gcs.py --gcs_path gcs://demo-dino2