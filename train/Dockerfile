# ockerfile for kubeflow distributed training
# Download base image ubuntu 16.04
FROM python:3.6

# Only for local testing with docker
ENV TF_CONFIG '{"cluster":{"master":["localhost:2220"],"ps":["localhost:2221"],"worker":["localhost:2222", "localhost:2223"]},"task":{"type":"ps","index":"0"},"environment":"cloud"}'

# Copy Gandalf files
ADD *.csv train_kubeflow.py /app/

# Change directory
WORKDIR /app

# Installing dependencies with pip
RUN pip3 install tensorflow numpy pandas

CMD python3 train_kubeflow.py