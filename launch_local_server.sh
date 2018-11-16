## to build from tfserving image and local model
#docker run -p 8501:8501 --mount type=bind,source=$(pwd)/models/trex,target=/models/trex -e MODEL_NAME=trex -t tensorflow/serving
## to build from docker hub public image ivancanaveral/trex
docker run -p 8501:8501 --name tfserving-trex ivancanaveral/trex:first