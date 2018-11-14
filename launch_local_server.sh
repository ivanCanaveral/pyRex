docker run -p 8501:8501 --mount type=bind,source=$(pwd)/models/trex,target=/models/trex -e MODEL_NAME=trex -t tensorflow/serving
