# Tests cheatsheet


## Minikube

```
minikube start --extra-config=kubelet.housekeeping-interval=2s

minikube dashboard
minikube addons list
minikube addons enable metrics-server
```

#### monitor

```
watch --interval 1 "kubectl top pods"
watch --interval 1 "kubectl get hpa"
```

##### resolution interval

```
kubectl -n kube-system get deployments.apps

kubectl -n kube-system edit deployments.apps metrics-server
````

```yaml
    spec:
      containers:
      - command:
        - /metrics-server
        - --metric-resolution=15s
        - --source=kubernetes.summary_api:https://kubernetes.default?kubeletHttps=true&kubeletPort=10250&insecure=true
        image: k8s.gcr.io/metrics-server-amd64:v0.2.1
        imagePullPolicy: Always
        name: metrics-server
```

or

```yaml
  template:
    metadata:
      creationTimestamp: null
      labels:
        k8s-app: metrics-server
      name: metrics-server
    spec:
      containers:
      - args:
        - --cert-dir=/tmp
        - --secure-port=4443
        - --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
        - --kubelet-use-node-status-port
        - --metric-resolution=5s
        - --kubelet-insecure-tls
        image: k8s.gcr.io/metrics-server/metrics-server:v0.6.1@sha256:5ddc6458eb95f5c70bd13fdab90cbd7d6ad1066e5b528ad1dcb28b76c5fb2f00
        imagePullPolicy: IfNotPresent
```

## Tf Serving


#### some requests

```
kubectl exec -it ubuntu -- bash

apt update
apt install -y curl apache2-utils
apt install curl

curl http://172.17.0.4:8501/v1/models/trex

curl http://10.104.152.172:8501/v1/models/trex

curl http://10.103.114.52:8501/v1/models/trex/metadata

curl -d '{"instances": [[0.0, 0.0, 0.0], [0.9, 0.9, 0.9]]}' -X POST http://localhost:8501/v1/models/trex:predict
```

#### scale tests

```
install apache2-utils

ab -n 100 -c 10 http://10.103.114.52:8501/v1/models/trex/metadata

echo '{"instances": [[0.0, 0.0, 0.0], [0.9, 0.9, 0.9]]}' > data.json

echo '{"instances": [[0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9]]}' > bdata.json

ab -n 100000 -c 100 -p data.json -rk http://localhost:8501/v1/models/trex:predict
ab -n 100000 -c 100 -p bdata.json -rk http://10.103.168.22:8501/v1/models/trex:predict
ab -n 1000000 -c 100 -p bdata.json http://10.103.114.52:8501/v1/models/trex:predict
```

## Urls

https://minikube.sigs.k8s.io/docs/start/
https://minikube.sigs.k8s.io/docs/handbook/accessing/
https://www.tensorflow.org/tfx/serving/api_rest