python train_cluster.py \
     --ps_hosts=localhost:3220,localhost:3221 \
     --worker_hosts=localhost:3222,localhost:3223,localhost:3224 \
     --job_name=ps --task_index=0 &

python train_cluster.py \
     --ps_hosts=localhost:3220,localhost:3221 \
     --worker_hosts=localhost:3222,localhost:3223,localhost:3224 \
     --job_name=ps --task_index=1 &

python train_cluster.py \
     --ps_hosts=localhost:3220,localhost:3221 \
     --worker_hosts=localhost:3222,localhost:3223,localhost:3224 \
     --job_name=worker --task_index=0 &

python train_cluster.py \
     --ps_hosts=localhost:3220,localhost:3221 \
     --worker_hosts=localhost:3222,localhost:3223,localhost:3224 \
     --job_name=worker --task_index=1 &

python train_cluster.py \
     --ps_hosts=localhost:3220,localhost:3221 \
     --worker_hosts=localhost:3222,localhost:3223,localhost:3224 \
     --job_name=worker --task_index=2 &