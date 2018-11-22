import argparse
import os
import sys
import json

import pandas as pd
import tensorflow as tf

FLAGS = None

n_vars = 3
n_layer1 = 30
n_layer2 = 10
n_outputs = 3
learning_rate = 0.01

n_epochs = 4
batch_size = 10

data = pd.read_csv('clean_batch_large.csv', dtype={'cactus':'float32', 'pterax':'float32', 'pteray':'float32', 'isJumping':'int32', 'isDucking':'int32'})
data['y'] = data['isDucking'] + 2*data['isJumping']
data[data['y'] > 2] = 2

print(data.columns)
features = ['cactus', 'pterax', 'pteray']

X_train = data[features].values
y_train = data['y'].values

print('data : {}'.format(X_train.shape))

def main(_):
  tf_config_json = os.environ.get("TF_CONFIG", "{}")
  tf_config = json.loads(tf_config_json)

  task = tf_config.get("task", {})
  cluster_spec = tf_config.get("cluster", {})
  cluster = tf.train.ClusterSpec(cluster_spec)
  job_name = task["type"]
  task_index = task["index"]

  #ps_hosts = FLAGS.ps_hosts.split(",")
  #worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  #cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=job_name,
                           task_index=task_index)

  if job_name == "ps":
    server.join()
  elif job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % task_index,
        cluster=cluster)):
      
      # Create dataset...
      dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().batch(batch_size)
      iterator = dataset.make_one_shot_iterator()
      X, y = iterator.get_next()
      
      def add_layer(X, n_units, name, activation=None):
        with tf.name_scope(name):
          input_size = int(X.get_shape()[1])
          sigma = 2 / (input_size + n_units)
          weights = tf.truncated_normal((input_size, n_units), stddev=sigma)
          W = tf.Variable(weights, name='W')
          b = tf.Variable(tf.zeros([n_units]), name = 'b')
          output = tf.matmul(X, W) + b
          if activation is not None:
            return activation(output)
          else:
            return output

      with tf.name_scope("mlp"):
        layer1 = add_layer(X, n_layer1, 'layer1_output', activation=tf.nn.relu)
        layer2 = add_layer(layer1, n_layer2, 'layer2_output', activation=tf.nn.relu)
        logits = add_layer(layer2, n_outputs, 'logits')

      with tf.name_scope("loss"):
        cross_entropy_result = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(cross_entropy_result)

      with tf.name_scope("evaluation"):
        correct = tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), y)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

      # Build model...
      global_step = tf.contrib.framework.get_or_create_global_step()

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=100000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(task_index == 0),
                                           checkpoint_dir="./tmp/train_logs",
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # mon_sess.run handles AbortedError in case of preempted PS.
        mon_sess.run(train_op)
        print('[{}:{}] loss:{} acc:{}'.format(job_name, task_index, mon_sess.run(loss), mon_sess.run(accuracy)))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)