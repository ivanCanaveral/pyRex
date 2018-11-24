import argparse
import os
import sys
import json

import pandas as pd
import tensorflow as tf

class SaveAtEnd(tf.train.SessionRunHook):
  '''a training hook for saving the final variables'''

  def __init__(self, path, X, logits):
    '''hook constructor

    Args:
        filename: where the model will be saved
        variables: the variables that will be saved'''

    self.path = path
    self.X = X
    self.logits = logits

  def end(self, session):
    '''this will be run at session closing'''
    print('Saving model...')
    session.graph._unsafe_unfinalize()
    tf.saved_model.simple_save(
        session,
        self.path,
        inputs={"X": self.X},
        outputs={"logits": self.logits}
      )
    print('Model saved!')

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
  task_index = int(task["index"])
  print('[{}:{}] Starting...'.format(job_name, task_index))

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
    print('Executing worker code')
    
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % task_index,
        cluster=cluster)):
      
      # Create dataset...
      print('Creating dataset...')
      dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().batch(batch_size)
      iterator = dataset.make_one_shot_iterator()
      X, y = iterator.get_next()
      
      def add_layer(X, n_units, name, activation=None):
        print('Adding layer {}'.format(name))
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
      print('Building model...')
      global_step = tf.contrib.framework.get_or_create_global_step()

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    # X = mon_sess.graph.get_tensor_by_name('IteratorGetNext' + ':0')
    # logits = mon_sess.graph.get_tensor_by_name('mlp/logits/add' + ':0')
    hooks=[tf.train.StopAtStepHook(last_step=1000), SaveAtEnd("gs://demo-dino2/1", X, logits)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(task_index == 0),
                                           hooks=hooks) as mon_sess:
      print('Training...')
      step = 0
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # mon_sess.run handles AbortedError in case of preempted PS.
        mon_sess.run(train_op)
        step = step + 1
        if (not mon_sess.should_stop()):
          print('[{}:{}] acc:{}'.format(job_name, task_index, mon_sess.run(accuracy)))

      print('Training finished')
      # if task_index == 0:
        # print('Chief preparing to save...')
        # export_path = "gs://demo-dino2/1"
        # X = mon_sess.graph.get_tensor_by_name('IteratorGetNext' + ':0')
        # logits = mon_sess.graph.get_tensor_by_name('mlp/logits/add' + ':0')
        
        # mon_sess.graph._unsafe_unfinalize()
        # tf.saved_model.simple_save(
        #   mon_sess,
        #   export_path,
        #   inputs={"X": X},
        #   outputs={"logits": logits}
        # )
        
        #print('\n Tensors info')
        # model_input = tf.saved_model.build_tensor_info(X)
        #print(model_input)
        # model_output = tf.saved_model.build_tensor_info(logits)
        #print(model_output)

        # print('Signature')
        # Create a signature definition for tfserving
        # signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
        #     inputs={'X': model_input},
        #     outputs={'logits': model_output},
        #     method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        # print(signature_definition)

        # builder = tf.saved_model.Builder(export_path)

        # print('\n Adding meta graph and variables')
        #mon_sess.graph._unsafe_unfinalize()
        # builder.add_meta_graph_and_variables(
        #     mon_sess, [tf.saved_model.tag_constants.SERVING],
        #     signature_def_map={
        #         tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
        #             signature_definition
        #     })

        # Save the model so we can serve it with a model server :)
        # print('Saving...')
        # builder.save()
        # print('Saved_model done!')


if __name__ == "__main__":
  print('Executing main...')
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
