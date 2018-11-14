import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

n_vars = 3
n_layer1 = 30
n_layer2 = 10
n_outputs = 2
learning_rate = 0.01

n_epochs = 2
batch_size = 10

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "../records/tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

data = pd.read_csv('../records/datasets/clean_batch_large.csv', dtype={'cactus':'float32', 'pterax':'float32', 'pteray':'float32', 'y':'int32'})
#data['output'] = 2*data['isJumping'] + data['isDucking']
#data[data['output'] > 2] = 2
#train_cols = ['cactus1', 'cactus2', 'cactus3', 'ptera1x', 'ptera1y', 'ptera2x', 'ptera2y', 'ptera3x', 'ptera3y']
print(data.columns)
features = ['cactus', 'pterax', 'pteray']

X_train = data[features].values
y_train = data['y'].values

print('data : {}'.format(X_train.shape))

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

with tf.name_scope("optimization"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("evaluation"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("utils"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

with tf.name_scope("board"):
    acc_summary = tf.summary.scalar('accuracy', accuracy)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

print('graph done')

if __name__ == '__main__':
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(len(y_train) // batch_size):
              sess.run(training_op)
            acc_train = sess.run(accuracy)#.eval()
            print("[{}] accuracy: {}".format(epoch, acc_train))
            acc_str = acc_summary.eval()
            step = epoch * len(y_train) + batch_size
            file_writer.add_summary(acc_str, step) 
            #if epoch % 50 == 0:
            #   modify this to support versions
            #   based on number of iterations
            #    last_check_point_path = saver.save(sess, './checkpoints/checkpoint_{}'.format(epoch))
            #    print('Checkpoint created')
        
        #file_writer.close()
        #save_path = saver.save(sess, 'models/model.ckpt')
        #print('Model saved at {}'.format(save_path))

        # Pick out the model input and output
        #print('\n Tensor extracted from closed session using its names')
        X = sess.graph.get_tensor_by_name('IteratorGetNext' + ':0')
        #print(X)
        logits = sess.graph.get_tensor_by_name('mlp/logits/add' + ':0')
        #print(logits)

        #print('\n Tensors info')
        model_input = tf.saved_model.build_tensor_info(X)
        #print(model_input)
        model_output = tf.saved_model.build_tensor_info(logits)
        #print(model_output)

        print('\n Signature')
        # Create a signature definition for tfserving
        signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'X': model_input},
            outputs={'logits': model_output},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        print(signature_definition)

        builder = tf.saved_model.Builder('../models/trex/1')

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature_definition
            })

        # Save the model so we can serve it with a model server :)
        builder.save()
        print('saved_model done!')

'''
# start
docker run -p 8501:8501 --mount type=bind,source=/Users/ivancanaveralsanchez/Desktop/serving/models/trex,target=/models/trex -e MODEL_NAME=trex -t tensorflow/serving

#GET request (also in chrome)
http://localhost:8501/v1/models/trex
curl http://localhost:8501/v1/models/trex -X GET

#POST request
curl -d '{"instances": [[0.0, 0.0, 0.0], [0.9, 0.9, 0.9]]}' -X POST http://localhost:8501/v1/models/trex:predict

## Python
import requests

url = "http://localhost:8501/v1/models/trex:predict"

payload = "{\"instances\": [[0.0, 0.0, 0.0], [0.9, 0.9, 0.9]]}"
headers = {
    'Content-Type': "application/json",
    'cache-control': "no-cache",
    'Postman-Token': "ce89024a-3c88-4a33-98a8-96b131afa16c"
    }

response = requests.request("POST", url, data=payload, headers=headers)

print(response.text)
'''
