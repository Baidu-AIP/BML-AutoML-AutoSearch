# -*- coding:utf-8 -*-
""" tensorflow1 train demo """
import os
import tensorflow as tf
import numpy as np
import time
from tensorflow import keras
import os
import argparse
from rudder_autosearch.sdk.amaas_tools import AMaasTools

tf.logging.set_verbosity(tf.logging.INFO)

def parse_arg():
    """parse arguments"""
    parser = argparse.ArgumentParser(description='tensorflow1.13.2 mnist Example')
    parser.add_argument('--train_dir', type=str, default='./train_data',
                        help='input data dir for training (default: ./train_data)')
    parser.add_argument('--test_dir', type=str, default='./test_data',
                        help='input data dir for test (default: ./test_data)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='output dir for auto_search job (default: ./output)')
    parser.add_argument('--job_id', type=str, default="job-1234",
                        help='auto_search job id (default: "job-1234")')
    parser.add_argument('--trial_id', type=str, default="0-0",
                        help='auto_search id of a single trial (default: "0-0")')
    parser.add_argument('--metric', type=str, default="acc",
                        help='evaluation metric of the model')
    parser.add_argument('--data_sampling_scale', type=float, default=1.0,
                        help='sampling ratio of the data (default: 1.0)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of images input in an iteration (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate of the training (default: 0.001)')
    parser.add_argument('--last_step', type=int, default=20000,
                        help='number of steps to train (default: 20000)')
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.job_id, args.trial_id)
    print("job_id: {}, trial_id: {}".format(args.job_id, args.trial_id))
    return args

def load_data(data_sampling_scale):
    """ load data """
    work_path = os.getcwd()
    (x_train, y_train), (x_test, y_test) = \
        keras.datasets.mnist.load_data('%s/train_data/mnist.npz' % work_path)
    # sample training data
    np.random.seed(0)
    sample_data_num = int(data_sampling_scale * len(x_train))
    idx = np.arange(len(x_train))
    np.random.shuffle(idx)
    x_train, y_train = x_train[0:sample_data_num], y_train[0:sample_data_num]
    # The shape of downloaded data is (-1, 28, 28), hence we need to reshape it
    # into (-1, 784) to feed into our network. Also, need to normalize the
    # features between 0 and 1.
    x_train = np.reshape(x_train, (-1, 784)) / 255.0
    x_test = np.reshape(x_test, (-1, 784)) / 255.0
    return (x_train, x_test), (y_train, y_test)

def train_input_generator(x_train, y_train, batch_size=64):
    """train_input_generator"""
    assert len(x_train) == len(y_train)
    while True:
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]
        index = 0
        while index <= len(x_train) - batch_size:
            yield x_train[index:index + batch_size], \
                  y_train[index:index + batch_size],
            index += batch_size

def conv_model(feature, target, mode):
    """2-layer convolution model."""
    # Convert the target to a one-hot tensor of shape (batch_size, 10) and
    # with a on-value of 1 for each one-hot vector of length 10.
    target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)
    # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
    # image width and height final dimension being the number of color channels.
    feature = tf.reshape(feature, [-1, 28, 28, 1])
    # First conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = tf.layers.conv2d(feature, 32, kernel_size=[5, 5],
                                   activation=tf.nn.relu, padding="SAME")
        h_pool1 = tf.nn.max_pool(
            h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Second conv layer will compute 64 features for each 5x5 patch.
    with tf.variable_scope('conv_layer2'):
        h_conv2 = tf.layers.conv2d(h_pool1, 64, kernel_size=[5, 5],
                                   activation=tf.nn.relu, padding="SAME")
        h_pool2 = tf.nn.max_pool(
            h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    # Densely connected layer with 1024 neurons.
    h_fc1 = tf.layers.dropout(
        tf.layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu),
        rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Compute logits (1 per class) and compute loss.
    logits = tf.layers.dense(h_fc1, 10, activation=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)
    return tf.argmax(logits, 1), loss

class Model():
    def __init__(self, args, train_test_data):
        self.args = args
        self.create_model()
        (self.x_train, self.x_test), (self.y_train, self.y_test) = train_test_data

    def create_model(self):
        """create_model"""
        with tf.name_scope('input'):
            self.image = tf.placeholder(tf.float32, [None, 784], name='image')
            self.label = tf.placeholder(tf.float32, [None], name='label')
        self.predict, self.loss = conv_model(self.image, self.label, tf.estimator.ModeKeys.TRAIN)
        opt = tf.train.RMSPropOptimizer(self.args.lr)
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op = opt.minimize(self.loss, global_step=self.global_step)

    def run_train(self):
        """run_train"""
        hooks = [
            tf.train.StopAtStepHook(last_step=self.args.last_step),
            tf.train.LoggingTensorHook(tensors={'step': self.global_step, 'loss': self.loss},
                                       every_n_iter=10),
        ]
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = '0'
        # Horovod: save checkpoints only on worker 0 to prevent other workers from
        # corrupting them.
        self.checkpoint_dir = '/checkpoints'
        os.system("rm -rf " + self.checkpoint_dir)
        training_batch_generator = train_input_generator(self.x_train,
                                                         self.y_train, batch_size=self.args.batch_size)
        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(checkpoint_dir=self.checkpoint_dir,
                                               hooks=hooks,
                                               config=config) as mon_sess:
            while not mon_sess.should_stop():
                # Run a training step synchronously.
                image_, label_ = next(training_batch_generator)
                mon_sess.run(self.train_op, feed_dict={self.image: image_, self.label: label_})

    def save_model(self):
        """save_model"""
        saver = tf.train.Saver()
        inputs_classes = tf.saved_model.utils.build_tensor_info(self.image)
        outputs_classes = tf.saved_model.utils.build_tensor_info(self.predict)
        signature = (tf.saved_model.signature_def_utils.build_signature_def(
            inputs={tf.saved_model.signature_constants.CLASSIFY_INPUTS: inputs_classes},
            outputs={tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: outputs_classes},
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

        with tf.Session() as sess:
            sess.run([tf.local_variables_initializer(), tf.tables_initializer()])
            saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_dir))
            model_output_dir = self.args.output_dir
            builder = tf.saved_model.builder.SavedModelBuilder(model_output_dir)
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                                 signature_def_map={'predict_images': signature},
                                                 legacy_init_op=legacy_init_op)
            builder.save()

    def evaluate(self):
        """evaluate"""
        with tf.Session() as sess:
            sess.run([tf.local_variables_initializer(), tf.tables_initializer()])
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_dir))
            y_pred = sess.run(self.predict, feed_dict={self.image: self.x_test})
            self.acc = sum(y_pred == self.y_test) / len(y_pred)
        print("accuracy: %f" % self.acc)
        return self.acc

def report_final(args, metric):
    """report_final_result"""
    # 结果上报sdk
    amaas_tools = AMaasTools(args.job_id, args.trial_id)
    metric_dict = {args.metric: metric}
    for i in range(3):
        flag, ret_msg = amaas_tools.report_final_result(metric=metric_dict,
                                                             export_model_path=args.output_dir,
                                                             checkpoint_path="")
        print("End Report, metric:{}, ret_msg:{}".format(metric, ret_msg))
        if flag:
            break
        time.sleep(1)
    assert flag, "Report final result to manager failed! Please check whether manager'address or manager'status " \
                 "is ok! "

def main(_):
    """main"""
    # 获取参数
    args = parse_arg()
    # 加载数据集
    train_test_data = load_data(args.data_sampling_scale)
    # 模型定义
    model = Model(args, train_test_data)
    # 模型训练
    model.run_train()
    # 模型保存
    model.save_model()
    # 模型评估
    acc = model.evaluate()
    # 上报结果
    report_final(args, metric=acc)

if __name__ == "__main__":
    tf.app.run()