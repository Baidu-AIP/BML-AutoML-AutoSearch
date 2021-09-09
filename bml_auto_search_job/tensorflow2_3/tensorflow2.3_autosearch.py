# -*- coding:utf-8 -*-
""" tensorflow2 train demo """
import tensorflow as tf
import os
import numpy as np
import time
import argparse
from rudder_autosearch.sdk.amaas_tools import AMaasTools

def parse_arg():
    """parse arguments"""
    parser = argparse.ArgumentParser(description='tensorflow2.3 mnist Example')
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
    parser.add_argument('--epoch', type=int, default=5,
                        help='number of epochs to train (default: 5)')
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.job_id, args.trial_id)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print("job_id: {}, trial_id: {}".format(args.job_id, args.trial_id))
    return args

def load_data(data_sampling_scale):
    """ load data """
    mnist = tf.keras.datasets.mnist
    work_path = os.getcwd()
    (x_train, y_train), (x_test, y_test) = mnist.load_data('%s/train_data/mnist.npz' % work_path)
    # sample training data
    np.random.seed(0)
    sample_data_num = int(data_sampling_scale * len(x_train))
    idx = np.arange(len(x_train))
    np.random.shuffle(idx)
    x_train, y_train = x_train[0:sample_data_num], y_train[0:sample_data_num]
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, x_test), (y_train, y_test)

def Model(learning_rate):
    """Model"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def evaluate(model, x_test, y_test):
    """evaluate"""
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print("accuracy: %f" % acc)
    return acc

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

def main():
    """main"""
    # 获取参数
    args = parse_arg()
    # 加载数据集
    (x_train, x_test), (y_train, y_test) = load_data(args.data_sampling_scale)
    # 模型定义
    model = Model(args.lr)
    # 模型训练
    model.fit(x_train, y_train, epochs=args.epoch, batch_size=args.batch_size)
    # 模型保存
    model.save(args.output_dir)
    # 模型评估
    acc = evaluate(model, x_test, y_test)
    # 上报结果
    report_final(args, metric=acc)

if __name__ == '__main__':
    main()