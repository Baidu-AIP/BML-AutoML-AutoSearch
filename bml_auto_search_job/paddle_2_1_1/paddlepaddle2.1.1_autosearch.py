# -*- coding:utf-8 -*-
""" paddle train demo """
import os
import numpy as np
import paddle # 导入paddle模块
import paddle.fluid as fluid
import gzip
import struct
import argparse
import time
from rudder_autosearch.sdk.amaas_tools import AMaasTools

def parse_arg():
    """parse arguments"""
    parser = argparse.ArgumentParser(description='paddle2.1.1 mnist Example')
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
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of images input in an iteration (default: 64)')
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

def load_data(file_dir, is_train=True):
    """
    :param file_dir:
    :param is_train:
    :return:
    """
    if is_train:
        image_path = file_dir + '/train-images-idx3-ubyte.gz'
        label_path = file_dir + '/train-labels-idx1-ubyte.gz'
    else:
        image_path = file_dir + '/t10k-images-idx3-ubyte.gz'
        label_path = file_dir + '/t10k-labels-idx1-ubyte.gz'
    with open(image_path.replace('.gz', ''), 'wb') as out_f, gzip.GzipFile(image_path) as zip_f:
        out_f.write(zip_f.read())
        # os.unlink(image_path)
    with open(label_path.replace('.gz', ''), 'wb') as out_f, gzip.GzipFile(label_path) as zip_f:
        out_f.write(zip_f.read())
        # os.unlink(label_path)
    with open(label_path[:-3], 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(image_path[:-3], 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

def reader_creator(file_dir, is_train=True, buffer_size=100, data_sampling_scale=1):
    """
    :param file_dir:
    :param is_train:
    :param buffer_size:
    :return:
    """
    images, labels = load_data(file_dir, is_train)
    if is_train:
        np.random.seed(0)
        sample_data_num = int(data_sampling_scale * len(images))
        idx = np.arange(len(images))
        np.random.shuffle(idx)
        images, labels = images[0:sample_data_num], labels[0:sample_data_num]
    def reader():
        """
        :return:
        """
        for num in range(int(len(labels) / buffer_size)):
            for i in range(buffer_size):
                yield images[num * buffer_size + i, :], int(labels[num * buffer_size + i])
    return reader

def reader_load(args):
    """reader_load"""
    # 每次读取训练集中的500个数据并随机打乱，传入batched reader中，batched reader 每次 yield args.batch_size个数据
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader_creator(args.train_dir, is_train=True, buffer_size=100,
                           data_sampling_scale=args.data_sampling_scale), buf_size=500),
        batch_size=args.batch_size)
    # 读取测试集的数据，每次 yield 64个数据
    test_reader = paddle.batch(
        reader_creator(args.test_dir, is_train=False, buffer_size=100), batch_size=args.batch_size)
    return train_reader, test_reader

def softmax_regression():
    """
    定义softmax分类器：
        一个以softmax为激活函数的全连接层
    Return:
        predict_image -- 分类的结果
    """
    # 输入的原始图像数据，大小为28*28*1
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # 以softmax为激活函数的全连接层，输出层的大小必须为数字的个数10
    predict = fluid.layers.fc(
        input=img, size=10, act='softmax')
    return predict

def multilayer_perceptron():
    """
    定义多层感知机分类器：
        含有两个隐藏层（全连接层）的多层感知器
        其中前两个隐藏层的激活函数采用 ReLU，输出层的激活函数用 Softmax
    Return:
        predict_image -- 分类的结果
    """
    # 输入的原始图像数据，大小为28*28*1
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # 第一个全连接层，激活函数为ReLU
    hidden = fluid.layers.fc(input=img, size=200, act='relu')
    # 第二个全连接层，激活函数为ReLU
    hidden = fluid.layers.fc(input=hidden, size=200, act='relu')
    # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    return prediction

def convolutional_neural_network():
    """
    定义卷积神经网络分类器：
        输入的二维图像，经过两个卷积-池化层，使用以softmax为激活函数的全连接层作为输出层
    Return:
        predict -- 分类的结果
    """
    # 输入的原始图像数据，大小为28*28*1
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # 第一个卷积-池化层
    # 使用20个5*5的滤波器，池化大小为2，池化步长为2，激活函数为Relu
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    # 第二个卷积-池化层
    # 使用20个5*5的滤波器，池化大小为2，池化步长为2，激活函数为Relu
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return prediction

def train_program():
    """
    配置train_program
    Return:
        predict -- 分类的结果
        avg_cost -- 平均损失
        acc -- 分类的准确率
    """
    paddle.enable_static()
    # 标签层，名称为label,对应输入图片的类别标签
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    # predict = softmax_regression() # 取消注释将使用 Softmax回归
    # predict = multilayer_perceptron() # 取消注释将使用 多层感知器
    predict = convolutional_neural_network() # 取消注释将使用 LeNet5卷积神经网络
    # 使用类交叉熵函数计算predict和label之间的损失函数
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    # 计算平均损失
    avg_cost = fluid.layers.mean(cost)
    # 计算分类准确率
    acc = fluid.layers.accuracy(input=predict, label=label)
    return predict, [avg_cost, acc]

def optimizer_program():
    """
    :return:
    """
    return fluid.optimizer.Adam(learning_rate=0.001)

def event_handler(pass_id, batch_id, cost):
    """event_handler"""
    # 打印训练的中间结果，训练轮次，batch数，损失函数
    print("Pass %d, Batch %d, Cost %f" % (pass_id, batch_id, cost))

def train_test(train_test_program,
                   train_test_feed, train_test_reader, executor, fetch_list):
    """train_test"""
    # 将分类准确率存储在acc_set中
    acc_set = []
    # 将平均损失存储在avg_loss_set中
    avg_loss_set = []
    # 将测试 reader yield 出的每一个数据传入网络中进行训练
    for test_data in train_test_reader():
        avg_loss_np, acc_np = executor.run(
            program=train_test_program,
            feed=train_test_feed.feed(test_data),
            fetch_list=fetch_list)
        acc_set.append(float(acc_np))
        avg_loss_set.append(float(avg_loss_np))
    # 获得测试数据上的准确率和损失值
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()
    # 返回平均损失值，平均准确率
    return avg_loss_val_mean, acc_val_mean

class Model():
    def __init__(self, args, train_reader, test_reader):
        self.args = args
        self.create_model()
        self.train_reader = train_reader
        self.test_reader = test_reader

    def create_model(self):
        """create_model"""
        # 该模型运行在单个CPU上
        self.place = fluid.CPUPlace()
        # 调用train_program 获取预测值，损失值
        self.prediction, [self.avg_loss, self.acc] = train_program()
        # 输入的原始图像数据，大小为28*28*1
        img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
        # 标签层，名称为label,对应输入图片的类别标签
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        # 告知网络传入的数据分为两部分，第一部分是img值，第二部分是label值
        self.feeder = fluid.DataFeeder(feed_list=[img, label], place=self.place)
        # 选择Adam优化器
        optimizer = fluid.optimizer.Adam(learning_rate=self.args.lr)
        optimizer.minimize(self.avg_loss)

    def run_train(self):
        PASS_NUM = self.args.epoch
        epochs = [epoch_id for epoch_id in range(PASS_NUM)]

        self.exe = fluid.Executor(self.place)
        self.exe.run(fluid.default_startup_program())
        main_program = fluid.default_main_program()
        step = 0
        for epoch_id in epochs:
            print("Epoch %d:" % (epoch_id))
            for step_id, data in enumerate(self.train_reader()):
                metrics = self.exe.run(main_program,
                                  feed=self.feeder.feed(data),
                                  fetch_list=[self.avg_loss, self.acc])
                if step % 100 == 0:  # 每训练100次 更新一次图片
                    event_handler(step, epoch_id, metrics[0])
                step += 1

    def save_model(self):
        """save_model"""
        # 将模型参数存储在名为 save_dirname 的文件中
        save_dirname = self.args.output_dir
        fluid.io.save_inference_model(save_dirname,
                                      ["img"], [self.prediction], self.exe,
                                      model_filename='model',
                                      params_filename='params')

    def evaluate(self):
        """evaluate"""
        test_program = fluid.default_main_program().clone(for_test=True)
        avg_loss_val, acc_val = train_test(train_test_program=test_program,
                                           train_test_reader=self.test_reader,
                                           train_test_feed=self.feeder,
                                           executor=self.exe,
                                           fetch_list=[self.avg_loss, self.acc])
        print("accuracy: %f" % acc_val)
        return acc_val

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
    train_reader, test_reader = reader_load(args)
    # 模型定义
    model = Model(args, train_reader, test_reader)
    # 模型训练
    model.run_train()
    # 模型保存
    model.save_model()
    # 模型评估
    acc = model.evaluate()
    # 上报结果
    report_final(args, metric=acc)

if __name__ == '__main__':
    main()
