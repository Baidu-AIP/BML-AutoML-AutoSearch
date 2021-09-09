# -*- coding:utf-8 -*-
""" pytorch train demo """
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
import codecs
import errno
import gzip
import numpy as np
import os
import time
from PIL import Image
from rudder_autosearch.sdk.amaas_tools import AMaasTools

def parse_arg():
    """parse arguments"""
    parser = argparse.ArgumentParser(description='PyTorch1.7.1 MNIST Example')
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
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--perturb_interval', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--resume_checkpoint_path', type=str, default="",
                        help='inherit the initial weight of the previous trial')
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.job_id, args.trial_id)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.cuda = not args.no_cuda and torch.cuda.is_available()


    print("job_id: {}, trial_id: {}".format(args.job_id, args.trial_id))
    return args

# 定义MNIST数据集的dataset
class MNIST(data.Dataset):
    """
    MNIST dataset
    """
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, train=True, transform=None, target_transform=None, data_sampling_scale=1):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.data_sampling_scale = data_sampling_scale
        self.preprocess(root, train, False)
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        """
        raw folder
        """
        return os.path.join('/tmp', 'raw')

    @property
    def processed_folder(self):
        """
        processed folder
        """
        return os.path.join('/tmp', 'processed')

    # data preprocessing
    def preprocess(self, train_dir, train, remove_finished=False):
        """
        preprocess
        """
        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)
        train_list = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz']
        test_list = ['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
        zip_list = train_list if train else test_list
        for zip_file in zip_list:
            print('Extracting {}'.format(zip_file))
            zip_file_path = os.path.join(train_dir, zip_file)
            raw_folder_path = os.path.join(self.raw_folder, zip_file)
            with open(raw_folder_path.replace('.gz', ''), 'wb') as out_f, gzip.GzipFile(zip_file_path) as zip_f:
                out_f.write(zip_f.read())
            if remove_finished:
                os.unlink(zip_file_path)
        if train:
            x_train = read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte'))
            y_train = read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
            np.random.seed(0)
            sample_data_num = int(self.data_sampling_scale * len(x_train))
            idx = np.arange(len(x_train))
            np.random.shuffle(idx)
            x_train, y_train = x_train[0:sample_data_num], y_train[0:sample_data_num]
            training_set = (x_train, y_train)
            with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
                torch.save(training_set, f)
        else:
            test_set = (
                read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
                read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
            )
            with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
                torch.save(test_set, f)

def get_int(b):
    """
    get int
    """
    return int(codecs.encode(b, 'hex'), 16)

def read_label_file(path):
    """
    read label file
    """
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()

def read_image_file(path):
    """
    read image file
    """
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)

def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

def load_data(args):
    """load_data"""
    # 若无测试集，训练集做验证集
    if not os.path.exists(args.test_dir) or not os.listdir(args.test_dir):
        args.test_dir = args.train_dir
    # 将数据进行转化，从PIL.Image/numpy.ndarray的数据进转化为torch.FloadTensor
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = MNIST(root=args.train_dir, train=True, transform=trans, data_sampling_scale=args.data_sampling_scale)
    test_set = MNIST(root=args.test_dir, train=False, transform=trans)
    # 定义data reader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False)
    return train_loader, test_loader

# 定义网络模型
class Net(nn.Module):
    """
    Net
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        forward
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def load_state_dict(model, resume_checkpoint_path):
    """load_state_dict"""
    if resume_checkpoint_path:
        model.load_state_dict(torch.load(resume_checkpoint_path))

def run_train(model, args, train_loader):
    """run_train"""
    if args.cuda:
        # Move model to GPU.
        model.cuda()
    # 选择优化器
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
    for epoch in range(1, args.perturb_interval + 1):
        train(model, args, train_loader, optimizer, epoch)

def train(model, args, train_loader, optimizer, epoch):
    """
    train
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)  # 获取预测值
        loss = F.nll_loss(output, target)  # 计算loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))

def evaluate(model, args, test_loader):
    """evaluate"""
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()
    test_loss /= len(test_loader) * args.batch_size
    test_accuracy /= len(test_loader) * args.batch_size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, 100. * test_accuracy))
    return float(test_accuracy)

def save(model, output_dir):
    """
    save
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 保存模型
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pkl'))

def report_final(args, metric):
    """report_final_result"""
    # 结果上报sdk
    amaas_tools = AMaasTools(args.job_id, args.trial_id)
    metric_dict = {args.metric: metric}
    checkpoint_path = os.path.join(args.output_dir, 'model.pkl')
    for i in range(3):
        flag, ret_msg = amaas_tools.report_final_result(metric=metric_dict,
                                                        export_model_path=args.output_dir,
                                                        checkpoint_path=checkpoint_path)
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
    train_loader, test_loader = load_data(args)
    # 模型定义
    model = Net()
    # 继承之前实验的模型参数
    load_state_dict(model, args.resume_checkpoint_path)
    # 模型训练
    run_train(model, args, train_loader)
    # 模型保存
    save(model, args.output_dir)
    # 模型评估
    acc = evaluate(model, args, test_loader)
    # 上报结果
    report_final(args, metric=acc)

if __name__ == '__main__':
    main()