# -*- coding:utf-8 -*-
""" sklearn train demo """
import os
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import svm
import pandas as pd
import numpy as np
from rudder_autosearch.sdk.amaas_tools import AMaasTools

def parse_arg():
    """parse arguments"""
    parser = argparse.ArgumentParser(description='Sklearn iris Example')
    parser.add_argument('--train_dir', type=str, default='./train_data',
                        help='input data dir for training (default: ./train_data)')
    parser.add_argument('--test_dir', type=str, default='./test_data',
                        help='input data dir for test (default: ./test_data)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='output dir for auto_search job (default: ./output)')
    parser.add_argument('--job_id', type=str, default="job-1234",
                        help='auto_search job id')
    parser.add_argument('--trial_id', type=str, default="0-0",
                        help='auto_search id of a single trial')
    parser.add_argument('--metric', type=str, default="f1_score",
                        help='evaluation metric of the model')
    parser.add_argument('--data_sampling_scale', type=float, default=1.0,
                        help='sampling ratio of the dataset for auto_search (default: 1.0)')
    parser.add_argument('--kernel', type=str, default='linear',
                        help='kernel function (default: "linear")')
    parser.add_argument('--C', type=float, default=1,
                        help='penalty term (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='parameter of the kernel (default: 0.5)')

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.job_id, args.trial_id)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print("job_id: {}, trial_id: {}".format(args.job_id, args.trial_id))
    return args

def load_data(train_dir, data_sampling_scale):
    """ load data """
    # 共150条数据，训练120条，测试30条，进行2,8分进行模型训练
    # 每条数据类型为 x{nbarray} [6.4, 3.1, 5.5, 1.8]
    # 上传的数据储存在./train_data和./test_data中
    inputdata = pd.read_csv(train_dir + "/iris.csv")
    target = inputdata["Species"]
    inputdata = inputdata.drop(columns=["Species"])
    # 切分，测试训练2,8分
    x_train, x_test, y_train, y_test = train_test_split(inputdata, target, test_size=0.2, random_state=0)
    train_data = np.concatenate([x_train, y_train.ravel().reshape([-1, 1])], axis=1)
    np.random.seed(0)
    np.random.shuffle(train_data)
    train_data = train_data[0:int(data_sampling_scale * len(train_data))]
    x_train, y_train = train_data[:, 0:-1], train_data[:, -1]
    return (x_train, x_test), (y_train, y_test)

def save_model(model, output_dir):
    """ save model with pickle format """
    import pickle
    with open(output_dir + '/clf.pickle', 'wb') as f:
        pickle.dump(model, f)

def save_model_joblib(model, output_dir):
    """ save model with joblib format """
    try:
        import joblib
    except:
        from sklearn.externals import joblib
    joblib.dump(model, output_dir + '/clf.pkl')

def evaluate(model, x_test, y_test):
    """evaluate"""
    # 多分类f1_score指标
    predict = model.predict(x_test)
    f1 = f1_score(y_test, predict, average="micro")
    print("f1_score: %f" % f1)
    return f1

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
    """ main """
    # 获取参数
    args = parse_arg()
    # 加载数据集
    (x_train, x_test), (y_train, y_test) = load_data(args.train_dir, args.data_sampling_scale)
    # 模型定义
    model = svm.SVC(C=args.C, kernel=args.kernel, gamma=args.gamma)
    # 模型训练
    model.fit(x_train, y_train)
    # 模型保存
    save_model(model, args.output_dir)
    # 模型评估
    f1 = evaluate(model, x_test, y_test)
    # 上报结果
    report_final(args, metric=f1)

if __name__ == '__main__':
    main()