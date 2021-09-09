# -*- coding:utf-8 -*-
""" xgboost train demo """
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import time
import argparse
from rudder_autosearch.sdk.amaas_tools import AMaasTools

def parse_arg():
    """parse arguments"""
    parser = argparse.ArgumentParser(description='xgboost boston Example')
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
    parser.add_argument('--metric', type=str, default="mse",
                        help='evaluation metric of the model')
    parser.add_argument('--data_sampling_scale', type=float, default=1.0,
                        help='sampling ratio of the dataset for auto_search (default: 1.0)')
    parser.add_argument('--max_depth', type=int, default=6,
                        help='maximum depth of the tree (default: 6)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='minimum loss reduction required for further splitting (default: 0.1)')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--num_round', type=int, default=10,
                        help='number of trees (default: 10)')
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.job_id, args.trial_id)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print("job_id: {}, trial_id: {}".format(args.job_id, args.trial_id))
    return args

def load_data(data_sampling_scale):
    """ load data """
    boston = datasets.load_boston()
    X, Y = boston.data, boston.target
    # 切分，测试训练2,8分
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    train_data = np.concatenate([x_train, y_train.reshape([-1, 1])], axis=1)
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
    import joblib
    joblib.dump(model, output_dir + '/clf.pkl')

def evaluate(model, x_test, y_test):
    """evaluate"""
    # 回归mean_squared_error指标
    deval = xgb.DMatrix(x_test)
    predict = model.predict(deval)
    mse = mean_squared_error(y_test, predict)
    print("mean_squared_error: %f" % mse)
    return mse

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
    (x_train, x_test), (y_train, y_test) = load_data(args.data_sampling_scale)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    # 模型参数定义
    param = {"gamma": args.gamma, 'max_depth': args.max_depth,
             'eta': args.eta, 'objective': 'reg:squarederror'}
    # 模型训练
    model = xgb.train(param, dtrain, args.num_round)
    # 模型保存
    save_model_joblib(model, args.output_dir)
    # 模型评估
    mse = evaluate(model, x_test, y_test)
    # 上报结果
    report_final(args, metric=mse)

if __name__ == '__main__':
    main()

