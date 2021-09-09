
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license: Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved.
@desc: 图像预测算法示例
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import base64
import json
from PIL import Image
from io import BytesIO
from torchvision import datasets, models, transforms
MODEL_FILE_NAME = 'model.pkl'  # 模型文件名称
def get_image_transform():
    """获取图片处理的transform
    Args:
        data_type: string, type of data(train/test)
    Returns:
        torchvision.transforms.Compose
    """
    trans = transforms.Compose([transforms.Resize((28, 28)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1.0,))])
    return trans
def model_fn(model_dir):
    """模型加载
    Args:
        model_dir: 模型路径，该目录存储的文件为在自动搜索作业中选择的输出路径下产出的文件
    Returns:
        加载好的模型对象
    """
    class Net(nn.Module):
        """Net"""
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
    model = Net()
    meta_info_path = "%s/%s" % (model_dir, MODEL_FILE_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(meta_info_path, map_location=device))
    model.to(device)
    logging.info("device type: " + str(device))
    return model
def input_fn(request):
    """对输入进行格式化，处理为预测需要的输入格式
    Args:
        request: api请求的json
    Returns:
        预测需要的输入数据，一般为tensor
    """
    instances = request['instances']
    transform_composes = get_image_transform()
    arr_tensor_data = []
    for instance in instances:
        decoded_data = base64.b64decode(instance['data'].encoding("utf8"))
        byte_stream = BytesIO(decoded_data)
        roiImg = Image.open(byte_stream)
        target_data = transform_composes(roiImg)
        arr_tensor_data.append(target_data)
    tensor_data = torch.stack(arr_tensor_data, dim=0)
    return tensor_data
def output_fn(predict_result):
    """进行输出格式化
    Args:
        predict_result: 预测结果
    Returns:
        格式化后的预测结果，需能够json序列化以便接口返回
    """
    js_str = None
    if type(predict_result) == torch.Tensor:
        list_prediction = predict_result.detach().cpu().numpy().tolist()
        js_str = json.dumps(list_prediction)
    return js_str