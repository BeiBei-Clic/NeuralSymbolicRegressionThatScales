import os
import sys
import pandas as pd
import numpy as np
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from nesymres.architectures.model import Model
from nesymres.dclasses import FitParams, NNEquation, BFGSParams
from pathlib import Path
from functools import partial
from sympy import lambdify
import json
import omegaconf

def load_feynman_dataset(dataset_name, num_samples=100, train_ratio=0.8):
    """
    从Feynman_with_units目录加载指定的数据集
    
    Args:
        dataset_name: 数据集名称，如 "1.6.2", "1.6.2a" 等
        num_samples: 使用的数据样本数量，默认100
        train_ratio: 训练集比例，默认0.8
    
    Returns:
        X_train: 训练集输入特征 (torch.Tensor)
        y_train: 训练集目标值 (torch.Tensor)
        X_test: 测试集输入特征 (torch.Tensor)
        y_test: 测试集目标值 (torch.Tensor)
    """
    # 将数据集名称转换为文件路径格式
    if dataset_name.startswith("1."):
        file_name = f"I.{dataset_name[2:]}"
    else:
        file_name = dataset_name
    
    dataset_path = f"/home/xyh/NeuralSymbolicRegressionThatScales/dataset/Feynman_with_units/{file_name}"
    
    # 读取数据，只取前num_samples条
    data = np.loadtxt(dataset_path)
    data = data[:num_samples]
    
    # 分离输入特征和目标值
    X = data[:, :-1]  # 除最后一列外的所有列作为输入特征
    y = data[:, -1]   # 最后一列作为目标值
    
    # 计算训练集和测试集的分割点
    train_size = int(len(data) * train_ratio)
    
    # 划分训练集和测试集
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # 转换为torch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    return X_train, y_train, X_test, y_test

## Load equation configuration and architecture configuration
# Adjust path to be relative to the project root
with open('jupyter/100M/eq_setting.json', 'r') as json_file:
  eq_setting = json.load(json_file)

# Adjust path to be relative to the project root
cfg = omegaconf.OmegaConf.load("jupyter/100M/config.yaml")

## Set up BFGS load rom the hydra config yaml
bfgs = BFGSParams(
        activated= cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )

params_fit = FitParams(word2id=eq_setting["word2id"], 
                            id2word={int(k): v for k,v in eq_setting["id2word"].items()}, 
                            una_ops=eq_setting["una_ops"], 
                            bin_ops=eq_setting["bin_ops"], 
                            total_variables=list(eq_setting["total_variables"]),
                            total_coefficients=list(eq_setting["total_coefficients"]),
                            rewrite_functions=list(eq_setting["rewrite_functions"]),
                            bfgs=bfgs,
                            beam_size=cfg.inference.beam_size #This parameter is a tradeoff between accuracy and fitting time
                            )

# Adjust path to be relative to the project root
weights_path = "weights/100M.ckpt"

## Load architecture, set into eval mode, and pass the config parameters
model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
model.eval()
if torch.cuda.is_available(): 
  model.cuda()

fitfunc = partial(model.fitfunc,cfg_params=params_fit)

if __name__ == "__main__":
    # 指定要处理的数据集
    datasets = ["1.6.2", "1.6.2a", "1.12.1", "1.12.5", "1.14.4", "1.25.13"]
    
    for dataset_name in datasets:
        print(f"\n正在处理数据集: {dataset_name}")
        print("=" * 50)
        
        # 加载数据集（前100条数据，80%训练，20%测试）
        X_train, y_train, X_test, y_test = load_feynman_dataset(dataset_name)
        
        print(f"数据集 {dataset_name} 加载完成")
        print(f"  训练集 X 形状: {X_train.shape}")
        print(f"  训练集 y 形状: {y_train.shape}")
        print(f"  测试集 X 形状: {X_test.shape}")
        print(f"  测试集 y 形状: {y_test.shape}")
        print(f"  训练集 X 范围: [{X_train.min():.4f}, {X_train.max():.4f}]")
        print(f"  训练集 y 范围: [{y_train.min():.4f}, {y_train.max():.4f}]")
        print(f"  测试集 X 范围: [{X_test.min():.4f}, {X_test.max():.4f}]")
        print(f"  测试集 y 范围: [{y_test.min():.4f}, {y_test.max():.4f}]")
        
        # 确保数据在正确的设备上
        if torch.cuda.is_available():
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            X_test = X_test.cuda()
            y_test = y_test.cuda()
        
        # 在训练集上运行符号回归
        print("在训练集上开始符号回归...")
        output = fitfunc(X_train, y_train)
        
        print(f"符号回归结果:")
        print(output)
        
        # 如果有最佳预测结果，可以在测试集上评估
        if 'best_bfgs_preds' in output and output['best_bfgs_preds']:
            print(f"\n在测试集上评估最佳预测结果...")
            best_pred = output['best_bfgs_preds'][0]
            print(f"最佳预测公式: {best_pred}")
            
        print("=" * 50)
