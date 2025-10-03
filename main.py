import os
import sys
import pandas as pd
import numpy as np
import torch
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from nesymres.architectures.model import Model
from nesymres.dclasses import FitParams, NNEquation, BFGSParams
from pathlib import Path
from functools import partial
from sympy import lambdify
import omegaconf

def load_feynman_dataset(dataset_name, num_samples=10, train_ratio=0.8, random_seed=0):
    """
    从Feynman_with_units目录加载指定的数据集，并进行标准化
    
    Args:
        dataset_name: 数据集名称，如 "1.6.2", "1.6.2a" 等
        num_samples: 使用的数据样本数量，默认100
        train_ratio: 训练集比例，默认0.8
        random_seed: 随机种子
    
    Returns:
        X_train: 标准化后的训练集输入特征 (torch.Tensor)
        y_train: 标准化后的训练集目标值 (torch.Tensor)
        X_test: 标准化后的测试集输入特征 (torch.Tensor)
        y_test: 标准化后的测试集目标值 (torch.Tensor)
        scaler_X: 输入特征的标准化器
        scaler_y: 目标值的标准化器
    """
    # 设置随机种子
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # 将数据集名称转换为文件路径格式
    if dataset_name.startswith("1."):
        file_name = f"I.{dataset_name[2:]}"
    else:
        file_name = dataset_name
    
    dataset_path = f"/home/xyh/NeuralSymbolicRegressionThatScales/dataset/Feynman_with_units/{file_name}"
    
    # 读取数据，只取前num_samples条
    data = np.loadtxt(dataset_path)
    data = data[:num_samples]
    
    # 随机打乱数据
    indices = np.random.permutation(len(data))
    data = data[indices]
    
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
    
    # 标准化输入特征
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std == 0] = 1  # 避免除零错误
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    
    # 标准化目标值
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std == 0:
        y_std = 1  # 避免除零错误
    y_train_scaled = (y_train - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std
    
    # 保存标准化参数
    scaler_X = {'mean': X_mean, 'std': X_std}
    scaler_y = {'mean': y_mean, 'std': y_std}
    
    # 转换为torch张量
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test = torch.tensor(y_test_scaled, dtype=torch.float32)
    
    return X_train, y_train, X_test, y_test, scaler_X, scaler_y

def calculate_mse(y_true, y_pred):
    """计算均方误差"""
    return torch.mean((y_true - y_pred) ** 2).item()

def save_experiment_results(dataset_name, results, results_dir):
    """保存实验结果到文件"""
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存为可读的JSON格式
    json_results = {}
    for percentile, data in results.items():
        # 处理无穷大值
        test_mse = data['test_mse']
        if test_mse == float('inf'):
            test_mse = "Infinity"
        elif test_mse == float('-inf'):
            test_mse = "-Infinity"
        else:
            # 确保是Python原生float类型
            test_mse = float(test_mse)
        
        # 确保适应度历史中的所有值都是Python原生float类型
        fitness_history = []
        if data['fitness_history']:
            fitness_history = [float(x) for x in data['fitness_history']]
        
        json_results[percentile] = {
            'seed': int(data['seed']),  # 确保是Python原生int类型
            'test_mse': test_mse,
            'best_equation': str(data['best_equation']) if data['best_equation'] else None,
            'fitness_history': fitness_history
        }
    
    json_file = os.path.join(results_dir, f"{dataset_name}_results.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {json_file}")

def run_single_experiment(dataset_name, seed, fitfunc):
    """运行单次实验"""
    print(f"  种子 {seed}: ", end="", flush=True)
    
    # 加载数据集
    X_train, y_train, X_test, y_test, scaler_X, scaler_y = load_feynman_dataset(
        dataset_name, random_seed=seed
    )
    
    # 确保数据在正确的设备上
    if torch.cuda.is_available():
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()
    
    # 运行符号回归
    output = fitfunc(X_train, y_train)
    
    # 计算测试集MSE
    test_mse = float('inf')
    best_equation = None
    fitness_history = []
    
    if 'best_bfgs_preds' in output and output['best_bfgs_preds']:
        # 获取最佳预测方程字符串
        best_equation = output['best_bfgs_preds'][0]
        
        # 使用sympy解析和计算MSE
        try:
            import sympy as sp
            import numpy as np
            
            # 定义变量（根据输入特征数量动态定义）
            n_features = X_test.shape[1]
            if n_features == 1:
                x_1 = sp.symbols('x_1')
                variables = [x_1]
            elif n_features == 2:
                x_1, x_2 = sp.symbols('x_1 x_2')
                variables = [x_1, x_2]
            elif n_features == 3:
                x_1, x_2, x_3 = sp.symbols('x_1 x_2 x_3')
                variables = [x_1, x_2, x_3]
            else:
                # 对于更多特征，动态创建变量
                variables = [sp.symbols(f'x_{i+1}') for i in range(n_features)]
            
            # 解析表达式
            expr = sp.sympify(best_equation)
            
            # 转换为可调用函数
            func = sp.lambdify(variables, expr, 'numpy')
            
            # 在测试集上进行预测
            X_test_np = X_test.cpu().numpy()
            y_test_np = y_test.cpu().numpy()
            
            if n_features == 1:
                y_pred = func(X_test_np[:, 0])
            elif n_features == 2:
                y_pred = func(X_test_np[:, 0], X_test_np[:, 1])
            elif n_features == 3:
                y_pred = func(X_test_np[:, 0], X_test_np[:, 1], X_test_np[:, 2])
            else:
                # 对于更多特征，使用*解包
                y_pred = func(*[X_test_np[:, i] for i in range(n_features)])
            
            # 确保预测结果是有限的
            if np.isfinite(y_pred).all():
                test_mse = np.mean((y_test_np - y_pred) ** 2)
            else:
                test_mse = float('inf')
                
        except Exception as e:
            # 如果解析或预测失败，保持MSE为无穷大
            test_mse = float('inf')
    
    # 提取适应度历史（使用训练损失）
    if 'all_bfgs_loss' in output and output['all_bfgs_loss']:
        # 将numpy数组转换为Python float列表
        fitness_history = [float(loss) for loss in output['all_bfgs_loss']]
    elif 'best_bfgs_loss' in output and output['best_bfgs_loss']:
        fitness_history = [float(output['best_bfgs_loss'][0])]
    
    # 显示结果
    if test_mse == float('inf'):
        print("MSE = inf")
    else:
        print(f"MSE = {test_mse:.6f}")
    
    return {
        'seed': seed,
        'test_mse': test_mse,
        'best_equation': best_equation,
        'fitness_history': fitness_history
    }

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
    
    # 结果保存目录
    results_dir = "/home/xyh/NeuralSymbolicRegressionThatScales/results"
    
    # 随机种子列表
    seeds = list(range(10))  # 0-9
    
    for dataset_name in datasets:
        print(f"\n正在处理数据集: {dataset_name}")
        print("=" * 60)
        
        # 存储所有实验结果
        all_experiments = []
        
        # 运行10次实验
        for seed in seeds:
            experiment_result = run_single_experiment(dataset_name, seed, fitfunc)
            all_experiments.append(experiment_result)
        
        # 按测试集MSE排序
        all_experiments.sort(key=lambda x: x['test_mse'])
        
        # 计算分位数索引
        n_experiments = len(all_experiments)
        percentile_25_idx = int(0.25 * (n_experiments - 1))
        percentile_50_idx = int(0.50 * (n_experiments - 1))
        percentile_75_idx = int(0.75 * (n_experiments - 1))
        
        # 选择三个分位数的实验结果
        selected_results = {
            'percentile_25': all_experiments[percentile_25_idx],
            'percentile_50': all_experiments[percentile_50_idx],
            'percentile_75': all_experiments[percentile_75_idx]
        }
        
        # 打印选择的结果
        print(f"\n数据集 {dataset_name} 的选择结果:")
        print(f"0.25分位数 (种子{selected_results['percentile_25']['seed']}): MSE = {selected_results['percentile_25']['test_mse']:.6f}")
        print(f"0.50分位数 (种子{selected_results['percentile_50']['seed']}): MSE = {selected_results['percentile_50']['test_mse']:.6f}")
        print(f"0.75分位数 (种子{selected_results['percentile_75']['seed']}): MSE = {selected_results['percentile_75']['test_mse']:.6f}")
        
        # 保存结果
        save_experiment_results(dataset_name, selected_results, results_dir)
        
        print("=" * 60)
