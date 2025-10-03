# NeuralSymbolicRegressionThatScales

ICML 2021 论文“Neural Symbolic Regression That Scales”的 Pytorch 实现和预训练模型。
我们基于深度学习的方法是第一个利用大规模预训练的符号回归方法。我们程序化地生成无限的方程集，并同时预训练一个 Transformer，以从相应的输入-输出对中预测符号方程。

有关详细信息，请参阅 **Neural Symbolic Regression That Scales**。 [[`arXiv`](https://arxiv.org/pdf/2106.06427.pdf)]


## 安装
请通过以下方式克隆并安装此存储库：

```
git clone https://github.com/xyh1999/NeuralSymbolicRegressionThatScales.git
cd NeuralSymbolicRegressionThatScales
pip install -r requirements.txt
```
此库需要 python>3.7

## 费曼数据集上的工作

我们正在将此项目应用于费曼数据集（Feynman Dataset），这是一个包含物理方程的数据集，旨在测试符号回归模型的泛化能力。

您可以在这里找到费曼数据集：[https://space.mit.edu/home/tegmark/aifeynman.html](https://space.mit.edu/home/tegmark/aifeynman.html)

`main.py` 文件是项目在费曼数据集上进行符号回归实验的核心脚本。它主要负责以下几个方面：

1.  **数据加载与预处理**：
    *   `load_feynman_dataset` 函数负责从 `dataset/Feynman_with_units` 目录加载指定的费曼数据集。
    *   它会根据配置的样本数量读取数据，并进行随机打乱。
    *   数据被划分为输入特征和目标值，并进一步分割为训练集和测试集。
    *   **关键步骤**：对输入特征和目标值进行**标准化**处理，这对于模型的稳定训练和泛化能力至关重要。
    *   最终，处理后的数据以 PyTorch 张量的形式返回，同时返回标准化参数以便后续反标准化。

2.  **单次实验运行**：
    *   `run_single_experiment` 函数执行单个数据集在特定随机种子下的完整实验流程。
    *   它首先调用 `load_feynman_dataset` 获取预处理数据。
    *   如果检测到 GPU，数据会被自动移动到 GPU 上进行加速计算。
    *   核心的符号回归拟合过程由 `fitfunc`（一个封装了模型推理逻辑的函数）完成。
    *   模型生成预测方程后，脚本会利用 `sympy` 库解析该方程，并将其转换为可执行的 Python 函数。
    *   随后，该函数在**测试集**上进行预测，并计算预测结果与真实值之间的**均方误差 (MSE)**，以此来量化模型的泛化性能。
    *   实验过程中还会记录模型的适应度历史（通常是训练损失）。

3.  **批量实验与结果分析**：
    *   `main.py` 的主执行块定义了一系列要进行实验的费曼数据集名称。
    *   为了确保结果的鲁棒性，每个数据集都会在多个随机种子下（默认为 10 个种子）独立运行实验。
    *   所有实验结果（包括每个种子的测试 MSE、最佳方程等）都会被收集。
    *   为了全面评估模型性能，脚本会根据测试 MSE 对所有实验结果进行排序，并提取 25%、50%（中位数）和 75% 分位数的实验结果。
    *   最后，`save_experiment_results` 函数将这些关键分位数的实验结果保存为 JSON 文件，便于后续的数据分析和报告。

通过上述流程，`main.py` 文件提供了一个系统化的框架，用于在费曼数据集上评估和分析符号回归模型的性能。



## 预训练模型
我们提供两种模型：“10M”和“100M”。两者都使用 **dataset_configuration.json**（包含数据集创建详细信息）和 **scripts/config.yaml**（包含模型训练详细信息）中显示的参数配置进行训练。“10M”模型使用 1000 万个数据集进行训练，“100M”模型使用 1 亿个数据集进行训练。

***更新***：
权重可以在这里找到：https://huggingface.co/TommasoBendinelli/NeuralSymbolicRegressionThatScales/tree/main

如果您想尝试这些模型，请查看 **jupyter/fit_func.ipynb**。在运行 notebook 之前，请务必先创建一个名为“weights”的文件夹，并将提供的检查点下载到其中。


## 数据集生成
在训练之前，您需要一个方程数据集。以下是需要遵循的步骤：

### 原始训练数据集生成
方程生成器脚本基于 [[SymbolicMathematics](https://github.com/facebookresearch/SymbolicMathematics)]
首先，如果您想更改默认值，请配置 dataset_configuration.json 文件：
```
{
    "max_len": 20, #方程的最大长度
    "operators": "add:10,mul:10,sub:5,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:2", #运算符非归一化概率
    "max_ops": 5, #最大操作数
    "rewrite_functions": "", #未使用，留空
    "variables": ["x_1","x_2","x_3"], #变量名，如果您想添加更多变量，请遵循约定，例如 x_4, x_5,... 等等
    "eos_index": 1,
    "pad_index": 0
}
```
有两种方法可以生成此数据集：

* 如果您在 Linux 上运行，可以在终端中使用 makefile，如下所示：
```
export NUM=${NumberOfEquations} #导出方程数量
make data/raw_datasets/${NUM}: #启动 makefile 命令
```
NumberOfEquations 可以用 K 或 M 后缀的两种格式定义。例如，100K 等于 100'000，而 10M 等于 10'0000000
例如，如果您想创建一个 10M 数据集，只需：

```
export NUM=10M #导出 NUM 变量
make data/raw_datasets/10M: #启动 makefile 命令
```

* 运行此脚本：
```
python3 scripts/data_creation/dataset_creation.py --number_of_equations NumberOfEquations --no-debug #将 NumberOfEquations 替换为您要生成的方程数量
```

此命令执行后，您将拥有一个名为 **data/raw_data/NumberOfEquations** 的文件夹，其中包含 .h5 文件。默认情况下，每个 .h5 文件最多包含 5e4 个方程。


### 原始测试数据集生成
此步骤是可选的。如果您想使用我们论文中使用的测试集（位于 **test_set/nc.csv**），则可以跳过此步骤。
使用与之前相同的命令生成验证数据集。此数据集中的所有方程都将在下一阶段从训练数据集中删除，因此此验证数据集应该很小。在我们的论文中，它包含 200 个方程。

```
#生成 150 个方程数据集的代码
python3 scripts/data_creation/dataset_creation.py --number_of_equations 150 --no-debug #此代码创建一个新文件夹 data/raw_datasets/150
```

如果您愿意，可以将新创建的验证数据集转换为 csv 格式。
为此，请运行：`python3 scripts/csv_handling/dataload_format_to_csv.py raw_test_path=data/raw_datasets/150`
此命令将在 test_set 文件夹中创建两个 csv 文件，分别名为 test_nc.csv（不带常量的方程）和 test_wc.csv（带常量的方程）。

### 从训练数据集中删除测试和数值问题方程
以下步骤将从训练集中删除验证方程，并删除始终为 nan、inf 等的方程。
* `path_to_data_folder=data/raw_datasets/100000` 如果您创建了 100K 数据集
* `path_to_csv=test_set/test_nc.csv` 如果您创建了 150 个方程用于验证。如果您想使用论文中的，请将其替换为 `nc.csv`
```
python3 scripts/data_creation/filter_from_already_existing.py --data_path path_to_data_folder --csv_path path_to_csv #如果您不想创建验证集，可以将 csv_path 留空
python3 scripts/data_creation/apply_filtering.py --data_path path_to_data_folder 
```
您现在应该有一个名为 data/datasets/100000 的文件夹。这将是训练文件夹。

## 训练
创建训练和验证数据集后，运行
```
python3 scripts/train.py
```
您可以配置 config.yaml，其中包含必要的选项。最重要的是，请确保您已正确设置 train_path 和 val_path。如果您遵循 100K 示例，则应将其设置为：
```
train_path:  data/datasets/100000
val_path: data/raw_datasets/150
```