# 🔬 红外光谱预测模型

基于深度学习的分子红外光谱预测系统，通过分子SMILES字符串预测其红外吸收光谱。

## ✨ 项目特色

- 🧬 **智能分子识别**: 从SMILES字符串自动提取分子特征
- 📊 **准确光谱预测**: 使用深度神经网络预测红外吸收光谱
- 🎨 **可视化分析**: 自动生成高质量的光谱图和训练过程可视化
- 💻 **交互式界面**: 支持实时输入SMILES并查看预测结果
- 📈 **性能评估**: 多维度评估模型性能，包括光谱相似度分析

## 🏗️ 项目结构

```
IR-Spectrum-Predictor/
├── preprocessing.py      # 数据预处理和特征提取工具函数库
├── train.py             # 模型训练脚本
├── predict.py           # 交互式预测脚本
├── README.md            # 项目说明文档
├── requirements.txt     # 依赖包列表
└── data/
    └── IR_database_full.csv  # 训练数据集（需自行准备）
```

## 📋 环境要求

### Python版本
- Python 3.7+

### 主要依赖库
```
tensorflow>=2.8.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
rdkit-pypi>=2022.3.0
scipy>=1.7.0
joblib>=1.1.0
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/your-username/IR-Spectrum-Predictor.git
cd IR-Spectrum-Predictor

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

准备包含SMILES和红外光谱数据的CSV文件，格式如下：
- 第一列：分子的SMILES字符串
- 其余列：对应波数点的红外吸收强度值

```csv
SMILES,500.0,501.0,502.0,...,3900.0
CCO,0.123,0.145,0.167,...,0.089
c1ccccc1,0.234,0.256,0.278,...,0.134
...
```

### 3. 模型训练

```bash
# 使用默认参数训练
python train.py

# 自定义参数训练
python train.py --data your_data.csv --epochs 300 --batch_size 64 --learning_rate 0.0005
```

训练过程中会自动：
- 提取分子特征（Morgan指纹 + 分子描述符）
- 数据预处理和标准化
- 模型训练和验证
- 生成训练历史图表
- 保存模型和标准化器

### 4. 交互式预测

```bash
python predict.py
```

然后按提示输入SMILES字符串，例如：
```
请输入SMILES: CCO
🔍 正在预测: CCO
✅ 预测成功!
   分子名称: CCO
   光谱维度: 1701
```

## 📊 模型架构

### 特征提取
- **Morgan指纹**: 半径2，2048位分子指纹
- **分子描述符**: 分子量、氢键供体/受体数、拓扑极性表面积等9个关键描述符
- **原子统计**: H, C, N, O, F, P, S, Cl, Br, I原子计数
- **键类型统计**: 单键、双键、三键、芳香键计数

### 神经网络结构
```
输入层 (2070维特征)
    ↓
全连接层 (1024神经元) + ReLU + BatchNorm + Dropout(0.3)
    ↓
全连接层 (1024神经元) + ReLU + BatchNorm + Dropout(0.3)
    ↓
输出层 (光谱维度) + 线性激活
```

### 训练策略
- **损失函数**: 稳健MSE损失
- **优化器**: Adam优化器
- **正则化**: Dropout + 批标准化
- **早停机制**: 验证损失30轮不改善时停止
- **学习率调度**: 损失平台期自动降低学习率

## 📈 性能评估

模型使用多个指标评估性能：

- **余弦相似度**: 衡量预测光谱与真实光谱的形状相似性
- **均方误差(MSE)**: 衡量预测值与真实值的数值差异
- **R²决定系数**: 衡量模型的解释能力
- **平均绝对误差(MAE)**: 衡量预测误差的绝对值

## 🎨 可视化功能

### 训练过程可视化
- 训练/验证损失曲线
- 训练/验证MAE曲线

### 预测结果可视化
- 光谱相似度分布直方图
- 真实vs预测光谱对比图
- 重要红外区域标注（C-H, C=O, O-H/N-H, C-O伸缩振动）

## 🔧 使用说明

### 训练参数说明
```bash
python train.py --help
```

主要参数：
- `--data`: 训练数据CSV文件路径
- `--epochs`: 训练轮次（默认200）
- `--batch_size`: 批次大小（默认32）
- `--learning_rate`: 学习率（默认0.0001）
- `--test_size`: 测试集比例（默认0.15）

### SMILES输入示例
```
乙醇: CCO
苯: c1ccccc1
丙酮: CC(=O)C
乙酸: CC(=O)O
甲苯: Cc1ccccc1
咖啡因: CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

## 📁 输出文件

训练完成后会生成以下文件：
- `robust_ir_predictor.keras`: 训练好的深度学习模型
- `feature_scaler.pkl`: 特征标准化器
- `spectra_scaler.pkl`: 光谱标准化器
- `training_history.png`: 训练历史曲线图
- `similarity_distribution.png`: 相似度分布图
- `spectra_predictions.png`: 光谱预测对比图

## ⚠️ 注意事项

1. **数据格式**: 确保CSV文件格式正确，第一列为SMILES，其余列为光谱数据
2. **内存需求**: 大型数据集可能需要16GB+内存
3. **训练时间**: 根据数据大小，训练时间从几分钟到几小时不等
4. **SMILES有效性**: 输入的SMILES必须是RDKit可解析的有效格式
5. **预测范围**: 模型预测效果取决于训练数据的覆盖范围

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📜 许可证

本项目采用MIT许可证。详见`LICENSE`文件。

## 🙏 致谢

- [RDKit](https://www.rdkit.org/): 分子信息学工具包
- [TensorFlow](https://tensorflow.org/): 深度学习框架
- [scikit-learn](https://scikit-learn.org/): 机器学习工具库

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至：your.email@example.com

---

⭐ 如果这个项目对您有帮助，请给个星标支持！
