"""
红外光谱预测模型 - 预处理工具函数库
包含所有数据预处理和特征提取功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
import joblib
from scipy import signal
import warnings

# 设置全局参数
warnings.filterwarnings('ignore')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black'
})
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.2})

def extract_molecular_features(smiles):
    """
    从SMILES中提取分子特征
    
    Args:
        smiles (str): 分子的SMILES字符串
    
    Returns:
        np.array: 分子特征向量，如果SMILES无效则返回None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Morgan指纹 (半径2, 2048位)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.array(fp)
    
    # 精选分子描述符
    desc_list = [
        'MolWt', 'HeavyAtomCount', 'NumRotatableBonds', 
        'TPSA', 'MolLogP', 'NumHDonors', 'NumHAcceptors',
        'NumValenceElectrons', 'NumAromaticRings'
    ]
    calc_desc = [Descriptors.__dict__[d] for d in desc_list]
    desc_vals = [d(mol) for d in calc_desc]
    
    # 原子组成统计
    atom_counts = [mol.GetNumAtoms(onlyExplicit=True)]
    for a in [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]:  # 包含H及其他元素
        atom_counts.append(len([x for x in mol.GetAtoms() if x.GetAtomicNum() == a]))
    
    # 键类型统计
    bond_types = [bt for bt in range(1, 5)]  # 单键、双键、三键、芳香键
    bond_counts = [len([b for b in mol.GetBonds() if b.GetBondTypeAsDouble() == bt]) for bt in bond_types]
    
    return np.concatenate([fp_array, desc_vals, atom_counts, bond_counts])


def load_and_preprocess_data(csv_file):
    """
    加载并预处理数据
    
    Args:
        csv_file (str): CSV文件路径
    
    Returns:
        tuple: (features, spectra, wavenumbers, valid_indices)
    """
    # 加载数据
    data = pd.read_csv(csv_file)
    
    # 提取SMILES和光谱数据
    smiles_list = data.iloc[:, 0].values
    spectra = data.iloc[:, 1:].values
    
    # 生成波数轴 (500-3900 cm^{-1})
    n_points = spectra.shape[1]
    wavenumbers = np.linspace(500, 3900, num=n_points)
    print(f"光谱点数: {n_points}, 波数范围: {wavenumbers[0]:.1f}-{wavenumbers[-1]:.1f} cm⁻¹")
    
    # 提取所有分子的特征
    features = []
    valid_indices = []
    
    for i, smi in enumerate(smiles_list):
        feat = extract_molecular_features(smi)
        if feat is not None:
            features.append(feat)
            valid_indices.append(i)
    
    # 过滤无效分子
    features = np.array(features)
    spectra = spectra[valid_indices]
    print(f"有效分子数量: {len(features)}, 特征维度: {features.shape}")
    
    return features, spectra, wavenumbers, valid_indices


def preprocess_features_and_spectra(features, spectra):
    """
    预处理特征和光谱数据
    
    Args:
        features (np.array): 分子特征矩阵
        spectra (np.array): 光谱数据矩阵
    
    Returns:
        tuple: (X_scaled, y_scaled, feature_scaler, spectra_scaler)
    """
    # 特征标准化
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(features)
    
    # 光谱预处理 - 全局归一化
    spectra_normalized = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        max_val = np.max(spectra[i])
        if max_val > 1e-6:
            spectra_normalized[i] = spectra[i] / max_val
        else:
            spectra_normalized[i] = spectra[i]
    
    # 全局标准化
    spectra_scaler = StandardScaler()
    y_scaled = spectra_scaler.fit_transform(spectra_normalized)
    
    return X_scaled, y_scaled, feature_scaler, spectra_scaler


def save_scalers(feature_scaler, spectra_scaler, feature_path='feature_scaler.pkl', spectra_path='spectra_scaler.pkl'):
    """
    保存标准化器
    
    Args:
        feature_scaler: 特征标准化器
        spectra_scaler: 光谱标准化器
        feature_path (str): 特征标准化器保存路径
        spectra_path (str): 光谱标准化器保存路径
    """
    joblib.dump(feature_scaler, feature_path)
    joblib.dump(spectra_scaler, spectra_path)
    print(f"标准化器已保存: {feature_path}, {spectra_path}")


def load_scalers(feature_path='feature_scaler.pkl', spectra_path='spectra_scaler.pkl'):
    """
    加载标准化器
    
    Args:
        feature_path (str): 特征标准化器路径
        spectra_path (str): 光谱标准化器路径
    
    Returns:
        tuple: (feature_scaler, spectra_scaler)
    """
    feature_scaler = joblib.load(feature_path)
    spectra_scaler = joblib.load(spectra_path)
    return feature_scaler, spectra_scaler


def spectral_similarity(y_true, y_pred):
    """
    计算光谱相似度 - 余弦相似度
    
    Args:
        y_true (np.array): 真实光谱
        y_pred (np.array): 预测光谱
    
    Returns:
        np.array: 相似度分数数组
    """
    epsilon = 1e-8
    similarity_scores = []
    for i in range(y_true.shape[0]):
        dot_product = np.dot(y_true[i], y_pred[i])
        norm_true = np.linalg.norm(y_true[i])
        norm_pred = np.linalg.norm(y_pred[i])
        similarity = dot_product / (norm_true * norm_pred + epsilon)
        similarity_scores.append(similarity)
    return np.array(similarity_scores)


def get_molecule_name(smiles):
    """
    获取分子名称（简化的SMILES）
    
    Args:
        smiles (str): SMILES字符串
    
    Returns:
        str: 分子名称
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        name = Chem.MolToSmiles(mol, canonical=True)
        return name if len(name) < 25 else name[:22] + "..."
    return "未知分子"


def plot_training_history(history, save_path='training_history.png'):
    """
    绘制训练历史曲线
    
    Args:
        history: Keras训练历史对象
        save_path (str): 保存路径
    """
    plt.figure(figsize=(12, 5), dpi=300)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Test')
    plt.title('Loss Curve')
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training')
    plt.plot(history.history['val_mae'], label='Test')
    plt.title('Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_similarity_distribution(similarity_scores, save_path='similarity_distribution.png'):
    """
    绘制光谱相似度分布
    
    Args:
        similarity_scores (np.array): 相似度分数数组
        save_path (str): 保存路径
    """
    if len(similarity_scores) > 0:
        plt.figure(figsize=(8, 5), dpi=300)
        sns.histplot(similarity_scores, bins=30, kde=True, color='#1f77b4', 
                     edgecolor='white', linewidth=0.5, alpha=0.8)
        
        mean_sim = np.nanmean(similarity_scores)
        median_sim = np.nanmedian(similarity_scores)
        plt.axvline(mean_sim, color='#d62728', linestyle='dashed', linewidth=1.5, 
                    label=f'Average: {mean_sim:.3f}')
        plt.axvline(median_sim, color='#2ca02c', linestyle='dashed', linewidth=1.5, 
                    label=f'Median: {median_sim:.3f}')
        
        plt.xlabel('Cosine Similarity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Spectral Similarity Distribution of Test Set', fontsize=14, pad=15)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.15)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()


def plot_spectrum_predictions(y_test_original, y_pred_original, similarity_scores, 
                            wavenumbers, test_smiles, save_path='spectra_predictions.png'):
    """
    绘制光谱预测对比图
    
    Args:
        y_test_original: 原始测试光谱
        y_pred_original: 原始预测光谱
        similarity_scores: 相似度分数
        wavenumbers: 波数轴
        test_smiles: 测试集SMILES列表
        save_path (str): 保存路径
    """
    if len(test_smiles) > 0:
        example_indices = list(range(min(5, len(test_smiles))))
        
        plt.figure(figsize=(10, 8), dpi=300)
        for i, idx in enumerate(example_indices):
            plt.subplot(len(example_indices), 1, i+1)
            
            actual = y_test_original[idx]
            pred = y_pred_original[idx]
            
            # 应用平滑
            actual_smooth = signal.savgol_filter(actual, 51, 3)
            pred_smooth = signal.savgol_filter(pred, 51, 3)
            
            name = get_molecule_name(test_smiles[idx])
            
            plt.plot(wavenumbers, actual_smooth, 'b-', lw=1.8, alpha=0.9, label='Experiment')
            plt.plot(wavenumbers, pred_smooth, 'r--', lw=1.8, alpha=0.9, label='Model')
            
            plt.gca().invert_xaxis()
            plt.xlim(wavenumbers[-1], wavenumbers[0])
            plt.title(f'{name} - Similarity: {similarity_scores[idx]:.3f}', fontsize=11)
            plt.grid(alpha=0.15)
            plt.legend(fontsize=9, loc='upper right')
            
            if i == len(example_indices)-1:
                plt.xlabel('wavenumbers', fontsize=11)
            plt.ylabel('Absorbance (a.u.)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()


def smooth_spectrum(spectrum, wavenumbers, window_size=None):
    """
    平滑光谱数据
    
    Args:
        spectrum (np.array): 光谱数据
        wavenumbers (np.array): 波数轴
        window_size (int): 窗口大小，默认自动计算
    
    Returns:
        np.array: 平滑后的光谱
    """
    if window_size is None:
        window_size = min(51, len(spectrum) // 10)
    if window_size % 2 == 0:
        window_size += 1  # 确保窗口大小为奇数
    
    return signal.savgol_filter(spectrum, window_size, 3)
