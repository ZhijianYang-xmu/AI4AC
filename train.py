"""
红外光谱预测模型训练程序
运行此程序进行模型训练
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import argparse
import os

# 导入预处理工具
from preprocessing import (
    load_and_preprocess_data, 
    preprocess_features_and_spectra,
    save_scalers,
    spectral_similarity,
    plot_training_history,
    plot_similarity_distribution,
    plot_spectrum_predictions
)


def robust_mse_loss(y_true, y_pred):
    """MSE损失函数"""
    epsilon = 1e-8
    squared_diff = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_diff) + epsilon


def create_robust_model(input_dim, output_dim):
    """创建模型架构"""
    input_layer = Input(shape=(input_dim,))
    
    # 特征提取层
    x = Dense(1024, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 输出层
    output = Dense(output_dim, activation='linear')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    return model


def train_model(data_file, model_save_path='robust_ir_predictor.keras', 
                test_size=0.15, validation_split=0.15, epochs=200, batch_size=32,
                learning_rate=0.0001, patience=30, verbose=1):
    """
    训练红外光谱预测模型
    
    Args:
        data_file (str): 训练数据CSV文件路径
        model_save_path (str): 模型保存路径
        test_size (float): 测试集比例
        validation_split (float): 验证集比例
        epochs (int): 训练轮次
        batch_size (int): 批次大小
        learning_rate (float): 学习率
        patience (int): 早停耐心值
        verbose (int): 训练详细程度
    
    Returns:
        dict: 训练结果统计
    """
    
    print("=== 开始模型训练 ===")
    
    # 1. 加载和预处理数据
    print("1. 加载数据...")
    features, spectra, wavenumbers, valid_indices = load_and_preprocess_data(data_file)
    
    print("2. 预处理特征和光谱...")
    X, y, feature_scaler, spectra_scaler = preprocess_features_and_spectra(features, spectra)
    
    # 保存标准化器
    save_scalers(feature_scaler, spectra_scaler)
    
    # 3. 划分训练集和测试集
    print("3. 划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    
    # 4. 创建模型
    print("4. 创建模型...")
    model = create_robust_model(X.shape[1], y.shape[1])
    print(f"模型参数数量: {model.count_params():,}")
    
    # 5. 编译模型
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=robust_mse_loss,
        metrics=['mae']
    )
    
    # 6. 设置回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]
    
    # 7. 训练模型
    print("5. 开始训练...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose
    )
    
    # 8. 保存模型
    model.save(model_save_path)
    print(f"模型已保存至: {model_save_path}")
    
    # 9. 评估模型
    print("6. 评估模型...")
    y_pred = model.predict(X_test, verbose=0)
    
    # 反标准化
    y_pred_original = spectra_scaler.inverse_transform(y_pred)
    y_test_original = spectra_scaler.inverse_transform(y_test)
    
    # 计算指标
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred_original)
    
    # 计算光谱相似度
    similarity_scores = spectral_similarity(y_test_original, y_pred_original)
    mean_similarity = np.nanmean(similarity_scores)
    
    # 打印结果
    print("\n=== 测试集评估结果 ===")
    print(f"光谱相似度(余弦): {mean_similarity:.4f}")
    print(f"均方误差(MSE): {mse:.4f}")
    print(f"均方根误差(RMSE): {rmse:.4f}")
    print(f"R²分数: {r2:.4f}")
    
    # 10. 生成可视化
    print("7. 生成可视化图表...")
    
    # 训练历史
    plot_training_history(history)
    
    # 相似度分布
    plot_similarity_distribution(similarity_scores)
    
    # 预测示例 - 需要获取测试集对应的SMILES
    data = pd.read_csv(data_file)
    smiles_list = data.iloc[:, 0].values
    test_smiles = [smiles_list[i] for i in valid_indices[len(X_train):]]
    plot_spectrum_predictions(y_test_original, y_pred_original, similarity_scores, 
                            wavenumbers, test_smiles)
    
    # 返回训练结果
    results = {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mean_similarity': mean_similarity,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_dim': X.shape[1],
        'spectra_dim': y.shape[1],
        'epochs_trained': len(history.history['loss'])
    }
    
    print("\n=== 训练完成 ===")
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练红外光谱预测模型')
    parser.add_argument('--data', type=str, default='IR_database_full.csv', 
                       help='训练数据CSV文件路径')
    parser.add_argument('--model', type=str, default='robust_ir_predictor.keras',
                       help='模型保存路径')
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='学习率')
    parser.add_argument('--test_size', type=float, default=0.15,
                       help='测试集比例')
    parser.add_argument('--patience', type=int, default=30,
                       help='早停耐心值')
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data):
        print(f"错误: 数据文件 '{args.data}' 不存在")
        print("请确保数据文件存在，或使用 --data 参数指定正确的文件路径")
        return
    
    # 开始训练
    try:
        results = train_model(
            data_file=args.data,
            model_save_path=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            test_size=args.test_size,
            patience=args.patience
        )
        
        print(f"\n训练成功完成!")
        print(f"最终模型性能: R² = {results['r2']:.4f}, 相似度 = {results['mean_similarity']:.4f}")
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        print("请检查数据格式和参数设置")


if __name__ == "__main__":
    import pandas as pd  # 在main中导入pandas避免循环导入
    main()
