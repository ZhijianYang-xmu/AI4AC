import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from flask import Flask, render_template, request, send_file, jsonify
from tensorflow.keras.models import load_model

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入预处理工具
from preprocessing import (
    extract_molecular_features,
    load_scalers,
    get_molecule_name,
    smooth_spectrum
)

app = Flask(__name__)

# 设置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class IRPredictor:
    """红外光谱预测器"""
    
    def __init__(self, model_path='robust_ir_predictor.keras', 
                 feature_scaler_path='feature_scaler.pkl',
                 spectra_scaler_path='spectra_scaler.pkl'):
        """
        初始化预测器
        """
        print("正在初始化红外光谱预测器...")
        
        # 检查文件是否存在
        missing_files = []
        if not os.path.exists(model_path):
            missing_files.append(f"模型文件: {model_path}")
        if not os.path.exists(feature_scaler_path):
            missing_files.append(f"特征标准化器: {feature_scaler_path}")
        if not os.path.exists(spectra_scaler_path):
            missing_files.append(f"光谱标准化器: {spectra_scaler_path}")
        
        if missing_files:
            error_msg = "❌❌ 错误: 以下必需文件缺失:"
            for file in missing_files:
                error_msg += f"\n   - {file}"
            error_msg += "\n\n请确保已完成模型训练并生成了所有必需文件"
            raise FileNotFoundError(error_msg)
        
        try:
            # 加载模型和标准化器
            self.model = load_model(model_path, compile=False)
            self.feature_scaler, self.spectra_scaler = load_scalers(
                feature_scaler_path, spectra_scaler_path
            )
            
            # 生成波数轴
            self.wavenumbers = np.linspace(500, 3900, num=self.model.output_shape[1])
            
            print("✅ 预测器初始化成功!")
            print(f"   模型输出维度: {self.model.output_shape[1]}")
            print(f"   波数范围: {self.wavenumbers[0]:.1f}-{self.wavenumbers[-1]:.1f} cm⁻¹")
            print("-" * 60)
            
        except Exception as e:
            raise RuntimeError(f"❌❌ 初始化失败: {str(e)}")
    
    def predict_spectrum(self, smiles):
        """
        预测给定SMILES的红外光谱
        
        Args:
            smiles (str): 分子的SMILES字符串
        
        Returns:
            tuple: (spectrum, success, error_message, mol_name)
        """
        try:
            # 提取分子特征
            features = extract_molecular_features(smiles)
            if features is None:
                return None, False, "无效的SMILES字符串", None
                
            # 预处理
            features = self.feature_scaler.transform([features])
            
            # 预测
            pred_norm = self.model.predict(features, verbose=0)
            
            # 反标准化
            spectrum = self.spectra_scaler.inverse_transform(pred_norm)[0]
            
            # 获取分子名称
            mol_name = get_molecule_name(smiles)
            
            return spectrum, True, None, mol_name
            
        except Exception as e:
            return None, False, f"预测过程出错: {str(e)}", None

# 初始化预测器
try:
    predictor = IRPredictor()
except Exception as e:
    print(e)
    sys.exit(1)

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求"""
    data = request.get_json()
    smiles = data.get('smiles', '').strip()
    
    if not smiles:
        return jsonify({
            'success': False,
            'error': '请输入SMILES字符串'
        })
    
    # 进行预测
    spectrum, success, error_msg, mol_name = predictor.predict_spectrum(smiles)
    
    if not success:
        return jsonify({
            'success': False,
            'error': error_msg
        })
    
    # 应用平滑
    spectrum_smooth = smooth_spectrum(spectrum, predictor.wavenumbers)
    
    # 准备返回数据
    response_data = {
        'success': True,
        'mol_name': mol_name,
        'smiles': smiles,
        'wavenumbers': predictor.wavenumbers.tolist(),
        'spectrum': spectrum.tolist(),
        'spectrum_smooth': spectrum_smooth.tolist()
    }
    
    return jsonify(response_data)

@app.route('/download_plot', methods=['POST'])
def download_plot():
    """下载光谱图"""
    data = request.get_json()
    smiles = data.get('smiles', '')
    mol_name = data.get('mol_name', 'Unknown')
    wavenumbers = np.array(data.get('wavenumbers', []))
    spectrum_smooth = np.array(data.get('spectrum_smooth', []))
    
    # 创建图形
    plt.figure(figsize=(12, 7), dpi=100)
    
    # 绘制光谱
    plt.plot(wavenumbers, spectrum_smooth, 'b-', linewidth=2, label='预测光谱')
    
    # 设置图形属性
    plt.title(f'红外光谱预测\n分子: {mol_name}\nSMILES: {smiles}', 
             fontsize=14, pad=20)
    plt.xlabel('波数 (cm⁻¹)', fontsize=12)
    plt.ylabel('归一化吸光度', fontsize=12)
    
    # 反转X轴（红外光谱习惯）
    plt.gca().invert_xaxis()
    plt.xlim(wavenumbers[-1], wavenumbers[0])
    
    # 设置网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加重要红外区域标注
    regions = [
        (2800, 3000, 'C-H伸缩', 'red'),
        (1600, 1700, 'C=O伸缩', 'green'), 
        (3200, 3600, 'O-H/N-H伸缩', 'blue'),
        (1000, 1300, 'C-O伸缩', 'orange')
    ]
    
    for start, end, label, color in regions:
        if start >= wavenumbers[0] and end <= wavenumbers[-1]:
            plt.axvspan(start, end, alpha=0.1, color=color, label=label)
    
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    # 保存图像到内存
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    # 返回图像文件
    return send_file(
        img_buffer,
        mimetype='image/png',
        as_attachment=True,
        download_name=f'{mol_name.replace("/", "_")}_ir_spectrum.png'
    )

@app.route('/download_data', methods=['POST'])
def download_data():
    """下载原始数据"""
    data = request.get_json()
    smiles = data.get('smiles', '')
    mol_name = data.get('mol_name', 'Unknown')
    wavenumbers = np.array(data.get('wavenumbers', []))
    spectrum = np.array(data.get('spectrum', []))
    
    # 创建文本数据
    txt_buffer = StringIO()
    txt_buffer.write(f"分子名称: {mol_name}\n")
    txt_buffer.write(f"SMILES: {smiles}\n\n")
    txt_buffer.write("波数(cm⁻¹)\t吸光度\t吸光度(平滑)\n")
    
    spectrum_smooth = smooth_spectrum(spectrum, wavenumbers)
    
    for i in range(len(wavenumbers)):
        txt_buffer.write(f"{wavenumbers[i]:.2f}\t{spectrum[i]:.6f}\t{spectrum_smooth[i]:.6f}\n")
    
    txt_buffer.seek(0)
    
    # 返回文本文件
    return send_file(
        BytesIO(txt_buffer.getvalue().encode('utf-8')),
        mimetype='text/plain',
        as_attachment=True,
        download_name=f'{mol_name.replace("/", "_")}_ir_data.txt'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)