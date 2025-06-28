"""
红外光谱交互式预测程序
运行后输入SMILES，实时预测并显示红外光谱图
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
import sys

# 导入预处理工具
from preprocessing import (
    extract_molecular_features,
    load_scalers,
    get_molecule_name,
    smooth_spectrum
)

# 设置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class InteractiveIRPredictor:
    """交互式红外光谱预测器"""
    
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
            print("❌ 错误: 以下必需文件缺失:")
            for file in missing_files:
                print(f"   - {file}")
            print("\n请确保已完成模型训练并生成了所有必需文件")
            sys.exit(1)
        
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
            print(f"❌ 初始化失败: {str(e)}")
            sys.exit(1)
    
    def predict_spectrum(self, smiles):
        """
        预测给定SMILES的红外光谱
        
        Args:
            smiles (str): 分子的SMILES字符串
        
        Returns:
            tuple: (spectrum, success, error_message)
        """
        try:
            # 提取分子特征
            features = extract_molecular_features(smiles)
            if features is None:
                return None, False, "无效的SMILES字符串"
                
            # 预处理
            features = self.feature_scaler.transform([features])
            
            # 预测
            pred_norm = self.model.predict(features, verbose=0)
            
            # 反标准化
            spectrum = self.spectra_scaler.inverse_transform(pred_norm)[0]
            
            return spectrum, True, None
            
        except Exception as e:
            return None, False, f"预测过程出错: {str(e)}"
    
    def plot_spectrum(self, smiles, spectrum, save_fig=False):
        """
        绘制红外光谱图
        
        Args:
            smiles (str): SMILES字符串
            spectrum (np.array): 光谱数据
            save_fig (bool): 是否保存图片
        """
        # 应用平滑
        spectrum_smooth = smooth_spectrum(spectrum, self.wavenumbers)
        
        # 获取分子名称
        mol_name = get_molecule_name(smiles)
        
        # 创建图形
        plt.figure(figsize=(12, 7), dpi=100)
        
        # 绘制光谱
        plt.plot(self.wavenumbers, spectrum_smooth, 'b-', linewidth=2, label='Model')
        
        # 设置图形属性
        plt.title(f'红外光谱预测\n分子: {mol_name}\nSMILES: {smiles}', 
                 fontsize=14, pad=20)
        plt.xlabel('Wavenumbers', fontsize=12)
        plt.ylabel('Absorbance (a.u.)', fontsize=12)
        
        # 反转X轴（红外光谱习惯）
        plt.gca().invert_xaxis()
        plt.xlim(self.wavenumbers[-1], self.wavenumbers[0])
        
        # 设置网格
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 添加重要红外区域标注
        regions = [
            (2800, 3000, 'C-H Stretching', 'red'),
            (1600, 1700, 'C=O Stretching', 'green'), 
            (3200, 3600, 'O-H/N-H Stretching', 'blue'),
            (1000, 1300, 'C-O Stretching', 'orange')
        ]
        
        for start, end, label, color in regions:
            if start >= self.wavenumbers[0] and end <= self.wavenumbers[-1]:
                plt.axvspan(start, end, alpha=0.1, color=color, label=label)
        
        plt.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        
        # 保存图片
        if save_fig:
            filename = f"{mol_name.replace('/', '_')}_{hash(smiles) % 10000:04d}_spectrum.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   图片已保存为: {filename}")
        
        plt.show()
    
    def run_interactive(self):
        """
        运行交互式预测
        """
        print("🔬 红外光谱预测器 - 交互模式")
        print("=" * 60)
        print("输入分子的SMILES字符串，程序将预测并显示红外光谱图")
        print("输入 'quit', 'exit' 或 'q' 退出程序")
        print("输入 'help' 查看使用帮助")
        print("-" * 60)
        
        while True:
            try:
                # 获取用户输入
                smiles = input("\n请输入SMILES: ").strip()
                
                # 检查退出命令
                if smiles.lower() in ['quit', 'exit', 'q']:
                    print("👋 谢谢使用，再见!")
                    break
                
                # 显示帮助
                if smiles.lower() == 'help':
                    self.show_help()
                    continue
                
                # 检查空输入
                if not smiles:
                    print("⚠️  请输入有效的SMILES字符串")
                    continue
                
                print(f"\n🔍 正在预测: {smiles}")
                
                # 预测光谱
                spectrum, success, error_msg = self.predict_spectrum(smiles)
                
                if success:
                    mol_name = get_molecule_name(smiles)
                    print(f"✅ 预测成功!")
                    print(f"   分子名称: {mol_name}")
                    print(f"   光谱维度: {len(spectrum)}")
                    
                    # 询问是否保存图片
                    save_choice = input("是否保存图片? (y/n, 默认n): ").strip().lower()
                    save_fig = save_choice in ['y', 'yes', '是']
                    
                    # 绘制光谱
                    self.plot_spectrum(smiles, spectrum, save_fig)
                    
                else:
                    print(f"❌ 预测失败: {error_msg}")
                    print("   请检查SMILES格式是否正确")
                
            except KeyboardInterrupt:
                print("\n\n👋 程序被用户中断，再见!")
                break
            except Exception as e:
                print(f"❌ 发生意外错误: {str(e)}")
                print("   请重试或联系开发者")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
📖 使用帮助
═══════════════════════════════════════════════════════════════

SMILES输入示例:
  - 乙醇: CCO
  - 苯: c1ccccc1  
  - 丙酮: CC(=O)C
  - 乙酸: CC(=O)O
  - 甲苯: Cc1ccccc1

功能说明:
  ✓ 输入SMILES后会自动识别分子名称
  ✓ 预测并显示红外光谱图
  ✓ 光谱图包含重要官能团区域标注
  ✓ 可选择是否保存预测图片

命令说明:
  - quit/exit/q: 退出程序
  - help: 显示此帮助信息

注意事项:
  ⚠️  请确保SMILES格式正确
  ⚠️  复杂分子预测时间可能较长
  ⚠️  预测结果仅供参考

═══════════════════════════════════════════════════════════════
        """
        print(help_text)


def main():
    """主函数"""
    print("🚀 启动红外光谱预测器...")
    
    try:
        # 初始化预测器
        predictor = InteractiveIRPredictor()
        
        # 运行交互式界面
        predictor.run_interactive()
        
    except Exception as e:
        print(f"❌ 程序启动失败: {str(e)}")
        print("请检查相关文件是否存在")


if __name__ == "__main__":
    main()
