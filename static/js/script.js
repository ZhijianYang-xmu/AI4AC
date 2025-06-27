document.addEventListener('DOMContentLoaded', function() {
    // 获取DOM元素
    const smilesInput = document.getElementById('smilesInput');
    const predictBtn = document.getElementById('predictBtn');
    const predictText = document.getElementById('predictText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorAlert = document.getElementById('errorAlert');
    const moleculeInfo = document.getElementById('moleculeInfo');
    const downloadPlotBtn = document.getElementById('downloadPlotBtn');
    const downloadDataBtn = document.getElementById('downloadDataBtn');
    
    // 图表配置
    const ctx = document.getElementById('spectrumChart').getContext('2d');
    let spectrumChart = null;
    
    // 当前预测数据
    let currentPrediction = null;
    
    // 初始化图表
    function initChart() {
        if (spectrumChart) {
            spectrumChart.destroy();
        }
        
        spectrumChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: '红外光谱',
                    data: [],
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: '波数 (cm⁻¹)',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        reverse: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: '归一化吸光度',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            title: function(tooltipItems) {
                                return '波数: ' + tooltipItems[0].label + ' cm⁻¹';
                            },
                            label: function(context) {
                                return '吸光度: ' + context.parsed.y.toFixed(4);
                            }
                        }
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeOutQuart'
                }
            }
        });
    }
    
    // 预测按钮点击事件
    predictBtn.addEventListener('click', function() {
        const smiles = smilesInput.value.trim();
        
        if (!smiles) {
            showError('请输入SMILES字符串');
            return;
        }
        
        // 显示加载状态
        predictBtn.disabled = true;
        predictText.textContent = '预测中...';
        loadingSpinner.classList.remove('d-none');
        hideError();
        
        // 发送预测请求
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ smiles: smiles })
        })
        .then(response => response.json())
        .then(data => {
            // 恢复按钮状态
            resetButtonState();
            
            if (data.success) {
                // 保存当前预测数据
                currentPrediction = data;
                
                // 更新分子信息
                moleculeInfo.innerHTML = `
                    <strong>分子名称:</strong> ${data.mol_name}<br>
                    <strong>SMILES:</strong> ${data.smiles}
                `;
                
                // 更新图表
                updateChart(data.wavenumbers, data.spectrum_smooth);
                
                // 启用下载按钮
                downloadPlotBtn.disabled = false;
                downloadDataBtn.disabled = false;
            } else {
                showError(data.error || '预测失败，请重试');
            }
        })
        .catch(error => {
            resetButtonState();
            showError('网络错误: ' + error.message);
        });
    });
    
    // 下载光谱图按钮
    downloadPlotBtn.addEventListener('click', function() {
        if (!currentPrediction) return;
        
        // 发送下载请求
        fetch('/download_plot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                smiles: currentPrediction.smiles,
                mol_name: currentPrediction.mol_name,
                wavenumbers: currentPrediction.wavenumbers,
                spectrum_smooth: currentPrediction.spectrum_smooth
            })
        })
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${currentPrediction.mol_name.replace(/\//g, '_')}_ir_spectrum.png`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
        })
        .catch(error => {
            showError('下载失败: ' + error.message);
        });
    });
    
    // 下载数据按钮
    downloadDataBtn.addEventListener('click', function() {
        if (!currentPrediction) return;
        
        // 发送下载请求
        fetch('/download_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                smiles: currentPrediction.smiles,
                mol_name: currentPrediction.mol_name,
                wavenumbers: currentPrediction.wavenumbers,
                spectrum: currentPrediction.spectrum
            })
        })
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${currentPrediction.mol_name.replace(/\//g, '_')}_ir_data.txt`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
        })
        .catch(error => {
            showError('下载失败: ' + error.message);
        });
    });
    
    // 更新图表函数
    function updateChart(wavenumbers, spectrum) {
        if (!spectrumChart) initChart();
        
        // 更新图表数据
        spectrumChart.data.labels = wavenumbers;
        spectrumChart.data.datasets[0].data = spectrum;
        
        // 更新图表配置
        spectrumChart.options.scales.x.min = Math.min(...wavenumbers);
        spectrumChart.options.scales.x.max = Math.max(...wavenumbers);
        
        // 添加区域注释
        addRegionAnnotations(wavenumbers);
        
        // 更新图表
        spectrumChart.update();
    }
    
    // 添加红外区域标注
    function addRegionAnnotations(wavenumbers) {
        const regions = [
            {xMin: 2800, xMax: 3000, label: 'C-H伸缩', color: 'rgba(255, 0, 0, 0.1)'},
            {xMin: 1600, xMax: 1700, label: 'C=O伸缩', color: 'rgba(0, 128, 0, 0.1)'},
            {xMin: 3200, xMax: 3600, label: 'O-H/N-H伸缩', color: 'rgba(0, 0, 255, 0.1)'},
            {xMin: 1000, xMax: 1300, label: 'C-O伸缩', color: 'rgba(255, 165, 0, 0.1)'}
        ];
        
        const annotations = {};
        
        regions.forEach((region, index) => {
            // 只添加在范围内的区域
            if (region.xMin >= Math.min(...wavenumbers) && region.xMax <= Math.max(...wavenumbers)) {
                annotations[`region${index}`] = {
                    type: 'box',
                    xMin: region.xMin,
                    xMax: region.xMax,
                    backgroundColor: region.color,
                    borderColor: region.color.replace('0.1', '0.5'),
                    borderWidth: 1,
                    label: {
                        display: true,
                        content: region.label,
                        position: 'start',
                        backgroundColor: region.color.replace('0.1', '0.8'),
                        font: {
                            size: 10
                        }
                    }
                };
            }
        });
        
        spectrumChart.options.plugins.annotation = {
            annotations: annotations
        };
    }
    
    // 重置按钮状态
    function resetButtonState() {
        predictBtn.disabled = false;
        predictText.textContent = '预测红外光谱';
        loadingSpinner.classList.add('d-none');
    }
    
    // 显示错误信息
    function showError(message) {
        errorAlert.textContent = message;
        errorAlert.classList.remove('d-none');
    }
    
    // 隐藏错误信息
    function hideError() {
        errorAlert.classList.add('d-none');
    }
    
    // 初始化图表
    initChart();
});