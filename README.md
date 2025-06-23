# House-Price-Interval-Prediction-Based-on-Hybrid-Decision-Trees

# 基于混合决策树的房价区间预测

## 项目简介
本项目实现了一个基于混合决策树的房价区间预测系统，提供了完整的数据清洗、特征工程和预测分析流程，重点关注异常值和缺失值的稳健处理。项目还会自动生成可视化图表和数据洞察报告，助力房地产定价的数据驱动决策。

## 主要功能
- 自动化数据清洗（缺失值、异常值、重复值处理）
- 特征工程与特征选择
- 基于混合决策树的区间预测（可扩展建模）
- 全面的数据分析与可视化
- 生成详细的数据洞察与业务建议

## 环境依赖与安装
**依赖环境：**
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

**安装依赖：**
```bash
pip install -r requirements.txt
```

## 数据说明
- `dataset.csv`：原始训练数据
- `test.csv`：原始测试数据
- 数据集应包含房屋特征及目标价格列（ `sale_price`）

## 使用方法
1. **准备数据：** 将 `dataset.csv` 和 `test.csv` 放在项目根目录下。
2. **运行主脚本：**
   ```bash
   python data.py
   ```
3. **输出文件：**
   - `train_cleaned.csv`、`test_cleaned.csv`：清洗后的数据集
   - `missing_values_analysis.png`：缺失值分布图
   - `numeric_features_distribution.png`：数值特征箱线图
   - `cleaning_comparison.png`：清洗前后对比图
   - `feature_importance.png`：特征重要性排名
   - `data_insights_report.md`：详细数据洞察报告

## 可视化与报告
项目会自动生成可视化图表和 markdown 格式的数据洞察报告，内容涵盖数据质量问题、特征重要性和业务建议，帮助用户快速理解数据集并指导后续建模。

## 项目扩展
你可以在本流程基础上，集成自己的混合决策树模型或其他机器学习算法，实现更丰富的区间预测功能。


