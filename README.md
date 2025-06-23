# House-Price-Interval-Prediction-Based-on-Hybrid-Decision-Trees


# 基于混合决策树的房价区间预测

## 项目简介
本项目实现了一个基于混合决策树的房价区间预测系统，包含数据清洗、特征工程、区间预测建模与多方法对比分析。项目自动生成可视化图表和数据洞察报告，助力房地产定价的数据驱动决策。

## 主要模块
- **data.py**：数据加载、探索、清洗、特征工程、可视化与报告生成。
- **module.py**：混合决策树区间预测模型，支持多种不确定性建模方法（超快、改进、自适应），并输出预测区间及可视化。
- **compare_methods.py**：对比不同区间预测方法的效果和效率，输出对比结果。

## 环境依赖与安装
**依赖环境：**
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm

**安装依赖：**
```bash
pip install -r requirements.txt
```

## 数据说明
- `dataset.csv`：原始训练数据
- `test.csv`：原始测试数据
- 数据集应包含房屋特征及目标价格列（如 `sale_price`）

## 使用方法
### 1. 数据清洗与特征工程
运行 data.py，完成数据预处理、特征工程和基础可视化：
```bash
python data.py
```
输出：
- `train_cleaned.csv`、`test_cleaned.csv`：清洗后的数据集
- `missing_values_analysis.png`、`numeric_features_distribution.png`、`cleaning_comparison.png`、`feature_importance.png`：数据分析可视化
- `data_insights_report.md`：数据洞察报告

### 2. 区间预测建模
运行 module.py，进行区间预测建模与分析：
```bash
python module.py
```
输出：
- `submission.csv`：预测区间结果（包含下界、上界）
- `prediction_intervals_analysis.png`：区间分布与分析可视化

### 3. 多方法对比分析
运行 compare_methods.py，对比不同区间预测方法的效果和效率：
```bash
python compare_methods.py
```
输出：
- 控制台打印各方法的运行时间、区间宽度等对比结果

## 可视化与报告
项目自动生成多种可视化图表和 markdown 格式报告，涵盖数据质量、特征重要性、预测区间分布等，帮助用户快速理解数据和模型表现。

## 项目扩展
你可以在本流程基础上，集成自己的混合决策树模型或其他机器学习算法，实现更丰富的区间预测功能。

