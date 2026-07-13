# 数学建模竞赛选题

本项目包含两个选题的解答工作。

---

## 选题 A：论文质量智能评估与优化

基于《新一代人工智能发展规划》，构建数学建模论文的智能评估系统。

| 内容 | 状态 | 说明 |
|------|:----:|------|
| 问题1：综合评价指标体系与自动评分模型 | ✅ 已完成 | 5个一级指标、20个二级指标，AHP+熵权法组合赋权 |
| 问题2：论文质量与文本特征关联分析 | ✅ 已完成 | 特征提取与相关性分析 |
| 问题3：论文优化策略与AI辅助程度评估 | ✅ 已完成 | 优化策略与评估方法 |
| 求解代码 | ✅ 已完成 | `solve_problems.py`、`extract_features.py`、`generate_report.py` |
| 结果报告 | ✅ 已完成 | 见 `A-Answer/求解结果报告.md` |
| 可视化图表 | ✅ 已完成 | `analysis_charts.png`、`optimization_charts.png` |

> 📂 文件目录：[选题A/](选题A/) — 题目与附件 | [A-Answer/](A-Answer/) — 解答与代码

---

## 选题 D：集装箱航运网络优化

**状态：⏳ 待开始**

- [ ] 题目理解与数据探索
- [ ] 问题分析与建模
- [ ] 求解代码编写
- [ ] 结果报告撰写

> 📂 文件目录：[选题D/](选题D/) — 题目、数据集与数据说明

---

## 仓库结构

```
.
├── README.md               # 本文件 — 进度跟踪
├── 选题A/                   # 题目A原始资料
│   ├── 选题A.pdf
│   ├── 附件1/  (30篇论文)
│   ├── 附件2/  (10篇待评分论文)
│   └── 附件3/  (评价标准)
├── A-Answer/               # 题目A解答
│   ├── 求解结果报告.md
│   ├── solve_problems.py
│   ├── extract_features.py
│   ├── generate_report.py
│   ├── paper_features.json
│   ├── evaluation_results.json
│   ├── analysis_charts.png
│   └── optimization_charts.png
└── 选题D/                   # 题目D原始资料
    ├── 选题D.pdf
    ├── 数据集说明.pdf
    ├── 集装箱数据描述 - 初赛.pdf
    ├── test_result.csv
    └── 数据集3713/
```
