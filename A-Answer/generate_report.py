"""
第三阶段：生成完整的结果报告和可视化
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Setup Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

with open('paper_features.json', 'r', encoding='utf-8') as f:
    all_papers = json.load(f)
with open('evaluation_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# ============================================================
# 图1：30篇论文质量分级分布饼图
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1.1 分级分布饼图
ax = axes[0, 0]
grades = results['问题1_评价体系']['分级统计']
grade_order = ['优秀', '良好', '中等', '及格', '不及格']
sizes = [grades.get(g, 0) for g in grade_order]
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']
explode = (0.05, 0.02, 0, 0, 0.05)
wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=grade_order,
                                   colors=colors, autopct='%1.1f%%',
                                   shadow=False, startangle=90)
ax.set_title('30篇论文质量分级分布', fontsize=14, fontweight='bold')

# 1.2 评分排序条形图
ax = axes[0, 1]
paper_names = [r['论文'] for r in results['问题1_评价体系']['评分结果']]
scores = [r['得分'] for r in results['问题1_评价体系']['评分结果']]
bars = ax.bar(range(len(paper_names)), scores,
              color=[colors[grade_order.index(r['等级'])] for r in results['问题1_评价体系']['评分结果']])
ax.set_xlabel('论文编号')
ax.set_ylabel('综合得分')
ax.set_title('30篇论文综合评分排名', fontsize=14, fontweight='bold')
ax.set_xticks(range(0, 30, 5))
ax.set_xticklabels([paper_names[i].replace('.pdf', '') for i in range(0, 30, 5)])
ax.axhline(y=85, color='green', linestyle='--', alpha=0.5, label='优秀线(85)')
ax.axhline(y=70, color='blue', linestyle='--', alpha=0.5, label='良好线(70)')
ax.axhline(y=55, color='orange', linestyle='--', alpha=0.5, label='中等线(55)')
ax.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='及格线(40)')
ax.legend(loc='upper right', fontsize=7)

# 1.3 一级指标与总分相关性
ax = axes[0, 2]
dimensions = ['逻辑严密性', '方法合理性', '结构规范性', '内容充实度', '表达规范性']
correlations = [0.798, 0.540, 0.750, 0.039, 0.759]
bars = ax.bar(dimensions, correlations, color=['#2c3e50', '#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
ax.set_ylabel('Pearson相关系数')
ax.set_title('各维度与总分的相关性', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1)
for bar, val in zip(bars, correlations):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}',
            ha='center', va='bottom', fontsize=10)

# 2.1 问题2关键特征相关系数
ax = axes[1, 0]
key_features_10 = results['问题2_关联分析']['关键特征'][:8]
feat_names = [f[0][:20] for f in key_features_10]
feat_corrs = [f[1] for f in key_features_10]
colors_corr = ['#e74c3c' if c < 0 else '#2ecc71' for c in feat_corrs]
bars = ax.barh(range(len(feat_names)), feat_corrs, color=colors_corr)
ax.set_yticks(range(len(feat_names)))
ax.set_yticklabels(feat_names, fontsize=8)
ax.set_xlabel('Pearson相关系数')
ax.set_title('关键文本特征与质量的相关系数', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.5)
for bar, val in zip(bars, feat_corrs):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:+.3f}', va='center', fontsize=9)

# 2.2 质量调整因子图
ax = axes[1, 1]
a2_papers = ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']
alphas = [1.434, 0.949, 0.623, 0.872, 1.190, 1.440, 1.196, 0.0, 1.359, 0.938]
bars = ax.bar(a2_papers, alphas,
              color=['#2ecc71' if a > 1 else '#e74c3c' for a in alphas])
ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
ax.set_ylabel('质量调整因子α')
ax.set_title('附件2论文质量调整因子', fontsize=14, fontweight='bold')
for bar, val in zip(bars, alphas):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=8)

# 2.3 预测模型系数
ax = axes[1, 2]
coeffs = results['问题2_关联分析']['特征系数']
coef_names = ['字数', '参考文献', '方法多样', '段落变异', '关键词', '有参考文献', '模型假设', '有结论']
coef_vals = list(coeffs.values())
bars = ax.barh(range(len(coef_names)), coef_vals,
               color=['#2ecc71' if v > 0 else '#e74c3c' for v in coef_vals])
ax.set_yticks(range(len(coef_names)))
ax.set_yticklabels(coef_names, fontsize=8)
ax.set_xlabel('回归系数')
ax.set_title('岭回归模型特征系数', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.5)

plt.tight_layout(pad=3.0)
plt.savefig('analysis_charts.png', dpi=150, bbox_inches='tight')
print("图表已保存至 analysis_charts.png")

# ============================================================
# 图2：问题3的详细图表
# ============================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

p3_papers = ['3-1 (海口)', '3-2 (太原)', '3-3 (北京)']
p3_scores = [68.94, 72.90, 79.34]
p3_ai = [23.93, 30.31, 27.40]
p3_methods = [10, 10, 13]

# 质量得分对比
ax = axes2[0]
bars = ax.bar(p3_papers, p3_scores, color=['#f39c12', '#f39c12', '#3498db'])
ax.set_ylabel('综合得分')
ax.set_title('附件3论文当前得分对比', fontsize=12, fontweight='bold')
ax.axhline(y=70, color='green', linestyle='--', label='良好线')
ax.axhline(y=55, color='orange', linestyle='--', label='中等线')
ax.legend(fontsize=8)
for bar, val in zip(bars, p3_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', fontsize=10)

# AI痕迹得分
ax = axes2[1]
bars = ax.bar(p3_papers, p3_ai, color=['#3498db', '#e74c3c', '#f39c12'])
ax.set_ylabel('AI生成痕迹 (%)')
ax.set_title('AI生成痕迹检测结果', fontsize=12, fontweight='bold')
for bar, val in zip(bars, p3_ai):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', fontsize=10)

# 方法多样性
ax = axes2[1]
bars = ax.bar(p3_papers, p3_methods, color=['#2c3e50', '#2c3e50', '#2c3e50'])
ax.set_ylabel('方法多样性')
ax.set_title('建模范式多样性', fontsize=12, fontweight='bold')
for bar, val in zip(bars, p3_methods):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val}', ha='center', fontsize=10)

# 优化预测对比
ax = axes2[2]
before_scores = [68.94, 72.90, 79.34]
after_scores = [71.94, 73.90, 79.34]
x_pos = range(len(p3_papers))
width = 0.35
bars1 = ax.bar([x - width/2 for x in x_pos], before_scores, width, label='优化前', color='#e74c3c')
bars2 = ax.bar([x + width/2 for x in x_pos], after_scores, width, label='优化后', color='#2ecc71')
ax.set_ylabel('综合得分')
ax.set_title('优化前后得分预测对比', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(p3_papers)
ax.legend(fontsize=8)
for bar, val in zip(bars1, before_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}', ha='center', fontsize=8)
for bar, val in zip(bars2, after_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('optimization_charts.png', dpi=150, bbox_inches='tight')
print("优化图表已保存至 optimization_charts.png")

print("\n所有结果已生成完毕！")
