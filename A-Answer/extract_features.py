"""
第一阶段：提取所有PDF论文的文本内容和量化特征
"""
import PyPDF2
import os
import re
import json
import numpy as np
from collections import Counter

def extract_text_from_pdf(filepath):
    """从PDF提取全部文本"""
    try:
        reader = PyPDF2.PdfReader(filepath)
        full_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                full_text += t + "\n"
        return full_text, len(reader.pages)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return "", 0

def compute_features(text, num_pages):
    """计算论文的量化文本特征"""
    features = {}

    # 1. 篇幅特征
    features['total_chars'] = len(text)
    features['total_pages'] = num_pages
    features['total_lines'] = len(text.split('\n'))
    # 估算字数（中文字符数）
    chinese_chars = len(re.findall(r'[一-鿿]', text))
    features['chinese_chars'] = chinese_chars

    # 2. 结构特征
    # 检测摘要
    features['has_abstract'] = 1 if '摘要' in text[:2000] else 0
    features['abstract_position'] = text[:2000].find('摘要') if '摘要' in text[:2000] else -1

    # 检测关键词
    keywords_match = re.findall(r'关键词[：:]\s*(.+)', text[:3000])
    features['has_keywords'] = 1 if keywords_match else 0
    features['keyword_count'] = len(keywords_match[0].split('；')) if keywords_match else 0

    # 检测问题重述
    features['has_problem_restate'] = 1 if ('问题重述' in text or '问题提出' in text) else 0

    # 检测模型假设
    features['has_model_assumption'] = 1 if ('模型假设' in text or '基本假设' in text) else 0

    # 检测模型建立
    features['has_model_building'] = 1 if ('模型建立' in text or '模型构建' in text or '建模' in text) else 0

    # 检测模型求解
    features['has_model_solving'] = 1 if ('模型求解' in text or '求解' in text) else 0

    # 检测结果分析
    features['has_result_analysis'] = 1 if ('结果分析' in text or '分析' in text) else 0

    # 检测结论
    features['has_conclusion'] = 1 if '结论' in text else 0

    # 检测参考文献
    features['has_references'] = 1 if '参考文献' in text else 0
    ref_count = len(re.findall(r'\[\d+\]', text))
    features['reference_count'] = ref_count

    # 检测附录
    features['has_appendix'] = 1 if '附录' in text else 0

    # 3. 公式特征
    # 检测数学符号和公式
    math_symbols = len(re.findall(r'[=+\-×÷∑∏∫∂√∞≈≤≥≠∈∉⊂⊃∪∩]', text))
    features['math_symbol_count'] = math_symbols

    # LaTeX公式检测
    latex_formulas = len(re.findall(r'\$[^$]+\$', text))
    features['latex_formula_count'] = latex_formulas

    # 数学表达式检测
    equation_patterns = len(re.findall(r'[a-zA-Z]\s*=\s*[^,\n]{5,}', text))
    features['equation_count'] = equation_patterns

    # 公式密度
    if chinese_chars > 0:
        features['formula_density'] = (math_symbols + latex_formulas + equation_patterns) / chinese_chars
    else:
        features['formula_density'] = 0

    # 4. 逻辑连接词特征
    logical_connectors = [
        '因此', '所以', '故', '由此可见', '综上所述', '综上',
        '首先', '其次', '再次', '最后', '第一', '第二', '第三',
        '然而', '但是', '但', '不过', '尽管', '虽然',
        '因为', '由于', '基于', '根据',
        '如果', '假设', '若', '则',
        '不仅', '而且', '同时', '此外', '另外',
        '换言之', '即', '也就是说',
        '一方面', '另一方面',
        '具体来说', '例如', '比如',
        '相比之下', '相反', '反而',
        '进而', '从而', '因而',
        '显然', '明显', '不难发现',
        '根据.*可以', '由.*可得'
    ]
    logic_count = 0
    for connector in logical_connectors:
        logic_count += len(re.findall(connector, text))
    features['logic_connector_count'] = logic_count
    if chinese_chars > 0:
        features['logic_connector_density'] = logic_count / chinese_chars
    else:
        features['logic_connector_density'] = 0

    # 5. 图表特征
    figure_count = len(re.findall(r'图\d+', text)) + len(re.findall(r'Fig\.?\s*\d+', text, re.IGNORECASE))
    table_count = len(re.findall(r'表\d+', text)) + len(re.findall(r'Table\s*\d+', text, re.IGNORECASE))
    features['figure_count'] = figure_count
    features['table_count'] = table_count
    if chinese_chars > 0:
        features['figure_density'] = figure_count / chinese_chars * 10000
        features['table_density'] = table_count / chinese_chars * 10000
    else:
        features['figure_density'] = 0
        features['table_density'] = 0

    # 6. 方法多样性特征
    methods = [
        '层次分析', 'AHP', '熵权法', 'TOPSIS', '灰色关联', '模糊',
        '神经网络', '深度学习', '机器学习', '随机森林', 'SVM', '支持向量机',
        '回归', '聚类', '主成分', 'PCA', '因子分析',
        '线性规划', '非线性规划', '整数规划', '动态规划',
        '遗传算法', '粒子群', '模拟退火', '蚁群算法',
        '蒙特卡洛', '时间序列', 'ARIMA', '马尔可夫',
        '博弈论', '排队论', '图论', '网络流',
        '德尔菲', 'Delphi', '变异系数', 'CRITIC',
        'XGBoost', 'LightGBM', '随机森林', '决策树',
        '逻辑回归', '多元回归', '岭回归', 'Lasso',
        '加权平均', '最小二乘', '最大似然',
        '系统动力学', '耦合协调', '综合评价'
    ]
    method_count = 0
    methods_found = []
    for method in methods:
        cnt = len(re.findall(method, text, re.IGNORECASE))
        if cnt > 0:
            method_count += 1
            methods_found.append(method)
    features['method_diversity'] = method_count
    features['methods_found'] = methods_found

    # 7. 创新性特征（通过独特词汇量衡量）
    unique_chars = len(set(re.findall(r'[一-鿿]', text)))
    if chinese_chars > 0:
        features['vocabulary_richness'] = unique_chars / chinese_chars
    else:
        features['vocabulary_richness'] = 0

    # 8. 规范性特征
    # 段落长度一致性
    paragraphs = [p for p in text.split('\n') if len(p.strip()) > 10]
    if paragraphs:
        para_lengths = [len(p) for p in paragraphs]
        features['avg_paragraph_length'] = np.mean(para_lengths)
        features['std_paragraph_length'] = np.std(para_lengths)
    else:
        features['avg_paragraph_length'] = 0
        features['std_paragraph_length'] = 0

    # 编号规范性
    numbered_items = len(re.findall(r'^\s*[\d一二三四五六七八九十]+[、．.)]', text, re.MULTILINE))
    features['numbered_items'] = numbered_items

    # 9. 代码/算法特征
    features['has_algorithm'] = 1 if ('算法' in text and ('步骤' in text or '伪代码' in text or '流程图' in text)) else 0
    features['has_pseudocode'] = 1 if '伪代码' in text else 0

    # 10. 数据特征
    features['has_data_source'] = 1 if ('数据来源' in text or '数据来源' in text or '数据采集' in text) else 0

    # 摘要质量初评
    abstract_text = text[:3000]
    abstract_quality = 0
    if '摘要' in abstract_text:
        # 摘要是否包含问题、方法、结果、结论
        if any(w in abstract_text for w in ['问题', '针对']):
            abstract_quality += 1
        if any(w in abstract_text for w in ['模型', '方法', '算法']):
            abstract_quality += 1
        if any(w in abstract_text for w in ['结果', '得出', '发现']):
            abstract_quality += 1
        if any(w in abstract_text for w in ['结论', '建议', '表明']):
            abstract_quality += 1
    features['abstract_quality'] = abstract_quality

    return features

def main():
    base_dir = '选题A'

    all_papers = {}

    # 处理附件1
    print("=" * 60)
    print("处理附件1：30篇竞赛论文")
    print("=" * 60)
    for i in range(1, 31):
        fname = f'{i:02d}.pdf'
        fpath = os.path.join(base_dir, '附件1', fname)
        if os.path.exists(fpath):
            text, pages = extract_text_from_pdf(fpath)
            features = compute_features(text, pages)
            features['filename'] = fname
            features['attachment'] = '附件1'
            all_papers[f'附件1/{fname}'] = features
            print(f"  {fname}: {pages}页, {features['chinese_chars']}字, "
                  f"方法{features['method_diversity']}种, "
                  f"逻辑词{features['logic_connector_count']}个, "
                  f"公式{features['equation_count']}个")

    # 处理附件2
    print("\n" + "=" * 60)
    print("处理附件2：10篇同题论文")
    print("=" * 60)
    for i in range(1, 11):
        fname = f'2-{i}.pdf'
        fpath = os.path.join(base_dir, '附件2', fname)
        if os.path.exists(fpath):
            text, pages = extract_text_from_pdf(fpath)
            features = compute_features(text, pages)
            features['filename'] = fname
            features['attachment'] = '附件2'
            all_papers[f'附件2/{fname}'] = features
            print(f"  {fname}: {pages}页, {features['chinese_chars']}字, "
                  f"方法{features['method_diversity']}种, "
                  f"逻辑词{features['logic_connector_count']}个")

    # 处理附件3
    print("\n" + "=" * 60)
    print("处理附件3：3篇中等质量论文")
    print("=" * 60)
    for i in range(1, 4):
        fname = f'3-{i}.pdf'
        fpath = os.path.join(base_dir, '附件3', fname)
        if os.path.exists(fpath):
            text, pages = extract_text_from_pdf(fpath)
            features = compute_features(text, pages)
            features['filename'] = fname
            features['attachment'] = '附件3'
            all_papers[f'附件3/{fname}'] = features
            print(f"  {fname}: {pages}页, {features['chinese_chars']}字, "
                  f"方法{features['method_diversity']}种, "
                  f"逻辑词{features['logic_connector_count']}个")

    # 保存特征数据
    with open('paper_features.json', 'w', encoding='utf-8') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        json.dump(convert(all_papers), f, ensure_ascii=False, indent=2)

    print(f"\n特征提取完成，共处理{len(all_papers)}篇论文")
    print("特征数据已保存至 paper_features.json")

if __name__ == '__main__':
    main()
