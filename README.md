# 黏菌自适应网络项目 — README

生物数学建模期末作业：复现 Tero et al. (2010) Science 论文，并做假说实验。

## 项目结构

```
project/
├── src/
│   ├── physarum.py            核心求解器（Hagen-Poiseuille + 自适应 ODE）
│   ├── tokyo_data.py          36 个东京都市圈城市坐标
│   ├── beijing_data.py        36 个北京地标坐标
│   ├── metrics.py             TL/MD/FT 三个网络质量指标 + MST 基线
│   └── viz.py                 可视化函数库
├── experiments/
│   ├── reproduce_paper.py     复现论文 Fig.3 (参数扫描) + Fig.4 (网络演化)
│   ├── hypothesis1_initial_topology.py
│   │                          H1: 收敛网络是否依赖初始拓扑
│   ├── hypothesis2_beijing.py
│   │                          H2: 模型在北京能否生成类似真实地铁的网络
│   ├── hypothesis3_robustness.py
│   │                          H3: 渐进失效下 Physarum vs MST vs Delaunay
│   └── bonus_steiner.py       彩蛋: Steiner 点的涌现
├── figures/                   所有生成的图（PNG）
├── report/REPORT.md           最终报告（讲解给别人听的主文档）
├── submission/               提交版报告（Markdown、HTML、PDF）及配图
├── requirements.txt          Python 依赖清单
└── paper_text.txt             抽取自原论文 PDF 的文本（参考用）
```

## 一键复现

```bash
git clone https://github.com/CyanTachyon/bio-final.git
cd bio-final
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 1. 复现论文核心结果（约 3 分钟）
python3 experiments/reproduce_paper.py

# 2. 三个假说实验（每个约 1-2 分钟）
python3 experiments/hypothesis1_initial_topology.py
python3 experiments/hypothesis2_beijing.py
python3 experiments/hypothesis3_robustness.py

# 3. Steiner 点彩蛋（约 1 分钟）
python3 experiments/bonus_steiner.py
```

## 依赖

```
python>=3.9
numpy scipy matplotlib networkx pdfplumber
```

## 三个核心收获

1. **三个方程就能建模黏菌的"网络智慧"** — Hagen-Poiseuille + Kirchhoff + 一个自适应 ODE
2. **论文所有定性预测都被复现** — I0↔密度、γ↔拓扑、TL-MD-FT 三角权衡
3. **三个新假说得到验证**：
   - H1: 收敛是拓扑稳态（Jaccard 0.77-0.81）
   - H2: 模型对北京同样适用（边数 44 vs 46，Jaccard 28.6%）
   - H3: Physarum 比 MST 鲁棒 1.7-1.8 倍（用 20% 多的边换得）

详细分析、数据、图表全部在 `report/REPORT.md`。
