# CellOracle 使用说明（现代化版 v0.22.2）

本说明适用于 `CellOracle-modern`（v0.22.2）：支持 **NumPy 2.x / pandas 2–3.x**，无需编译 `velocyto`，
并可在 Apple Silicon 上使用 **Accelerate（CPU 矩阵加速）** 与 **Metal GPU（通过 MLX）** 加速。
关于改造细节见 [`MODERNIZATION.md`](MODERNIZATION.md)。

---

## 目录

1. [环境要求](#1-环境要求)
2. [安装](#2-安装)
3. [验证安装与加速器](#3-验证安装与加速器)
4. [硬件加速](#4-硬件加速)
5. [典型工作流](#5-典型工作流)
6. [保存与读取](#6-保存与读取)
7. [迁移到新机器 / 离线数据](#7-迁移到新机器--离线数据)
8. [常见问题](#8-常见问题)

---

## 1. 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | macOS（Apple Silicon 推荐）/ Linux |
| Python | 3.9 – 3.13（实测 3.10、3.12） |
| 核心依赖 | NumPy ≥ 1.24、pandas ≥ 2.0、SciPy、scikit-learn、scanpy、anndata、h5py、pyarrow、matplotlib、seaborn、networkx、tqdm、joblib、goatools、umap-learn、igraph |
| 可选 | `mlx`（GPU）、`genomepy`+`gimmemotifs`+`pybedtools`（motif 分析）、`ipython`+`ipywidgets`（notebook 交互） |

> 提示：无需再手动安装 `velocyto` / `cython`。速度相关的相关性内核已内置在
> `celloracle/trajectory/_velocity_compat.py`。

---

## 2. 安装

### 2.1 用 wheel 安装（推荐）

```bash
# 基础安装
pip install /path/to/CellOracle-modern/dist/celloracle-0.22.2-py3-none-any.whl

# 需要 GPU（Apple Silicon）
pip install "celloracle[gpu] @ file:///path/to/CellOracle-modern/dist/celloracle-0.22.2-py3-none-any.whl"

# 或用 uv（支持直接对 wheel 追加 extra）
uv pip install "/path/to/CellOracle-modern/dist/celloracle-0.22.2-py3-none-any.whl[all]"
```

### 2.2 用 uv 建环境（推荐做法）

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install "/path/to/CellOracle-modern/dist/celloracle-0.22.2-py3-none-any.whl[all]"
```

### 2.3 从源码安装（开发模式）

```bash
cd CellOracle-modern
pip install -e ".[all]"        # 或仅 pip install -e .
```

### 2.4 可选的 extra

| extra | 内容 | 用途 |
|---|---|---|
| （无） | 核心依赖 | GRN 推断、模拟、可视化 |
| `gpu` | `mlx` | Apple Metal GPU 加速 |
| `motif` | `genomepy` `gimmemotifs` `pybedtools` | ATAC-seq motif 扫描 |
| `notebook` | `ipython` `ipywidgets` | Notebook 交互组件 |
| `all` | 以上全部 | 完整功能 |

```bash
pip install "celloracle[gpu]"
pip install "celloracle[motif]"
pip install "celloracle[all]"
```

### 2.5 从源码自行打包

```bash
cd CellOracle-modern
./scripts/make_bundle.sh        # 生成 wheel + sdist + 离线数据包到 ./dist
```

---

## 3. 验证安装与加速器

```python
import celloracle as co
import numpy as np, pandas as pd, scipy, sklearn

print("celloracle", co.__version__)
print("numpy", np.__version__, "| pandas", pd.__version__,
      "| scipy", scipy.__version__, "| sklearn", sklearn.__version__)

info = co.accelerator_info()
print("backend   :", info["backend"])          # 'cpu' 或 'gpu'
print("BLAS      :", info["blas"]["blas"])     # macOS 上通常为 'accelerate'
print("MLX 版本  :", info["mlx_version"])
print("设备      :", info["device"])
```

运行自带的兼容性与加速测试：

```bash
pytest celloracle/tests/test_accelerator.py celloracle/tests/test_modern_stack.py
```

---

## 4. 硬件加速

### 4.1 Apple 矩阵加速器（Accelerate / vecLib）

macOS 上官方 NumPy 已把 BLAS/LAPACK 链接到 Accelerate，因此 `simulate_shift`、
相关性计算等密集矩阵运算默认就走矩阵加速器。可用
`co.accelerator_info()["blas"]` 确认（返回 `accelerate`）。

### 4.2 Apple GPU（Metal / MLX）

安装 `mlx`（Apple Silicon）后，以下两个最耗时的内核会自动使用 GPU：

- **信号传播**：`Oracle.simulate_shift` → `_do_simulation`
- **转移概率相关性**：`Oracle.estimate_transition_prob`

控制方式：

```python
import celloracle as co

co.set_accelerator("auto")   # 默认：有可用 GPU 就用，小问题仍走 CPU
co.set_accelerator("gpu")    # 强制 GPU（不可用时抛错）
co.set_accelerator("cpu")    # 强制 CPU（Accelerate）
print(co.get_accelerator())
```

或使用环境变量：

```bash
CELLORACLE_ACCELERATOR=cpu  python analysis.py    # auto | cpu | gpu
CELLORACLE_DISABLE_MLX=1    python analysis.py    # 完全禁用 MLX
CELLORACLE_USE_VELOCYTO=1   python analysis.py    # 改用已安装的 velocyto（默认用内置实现）
```

> GPU 路径使用 float32；与 float64 CPU 路径相比，相关性偏差约 1e-7，
> 传播偏差约 1e-5。对精度有严格要求时可 `co.set_accelerator("cpu")`。

---

## 5. 典型工作流

下面用一个可离线运行的完整示例走一遍标准流程（使用仓库自带教程数据）。
若使用自己的数据，只需替换第 5.1 步。

### 5.1 载入数据与 base GRN

```python
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import celloracle as co

# 教程 scRNA-seq 数据（首次运行会自动下载到 ~/celloracle_data）
adata = co.data.load_Paul2015_data()

# base GRN：来自小鼠 scATAC-seq atlas 的 TF–peak 连接
base_GRN = co.data.load_mouse_scATAC_atlas_base_GRN()

# （可选）下采样以加快演示
cells = adata.obs.index[adata.obs.louvain_annot.isin(["MEP_0", "GMPl_0"])]
adata = adata[cells, :].copy()

# CellOracle 使用未标准化的 count 作为输入
adata.X = adata.layers["raw_count"].copy()
print(adata.shape)          # (细胞数, 基因数)
base_GRN.head()
```

**使用自己的数据时**：先做 QC、过滤、归一化、log 转换（但**不要**做 scaling/centering），
并确保 `adata.obs` 中有聚类列、`adata.obsm` 中有降维坐标（如 `X_umap`）。

### 5.2 创建 Oracle 对象并导入 TF 信息

```python
oracle = co.Oracle()

# 以 raw count 导入；也可用 import_anndata_as_normalized_count(...)
oracle.import_anndata_as_raw_count(
    adata=adata,
    cluster_column_name="louvain_annot",   # 聚类列
    embedding_name="X_draw_graph_fa",      # 降维坐标
)

# 导入 base GRN
oracle.import_TF_data(TF_info_matrix=base_GRN)

# （可选）手动补充已知 TF→靶基因 关系
# TG_to_TF = co.utility.inverse_dictionary(TF_to_TG_dictionary)
# oracle.addTFinfo_dictionary(TG_to_TF)
```

### 5.3 PCA 与 KNN 插补

```python
oracle.perform_PCA()

# 选择用于插补的 PC 数（这里用简单启发式）
n_comps = min(
    np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_)) > 0.002))[0][0],
    50,
)

n_cell = oracle.adata.shape[0]
k = int(0.025 * n_cell)

oracle.knn_imputation(
    n_pca_dims=n_comps,
    k=k,
    balanced=True,
    b_sight=k * 8,
    b_maxl=k * 4,
    n_jobs=4,
)
```

### 5.4 GRN 推断与网络分析（Links）

```python
links = oracle.get_links(
    cluster_name_for_GRN_unit="louvain_annot",
    alpha=10,
    bagging_number=20,      # 正式分析可设 200 或更高
    verbose_level=1,
)

# 过滤 + 网络打分
links.filter_links(p=0.001, weight="coef_abs", threshold_number=2000)
links.get_network_score()
print(links.merged_score.head())

# 可视化
links.plot_degree_distributions(plot_model=True)
links.plot_scores_as_rank(cluster="MEP_0", n_gene=30)
links.plot_score_per_cluster(goi="Gata2")
```

### 5.5 构建用于模拟的 cluster 特异性 GRN

```python
oracle.get_cluster_specific_TFdict_from_Links(links_object=links)

oracle.fit_GRN_for_simulation(
    alpha=10,
    use_cluster_specific_TFdict=True,
    verbose_level=1,
)
```

### 5.6 基因扰动模拟（in silico perturbation）

```python
goi = "Gata1"

# 条件：{"基因": 目标表达值}。0.0 表示完全敲除（KO）
oracle.simulate_shift(
    perturb_condition={goi: 0.0},
    n_propagation=3,
    ignore_warning=True,     # 或先检查基因表达分布
)

# 查看结果：模拟表达量与相对变化
sim = oracle.adata.to_df(layer="simulated_count")
delta = oracle.adata.to_df(layer="delta_X")
print(delta[[goi, "Gata2", "Klf1"]].describe())
```

### 5.7 计算向量场并可视化

```python
# 1) 转移概率（GPU 会自动加速）
oracle.estimate_transition_prob(
    n_neighbors=200,
    knn_random=True,
    sampled_fraction=1,
)

# 2) 投射到降维空间，得到每个细胞的位移向量
oracle.calculate_embedding_shift(sigma_corr=0.05)

# 3) 点图（quiver）
fig, ax = plt.subplots(1, 2, figsize=[13, 6])
oracle.plot_quiver(scale=25, ax=ax[0]);       ax[0].set_title(f"{goi} KO")
oracle.plot_quiver_random(scale=25, ax=ax[1]); ax[1].set_title("Randomized control")
plt.show()

# 4) 向量场（网格化平滑）
oracle.calculate_p_mass(smooth=0.8, n_grid=40, n_neighbors=200)
oracle.suggest_mass_thresholds(n_suggestion=12)
oracle.calculate_mass_filter(min_mass=0.01, plot=True)

fig, ax = plt.subplots(1, 2, figsize=[13, 6])
oracle.plot_simulation_flow_on_grid(scale=0.5, ax=ax[0])
oracle.plot_simulation_flow_random_on_grid(scale=0.5, ax=ax[1])
plt.show()
```

### 5.8 发育轨迹比较（可选）

```python
from celloracle.applications import Gradient_calculator, Oracle_development_module

# 参考向量场：由 pseudotime 计算得到的发育方向
gradient = Gradient_calculator(oracle_object=oracle, pseudotime_key="Pseudotime")
gradient.calculate_p_mass(smooth=0.8, n_grid=40, n_neighbors=200)
gradient.calculate_mass_filter(min_mass=0.01, plot=True)
gradient.transfer_data_into_grid(args={"method": "polynomial", "n_poly": 3}, plot=True)
gradient.calculate_gradient()

# 比较“扰动模拟向量场”与“发育向量场”的内积（扰动评分）
dev = Oracle_development_module()
dev.load_differentiation_reference_data(gradient_object=gradient)
dev.load_perturb_simulation_data(oracle_object=oracle)
dev.calculate_inner_product()
dev.calculate_digitized_ip(n_bins=10)
dev.plot_inner_product_on_grid(vm=0.02, s=50)
```

### 5.9 独立使用 Net（仅做 GRN 推断）

```python
gem = oracle.adata.to_df(layer="imputed_count")

net = co.Net(gene_expression_matrix=gem, TFinfo_matrix=base_GRN, verbose=True)
net.fit_All_genes(bagging_number=200, model_method="bagging_ridge")  # 或 "bayesian_ridge"
net.updateLinkList()
print(net.linkList.head())

# 仅拟合部分基因
net.fit_genes(target_genes=["Gata1", "Gata2"], model_method="bayesian_ridge")
```

---

## 6. 保存与读取

文件扩展名决定对象类型（`.celloracle.oracle` / `.celloracle.links` / `.celloracle.net` 等）。

```python
# Oracle
oracle.to_hdf5("Paul15.celloracle.oracle")
oracle = co.load_hdf5("Paul15.celloracle.oracle")

# Links
links.to_hdf5("links.celloracle.links")
links = co.load_hdf5("links.celloracle.links")

# Net（HDF5）
net.to_hdf5("net.celloracle.net")
net = co.load_hdf5("net.celloracle.net")

# Net（parquet，体积更小）
net._save_as_parquet("net_folder")
from celloracle.network import load_net_from_patquets
net = load_net_from_patquets("net_folder")

# 通用加载
obj = co.load_hdf5("anything.celloracle.oracle")
```

也可以显式指定类型：`co.load_hdf5(path, object_class_name="Links")`。

---

## 7. 迁移到新机器 / 离线数据

`dist/` 中有三个产物：

| 文件 | 说明 |
|---|---|
| `celloracle-0.22.2-py3-none-any.whl` | 可安装 wheel（代码 + 小幅资源） |
| `celloracle-0.22.2.tar.gz` | 源码包 |
| `celloracle-data-0.22.2.tar.gz` | 离线数据（约 620 MB，可选） |

### 7.1 在线迁移（最小体积）

安装 wheel 即可。首次调用 `co.data.load_*` 时，教程/base GRN 数据会自动下载到
`~/celloracle_data/`（由 `celloracle/data/config.py` 的 `CELLORACLE_DATA_DIR` 指定）。

### 7.2 完全离线

```bash
mkdir -p ~/celloracle_data
tar -xzf celloracle-data-0.22.2.tar.gz -C ~/celloracle_data
# 解压后结构：~/celloracle_data/{anndata,TFinfo_data,promoter_base_GRN,tutorial_data,motif_data}
```

之后所有 `co.data.load_*` 都会优先命中本地文件，无需联网。

### 7.3 复制整个源码目录

直接把 `CellOracle-modern/` 拷贝到新机器也可以（`celloracle/data/` 里已包含全部自带数据），
再 `pip install -e .` 或 `pip install ./dist/*.whl`。

---

## 8. 常见问题

**Q：`import celloracle` 报 `No module named 'velocyto'`？**
A：v0.22.2 已移除该依赖。若仍在旧环境，请安装本版本的 wheel / 源码，不要保留旧的 editable 安装。

**Q：`ImportError: CellOracle's motif_analysis module requires ...`？**
A：motif 分析是可选功能，安装 `pip install "celloracle[motif]"`，或使用预计算的
`base_GRN`（如 `co.data.load_mouse_scATAC_atlas_base_GRN()`），无需 motif 扫描。

**Q：想用 GPU，但 `accelerator_info()["backend"]` 一直是 `cpu`？**
A：`pip install "celloracle[gpu]"`（仅 Apple Silicon），并确认 `co.accelerator_info()["gpu_available"]`
为 `True`、`mlx_error` 为 `None`。Intel Mac / Linux 会自动回退到 CPU。

**Q：结果与旧版本有细微差异？**
A：GPU 路径使用 float32。需要与 float64 完全一致时：
`co.set_accelerator("cpu")`，或设置环境变量 `CELLORACLE_ACCELERATOR=cpu`。

**Q：`plot_simulation_flow_on_grid()` 报 `UnboundLocalError: local variable 'gridpoints_coordinates' referenced before assignment`？**
A：这是**上游原版的绘图 bug**，v0.22.2 已修复。原因是绘图包装函数把网格坐标的判断条件写成了
`elif hasattr(self, "mass_filter")`（应为 `flow_grid`），而 `Oracle` 的网格存在 `flow_grid` 里；
只有当之前恰好调用过 `calculate_mass_filter()`（教程里总是会调）时才被掩盖。
同时 `mass_filter` 这个局部变量也没有兜底，缺省时同样会报未赋值。
现在的行为：网格从 `gridpoints_coordinates` 或 `flow_grid` 解析；没有 mass filter 时默认
“保留全部网格点”。因此 `calculate_p_mass()` 之后可以直接绘图，**不必**先调 `calculate_mass_filter()`：

```python
oracle.calculate_p_mass(smooth=0.8, n_grid=40, n_neighbors=200)
oracle.plot_simulation_flow_on_grid(scale=0.5)      # v0.22.2 不再报错
# 需要过滤稀疏区域时再调用：
# oracle.calculate_mass_filter(min_mass=0.01)
```

若连 `flow` / `flow_grid` 都没有，会抛出指明需先运行 `calculate_p_mass()` 的 `AttributeError`，
不再是难以定位的 `UnboundLocalError`。

**Q：`plot_quiver` / 绘图没有反应？**
A：v0.22.2 不再在 import 时强制 `Agg` 后端。脚本环境如无显示设备，请自行
`import matplotlib; matplotlib.use("Agg")`。

**Q：pandas 3.0 下部分旧 `.pkl`（pickle）加载失败？**
A：这是旧对象序列化格式问题。请使用 v202204 及以后的 HDF5/parquet 数据，
或 `co.data.load_tutorial_oracle_object()` 等会自动选择与现代 pandas 匹配的数据版本。

---

## 附：完整可运行脚本

```python
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无显示环境
import matplotlib.pyplot as plt
import celloracle as co

print("backend:", co.get_accelerator())

# 1. 数据
oracle = co.data.load_tutorial_oracle_object()
links = co.data.load_tutorial_links_object()

# 2. 模拟用 GRN
links.filter_links()
oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True, verbose_level=1)

# 3. 扰动
oracle.simulate_shift(perturb_condition={"Gata1": 0.0}, n_propagation=3, ignore_warning=True)

# 4. 向量场
oracle.estimate_transition_prob(n_neighbors=200, knn_random=True, sampled_fraction=1)
oracle.calculate_embedding_shift(sigma_corr=0.05)

# 5. 可视化
os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 6))
oracle.plot_quiver(scale=25, ax=ax)
ax.set_title("Gata1 KO")
fig.savefig("figures/gata1_ko_quiver.png", dpi=200)

oracle.calculate_p_mass(smooth=0.8, n_grid=40, n_neighbors=200)
oracle.calculate_mass_filter(min_mass=0.01)
fig, ax = plt.subplots(figsize=(6, 6))
oracle.plot_simulation_flow_on_grid(scale=0.5, ax=ax)
fig.savefig("figures/gata1_ko_flow.png", dpi=200)

print("done: delta_embedding", oracle.delta_embedding.shape)
```
