# Manifold Malliavin implementation

## 方針

実装順序は次で固定する。

1. De Bortoli の base-manifold forward/reverse SDE に、一般形の離散 Malliavin--Skorokhod teacher を接続する。
2. 同じ teacher backend を Park 型 horizontal development の endpoint map に接続する。
3. Park 論文の Euclidean-score-lift 命題は使用しない。

現段階では GPU サーバーで検証するための exact reference 実装を優先している。ローカル環境では実行していない。

## 実装構成

### 共通 teacher backend

`src/scoremodel_ext/manifold/malliavin_teacher.py`

標準 Gaussian noise `Z` から endpoint `F(Z)` への微分可能写像だけを要求する。内部で

```text
J = dF/dZ
C = J J^T
U = J^T (C + lambda I)^(-1) V
delta(U) = U^T Z - div_Z U
```

を計算する。`V` は endpoint tangent space を張るベクトル場であり、必要なら多様体体積に関する `div V` も明示的に与える。

この backend は以下を仮定しない。

- 拡散係数が状態非依存であること
- endpoint が Euclidean 空間にあること
- Euclidean endpoint density と frame-bundle density が一致すること

### De Bortoli / S2 adapter

`src/scoremodel_ext/manifold/s2_malliavin.py`

- base manifold 上の geodesic random walk
- projected ambient generating fields `V_j(x)=(I-xx^T)e_j`
- `div V_j=-2x_j`
- full discrete Skorokhod teacher
- S2 spectral heat-kernel score
- Varadhan small-time score
- arbitrary data initial points からの marginal teacher dataset builder
- `E[delta | X_t]` を tangent score に変換する model wrapper
- learned score を受け取る De Bortoli reverse GRW

を実装している。teacher の endpoint は frame bundle ではなく `X_t in S2` である。

### Park horizontal development adapter

`src/scoremodel_ext/manifold/horizontal_development.py`

- local metric / Christoffel interface
- horizontal lift
- differentiable metric orthonormalisation
- Stratonovich Heun integrator
- base endpoint `X_t` と frame-bundle endpoint `U_t` の選択
- 共通 Malliavin backend への adapter

を実装している。この層には score-lift のコードは存在しない。

## GPU サーバーでの検証順序

### 1. 単体テスト

```bash
cd /path/to/scoremodel
PYTHONPATH=src python -m pytest \
  tests/test_manifold_malliavin_teacher.py \
  tests/test_horizontal_development.py -v
```

### 2. 小規模 exact teacher

```bash
PYTHONPATH=src python -m \
  scoremodel_ext.manifold.experiment_s2_malliavin_teacher \
  --device cuda \
  --dtype float64 \
  --n-paths 64 \
  --n-steps 8 \
  --time 0.3 \
  --knn-k 8 \
  --outdir results/s2_malliavin_teacher_exact
```

`torch.autograd.functional.jacobian(..., vectorize=True)` の backend coverage に問題がある場合は、次を追加する。

```bash
--no-vectorize-jacobian
```

出力:

- `teacher_dataset.pt`
- `metrics.json`
- `config.json`

### 3. データ分布からの marginal score 学習

Python 側で S² data tensor `initial_points: [n_paths,3]` を用意し、次を実行する。

```python
from scoremodel_ext.manifold.experiment_s2_malliavin_teacher import (
    generate_s2_marginal_teacher_dataset,
    train_s2_marginal_score,
)
from scoremodel_ext.manifold.s2_malliavin import s2_reverse_grw

times = torch.empty(len(initial_points), device="cuda").uniform_(0.05, 1.0)
dataset = generate_s2_marginal_teacher_dataset(
    initial_points,
    times,
    n_steps=16,
    covariance_regularization=1e-6,
)
score_model = train_s2_marginal_score(dataset, device="cuda")

terminal = torch.randn(1024, 3, device="cuda")
terminal = terminal / terminal.norm(dim=1, keepdim=True)  # uniform on S2
samples = s2_reverse_grw(
    terminal,
    score_model,
    terminal_time=1.0,
    n_steps=500,
)
```

### 4. 最初に確認する診断値

- `max_endpoint_norm_error`
- `max_tangent_residual`
- Malliavin covariance の2つの tangent eigenvalues
- normal eigenvalue が数値誤差程度であること
- `nan_rate`
- `mean_cosine_knn_vs_heat`
- `rmse_knn_vs_heat`

raw path weight と heat-kernel score は path ごとには一致しない。比較対象は `E[weight | X_t]` なので、診断 CLI は leave-one-out kNN conditional mean を使用する。

## 次の実装単位

1. marginal teacher builder、Algorithm 6 training、reverse GRW を一括実行する experiment CLI
2. exact divergence を基準にした Hutchinson/VJP divergence
3. S2 上で base endpoint と horizontal endpoint の teacher の一致検証

full frame-bundle endpoint は小時間で hypoelliptic covariance が悪条件化しやすい。Park adapter の最初の検証でも、まず `return_frame_bundle=False` として base endpoint teacher を使い、その後に full `U_t` teacher へ進む。
