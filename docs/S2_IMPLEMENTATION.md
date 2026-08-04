# S² Earthquake 実装: 数式・アルゴリズム・コード対応

## 0. この文書の範囲

この文書は Earthquake データを単位球面

$$
\mathbb S^2=\{x\in\mathbb R^3:\|x\|_2=1\}
$$

上の経験分布として扱う実装について、forward diffusion、三種類の score teacher、ニューラルネット、reverse sampling、checkpoint replay の対応を説明する。

中心となる実行ファイルは [`scripts/experiment_earthquake_teacher_compare_smoke.py`](../scripts/experiment_earthquake_teacher_compare_smoke.py) である。幾何と確率過程は [`s2_malliavin.py`](../src/scoremodel_ext/manifold/s2_malliavin.py)、離散 Malliavin--Skorokhod 計算は [`malliavin_teacher.py`](../src/scoremodel_ext/manifold/malliavin_teacher.py)、ネットワークは [`models.py`](../src/scoremodel_ext/malliavin/models.py) に分離されている。

本実装の全体像は次の通りである。

$$
X_0\sim p_{\mathrm{data}}
\xrightarrow[\texttt{s2\_grw\_endpoint}]{\text{forward GRW}}
(t,X_t,\text{teacher})
\xrightarrow{\text{regression}}
s_\theta(t,x)
\xrightarrow[\texttt{s2\_reverse\_grw}]{\text{reverse GRW}}
Y_T\approx p_{\mathrm{data}}.
$$
ここで score は常に球面体積測度に関する密度の Riemannian gradient

$$
s(t,x)=\nabla_{\mathbb S^2}\log p_t(x)\in T_x\mathbb S^2
$$

を意味する。

## 1. 球面上の幾何

### 1.1 接空間と射影

点 \(x\in\mathbb S^2\) の接空間と直交射影は

$$
T_x\mathbb S^2=\{v\in\mathbb R^3:x^\top v=0\},\qquad
P_x=I_3-xx^\top
$$

である。

コード対応:

- `s2_projector(x)` は単一点の \(P_x\) を返す。
- `_batched_s2_projector(x)` は batch ごとの \(P_x\) を返す。
- `s2_to_tangent(v, x)` は \(P_xv\) を返す。

### 1.2 指数写像

\(v\in T_x\mathbb S^2\)、\(r=\|v\|\) に対して

$$
\operatorname{Exp}_x(v)
=\cos(r)x+\frac{\sin r}{r}v,
$$

ただし \(r=0\) では \(\sin r/r=1\) と連続延長する。

`s2_exp(base_point, tangent_vector)` は最初に引数を接空間へ再射影し、`torch.sinc(r / pi)` により \(\sin r/r\) を安定に計算する。最後の再正規化は解析的な写像を変えるためではなく、浮動小数点誤差による球面からのずれを抑えるためである。

## 2. Forward SDE と geodesic random walk

### 2.1 Stratonovich SDE

標準基底を \(e_a\in\mathbb R^3\)、球面上の生成ベクトル場を

$$
V_a(x)=P_xe_a,\qquad a=1,2,3
$$

とする。forward process は外在的な Stratonovich SDE

$$
dX_t=\sum_{a=1}^3V_a(X_t)\circ dW_t^a
=P_{X_t}\circ dW_t
$$

で表され、その生成作用素は

$$
\mathcal L=\frac12\sum_{a=1}^3V_a^2
=\frac12\Delta_{\mathbb S^2}
$$

である。等価な Itô 表現は、\(\dim\mathbb S^2=2\) より

$$
dX_t=P_{X_t}\,dW_t-X_t\,dt
$$

となる。ただし実装はこの Itô drift を Euler 更新するのではなく、指数写像を使う intrinsic な geodesic random walk (GRW) を用いる。

### 2.2 GRW 離散化

\(T>0\)、step 数 \(K\)、\(\Delta t=T/K\)、独立な \(Z_k\sim\mathcal N(0,I_3)\) に対して

$$
X_{k+1}
=\operatorname{Exp}_{X_k}
\left(\sqrt{\Delta t}\,P_{X_k}Z_k\right).
$$

これは \(K\to\infty\) で generator \(\tfrac12\Delta_{\mathbb S^2}\) の Brownian motion に収束する。

### 2.3 `s2_grw_endpoint()` との逐語対応

関数: `s2_grw_endpoint(initial_point, standard_noise, terminal_time)`  
ファイル: [`src/scoremodel_ext/manifold/s2_malliavin.py`](../src/scoremodel_ext/manifold/s2_malliavin.py)

| 数式 | コード上の量 | 処理 |
|---|---|---|
| \(K\) | `standard_noise.shape[0]` | forward step 数 |
| \(T\) | `terminal_time` | path の終端時刻 |
| \(\sqrt{\Delta t}\) | `sqrt_dt = sqrt(terminal_time / K)` | noise scaling |
| \(Z_k\) | `increment` | `[3]` の standard normal |
| \(P_{X_k}Z_k\) | `s2_to_tangent(increment, x)` | ambient noise の接空間射影 |
| \(\sqrt{\Delta t}P_{X_k}Z_k\) | `tangent_increment` | 指数写像へ渡す接ベクトル |
| \(X_{k+1}\) | `x = s2_exp(x, tangent_increment)` | 一段の GRW 更新 |

Earthquake runner の `build_teacher_dataset()` は、Heat/Varadhan ではこの関数を直接呼ぶ。Malliavin では `s2_discrete_malliavin_teacher()` 内の endpoint map として同じ関数を一度だけ呼び、微分対象の path と teacher の endpoint を一致させる。

## 3. Reverse process

### 3.1 連続時間の reverse SDE

forward density を \(p_t\)、終端時刻を \(T\) とする。reverse-time parameter を \(\tau\in[0,T]\)、対応する forward time を \(t=T-\tau\) と置くと、Brownian forward process の時間反転は

$$
dY_\tau
=s(T-\tau,Y_\tau)\,d\tau
+\sum_{a=1}^3V_a(Y_\tau)\circ d\bar W_\tau^a,
\qquad
s(t,x)=\nabla_{\mathbb S^2}\log p_t(x).
$$

この式の drift の符号が正であるのは、積分変数を decreasing forward time \(t\) ではなく increasing reverse time \(\tau\) にしたためである。

runner は \(p_T\) の近似として、三次元 Gaussian を正規化して得る球面一様分布から `terminal_samples.pt` を作る。

### 3.2 Reverse GRW

\(N\) step、\(\Delta t=T/N\) として

$$
Y_{k+1}
=\operatorname{Exp}_{Y_k}\!\left(
\Delta t\,s(t_k,Y_k)
+\sqrt{\Delta t}\,P_{Y_k}Z_k
\right),
$$

$$
t_k=\max(T-k\Delta t,t_{\min}).
$$

`minimum_forward_time=t_min` は \(t=0\) 近傍で score が特異・不安定になることを避ける clamp である。loop は \(k=0,\ldots,N-1\) なので、最後に評価される時刻は

$$
\max(\Delta t,t_{\min})
$$

であってゼロではない。

### 3.3 `s2_reverse_grw()` の各段階

関数: `s2_reverse_grw()`  
ファイル: [`src/scoremodel_ext/manifold/s2_malliavin.py`](../src/scoremodel_ext/manifold/s2_malliavin.py)

| 順序 | コード上の量 | 数式・役割 |
|---:|---|---|
| 1 | `points = terminal_points / norm(...)` | \(Y_0\in\mathbb S^2\) を保証 |
| 2 | `dt = terminal_time / n_steps` | \(\Delta t=T/N\) |
| 3 | `sqrt_dt = sqrt(dt)` | \(\sqrt{\Delta t}\) |
| 4 | `forward_time` | \(t_k=\max(T-k\Delta t,t_{\min})\) |
| 5 | `time_batch` | 全 sample に同じ \(t_k\) を与える batch |
| 6 | `raw_score = score_fn(time_batch, points)` | network の \(s_\theta(t_k,Y_k)\) |
| 7 | `projector = _batched_s2_projector(points)` | \(P_{Y_k}\) |
| 8 | `projected_score = projector @ raw_score` | 数値的に score を接空間へ強制 |
| 9 | `raw_noise = standard_noise[step]` | \(Z_k\sim N(0,I_3)\) |
| 10 | `projected_noise = projector @ raw_noise` | \(P_{Y_k}Z_k\) |
| 11 | `tangent_increment` | \(\Delta tP_Ys_\theta+\sqrt{\Delta t}P_YZ_k\) |
| 12 | `s2_exp(point, increment)` | \(Y_{k+1}=\operatorname{Exp}_{Y_k}(\cdot)\) |

`S2SkorokhodScoreModel` 自体も tangent projection を行うが、reverse integrator は任意の `score_fn` を受け付けるため、integrator 側でも再射影する。この二重射影は理論上同じベクトルを返し、球面法線方向の数値誤差を除去する。

`debug_output_dir` は opt-in で、step 0/1 の `raw_score`、projector、noise、tangent increment、output などを保存する。debug off/on で通常の更新式は同一であり、回帰テストは final output の完全一致を要求する。

## 4. Score teacher

Earthquake runner は同じ data split、forward time、forward noise を共有し、teacher label だけを `heat`、`varadhan`、`malliavin` で切り替える。データ点 \(X_0\sim p_{\mathrm{data}}\) と forward endpoint \(X_t\) から学ぶ conditional teacher は、二乗誤差回帰によって

$$
\mathbb E[\nabla_x\log p_{t|0}(X_t\mid X_0)\mid X_t=x]
=\nabla_x\log p_t(x)
$$

という marginal score を与える。

### 4.1 Heat teacher

generator が \(\tfrac12\Delta\) のとき、\(c=x_0^\top x\) と置いた球面 heat kernel は

$$
p_t(x\mid x_0)
=\frac1{4\pi}\sum_{\ell=0}^{\infty}
(2\ell+1)e^{-\ell(\ell+1)t/2}P_\ell(c).
$$

\(\nabla_{\mathbb S^2,x}c=P_xx_0=x_0-cx\) なので、教師 score は

$$
s_{\mathrm{heat}}(t,x;x_0)
=\nabla_{\mathbb S^2,x}\log p_t(x\mid x_0)
=\frac{\partial_c p_t(c)}{p_t(c)}(x_0-cx).
$$

`s2_heat_kernel_score()` は Legendre 多項式 \(P_\ell\) と微分 \(P_\ell'\) を recurrence で同時に評価し、`n_terms` で spectral sum を打ち切る。`acos` を微分しないため対角近傍で安定である。打切り密度が非正なら黙って別 teacher へ切り替えず `FloatingPointError` を送出する。

コード経路:

1. `build_teacher_dataset()` または `S2TeacherProvider.sample_batch()` が `s2_grw_endpoint()` で \(X_t\) を生成する。
2. `s2_heat_kernel_score(x0, endpoint, t, n_terms=heat_terms)` を `score_target` に保存する。
3. `train_s2_score_model()` が \((t,X_t)\mapsto\texttt{score_target}\) を直接回帰する。

### 4.2 Varadhan teacher

小時間では Varadhan asymptotics により

$$
\log p_t(x\mid x_0)
=-\frac{d_{\mathbb S^2}(x,x_0)^2}{2t}+O(\log t),
$$

したがって leading-order score は

$$
s_{\mathrm{Varadhan}}(t,x;x_0)
=\frac{\operatorname{Log}_x(x_0)}{t}.
$$

\(c=x^\top x_0\)、\(\theta=\arccos c\) とすると

$$
\operatorname{Log}_x(x_0)
=\frac{\theta}{\|x_0-cx\|}(x_0-cx).
$$

`s2_varadhan_score()` はこの式を実装し、対角近傍では scale を 1 に連続化する。これは有限時刻の exact heat score ではなく、小時間近似を独立の teacher として比較する経路である。

コード経路は Heat と同じで、label 計算だけが `s2_varadhan_score()` になる。学習は `train_s2_score_model()` による direct score regression である。

### 4.3 Malliavin teacher

Malliavin 経路は解析的な heat kernel を教師に使わない。standard Gaussian path noise \(Z\) から endpoint への微分可能写像

$$
F:Z\mapsto X_t=F(Z)
$$

を自動微分し、有限次元 Gaussian integration by parts から pathwise Skorokhod weight を作る。

Earthquake の主学習 target は `teacher_state.skorokhod` である。`teacher_state.score_weight` も保存されるが、これは診断・direct score 比較用であり、Malliavin 主経路の network target ではない。

コード経路:

1. `build_teacher_dataset()` が `s2_discrete_malliavin_teacher()` を呼ぶ。
2. 同関数が `s2_grw_endpoint()` を endpoint map \(F\) として `discrete_malliavin_skorokhod_teacher()` へ渡す。
3. dataset に `endpoint`、`skorokhod`、`score_target=score_weight` を保存する。
4. `train_s2_marginal_score()` が `skorokhod` を回帰する。
5. 戻り値 `S2SkorokhodScoreModel` が conditional mean を tangent score へ変換する。

## 5. Malliavin teacher の詳細

### 5.1 離散 Malliavin covariance

forward GRW が使う noise を平坦化して

$$
Z=(Z_0,\ldots,Z_{K-1})\in\mathbb R^q,\qquad q=3K
$$

とする。endpoint Jacobian と離散 Malliavin covariance（コードでは `covariance`）は

$$
J(Z)=\partial_ZF(Z)\in\mathbb R^{3\times q},
\qquad
\Gamma(Z)=J(Z)J(Z)^\top\in\mathbb R^{3\times3}.
$$

球面 endpoint の有効次元は 2 なので ambient \(\Gamma\) は法線方向に退化する。数値計算では

$$
\Gamma_\lambda=\Gamma+\lambda I
$$

を用いる。\(\lambda\) は `covariance_regularization` である。`covariance_eigenvalues`、`condition_number`、`right_inverse_residual` は、この regularized inverse と tangent coverage の診断値である。

`tangent_malliavin_skorokhod_teacher()` には orthonormal tangent basis で \(2\times2\) covariance を作る経路もある。ただし現在の Earthquake runner が呼ぶ `s2_discrete_malliavin_teacher()` は三つの projected ambient fields を用いる一般 backend である。

### 5.2 Covering control

endpoint 上の対象ベクトル場を列に並べた

$$
V(F)=[V_1(F),V_2(F),V_3(F)]=P_F
$$

に対し、regularized minimum-energy control を

$$
U(Z)=J^\top(\Gamma+\lambda I)^{-1}V(F)
\in\mathbb R^{q\times3}
$$

とする。理想的には \(JU=V\) であり、その誤差が `right_inverse_residual` である。

コードでは `_jacobian(flat_endpoint, z, ...)` が \(J\)、`_covering_from_endpoint_state()` が \(\Gamma_\lambda^{-1}V\) と \(U\) を計算する。

### 5.3 Skorokhod correction \(\delta\)

各列 \(U_a\) に対する有限次元 Gaussian divergence、すなわち Skorokhod integral は

$$
\delta(U_a)
=U_a(Z)^\top Z-\operatorname{div}_Z U_a(Z),
$$

$$
\operatorname{div}_ZU_a
=\sum_{i=1}^q\frac{\partial U_{ia}}{\partial Z_i}.
$$

対応するコード上の量は

$$
\texttt{gaussian\_pairing}=U^\top Z,
\qquad
\texttt{divergence\_term}=\operatorname{div}_ZU,
\qquad
\texttt{skorokhod}=U^\top Z-\operatorname{div}_ZU.
$$

重要なのは第二項を落とさないことである。\(U\) は nonlinear な GRW endpoint と \(Z\) に依存するため、一般には adapted Itô integral だけでは正しい weight にならない。実装は `covering_from_noise()` の Jacobian をさらに自動微分し、対角 trace を合計する。これは選んだ GRW 離散化に対する有限次元 Gaussian IBP として exact であり、連続時間 Malliavin weight に対しては time discretization を通じた近似である。

### 5.4 Field divergence と score 再構成

密度を球面体積測度に関して定義すると、各ベクトル場方向に

$$
V_a\log p(F)
=-\mathbb E[\delta(U_a)\mid F]
-\operatorname{div}_{\mathbb S^2}V_a(F).
$$

\(V_a(x)=P_xe_a=\nabla_{\mathbb S^2}x_a\) かつ

$$
\operatorname{div}_{\mathbb S^2}V_a
=\Delta_{\mathbb S^2}x_a=-2x_a
$$

なので pathwise directional label は

$$
w_a=-\delta(U_a)+2F_a.
$$

これが `directional_score_weight = -skorokhod - field_divergence` である。三つの \(V_a\) は二次元接空間を冗長に張るため、pseudoinverse で ambient tangent vector `score_weight` を再構成する。

Malliavin 主学習では先に

$$
N_\theta(t,x)\approx\mathbb E[\delta(U)\mid X_t=x]
$$

を学び、その後

$$
s_\theta(t,x)=-P_xN_\theta(t,x)
$$

とする。field-divergence vector は \(-2x\) に比例し \(P_xx=0\) なので、最終的な tangent vector 再構成では消える。これが `S2SkorokhodScoreModel.forward()` の `-projector @ delta` に対応する。

### 5.5 Mirafzali et al. との対応と相違

本実装は Mirafzali et al. の「Skorokhod/Malliavin weight を生成し、\((t,X_t)\) を入力としてその conditional mean を MSE 回帰する」という Algorithm 6 の学習原理を使う。`MirafzaliSkorokhodNet` と `train_mirafzali_skorokhod_net()` がこの回帰器である。

一方、S² adapter は論文の Euclidean additive-noise 公式をそのまま移植していない。相違は次の通りである。

- endpoint は Euclidean SDE の状態ではなく、base manifold 上の GRW endpoint \(F(Z)\in\mathbb S^2\) である。
- Malliavin derivative は `s2_grw_endpoint()` 全体を autograd して得る。
- score は Lebesgue 密度ではなく球面体積密度に関するもので、\(\operatorname{div}_{\mathbb S^2}V_a\) を含む。
- nonlinear path に必要な \(-\operatorname{div}_ZU\) を finite-dimensional Jacobian trace として明示的に計算する。
- frame-bundle density と base-manifold density を同一視せず、teacher endpoint は \(X_t\in\mathbb S^2\) である。

したがって「Mirafzali Algorithm 6 型の conditional-expectation regression を、De Bortoli 型 base-manifold GRW と一般的な離散 Gaussian IBP backend に接続したもの」と位置付けるのが正確である。

## 6. Neural network と学習

### 6.1 入出力

基礎 network は

$$
N_\theta:(t,x)\in\mathbb R\times\mathbb R^3\longmapsto y\in\mathbb R^3
$$

である。

- Heat/Varadhan: \(y\) は直接 score label を表し、返される model class は `NormalizedSkorokhodModel` である。
- Malliavin: 内側の \(y\) は \(\mathbb E[\delta(U)\mid t,x]\) を表す。返される class は `S2SkorokhodScoreModel(NormalizedSkorokhodModel(...))` で、外側 wrapper が \(-P_xy\) を score として返す。

この違いは checkpoint の key prefix と replay 時の class 構成にも影響する。

### 6.2 Normalization

training dataset から

$$
\hat x=\frac{x-x_{\mathrm{mean}}}{x_{\mathrm{std}}},\qquad
\hat t=\frac{t-t_{\mathrm{mean}}}{t_{\mathrm{std}}},\qquad
\hat y=\frac{y-y_{\mathrm{mean}}}{y_{\mathrm{std}}}
$$

を作り、network は \((\hat t,\hat x)\mapsto\hat y\) を学習する。推論時は

$$
y=N_\theta(\hat t,\hat x)\odot y_{\mathrm{std}}+y_{\mathrm{mean}}
$$

と逆変換する。std は `clamp_min(1e-6)` される。

`NormalizedSkorokhodModel` は次の六つを parameter ではなく persistent buffer として登録する。

- `x_mean`, `x_std`: shape `[1, 3]`
- `t_mean`, `t_std`: shape `[1, 1]`
- `y_mean`, `y_std`: shape `[1, 3]`

実 checkpoint で確認された代表値は次である。

```text
x_mean = [[ 0.0521,  0.1777,  0.3091]]
x_std  = [[ 0.5555,  0.6081,  0.4385]]
t_mean = [[ 0.1543]]
t_std  = [[ 0.0873]]
y_mean = [[-0.0248,  0.3799,  0.5936]]
y_std  = [[ 3.0537,  3.0087,  3.1578]]
```

値は説明用に四桁へ丸めている。replay の一致判定では丸めず、checkpoint tensor をそのまま比較する。

### 6.3 Network architecture

`MirafzaliSkorokhodNet` は次の構成である。

1. normalized input を \(z=[\hat t,\hat x]\in\mathbb R^4\) とする。
2. fixed random Fourier matrix `ff.B` により
   $$
   \phi(z)=[\sin(2\pi zB),\cos(2\pi zB)]
   $$
   を作る。
3. `[z, phi(z)]` を `in_layer` の Linear + SiLU へ入れる。
4. `n_blocks` 個の `ResidualBlock` を通す。各 block は Linear--SiLU--Linear と skip connection、最後の SiLU からなる。
5. `out_layer` で三次元出力を得る。

`ff.B` は parameter ではないが buffer なので `state_dict` に含まれ、replay で再現される。optimizer は AdamW、gradient norm clip は 1.0、scheduler は cosine annealing、loss は normalized target に対する MSE である。

### 6.4 学習関数の分岐

| teacher | 学習関数 | target | 保存時 model class |
|---|---|---|---|
| Heat | `train_s2_score_model()` | `score_target` | `NormalizedSkorokhodModel` |
| Varadhan | `train_s2_score_model()` | `score_target` | `NormalizedSkorokhodModel` |
| Malliavin | `train_s2_marginal_score()` | `skorokhod` | `S2SkorokhodScoreModel` |

`evaluate_dataset_loss()` もこの違いを保ち、Malliavin では外側 score ではなく `model.skorokhod_network(...)` と raw `skorokhod` target の MSE を評価する。

### 6.5 `model.pt`

通常学習経路は以下の outer keys を辞書として保存する。

```text
teacher
training_path
state_dict
hidden
n_blocks
num_frequencies
dtype
```

`training_path` は Heat/Varadhan で `direct_score`、Malliavin で `marginal_skorokhod` である。

Heat/Varadhan の `state_dict` key は

```text
x_mean, x_std, t_mean, t_std, y_mean, y_std
net.ff.B
net.in_layer.*
net.blocks.*
net.out_layer.*
```

である。Malliavin では外側 wrapper のため、すべて

```text
skorokhod_network.x_mean
...
skorokhod_network.net.*
```

という prefix を持つ。

## 7. Saved-model replay

### 7.1 Replay のデータフロー

evaluation-only 実行は

```text
model.pt + source run_config.json
  -> checkpoint metadataから学習時と同じclass構成を生成
  -> checkpoint["state_dict"]をstrict=Trueでload
  -> eval mode
  -> terminal_samples.ptをload
  -> original reverse_noise.ptまたはshared 1000-step poolをload
  -> s2_reverse_grw()
  -> generated_samples.pt
  -> MMD / 双方向NN距離 / density図
```

の順で進む。入口は `run_saved_model_evaluation()` である。

CLI の制約:

- `--skip-training` は `--model-path` を必須とする。
- `--skip-teacher-generation` 単独は許可せず、`--skip-training` と組み合わせる。
- replay は teacher dataset を生成せず、学習もしない。
- source run の設定から model、dtype、\(T\)、\(t_{\min}\)、sample 数を復元し、ablation で変更する数値設定は `--reverse-steps` に限定する。

model 構築は三経路で比較される。

- A: `build_model_from_run_config()`
- B: `build_model_from_training_checkpoint()` と同じ class/wrapper 構成
- C: `build_model_from_checkpoint_metadata()`

実際の replay は B を使用し、architecture の `hidden`、`n_blocks`、`num_frequencies`、`dtype` は checkpoint metadata を優先する。

### 7.2 Exact checkpoint gate

reverse sampling 直前に `require_exact_checkpoint_state()` が

- 全 key の存在
- shape
- dtype
- `torch.equal` による値の完全一致

を検査する。結果は `checkpoint_state_comparison.json` に key ごとの `max_abs_error` と `exact_equal` とともに保存される。許容条件は missing/unexpected key がなく、`overall_max_abs_error == 0.0`、`first_mismatching_key is None` である。

さらに `normalization_state_stages.json` は六つの normalization buffer を次の時点で checkpoint と比較する。

1. constructor 直後
2. `load_state_dict` 直後
3. wrapper 適用直後
4. fixed input 評価直前
5. reverse sampling 開始直前

constructor は placeholder の zero/one なので一致しなくてよい。stage 2 以降はすべて exact でなければならない。

### 7.3 `x_mean/x_std` buffer alias bug

#### 原因

旧 replay constructor は概念的に次のように placeholder を再利用していた。

```python
zero = torch.zeros(1, 3, ...)
one = torch.ones(1, 3, ...)
NormalizedSkorokhodModel(net, zero, one, ..., zero, one)
```

`register_buffer()` は渡された tensor を clone しない。このため `x_mean` と `y_mean`、`x_std` と `y_std` がそれぞれ同じ storage を指した。`load_state_dict()` は state key の値を既存 buffer へ順次 copy するため、先に正しい `x_*` を load しても、後の `y_*` copy が同じ storage を上書きした。

したがって restore 後は実質的に

$$
x_{\mathrm{mean}}\leftarrow y_{\mathrm{mean}},\qquad
x_{\mathrm{std}}\leftarrow y_{\mathrm{std}}
$$

となり、network weight、Fourier matrix、time normalization が完全一致していても fixed input の model output と reverse drift が変化した。

#### 機械的に観測された最初の差

旧 Heat artifact の `checkpoint_state_comparison.json` では、最初の不一致 key は `x_mean`、その最大絶対誤差は `0.28453699108305625` だった。`x_std` の誤差は `2.7193315930583974` で、これが checkpoint state 全体の最大誤差だった。一方、`t_mean`、`t_std`、`y_mean`、`y_std`、`net.ff.B`、`net.*` は一致していた。

同じ run の final generated samples は最大絶対誤差 `1.3654029964392962` となった。terminal samples と reverse noise が完全一致していたこと、および network parameter が一致していたことと合わせると、最初の差は reverse discretization ではなく checkpoint restore 中の `x_*` buffer 上書きである。

#### 修正

`_build_normalized_saved_model()` は六つの placeholder をそれぞれ別 tensor として生成する。

```python
x_mean = torch.zeros(...)
x_std  = torch.ones(...)
t_mean = torch.zeros(...)
t_std  = torch.ones(...)
y_mean = torch.zeros(...)
y_std  = torch.ones(...)
```

回帰テストは `x_mean.data_ptr() != y_mean.data_ptr()` と `x_std.data_ptr() != y_std.data_ptr()` を明示的に確認する。load 後、wrapper 後、fixed input 直前、reverse 直前の各 stage で checkpoint 値との exact equality も確認する。

#### 再現実験と合格条件

[`tests/test_earthquake_saved_model_reuse.py`](../tests/test_earthquake_saved_model_reuse.py) は実 checkpoint に近い、互いに明確に異なる六つの normalization 値を使い、Heat/Varadhan/Malliavin を別々に検証する。synthetic artifact では次を要求する。

- 全 state key の最大誤差 0
- fixed input output の最大誤差 \(<10^{-12}\)（float64）
- original terminal/noise を使った step 1 と final samples が `rtol=0, atol=1e-12` で一致
- debug on/off の final output が完全一致

実artifactについては `--replay-original-reverse-artifacts` が source run の `terminal_samples.pt` と `reverse_noise.pt` を集約せずそのまま用い、source `generated_samples.pt` と比較する。合格条件は最大絶対誤差 \(\le10^{-12}\) である。旧 alias-bug 実行の JSON は失敗結果を記録しているため、修正版で実artifact replay を再実行して合格 JSON を得るまでは reverse-step ablation を有効な結果として扱わない。

### 7.4 Terminal samples と reverse noise の共有

全 teacher・全 step 設定で terminal points を固定するため、`terminal_samples.pt` を共有する。

元 run の厳密再現では `--replay-original-reverse-artifacts` を指定し、元の step 数と同じ shape の `reverse_noise.pt` を `load_original_reverse_artifact()` でそのまま読む。この経路では noise coupling は exact である。

reverse-step ablation では最大 1000 step の standard normal pool

$$
Z_1,\ldots,Z_{1000}
$$

を一度 `reverse_noise_1000.pt` に保存する。`aggregate_reverse_noise_pool()` は cumulative fine Brownian path を構成し、coarse-grid 時刻で線形補間し、差分を \(\sqrt{T/N}\) で割って coarse standard-noise 表現へ戻す。

1000 が \(N\) で割り切れるとき、coarse noise は各 fine block \(I_j\) について厳密に

$$
\eta_j=\sqrt{\frac{N}{1000}}\sum_{k\in I_j}Z_k
$$

となる。128、256、512 のように割り切れない場合は、共通 1000-step cumulative path の線形補間を通じた近似 coupling であり、「同一 Brownian path の exact coarse coupling」ではない。

## 8. 評価と保存 artifact

### 8.1 MMD

`s2_rbf_mmd()` は ambient chordal squared distance を使う RBF kernel

$$
k(x,y)=\exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)
$$

の unbiased MMD estimate を計算する。計算量制御のため seed 固定の subsample を使う。

### 8.2 双方向 nearest-neighbor 距離

球面 geodesic distance

$$
d_{\mathbb S^2}(x,y)=\arccos(\operatorname{clip}(x^\top y,-1,1))
$$

を用いる。saved-model evaluation は

- generated \(\to\) target
- target \(\to\) generated

の両方向について mean、median、max を `metrics.json` に保存する。一方向だけでは mode dropping と spurious samples を区別しにくいためである。

### 8.3 Density 図

`generate_earthquake_density_plots()` は spherical KDE を緯度経度 grid 上で評価し、teacher ごとの `earthquake_density_<teacher>.png` と比較図 `earthquake_density_comparison.png` を保存する。実装は [`earthquake_smoke_viz.py`](../src/scoremodel_ext/manifold/earthquake_smoke_viz.py) にある。

主な artifact は次である。

| artifact | 内容 |
|---|---|
| `model.pt` | metadata と model `state_dict` |
| `run_config.json` | 実行設定と replay source |
| `terminal_samples.pt` | reverse process の共通初期点 |
| `reverse_noise.pt` | その run が実際に使用した reverse noise |
| `reverse_noise_1000.pt` | ablation 用 fine noise pool |
| `generated_samples.pt` | reverse GRW の最終 sample |
| `target_samples.pt` | held-out earthquake points |
| `metrics.json` | MMD、双方向 NN、有限性など |
| `normalization_state_stages.json` | normalization buffer の五段階比較 |
| `checkpoint_state_comparison.json` | checkpoint と最終 model の全 key 比較 |
| `reverse_debug_step_000.pt`, `001.pt` | opt-in reverse trace |

## 9. コード一覧

### 9.1 Geometry・forward・reverse

| 関数 / class | ファイル | 役割 |
|---|---|---|
| `s2_projector` | `src/scoremodel_ext/manifold/s2_malliavin.py` | \(P_x=I-xx^\top\) |
| `_batched_s2_projector` | 同上 | batched projector |
| `s2_to_tangent` | 同上 | ambient vector の tangent projection |
| `s2_exp` | 同上 | \(\mathbb S^2\) の指数写像 |
| `s2_grw_endpoint` | 同上 | forward GRW endpoint |
| `s2_reverse_grw` | 同上 | score-driven reverse GRW と opt-in trace |

### 9.2 Teachers

| 関数 / class | ファイル | 役割 |
|---|---|---|
| `s2_heat_kernel_score` | `src/scoremodel_ext/manifold/s2_malliavin.py` | spectral heat score |
| `s2_varadhan_score` | 同上 | small-time Varadhan score |
| `s2_projected_ambient_fields` | 同上 | \(V_a=P_xe_a\) |
| `s2_projected_ambient_field_divergence` | 同上 | \(\operatorname{div}V_a=-2x_a\) |
| `s2_discrete_malliavin_teacher` | 同上 | S² endpoint と共通 Malliavin backend の接続 |
| `discrete_malliavin_skorokhod_teacher` | `src/scoremodel_ext/manifold/malliavin_teacher.py` | \(J,\Gamma,U,\delta\), score weight |
| `tangent_malliavin_skorokhod_teacher` | 同上 | tangent-basis covariance 版 |
| `S2TeacherProvider.sample_batch` | `src/scoremodel_ext/manifold/earthquake_adapter.py` | empirical initial points から teacher batch |
| `build_teacher_dataset` | `scripts/experiment_earthquake_teacher_compare_smoke.py` | runner の三 teacher 分岐 |

### 9.3 Network・training

| 関数 / class | ファイル | 役割 |
|---|---|---|
| `FourierFeatures` | `src/scoremodel_ext/malliavin/models.py` | random Fourier embedding |
| `ResidualBlock` | 同上 | residual MLP block |
| `MirafzaliSkorokhodNet` | 同上 | \((t,x)\mapsto\mathbb R^3\) network |
| `NormalizedSkorokhodModel` | 同上 | 入出力 normalization と六 buffer |
| `train_mirafzali_skorokhod_net` | 同上 | normalized MSE training |
| `S2SkorokhodScoreModel` | `src/scoremodel_ext/manifold/s2_malliavin.py` | \(N_\theta\mapsto-P_xN_\theta\) |
| `train_s2_score_model` | `src/scoremodel_ext/manifold/experiment_s2_malliavin_teacher.py` | Heat/Varadhan direct score regression |
| `train_s2_marginal_score` | 同上 | Malliavin \(\delta\) regression + wrapper |

### 9.4 Checkpoint・replay・evaluation

| 関数 | ファイル | 役割 |
|---|---|---|
| `_build_normalized_saved_model` | `scripts/experiment_earthquake_teacher_compare_smoke.py` | alias-free placeholder model |
| `build_model_from_run_config` | 同上 | run config 経路の復元 |
| `build_model_from_training_checkpoint` | 同上 | checkpoint metadata と学習 class 経路の復元 |
| `build_model_from_checkpoint_metadata` | 同上 | metadata-only architecture 復元比較 |
| `compare_checkpoint_state` | 同上 | 全 state key の比較 |
| `require_exact_checkpoint_state` | 同上 | reverse 直前の exact gate |
| `build_model_from_training_checkpoint_with_normalization_trace` | 同上 | normalization 五段階追跡 |
| `load_original_reverse_artifact` | 同上 | 元 run noise の無変換 replay |
| `aggregate_reverse_noise_pool` | 同上 | 1000-step cumulative path の coarse 化 |
| `run_saved_model_evaluation` | 同上 | teacher生成・学習なしの reverse/evaluation |
| `compare_debug_directories` | `scripts/compare_earthquake_reverse_debug.py` | 最初に異なる step/tensor の自動特定 |
| `s2_rbf_mmd` | `src/scoremodel_ext/manifold/earthquake_adapter.py` | sample MMD |
| `nearest_neighbor_geodesic_summary` | 同上 | geodesic NN summary |
| `generate_earthquake_density_plots` | `src/scoremodel_ext/manifold/earthquake_smoke_viz.py` | spherical KDE density 図 |

## 10. 再現性チェックリスト

元 run の厳密 replay を主張するには、次を順に満たす必要がある。

1. checkpoint の全 key が exact: `overall_max_abs_error == 0`。
2. fixed input の復元経路 A/B/C が float64 で最大誤差 \(\le10^{-12}\)。
3. `terminal_samples.pt` が exact。
4. 元の `reverse_noise.pt` が exact。shared 1000-step pool はこの検査には使わない。
5. \(T\)、\(t_{\min}\)、reverse step 数、time grid が元 run と一致。
6. reverse debug step 0/1 の全 tensor が最大誤差 \(\le10^{-12}\)。
7. final `generated_samples.pt` が `rtol=0, atol=1e-12` で一致。

これらを満たした後にだけ 128/256/512/1000 step の shared-noise ablation を解釈する。

## 11. 参考文献

- Valentin De Bortoli, Emile Mathieu, Michael Hutchinson, James Thornton, Yee Whye Teh, Arnaud Doucet, [“Riemannian Score-Based Generative Modelling,” arXiv:2202.02763](https://arxiv.org/abs/2202.02763), 2022.
- Ehsan Mirafzali, Utkarsh Gupta, Patrick Wyrod, Frank Proske, Daniele Venturi, Razvan Marinescu, [“Malliavin-Bismut Score-based Diffusion Models,” arXiv:2503.16917](https://arxiv.org/abs/2503.16917), 2025.

本実装固有の離散 Malliavin backend については [`MANIFOLD_MALLIAVIN_IMPLEMENTATION.md`](MANIFOLD_MALLIAVIN_IMPLEMENTATION.md) も参照されたい。
