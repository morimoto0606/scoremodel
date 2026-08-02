Riemannian Malliavin Score-Based Generative Modeling 実装仕様書

1. 目的

本プロジェクトでは、Riemann 多様体上の Score-Based Generative Modeling に対して、Malliavin 解析に基づく score teacher を実装し、その理論的・数値的妥当性を検証する。

理論はプレプリントで導出した

[
V\log p_t(x)

E!\left[D^*u_t^V\mid X_t=x\right]

\operatorname{div}V(x)
]

に基づく。

ここで、各経路から得られる Malliavin weight 自体を score とみなすのではなく、その条件付き期待値

[
E!\left[H_t\mid X_t=x\right]
]

が score を与えるという関係を利用する。

実装は既存 upstream の Riemannian Score SDE コードを変更せず、原則として

scoremodel_ext

以下に追加する。

⸻

2. 基本方針
scriptの実行はコマンドをだす．
ユーザーがそれを自分で実行する．

現在実装済みの以下の機能は、そのまま利用する。

* geodesic random walk
* endpoint map
* heat kernel score
* Varadhan score
* reverse geodesic random walk
* tangent projection
* tangent basis
* S² 上の score reconstruction

既存実装を削除・置換せず、Malliavin teacher、学習用 dataset、adapter、評価機能を追加する。

特に、以下を遵守する。

* upstream は変更しない。
* scoremodel_ext 以下で拡張する。
* 既存 API の後方互換性を維持する。
* 数式上の定義と実装上の tensor shape を明示的に対応させる。
* 各 Phase で unit test と numerical experiment を分離する。
* unit test の成功と、数値精度の良さを混同しない。
* 単一の実験結果だけで収束や理論的妥当性を結論しない。

⸻

3. 理論上の基本構造

離散 noise を

[
Z\in\mathbb R^m
]

とし、endpoint map を

[
X_T=\Phi(Z)
]

とする。

endpoint Jacobian を

[
J

\frac{\partial X_T}{\partial Z}
]

とする。

多様体の接空間を表す orthonormal tangent basis を

[
E(X_T)
]

とし、接空間表示された Jacobian を

[
J_{\mathrm{tan}}

E(X_T)^\top J
]

とする。

Malliavin covariance は

[
\gamma

J_{\mathrm{tan}}
J_{\mathrm{tan}}^\top
]

で与える。

対象ベクトル場を列方向に並べて

[
V_{\mathrm{tan}}
\in
\mathbb R^{d\times n_{\mathrm{fields}}}
]

とする。

regularization parameter を (\lambda>0) とし、covering field を

[
U

J_{\mathrm{tan}}^\top
(\gamma+\lambda I)^{-1}
V_{\mathrm{tan}}
]

で計算する。

各対象ベクトル場 (V_j) に対して、

[
V_j\log p_T(x)

E[D^*u_T^{V_j}\mid X_T=x]

\operatorname{div}V_j(x)
]

を用いて directional score を構成する。

複数の directional score から、接ベクトルとしての score を再構成する。

⸻

4. Phase 1：Generic Malliavin teacher backend

4.1 目的

generic discrete Malliavin–Skorokhod backend を完成させる。

単一ベクトル場だけでなく、複数の target fields を同時に扱えるようにする。

4.2 target fields API

target_fields_fn(endpoint)

は、原則として

[ambient_dim, n_fields]

の tensor を返す。

単一ベクトル場を入力した既存 API も後方互換性のため維持する。

4.3 実装内容

以下を計算する。

[
J

\frac{\partial X_T}{\partial Z},
\qquad
J_{\mathrm{tan}}

E^\top J,
]

[
\gamma

J_{\mathrm{tan}}J_{\mathrm{tan}}^\top,
]

[
U

J_{\mathrm{tan}}^\top
(\gamma+\lambda I)^{-1}
V_{\mathrm{tan}}.
]

各 field について、

* Gaussian pairing
* divergence term
* Skorokhod divergence
* directional score weight

を同時に計算する。

4.4 teacher の返り値

最低限、以下を返す。

* endpoint
* endpoint Jacobian
* tangent Jacobian
* Malliavin covariance
* covariance eigenvalues
* condition number
* covering field
* Gaussian pairing
* divergence term
* Skorokhod divergence
* directional score weight
* reconstructed score weight
* right inverse residual

4.5 diagnostics

以下を数値診断として利用する。

* covariance eigenvalues
* condition number
* right inverse residual
* tangent residual
* endpoint norm error
* NaN rate
* Gaussian pairing
* divergence term

4.6 Unit Test

以下を確認する。

* Euclidean additive-noise case の厳密解
* tangent reduction
* redundant target fields
* projector
* tangent basis
* endpoint map
* endpoint Jacobian
* Malliavin covariance
* covering field
* exact Skorokhod divergence
* finite difference との一致
* regularization sweep
* single-field API の後方互換性
* endpoint Jacobian が不要な autograd graph を保持しないこと

4.7 現在の状態

Phase 1 は完了済みとする。

確認済み事項：

* generic backend 実装済み
* multi-field teacher 実装済み
* diagnostics 実装済み
* finite-difference tests 実装済み
* backward compatibility tests 実装済み
* High severity review 対応済み
* 全体 test suite 通過済み
* 対象 test file 31 tests passed

⸻

5. Phase 2：S² fixed-start validation

Phase 2 は、目的の異なる三つの段階に分ける。

⸻

5.1 Phase 2A：固定初期値・固定終端時刻での teacher 検証

目的

固定初期値

[
X_0=x_0\in S^2
]

および固定終端時刻 (T) に対して、

* heat kernel score
* Varadhan score
* Malliavin teacher の条件付き平均

を比較する。

これは、学習前の teacher 自体の検証である。

Target fields

(S^2\subset\mathbb R^3) 上では、

[
P(x)=I-xx^\top
]

を接空間 projector とし、

[
A_i(x)=P(x)e_i,
\qquad i=1,2,3
]

を用いる。

divergence は

[
\operatorname{div}A_i(x)=-2x_i
]

を利用する。

teacher は、

* directional_score_weight
* score_weight

の両方を返す。

Dataset

固定した (x_0)、(T) に対して複数経路を生成する。

各経路について、

* initial point
* endpoint
* noise
* directional score weight
* reconstructed score weight
* covariance diagnostics

を保存する。

使用する既存関数は、単一終端時刻の比較用として

generate_s2_teacher_dataset

とする。

条件付き期待値の数値近似

pathwise Malliavin weight を直接 heat score と比較しない。

endpoint (X_T=x) の近傍にある sample を用いて、k-nearest-neighbor 平均などにより

[
E[H_T\mid X_T=x]
]

を近似する。

比較対象

以下を比較する。

[
s_{\mathrm{Mall}}(T,x),
\qquad
s_{\mathrm{heat}}(T,x),
\qquad
s_{\mathrm{Varadhan}}(T,x).
]

評価項目

* Malliavin vs heat RMSE
* Malliavin vs heat cosine similarity
* Malliavin vs Varadhan RMSE
* Malliavin vs Varadhan cosine similarity
* Varadhan vs heat RMSE
* Varadhan vs heat cosine similarity
* geodesic distance ごとの RMSE
* geodesic distance ごとの cosine similarity
* 各 geodesic bin の sample count
* endpoint norm error
* tangent residual
* NaN rate
* covariance eigenvalues

主要な metric key は以下とする。

malliavin_vs_heat_rmse
malliavin_vs_heat_mean_cosine
malliavin_vs_varadhan_rmse
malliavin_vs_varadhan_mean_cosine
varadhan_vs_heat_rmse
varadhan_vs_heat_mean_cosine

後方互換用 alias がある場合は維持してよい。

数値実験の順序

Smoke run

小規模設定で以下を確認する。

* CUDA 上で実行できる
* dataset generation が完了する
* metrics computation が完了する
* artifact が保存される
* NaN がない
* endpoint が (S^2) 上にある
* score が接空間にある

Baseline run

path 数を増やし、

* RMSE
* cosine similarity
* geodesic bin ごとの精度
* runtime

を smoke run と比較する。

Long run

baseline が安定した場合のみ大規模実験を行う。

Parameter sweep

最終的には以下を変化させる。

* n_paths
* n_steps
* seed
* covariance_regularization
* terminal_time
* knn_k

単一 run では収束を結論しない。

現在の状態

Phase 2A の smoke run は完了済み。

例：

device: cuda
dtype: float64
n_paths: 64
n_steps: 8
terminal_time: 0.3

確認済み事項：

* CUDA 上で完走
* NaN rate = 0
* endpoint norm error は機械精度
* tangent residual は機械精度
* heat score と高い方向一致
* geodesic distance が大きい領域で RMSE が増える傾向

次は baseline run とする。

⸻

5.2 Phase 2B：固定初期値・可変時刻での marginal score 学習

目的

固定初期値

[
X_0=x_0
]

に対して、時刻 (t) を区間から変化させ、

[
s_\theta(t,x)
\approx
\nabla_x\log p_t(x\mid x_0)
]

を学習する。

reverse sampling に必要なのは、単一時刻の score ではなく、時間依存 score である。

Dataset

使用する既存関数は、

generate_s2_fixed_start_marginal_teacher_dataset

とする。

この dataset は、

* initial point は固定
* time は minimum_time と maximum_time の間で変化
* endpoint は各時刻の GRW endpoint
* target は Malliavin teacher

という構造を持つ。

学習

既存の score network を用い、

train_s2_marginal_score

または既存設計に対応する

train_s2_score_model

を使用する。

ネットワークは、

(time, endpoint)

を入力し、

score_target

を出力する。

出力は必ず接空間へ projection する。

評価

* train loss
* validation loss
* heat score に対する MSE
* cosine similarity
* time bin ごとの誤差
* geodesic bin ごとの誤差
* tangent residual
* norm statistics

注意事項

Malliavin teacher の pathwise sample は分散を持つため、学習 loss が小さいことだけでは score の精度を保証しない。

必ず heat kernel score との比較を行う。

⸻

5.3 Phase 2C：固定初期値に対する reverse sampling

目的

Phase 2B で学習した score を

s2_reverse_grw

へ渡し、reverse sampling が正しく動作することを確認する。

比較する score

* heat kernel score
* Varadhan score
* learned Malliavin score

使用する比較関数

既存の

build_s2_reference_score_functions
compare_s2_reverse_generators

を利用する。

評価項目

* generated samples の norm error
* initial point までの geodesic distance
* mean geodesic distance
* median geodesic distance
* RMSE geodesic distance
* max geodesic distance
* heat generator との pairwise geodesic distance
* Varadhan generator との pairwise geodesic distance
* learned generator との比較

完了条件

* 全 sample が (S^2) 上にある
* NaN がない
* heat score による reverse sampling が基準として妥当
* learned score が heat score に近い生成挙動を示す
* 複数 seed で結果が安定する

⸻

6. Phase 3：Mixture distribution

6.1 目的

初期分布を一点ではなく mixture に変更し、Malliavin teacher によって mixture の marginal score を学習できることを確認する。

初期分布を

[
\mu_0

\sum_{k=1}^K
w_k\delta_{x_k}
]

とする。

6.2 Dataset

既存の

generate_s2_mixture_marginal_teacher_dataset

を使用する。

各 sample について、

* component index
* initial point
* time
* endpoint
* score target

を保存する。

6.3 現在の状態

以下の基盤実装は完了済み。

* mixture component sampling
* component weight handling
* initial point reconstruction
* score target shape test

ただし、以下は未完了。

* 本格的な score 学習
* heat mixture score との比較
* reverse sampling
* 複数 seed 評価
* mixture weight 回復の検証

6.4 学習と評価

評価項目：

* train loss
* validation loss
* reference mixture score との MSE
* cosine similarity
* generated sample の component allocation
* MMD
* nearest-neighbor geodesic distance
* mode collapse の有無
* mixture weight の再現性

6.5 注意事項

「teacher が marginal score を学習する」とは、pathwise Malliavin weight 自体を score とすることではない。

ネットワークが conditional expectation を回帰することで、

[
E[H_t\mid X_t=x]
]

を近似し、marginal score を得る。

⸻

7. Phase 4：Earthquake adapter

7.1 目的

既存 upstream の Earthquake experiment に対して、upstream を変更せず adapter を介して Malliavin teacher を接続する。

7.2 基本方針

upstream の training loop、network、optimizer、dataset handling は変更しない。

adapter は upstream が要求する

(time, endpoint, score_target)

形式を返す。

7.3 Adapter API

既存の

S2TeacherProvider

を利用する。

teacher は切り替え可能とする。

malliavin
heat
varadhan

Malliavin teacher の場合のみ、必要に応じて以下も返す。

* directional score target
* Skorokhod divergence
* covariance diagnostics

7.4 現在の状態

以下の adapter 基盤は実装済み。

* S2TeacherProvider
* teacher switching
* Malliavin batch generation
* heat batch generation
* Varadhan batch generation
* tangent target check
* direct score model training
* score model evaluation
* S² RBF MMD
* nearest-neighbor geodesic summary
* train/validation split utility

ただし、本格的な Earthquake comparison experiment は未完了。

7.5 Earthquake experiment

teacher のみを切り替え、以下を固定する。

* network architecture
* optimizer
* learning rate
* batch size
* number of epochs
* train/validation split
* reverse sampler
* evaluation sample size

比較対象：

* heat kernel teacher
* Varadhan teacher
* Malliavin teacher

7.6 評価項目

* train loss
* validation loss
* score MSE
* generated samples
* MMD
* nearest-neighbor geodesic distance
* pairwise geodesic distance
* norm error
* tangent residual
* training time
* teacher generation time
* inference time

7.7 公平性

teacher 以外の条件を変更しない。

特に、

* Malliavin teacher だけ network を大きくしない。
* optimizer を変えない。
* epoch 数を変えない。
* evaluation sample size を変えない。
* Malliavin teacher だけ別の preprocessing を使用しない。

⸻

8. 実験 runner と CLI

8.1 基本方針

長時間実験は再現可能な CLI から実行する。

CLI は backend の数式を再実装せず、既存関数を呼び出すだけとする。

8.2 必須引数

最低限、以下を指定可能にする。

--n-paths
--n-steps
--terminal-time
--minimum-time
--maximum-time
--covariance-regularization
--heat-terms
--knn-k
--seed
--device
--dtype
--output-dir

実験の種類によって不要な引数は省略可能とする。

8.3 Device

GPU実験では、

--device cuda

を明示する。

CUDAを要求した場合は、CPUへ黙って fallback しない。

CUDAが利用できない場合は、明示的にエラーとする。

8.4 保存する成果物

各 run について、最低限以下を保存する。

teacher_dataset.pt
metrics.json
run_config.json
run.log

学習を含む場合は追加で、

model.pt
training_history.json
generated_samples.pt

を保存する。

8.5 ログ

最低限、以下を出力する。

* experiment start
* device
* dtype
* seed
* n_paths
* n_steps
* time range
* dataset generation start
* dataset generation completed
* training start
* training completed
* metric computation start
* metric computation completed
* artifact paths
* generation seconds
* training seconds
* metric seconds
* total seconds

8.6 完了判定

以下を満たした場合のみ、実験完了とする。

1. Python process の exit code が 0。
2. 必要な artifact が存在する。
3. metrics.json が JSON として読み込める。
4. 主要 metric が存在する。
5. NaN rate が記録されている。
6. 実行設定と所要時間が記録されている。

ツール側の

Completed with input

などのメッセージだけでは完了と判断しない。

⸻

9. テスト方針

9.1 Unit tests

原則として CPU で実行し、CI 互換性を維持する。

小規模 Jacobian、finite difference、shape、API compatibility を検証する。

9.2 CUDA smoke tests

CUDA device propagation を確認する専用 smoke test を追加してよい。

CUDAが利用できない場合は skip する。

確認項目：

* CUDA上で dataset generation が動く
* 出力が finite
* device propagation が維持される
* CPU tensor と CUDA tensor の混在がない

9.3 Numerical experiments

unit test とは分離する。

以下を数値的に確認する。

* path 数依存性
* time-step 数依存性
* seed 依存性
* regularization 依存性
* terminal time 依存性
* kNN parameter 依存性

⸻

10. 各 Phase 完了時の報告内容

各 Phase が完了したら、以下を報告する。

1. 変更したファイル
2. 数式との対応
3. 追加・変更した API
4. 実行した unit tests
5. unit test の結果
6. 実行した numerical experiments
7. numerical metrics
8. artifacts の保存先
9. 残っている課題
10. 次に実装または検証する内容

以下を明確に区別する。

* コードが実行可能であること
* unit tests が通ったこと
* 幾何学的制約を満たすこと
* heat score と数値的に一致すること
* path 数や step 数に対して収束すること
* reverse sampling が妥当であること

⸻

11. 現在の進捗

Phase 1

完了。

* generic backend
* multi-field teacher
* diagnostics
* finite-difference validation
* backward compatibility
* High severity review 対応
* full test suite 通過

Phase 2A

進行中。

* fixed-start, fixed-time experiment 実装済み
* CUDA smoke run 完了
* heat / Varadhan / Malliavin comparison 完了
* geodesic bin metrics 実装済み

次の作業：

* baseline run
* path 数比較
* step 数比較
* seed sweep
* regularization sweep

Phase 2B

基盤実装済み、数値検証未完了。

* fixed-start marginal dataset
* time-dependent score training
* validation utility

Phase 2C

smoke test 実装済み、本格検証未完了。

* reverse GRW
* heat / Varadhan / trained score comparison

Phase 3

dataset 基盤実装済み、本格学習未完了。

* mixture sampling
* component index
* marginal teacher target

Phase 4

adapter 基盤実装済み、本格 Earthquake comparison 未完了。

* teacher provider
* metrics
* evaluation utilities
* experiment script 接続

⸻

12. 最終目標

最終的に以下を達成する。

1. S² 上で Malliavin teacher が理論式と整合して動作することを確認する。
2. Malliavin weight の条件付き平均が heat kernel score を近似することを数値的に確認する。
3. path 数、step 数、seed、regularization に対して結果が安定することを確認する。
4. 固定初期値に対して時間依存 score model を学習する。
5. 学習した score を用いた reverse sampling が、heat kernel score を用いた reverse sampling に近いことを確認する。
6. mixture 初期分布に対して marginal score を学習し、複数 mode を再現する。
7. Earthquake dataset に adapter を介して接続する。
8. 以下を公平な条件で比較する。

* heat kernel teacher
* Varadhan teacher
* Malliavin teacher

9. 比較結果を以下の観点で整理する。

* score accuracy
* train / validation loss
* generated sample quality
* MMD
* geodesic distance
* computational cost
* numerical stability

⸻

13. 注意事項

* upstream は変更しない。
* scoremodel_ext 以下で実装する。
* 既存の GRW、reverse GRW、heat kernel score、Varadhan score を維持する。
* teacher の pathwise sample と conditional score を混同しない。
* unit test の成功だけで numerical accuracy を結論しない。
* 単一 seed の結果だけで収束を結論しない。
* 遠方 geodesic bin は sample count と併せて評価する。
* CUDA実験では device=cuda を明示する。
* 長時間 run の前に smoke run と baseline run を行う。
* artifact が保存されたことを確認してから完了と判断する。
* 各 Phase の検証を終えてから次の本格実験へ進む。