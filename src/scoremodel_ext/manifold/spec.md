# Riemannian Malliavin Score-Based Generative Modeling 実装仕様書

## 目的

本プロジェクトでは，Riemann 多様体上の Score-Based Generative Modeling に対して，
Malliavin 解析に基づく score teacher を実装する．

理論はプレプリントで導出した

\[
V\log p_t(x)
=
-
E[D^*u_t^V\mid X_t=x]
-
\operatorname{div}V(x)
\]

に基づく．

実装は既存の upstream (Riemannian Score SDE) を変更せず，
`scoremodel_ext` 以下のみで実装する。

---

# 基本方針

現在実装済みの

- geodesic random walk
- endpoint map
- heat kernel score
- Varadhan score
- reverse GRW

はそのまま利用する。

Malliavin teacher を追加し，
既存実装と比較できるようにする。

upstream のコードは変更しない。

---

# Phase 1 : Malliavin teacher の完成

## 目的

現在の tangent teacher は一つのベクトル場しか扱えない。

これを複数のベクトル場を同時に扱えるように変更する。

target_fields_fn(endpoint)

が

```
[ambient_dim, n_fields]
```

を返せるようにする。

内部では

\[
J=\frac{\partial X_T}{\partial Z}
\]

から

\[
J_{\mathrm{tan}}
=
E^\top J
\]

を構成する。

さらに

\[
\gamma
=
J_{\mathrm{tan}}J_{\mathrm{tan}}^\top
\]

を用いて

\[
U
=
J_{\mathrm{tan}}^\top
(\gamma+\lambda I)^{-1}
V_{\mathrm{tan}}
\]

を各ベクトル場について同時に計算する。

---

## diagnostics

teacher の返り値に以下を追加する。

- covariance
- covariance eigenvalues
- condition number
- right inverse residual
- gaussian pairing
- divergence term

論文用の解析にも利用できるようにする。

---

## Unit Test

以下を確認する。

- projector
- tangent basis
- exponential map
- endpoint Jacobian
- Malliavin covariance
- covering field
- Skorokhod divergence

有限差分との一致も確認する。

---

# Phase 2 : S² teacher

## Target fields

S² 上では

\[
A_i(x)=P(x)e_i
\]

を用いる。

divergence は

\[
\operatorname{div}A_i=-2x_i
\]

を利用する。

teacher は

- directional_score_weight
- score_weight

の両方を返す。

---

## heat kernel との比較

固定初期値

\[
X_0=x
\]

について

- heat kernel score
- Varadhan score
- Malliavin teacher

を比較する。

評価項目

- RMSE
- cosine similarity
- geodesic distance ごとの誤差

---

## Reverse sampling

学習した score を

```
s2_reverse_grw
```

へ渡し，

生成結果を

- heat kernel score
- Varadhan score

と比較する。

---

# Phase 3 : Mixture distribution

初期分布を一点ではなく mixture に変更する。

teacher が周辺 score を学習できることを確認する。

---

# Phase 4 : Earthquake adapter

ここで初めて upstream に接続する。

**upstream は一切変更しない。**

adapter を追加するだけにする。

例えば

```
MalliavinScoreProvider
```

のようなクラスを作り，

training loop が要求する

```
(time,
 endpoint,
 score_target)
```

を返す。

---

# Earthquake experiment

既存の Earthquake 実験を利用し，

teacher のみ

- heat kernel
- Varadhan
- Malliavin

に切り替えて比較する。

ネットワークや optimizer は変更しない。

比較項目

- train loss
- validation loss
- generated samples
- MMD
- geodesic distance
- 学習時間

---

# 実装方針

既存の

```
s2_malliavin.py
```

にある

- geodesic random walk
- reverse GRW
- heat kernel score
- Varadhan score

は削除・変更しない。

追加実装のみ行う。

---

# 各 Phase 完了時に報告すること

各 Phase が終わるたびに

1. 変更したファイル
2. 数式との対応
3. 実行したテスト
4. テスト結果
5. 残っている課題
6. 次に実装する内容

を報告する。

---

# 最終目標

最終的には

1. S² 上で Malliavin teacher が理論通り動作することを確認する。

2. heat-kernel score と比較して精度を評価する。

3. reverse sampling が正しく動作することを確認する。

4. Earthquake dataset に adapter を介して接続し，

- 既存手法
- Malliavin teacher

を公平な条件で比較する。

---

# 注意事項

- upstream は変更しない。
- `scoremodel_ext` 以下だけで実装する。
- 既存の geodesic random walk 実装は維持する。
- 各 Phase の Unit Test を通してから次へ進む。
- Earthquake 実験は最後に行う。