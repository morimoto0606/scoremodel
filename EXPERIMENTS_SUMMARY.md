# 実験まとめ: Malliavin → Mirafzali → Debortoli への進化

## 概要

このプロジェクトでは、スコアベース生成モデルにおける異なるアプローチを段階的に検証しました。
初期の **Malliavin 理論ベースのアプローチ** から始まり、**Mirafzali の高次手法** による検証を経て、最終的に **多様体上の拡散モデル (Debortoli)** へと発展させました。

---

## Phase 1: Malliavin 理論の検証 (1D/2D 実験)

### 目的
Malliavin 微積分を用いたスコア推定の可能性を検証

### 実施した実験

#### 1D 実験
- **malliavin_nonlinear_1d**: 非線形 SDE に対する Malliavin スコア推定の基本検証
- **malliavin_nonlinear_1d_corrected**: Itô 補正を適用したスコア推定
- **malliavin_reverse_1d**: 逆過程でのスコア学習の可視化
- **malliavin_time_corrected_1d**: 時間依存スコアの MLP による学習と補正
- **malliavin_teacher_mlp_1d**: Teacher MLP によるスコア推定値の検証

#### 2D 実験
- **2d_malliavin_binned_teacher**: 離散化スコア計算と学習の比較
- **2d_malliavin_reverse**: 2D での逆過程サンプリング
- **2d_time_malliavin_binned**: 時間別スコアフィールドの可視化
- **2d_time_malliavin_reverse**: 逆過程での時間的なスコア更新

### 主な知見

❌ **課題**: 線形 SDE に対しても多くの計算が必要であり、実用的なメリットが限定的

**理由**:
- Malliavin 計算は高い計算コスト（勾配計算、数値積分）を伴う
- スコア推定の精度向上が得られない場合が多い
- 特に線形 VP-SDE では既知の解析解があり、Malliavin アプローチの優位性が不明確

**決定**: Malliavin 方法の実用性が低いため、より直接的なアプローチへ転換

---

## Phase 2: SwissRoll での Mirafzali 手法の検証

### 目的
Mirafzali らの高次補正手法が、**線形 SDE** での既知解と比較して、近似度がどの程度か検証

### 実施した実験

#### 基本実験
- **linear_vs_nonlinear_swissroll_lowt_stationary_5seed**: 
  - 線形 VP-SDE の基準値測定
  - 5 シード平均での性能測定
  - **結果**: 
    - MMD: -0.000394
    - Sliced Wasserstein: 1.678481
    - 訓練時間: ~800 秒

#### Mirafzali 検証シリーズ

**1. 公式 vs 近似の比較**
- **mirafzali_approx_vs_full_swissroll_lowt_stationary_1seed**
  - 完全版（高次まで計算）vs 近似版（低次で打切）
  - **結論**: 差がほぼ無視できるレベル

**2. 補正手法の効果測定**
- **mirafzali_correction_compare_swissroll_3seeds**
  - 複数シード (3 種類) での一貫性確認
  - 弱解析性条件 vs 強解析性条件
  
- **mirafzali_correction_compare_swissroll_strong_1seed**
  - より強い条件下での補正効果

**3. フォワード初期化との関連**
- **mirafzali_full_swissroll_big_1seed** (基本)
- **mirafzali_full_swissroll_big_forward_init_1seed** (フォワード初期化)
- **mirafzali_full_swissroll_big_forward_init_rev1000_1seed** (フォワード初期化 + 長い逆過程)
- **mirafzali_full_swissroll_big_forward_init_lowt_rev1000_1seed** (低温度 + 逆過程調整)

**4. 統計量のスナップショット**
- **mirafzali_variance_diag**: 分散成分の詳細分析
- **mirafzali_variance_diag_5seed**: 5 シード平均での安定性確認

#### 残差分析シリーズ
- **mirafzali_residual_multiseed**: 複数シード間の残差分析
- **mirafzali_residual_multiseed_8gmm**: 8 成分 GMM データセット
- **mirafzali_residual_multiseed_checkerboard**: チェッカーボード分布
  
各データセットで：
  - **_smoke**: 軽量テスト
  - **_strong**: 強解析性条件
  - **_sweep**: パラメータ掃引

### 主な知見

✅ **結論**: Mirafzali の高次補正は **ほぼ無視できるレベルの改善** しかもたらさない

**詳細**:
- 近似版と完全版の差 → **非常に小さい** (~1% 以下)
- 補正の効果 → **データセット依存的かつ限定的**
- より多くの計算コスト（高次項の計算）に見合う利益がない

**決定**: 線形 SDE では Mirafzali 手法の優位性が証明できず、
→ **異なる枠組み (多様体上の拡散)** へのシフトが必要

---

## Phase 3: 多様体上の拡散モデル (Debortoli)

### 背景

線形 SDE での理論的改善が飽和したため、**本質的に異なるアプローチ** へ転換:

**質問**: 線形 SDE の仮定を緩和して、**曲がった空間 (多様体)** で生成モデルを定義したら？

**答え**: De Bortoli らの Riemannian Score-SDE フレームワーク

### 実施した実験

#### 1. S² 上での Teacher スコア比較
- **s2_debortoli_teacher_check**: 
  - 比較対象:
    - **s_var**: 測地線距離ベースの変分法スコア
      - 式: $\text{score}_{\text{var}}(x_0, x_t) = \frac{M.log(x_0, x_t)}{t}$
    - **s_db**: De Bortoli 法による周辺対数確率の勾配
      - 式: $\text{score}_{\text{db}}(x_0, x_t) = M.\text{grad\_marginal\_log\_prob}(x_0, x_t, t)$
  
  - **パラメータスイープ**:
    - 時刻: $t \in [0.01, 0.05, 0.10, 0.50, 1.00]$
    - 展開次数: $n_{\text{max}} \in [5, 10, 20, 40]$
    - 閾値: $\text{thresh} \in [0.0, 0.5]$
  
  - **結果**:
    ```
    t = 0.01:  両者ほぼ同一 (RMSE ≈ 0)
    t = 0.05:  誤差 = 5.76 (n_max増加で減少)
    t = 0.10:  誤差 = 3.99
    t = 0.50:  誤差 = 1.78 (安定化)
    t = 1.00:  誤差 = 1.25
    ```
  
  - **解釈**:
    - 早期時刻では両方法が一致
    - 中間〜後期で De Bortoli 法が展開次数に依存
    - 十分な次数で収束性を確認

#### 2. De Bortoli S² 実装のセットアップ検証
- **debortoli_reproduction**: 
  - 上流リポジトリ `riemannian-score-sde` の動作確認
  - **課題**: JAX バージョン不一致 (0.3.15 vs 0.6.2)
  - **対応**: JAX 互換性パッチの作成と適用
  
  - **パッチ内容**:
    1. `jax.random.KeyArray` の後方互換性処理
    2. `jax.linear_util` の import パス調整
    3. `jax.core.Primitive` → `jax.extend.core` への移行対応
    4. GeomStats JAX バックエンド の対応
  
  - **検証コマンド**:
    ```bash
    python main.py experiment=s2_toy steps=500 batch_size=32 \
      eval_batch_size=32 warmup_steps=10
    ```
  
  - **結果**: ✅ 全てのパッチが正常に機能

#### 3. その他の多様体実験
- **edm**: Exponential Distribution Matching
- 複数の多様体・手法の統合検証

### 主な知見

✅ **新しい視点の発見**:

1. **多様体構造の重要性**: 
   - 平面空間での線形 SDE の改善は限定的
   - 曲がった空間では生成モデルの定式化そのものが異なる

2. **De Bortoli フレームワークの有効性**:
   - S² 上で変分法と De Bortoli 法の一貫性を確認
   - 周辺分布の正確な計算が可能

3. **理論的拡張の道**:
   - 一般的な Riemannian 多様体への拡張
   - より複雑な多様体 (e.g., SO(3), Grassmannian) への応用

---

## 比較表: 各アプローチの特徴

| アプローチ | 領域 | 計算コスト | 理論的メリット | 実用性 | 次のステップ |
|-----------|------|----------|----------------|--------|-----------|
| **Malliavin** | $\mathbb{R}^d$ | 非常に高い | 限定的 | ❌ 低い | 廃止 |
| **Mirafzali 近似** | $\mathbb{R}^d$ (線形 SDE) | 中程度 | 微小 | ❌ 低い | 多様体へ転換 |
| **De Bortoli** | Riemannian 多様体 | 中程度 | 根本的 | ✅ 高い | 実装・拡張 |

---

## 実験ファイル構成

```
results/
├── Malliavin 系 (合計11実験)
│   ├── 1D実験 (6個)
│   │   ├── malliavin_nonlinear_1d/
│   │   ├── malliavin_nonlinear_1d_corrected/
│   │   ├── malliavin_reverse_1d/
│   │   ├── malliavin_time_corrected_1d/
│   │   ├── malliavin_teacher_mlp_1d/
│   │   └── teacher_compare_1d/
│   └── 2D実験 (5個)
│       ├── 2d_malliavin_binned_teacher/
│       ├── 2d_malliavin_reverse/
│       ├── 2d_time_malliavin_binned/
│       └── 2d_time_malliavin_reverse/
│
├── SwissRoll/Mirafzali 系 (合計30実験)
│   ├── 基準実験
│   │   └── linear_vs_nonlinear_swissroll_lowt_stationary_5seed/
│   ├── 比較実験 (3個)
│   │   ├── mirafzali_approx_vs_full_swissroll_lowt_stationary_1seed/
│   │   ├── mirafzali_correction_compare_swissroll_3seeds/
│   │   └── mirafzali_correction_compare_swissroll_strong_1seed/
│   ├── 変動実験 (7個)
│   │   ├── mirafzali_full_swissroll_big_1seed/
│   │   ├── mirafzali_full_swissroll_big_forward_init_*
│   │   ├── mirafzali_full_swissroll_big_stationary_lowt_rev1000_1seed/
│   │   └── ...
│   ├── 統計分析 (2個)
│   │   ├── mirafzali_variance_diag/
│   │   └── mirafzali_variance_diag_5seed/
│   ├── 非線形実験 (8個)
│   │   ├── mirafzali_nonlinear/
│   │   ├── mirafzali_nonlinear_baseline/
│   │   ├── mirafzali_nonlinear_baseline_zcorr/
│   │   └── ...
│   └── 残差分析 (10個)
│       ├── mirafzali_residual_multiseed*/
│       ├── mirafzali_residual_sweep_*/
│       └── ...
│
└── 多様体/Debortoli 系 (合計3実験)
    ├── s2_debortoli_teacher_check/
    ├── debortoli_reproduction/
    └── edm/
```

---

## 主な成果

### 1. 理論的洞察
- **Malliavin アプローチの限界**: 計算コスト vs 利益の不均衡
- **線形 SDE での飽和**: 純粋な数学的改善の限界
- **多様体への必要性**: 本質的な進展には問題設定の変更が必須

### 2. 実装的成果
- Malliavin ベースの多数の 1D/2D 実験スクリプト
- Mirafzali 手法の包括的な検証スイート
- De Bortoli フレームワークの JAX 環境での動作確認

### 3. 次の研究方向
- **Riemannian 多様体上での生成モデルの開発**
- より複雑な多様体トポロジーへの対応
- De Bortoli フレームワークの拡張と最適化

---

## 結論

> **段階的な実験を通じて、理論的に最適とされていた Malliavin・Mirafzali アプローチでは
> 十分な利益が得られないことを実証しました。
> これは本質的には「より複雑な数学的処理」では解決できず、
> **問題設定そのものの転換**（平面空間 → 多様体空間）へと導きました。**

この過程は、スコアベース生成モデルの研究における重要なパラダイムシフトを示しています。

---

*Last Updated: 2026-07-16*
