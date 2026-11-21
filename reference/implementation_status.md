# 実装と検証の進捗

機能ごとに、固定した公開 wheel `umap-learn==0.6.0` と比較した範囲と、移植側の回帰テストで確認した範囲を記録する。

| 機能 | 今回の実装範囲 | 状態 |
| --- | --- | --- |
| 公開のパラメータ付き距離 | Minkowski、重み付き Minkowski、標準化 Euclidean、Mahalanobis、Haversine、Poincaré、Hellinger、softmax Hellinger、対称 KL の9種 | float32 と float64 で公開版44件を比較。ゼロは厳密一致、それ以外は相対誤差 `1e-6` 以内。距離のビット一致と無効入力の例外互換性は検証継続 |
| `UMAP(metric=..., metric_kwds=...)` | パラメータ付き距離と Python 関数を密・CSR 特徴量の学習、外部変換、教師なし更新に使用。密な逆変換では供給勾配または数値微分を使用し、互換レイアウトで密度を扱う | 既存の公開 Minkowski `p=1` 照合に加え、CSR と密入力の学習・外部変換・更新・密度、密入力の勾配付き逆変換を確認。設定付き距離の公開版との逆変換・密度のビット比較は保留 |
| `nearest_neighbors` と `fuzzy_simplicial_set` | 特徴量からの厳密探索と、`query` を持つインデックスを公開 | 密 Minkowski と CSR Euclidean の学習・外部順位を SciPy の独立計算で検証。検索設定と近似バックエンドの低レベル接続は未実装 |
| `precomputed_knn` の外部変換 | 既存の Rust 変換に外部インデックスの順位を渡す | 厳密インデックスの座標と直列化、テスト用インデックスで切断値、欠損順位、無効列、特徴数を検証。Windows の NNDescent 実機確認では15列の順序が一致し、埋め込み6座標が有限。外部インデックス付きモデルの更新は未実装 |
| 近似探索の切り替え | 密な教師なし Euclidean・cosine・correlation で有効な行が4,096以上の場合、または明示的な強制指定時に Rust の近似探索を使用。Euclidean の分割が必要な学習では、検証済みの投影葉から候補を投入する | 4,095・4,096行の閾値、重複除去後の判定、外部変換、直列化を検証。公開版と一般の候補・問い合わせ順位を揃える作業は継続。バックエンド指定、近似モデルの更新、教師あり、密度、疎入力は未実装 |
| AlignedUMAP と ParametricUMAP の学習グラフ | `metric_kwds` と `force_approximation_algorithm` を共通の UMAP グラフ作成に渡す | 指定距離と強制近似の学習グラフを独立した UMAP 学習と比較。ParametricUMAP は学習器を置き換え、Keras への引き渡しまでを確認。公開版との各モードの組み合わせは互換性未検証 |
| シード省略と `RandomState` | NumPy の共有 RNG または渡された RNG を初期化と直列レイアウトに使う | random・spectral の座標と RNG 状態を同じシードで比較。単一行の RNG が進まないことを確認。公開版のシード省略時の実行経路は互換性未検証 |
| 公開距離関数の追加境界ケース | cosine・correlation・Canberra・Bray–Curtis・ブール距離の、混在整数・NaN ペイロード・ストライド付き入力 | 参照 wheel で照合済み。混在する符号付き・符号なし整数の Hamming 比較で両型を float64 に変換する規則を確認 |
| 再帰 PCA のサブサンプリング | `anchor_sample_size` を指定可能。選んだ行で PCA を行い、残りの行を学習グラフ上で隣接するアンカーの加重平均に置く。隣接アンカーがない行はアンカー全体の平均を使う | 17行・7アンカーの通常・加重グラフで公開版と照合し、座標差は絶対誤差 `2e-6` 以内。既存の全行 PCA の残差と、悪条件の弱い軸は継続 |
| densMAP の入力半径 | 設定付き距離・パック済み距離の距離を CSR の各辺に揃えて最適化と診断へ渡す。新規経路では平均を float64 に広げてから対数のオフセットを加える | 公開 `compute_densities` の1,024対照で広い演算を確認。既存の密・事前計算距離経路への一律適用は2件の座標比較を悪化させたため、新規経路に適用。密度勾配全体のビット差は継続 |
| AlignedUMAP の教師あり近傍数 | `target_n_neighbors` を各スライスの UMAP に渡す | スライスごとの連続ターゲットで確認。最後尾の非対応行を含む時間窓2の座標差は継続 |
| ParametricUMAP の学習設定・保存 | 学習エポック数、損失報告頻度、Keras オプティマイザーを指定できる。エンコーダー、デコーダー、学習モデルと推定器状態を保存・復元する | Windows CPU の実 Keras で19行を学習し、入力保存の有無の両方で復元後の埋め込みと推論をビット照合。比較スクリプトでも30予測を確認。公開版と多ステップ学習の座標には最大 `3.38e-5` の差が残る |

## 数値比較の再現

固定した公開 wheel から、直接距離44値と Minkowski の推定器の入出力を採取した。Windows と Ubuntu WSL で採取ファイルは同一だった。推定器は、学習 CSR 34重み、外部変換 COO 8項目、更新後 CSR 42重み、学習18・変換4・更新後22座標をビット比較する。学習後の外部変換には同距離の候補を含め、COO の格納順も照合する。

Windows と Ubuntu WSL の両方で、インストール済み Python wheel の4,789件、locked release の Rust 204件、C11 スモークテスト2本、警告をエラーにした Clippy が通過した。

ロック済みの参照環境から、単一スレッドで採取を再実行できる。

```sh
uv run --project reference --locked python -I reference/generate_configured_metric_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-configured-metrics-x86_64.json --check
python -I -m pytest -q tests/test_configured_metrics.py tests/test_row_distances.py tests/test_discrete_graph.py
cargo test -p umap-core --locked --test feature_metrics
```

次の比較では float32 の演算順と Mahalanobis の作業精度、推定器での残りの距離と失敗した再学習後の状態、近似探索の候補順とシード消費を扱う。

`ll_dirichlet`、残っている公開距離勾配と距離行列ヘルパー、出力空間の距離指定、新規距離の C ABI は引き続き実装対象。

## 残差の再調査

- 全行 PCA は列合計を float32 で求め、平均と中心化を float64 で行う。27行の PCA 単独ケースと、129行の悪条件2ケースで計828座標と RNG 状態が公開版に一致した。再帰収縮後の微小差は継続。 [PCA の記録](modern_recursive_rounding_reference.md)
- ParametricUMAP の一括損失計算を修正し、Windows CPU の学習済みモデルで公開版の162出力とビット一致した。凍結モデルの30予測と、保存・復元も確認した。 [学習の記録](parametric_reference.md)
- 密な特徴量の直列近似探索は Euclidean、correlation、cosine の計34ケースで学習と問い合わせの順位・距離・グラフ重みがビット一致した。cosine の4候補一括評価と学習時の候補ヒープ順を再現した。4,095行と4,096行の自動切り替え、呼び出し元の乱数状態、16特徴と64特徴の境界前後も検証した。 [近似探索の検証](approximate_search_parity.md)
- AlignedUMAP の時間窓2では、参照版の時間オフセットを境界チェック付きで再現した。末尾1行が未対応の8エポックケースの最大差は `0.5040512` から `3.58e-7` に縮小した。末尾2行が未対応の追加3ケースは全120座標が一致した。 [時間窓の記録](aligned_reference.md)
- densMAP は入力半径の標準化まで47値が一致した。3次元の初回密度更新を1辺ずつ比較すると最大 `1.34e-5` の差があり、勾配の演算順を引き続き調べる。 [勾配の記録](densmap_layout_reference.md)
