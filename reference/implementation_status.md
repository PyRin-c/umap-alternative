# 実装と検証の進捗

機能ごとに、固定した公開 wheel `umap-learn==0.6.0` と比較した範囲と、移植側の回帰テストで確認した範囲を記録する。

| 機能 | 今回の実装範囲 | 状態 |
| --- | --- | --- |
| 公開のパラメータ付き距離 | Minkowski、重み付き Minkowski、標準化 Euclidean、Mahalanobis、Haversine、Poincaré、Hellinger、softmax Hellinger、対称 KL の9種 | float32 と float64 で公開版44件を比較。ゼロは厳密一致、それ以外は相対誤差 `1e-6` 以内。距離のビット一致と無効入力の例外互換性は検証継続 |
| `UMAP(metric=..., metric_kwds=...)` | パラメータ付き距離と Python 関数を密な特徴量に使用。学習、教師ありグラフ、外部変換、教師なし更新に接続 | 公開版と Minkowski `p=1` の学習・同距離の外部変換・更新をビット照合。重み付き距離と勾配タプルを返す関数は、座標をスケールした Manhattan 対照で検証。新規距離の逆変換、疎入力、密度は未実装 |
| `nearest_neighbors` と `fuzzy_simplicial_set` | 特徴量からの厳密探索と、`query` を持つインデックスを公開 | 密 Minkowski と CSR Euclidean の学習・外部順位を SciPy の独立計算で検証。検索設定と近似バックエンドの低レベル接続は未実装 |
| `precomputed_knn` の外部変換 | 既存の Rust 変換に外部インデックスの順位を渡す | 厳密インデックスの座標と直列化、テスト用インデックスで切断値、欠損順位、無効列、特徴数を検証。Windows の NNDescent 実機確認では15列の順序が一致し、埋め込み6座標が有限。外部インデックス付きモデルの更新は未実装 |
| `force_approximation_algorithm=True` | 密な教師なし Euclidean・cosine・correlation の学習と外部変換に Rust の近似探索を使用 | 3距離のグラフ・埋め込み両モードで返却距離と直列化を検証。公開版との候補順位・座標比較は保留。行数閾値の自動切り替え、バックエンド指定、更新、教師あり、密度、疎入力は未実装 |
| AlignedUMAP と ParametricUMAP の学習グラフ | `metric_kwds` と `force_approximation_algorithm` を共通の UMAP グラフ作成に渡す | 指定距離と強制近似の学習グラフを独立した UMAP 学習と比較。ParametricUMAP は学習器を置き換え、Keras への引き渡しまでを確認。公開版との各モードの組み合わせは互換性未検証 |
| シード省略と `RandomState` | NumPy の共有 RNG または渡された RNG を初期化と直列レイアウトに使う | random・spectral の座標と RNG 状態を同じシードで比較。単一行の RNG が進まないことを確認。公開版のシード省略時の実行経路は互換性未検証 |
| 公開距離関数の追加境界ケース | cosine・correlation・Canberra・Bray–Curtis・ブール距離の、混在整数・NaN ペイロード・ストライド付き入力 | 参照 wheel で照合済み。混在する符号付き・符号なし整数の Hamming 比較で両型を float64 に変換する規則を確認 |

## 数値比較の再現

固定した公開 wheel から、直接距離44値と Minkowski の推定器の入出力を採取した。Windows と Ubuntu WSL で採取ファイルは同一だった。推定器は、学習 CSR 34重み、外部変換 COO 8項目、更新後 CSR 42重み、学習18・変換4・更新後22座標をビット比較する。学習後の外部変換には同距離の候補を含め、COO の格納順も照合する。

Windows と Ubuntu WSL の両方で、インストール済み Python wheel の4,777件、locked release の Rust 203件、C11 スモークテスト2本、警告をエラーにした Clippy が通過した。

ロック済みの参照環境から、単一スレッドで採取を再実行できる。

```sh
uv run --project reference --locked python -I reference/generate_configured_metric_fixture.py \
  --output reference/fixtures/umap-learn-0.6.0-configured-metrics-x86_64.json --check
python -I -m pytest -q tests/test_configured_metrics.py tests/test_row_distances.py tests/test_discrete_graph.py
cargo test -p umap-core --locked --test feature_metrics
```

次の比較では float32 の演算順と Mahalanobis の作業精度、推定器での残りの距離と失敗した再学習後の状態、近似探索の候補順とシード消費を扱う。

`ll_dirichlet`、残っている公開距離勾配と距離行列ヘルパー、出力空間の距離指定、新規距離の C ABI は引き続き実装対象。
