# umap-alternative

UMAP を Rust で独立実装するリポジトリです。`umap-learn` との数値の完全一致を検証したうえで高速化し、Rust ライブラリ、C ABI、PyO3 + Maturin による Python Wheel として提供します。

`crates/umap-core` が Rust 本体の置き場で、`crates/umap-python` が Python 拡張です。Python Wheel は `UMAP(transform_mode="graph")` のグラフ構築と変換、および固定シードを使った密な入力の埋め込みを提供します。`crates/umap-capi` の C ABI は、密な入力のグラフ構築と固定シードによる埋め込みに加え、学習済みデータに対する新しい入力の埋め込み変換と有向 COO グラフを提供します。数値のビット単位の一致は、記録済みの fixture で検証しています。

## 開発環境

- Rust 1.93.1 と Clippy、rustfmt は `rust-toolchain.toml` で指定しています。Rustup が取得します。
- 開発用の CPython は `.python-version` で 3.13.12 に固定しています。uv 0.10.8 で動作確認しています。
- Windows では Visual Studio 2022 Build Tools または Visual Studio の「C++ によるデスクトップ開発」と Windows SDK が必要です。Linux では C リンカーが必要です。

リポジトリのルートで実行します。

```sh
uv python install
uv sync --locked
cargo fmt --all -- --check
uv run --locked cargo clippy --workspace --all-targets --locked -- -D warnings
uv run --locked cargo test --workspace --locked
uv run --locked pytest
```

`uv sync --locked` は `.venv` に開発依存と編集可能な Python パッケージをインストールします。Rust を変更した後は `uv sync --locked --reinstall-package umap-alternative` で拡張を再ビルドします。

## Python Wheel

`from umap_alternative import ParametricUMAP` で Keras を使った埋め込みを学習できます。
複数行の学習には TensorFlow と Keras が必要です。ライブラリのインポート時には読み込みません。
`encoder` と `decoder` には Keras モデルを指定できます。`parametric_reconstruction=True`
の場合、`inverse_transform` は学習済みデコーダを使います。
[Windows CPU での検証結果](reference/parametric_reference.md)

```sh
uv run --locked maturin build --release --locked --interpreter python
```

Wheel は `target/wheels/` に生成されます。Python パッケージ名は `umap-alternative`、インポート名は `umap_alternative` です。Windows と Linux の CI では Wheel を隔離環境にインストールし、インポートまで確認します。

`UMAP(random_state=42, n_jobs=1, compatibility_layout=True).fit_transform(X)` は、密な特徴量から `float32` の `embedding_` を作り、学習時の CSR グラフを `graph_` に格納します。既定の `init="spectral"` は枝刈り後のグラフの連結成分を配置します。グラフが連結の場合、出力次元数はサンプル数未満にしてください。`init="random"` や、各行が異なり各列に幅がある NumPy 配列も指定できます。`n_epochs` には非負の整数を指定でき、小規模データでは `None` が 500 エポックになります。学習データとビット列が一致する入力の `transform` は `embedding_` を返します。

密な新しい入力の `transform` は、学習時の座標を固定したまま `float32` の座標を返します。`transform_seed` は学習の `random_state` とは独立した整数シードで、既定値は `42` です。明示した `n_epochs` の 3 分の 1 を切り捨てた回数で最適化し、`None` の場合は入力が 10,000 行以下なら 100 エポック、それより多ければ 30 エポックを使います。検証結果と適用範囲は [埋め込み変換の参照記録](reference/transform_embedding_reference.md) を参照してください。

密な Euclidean 特徴量で学習した互換レイアウトでは、`inverse_transform` が埋め込み座標から元の特徴量と同じ列数の `float32` 配列を返します。埋め込みの凸包の外にある点は外挿し、警告を出します。検証結果と適用範囲は [逆変換の参照記録](reference/inverse_transform_reference.md) を参照してください。

密な Euclidean 特徴量の教師なし学習では、`update(X)` が行を追加し、学習済みの近傍数を使って `graph_` と `embedding_` を更新します。学習済みの点も再配置の対象になり、`n_epochs=None` の更新は初期化まで進みます。事前計算近傍や有限の切断距離を使う更新は未対応です。[更新の参照記録](reference/update_reference.md)

`output_dens=True` を指定した密な Euclidean 埋め込みでは、`fit_transform` が埋め込み、元空間の対数半径、埋め込み空間の対数半径を返します。近傍を内部で計算し、有限の切断距離を使わない場合に利用できます。[半径の参照記録](reference/densmap_reference.md)

`densmap=True` の直列学習は、密な特徴量と密な事前計算距離に対応します。複数の epoch を指定すると、`embedding_list_` に各時点の座標が入ります。C の密入力は `umap_capi_fit_dense_densmap_seeded` で学習できます。[densMAP の比較結果と精度差](reference/densmap_layout_reference.md)

`metric="precomputed"` の外部変換は、問い合わせ行 × 学習行の密な距離行列を受け取り、COO グラフまたは埋め込み座標を返します。有限の切断距離を指定した学習では、正規化済みの学習行列の切断対象が `inf` に書き換わります。[事前計算距離の比較記録](reference/transform_reference.md#precomputed-query-distances)

密な特徴量では `fit(X, y)` と `fit_transform(X, y)` で教師あり埋め込みを作れます。`target_metric="categorical"` では `-1` が未ラベルを表します。連続値の `y` には `target_metric="euclidean"`、`"manhattan"`、`"chebyshev"` を指定できます。[教師あり埋め込みの参照記録](reference/supervised_embedding_reference.md)

密な入力の現代レイアウトは、`n_epochs=0` で `init="random"` または NumPy の初期座標を使えます。`compatibility_layout=False` か `optimizer="adam"` または `"momentum"` で選択し、`init_rescale` と `init_scale_factor` で初期座標を調整できます。[現代レイアウト初期化の比較記録](reference/modern_initialization_reference.md)

疎行列の埋め込み、および Adam と momentum による正のエポック数の学習は未実装です。固定シードの検証条件と適用範囲は [スペクトル初期化を使った学習の参照記録](reference/spectral_fit_reference.md) と [ランダム初期化と初期座標を指定した学習の参照記録](reference/seeded_fit_reference.md) を参照してください。

`UMAP(transform_mode="graph").fit(X)` と `fit_transform(X)` は、Euclidean、Manhattan、Chebyshev、cosine の教師なし学習で SciPy の CSR、CSC、COO 形式の疎な特徴量を受け取ります。入力を変更せずに CSR に変換し、`float32` の CSR グラフを作ります。Cosine は圧縮形式の行に格納された重複要素を保持して距離を計算します。距離ゼロの同順位により自分自身が近傍から外れる場合は `ValueError` になります。疎行列での事前計算済み距離と変換は未実装です。形式ごとの検証結果は [疎グラフの参照記録](reference/sparse_graph_reference.md) を参照してください。

`local_connectivity` は有限の非負値、`set_op_mix_ratio` は0から1までを受け付けます。Python の対応済みグラフ学習と固定シードの密な埋め込みに適用できます。C の `umap_capi_fit_dense_graph_with_options` でも密なグラフに両設定を渡せます。[C ヘッダー](crates/umap-capi/include/umap_capi.h)

`UMAP(transform_mode="graph").transform(X)` は、学習データと同じ特徴数の密な入力を受け取ります。学習データと `float32` 変換後のビット列が一致すれば `graph_` をそのまま返します。新しい入力に対しては、行が入力サンプル、列が学習サンプルに対応する `float32` の SciPy COO 行列を返し、近傍順と明示的なゼロ要素を保持します。

変換は Euclidean、Manhattan、Chebyshev と既存の別名に対応します。変換時の `local_connectivity` から1を引き、下限を0として外部入力のグラフに適用します。距離が等しい近傍は学習行の添字順で並べます。ビット単位の一致は記録済みの Euclidean fixture で検証しています。変換の対応範囲と公開 API の境界ケースは [変換の参照記録](reference/transform_reference.md) を参照してください。

密な `cosine` と `correlation` は、グラフモードの学習と新しい入力の変換で利用できます。既定では距離が2以上の辺を除外します。Windows と Ubuntu でのビット比較と適用範囲は [密な角距離の参照記録](reference/dense_angular_reference.md) を参照してください。

## 独立実装と数値互換性

実装側は公開論文と自前の仕様を参照し、設計判断の根拠を記録します。上流の実装コード・コメント・テストは閲覧や転用の対象から外します。クリーンルームで実装を担当する場合は、上流ソースの断片を含む過去の調査履歴も共有対象から外します。[UMAP 原論文](https://arxiv.org/abs/1802.03426)

検証側は `reference/` の専用環境で `umap-learn` 0.6.0 の公開 Wheel を使います。自作の入力を公開 API に渡し、入力、設定、期待出力、実行環境を JSON に記録して実装側に共有します。`reference/uv.lock` は NumPy、SciPy、Numba などの参照実行に使う依存を固定します。[PyPI の参照 Wheel](https://pypi.org/project/umap-learn/0.6.0/)

検証側で次のコマンドを実行すると、同じ固定シードでの出力がビット単位で再現することを確認し、期待値を `float32` のビット列として記録します。`reference/fixtures/` にある JSON は実装側で使う比較データです。

```sh
uv run --project reference --locked python -I reference/generate_fixture.py --output umap-reference.json
```

## ライセンス

このリポジトリは MIT License で公開します。[ライセンス本文](LICENSE)
