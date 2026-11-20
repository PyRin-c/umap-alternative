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

`UMAP(random_state=42, n_jobs=1, compatibility_layout=True).fit_transform(X)` は、密な特徴量、および Euclidean、Manhattan、Chebyshev の疎な特徴量から `float32` の `embedding_` を作り、学習時の CSR グラフを `graph_` に格納します。既定の `init="spectral"` は枝刈り後のグラフの連結成分を配置します。グラフが連結の場合、出力次元数はサンプル数未満にしてください。`init="random"` や、各行が異なり各列に幅がある NumPy 配列も指定できます。`n_epochs` には非負の整数を指定でき、小規模データでは `None` が 500 エポックになります。学習データと正規化後の格納内容が一致する入力の `transform` は学習済みの座標を返します。

密な新しい入力の `transform` は、学習時の座標を固定したまま `float32` の座標を返します。`transform_seed` は学習の `random_state` とは独立した整数シードで、既定値は `42` です。明示した `n_epochs` の 3 分の 1 を切り捨てた回数で最適化し、`None` の場合は入力が 10,000 行以下なら 100 エポック、それより多ければ 30 エポックを使います。検証結果と適用範囲は [埋め込み変換の参照記録](reference/transform_embedding_reference.md) を参照してください。

密な互換レイアウトでは、`inverse_transform` が元の特徴量と同じ列数の `float32` 配列を返します。Euclidean、Manhattan、Chebyshev、cosine、correlation を使えます。凸包の外にある点は外挿し、警告を出します。[逆変換の比較記録](reference/inverse_transform_reference.md)

密な教師なし互換レイアウトでは、`update(X)` が学習時の距離と有効近傍数を使って行を追加します。Euclidean、Manhattan、Chebyshev、cosine、correlation を使えます。更新時のグラフは距離を切断せずに再構築され、学習済みの点も再配置の対象になります。C の `umap_capi_update_dense_embedding_with_metric` には学習時の距離タグを渡します。[更新の比較記録](reference/update_reference.md)

疎な Euclidean、Manhattan、Chebyshev、cosine 特徴量の教師なし互換レイアウトでも、`update(X)` で行を追加できます。学習時の有効近傍数を保ち、CSR のグラフを再構築して追加前後の座標を更新します。[疎な特徴量の更新の比較記録](reference/update_reference.md#sparse-euclidean-updates)

`output_dens=True` を指定した密な Euclidean 埋め込みでは、`fit_transform` が埋め込み、元空間の対数半径、埋め込み空間の対数半径を返します。近傍を内部で計算し、有限の切断距離を使わない場合に利用できます。[半径の参照記録](reference/densmap_reference.md)

`densmap=True` の直列学習は、密な特徴量と密な事前計算距離に対応します。複数の epoch を指定すると、`embedding_list_` に各時点の座標が入ります。C の密入力は `umap_capi_fit_dense_densmap_seeded` で学習できます。[densMAP の比較結果と精度差](reference/densmap_layout_reference.md)

`metric="precomputed"` の学習と外部変換は、密行列と SciPy の CSR 距離行列を受け取ります。外部変換では列数を学習サンプル数に合わせます。CSR の各行には有効近傍数より多くの格納要素が必要で、明示的なゼロと重複要素も数えます。学習時は CSR の列インデックスをソートし、外部変換時は格納順を維持します。有限の切断距離を指定すると、密な学習行列の切断対象は `inf` に書き換わり、CSR では選択した近傍に切断を適用します。[事前計算距離の比較記録](reference/transform_reference.md#precomputed-query-distances)

密な特徴量では `fit(X, y)` と `fit_transform(X, y)` で教師あり埋め込みを作れます。`target_metric="categorical"` では `-1` が未ラベルを表します。連続値の `y` には `target_metric="euclidean"`、`"manhattan"`、`"chebyshev"` を指定できます。[教師あり埋め込みの参照記録](reference/supervised_embedding_reference.md)

密な入力の現代レイアウトは、乱数シードを指定し、連結グラフでの `init="spectral"`、`init="random"`、または NumPy の初期座標で学習できます。`compatibility_layout=False` か `optimizer="adam"` または `"momentum"` で選択し、`init_rescale` と `init_scale_factor` で初期座標を調整できます。エポック番号のリストを指定すると、対応する座標を `embedding_list_` に格納します。[現代レイアウトの比較記録](reference/modern_fit_reference.md) [スペクトル初期化の比較記録](reference/modern_spectral_fit_reference.md)

`init="recursive"` は、固定シードでグラフを縮約し、粗い段階を Adam で最適化してから元の解像度へ戻します。4段の縮約や事前計算距離による星型グラフの一部では、公開版と座標差が残ります。[再帰学習の比較記録](reference/modern_recursive_coarse_reference.md)

Euclidean、Manhattan、Chebyshev、cosine の互換レイアウトでは、CSR、CSC、COO の特徴量を学習できます。有限の切断距離で孤立した行の座標は `NaN` になります。これらの距離で疎な特徴量を学習したモデルは、密な入力と CSR、CSC、COO の入力を変換できます。グラフモードは近傍順の COO グラフを、埋め込みモードは座標を返します。[疎な Manhattan と Chebyshev の比較記録](reference/sparse_graph_reference.md#sparse-manhattan-and-chebyshev)・[疎な cosine の比較記録](reference/sparse_graph_reference.md#sparse-cosine)

`UMAP(transform_mode="graph").fit(X)` と `fit_transform(X)` は、Euclidean、Manhattan、Chebyshev、cosine の教師なし学習で SciPy の CSR、CSC、COO 形式の疎な特徴量を受け取ります。入力を変更せずに CSR に変換し、`float32` の CSR グラフを作ります。Cosine は圧縮形式の行に格納された重複要素を保持して距離を計算し、既定で距離が2以上の辺を除外します。切断距離を指定しない Euclidean のグラフ学習では、自分自身が距離ゼロの同順位で近傍から外れると `ValueError` になります。形式ごとの検証結果は [疎グラフの参照記録](reference/sparse_graph_reference.md) を参照してください。

`local_connectivity` は有限の非負値、`set_op_mix_ratio` は0から1までを受け付けます。Python の対応済みグラフ学習と固定シードの密な埋め込みに適用できます。C の `umap_capi_fit_dense_graph_with_options` でも密なグラフに両設定を渡せます。[C ヘッダー](crates/umap-capi/include/umap_capi.h)

`UMAP(transform_mode="graph").transform(X)` は、学習データと同じ特徴数の密な入力を受け取ります。学習データと `float32` 変換後のビット列が一致すれば `graph_` をそのまま返します。新しい入力に対しては、行が入力サンプル、列が学習サンプルに対応する `float32` の SciPy COO 行列を返し、近傍順と明示的なゼロ要素を保持します。

変換は Euclidean、Manhattan、Chebyshev と既存の別名に対応します。変換時の `local_connectivity` から1を引き、下限を0として外部入力のグラフに適用します。距離が等しい近傍は学習行の添字順で並べます。ビット単位の一致は記録済みの Euclidean fixture で検証しています。変換の対応範囲と公開 API の境界ケースは [変換の参照記録](reference/transform_reference.md) を参照してください。

密な `cosine` と `correlation` は、グラフモードと互換レイアウトの学習・変換で利用できます。既定では距離が2以上の辺を除外します。Python の変換では、結果に影響する同距離の順位を NumPy で決めます。C の密な学習・変換では `UMAP_CAPI_METRIC_COSINE` と `UMAP_CAPI_METRIC_CORRELATION` を指定できます。[密な角距離の比較記録](reference/dense_angular_reference.md)

学習済みの二つの `UMAP` は、同じ頂点数なら `+`、`*`、`-` でグラフを合成できます。結果は新しい `graph_` と `embedding_` を持ちます。減算では左の学習済みグラフにも重みの変更が入ります。[合成演算の参照記録](reference/composition_reference.md)

`unique=True` は重複をまとめてグラフを学習し、埋め込みと密度半径を元の入力行に戻します。疎行の代表行、逆対応、件数は `umap_alternative.utils.csr_unique` で取得できます。[重複行学習の参照記録](reference/unique_reference.md)

特徴量の距離には `canberra` と `braycurtis` も指定できます。密・CSR の互換レイアウト学習・変換・更新と、密な逆変換を公開版と比較しています。[距離の比較記録](reference/ratio_metrics_reference.md)

`hamming` は特徴量の値を比較します。`matching`、`jaccard`、`dice`、`russellrao`、`rogerstanimoto`、`sokalmichener`、`sokalsneath`、`yule`、`kulsinski` は各特徴量の値が非ゼロかどうかを比較します。これらの距離は密な特徴量のグラフ学習、互換レイアウトの学習、外部変換、更新で使えます。CSR の学習では `yule` が `ValueError` になります。逆変換を呼ぶと `ValueError` になります。[ブール距離の比較記録](reference/boolean_metric_reference.md)・[更新の比較記録](reference/boolean_update_reference.md)

パック済みのビット列には `bit_hamming` と `bit_jaccard` を指定できます。密な入力を `uint8` に変換し、教師なし学習と外部変換でビット単位の距離を計算します。[パック済みビット距離の比較記録](reference/packed_metric_reference.md)

`umap_alternative.umap_` で近傍距離の平滑化、メンバーシップ、近傍配列からのグラフ構築を利用できます。孤立した入力行は `umap_alternative.utils.disconnected_vertices` で確認できます。[グラフ補助関数の参照記録](reference/graph_primitives_reference.md)

同じ頂点を持つ疎グラフの有向和・重み付き積・局所接続度の再正規化は、`umap_alternative.umap_` の補助関数で実行できます。`umap_alternative.sparse` の演算は COO の値バッファへ直接書き込みます。[グラフ演算の比較記録](reference/graph_operators_reference.md)

既存の埋め込みから新しい行の初期座標を求めるには、順位と重みを受け取る `init_transform` と、疎グラフを受け取る `init_graph_transform` を利用できます。[初期化の比較記録](reference/public_transform_init_reference.md)

`umap_alternative.validation.trustworthiness_vector` で、元データの近傍が埋め込みに保たれているかを、近傍数ごとのスコアで確認できます。近傍を計算済みの場合は `trustworthiness_vector_bulk` や `make_trustworthiness_calculator` を利用できます。[信頼度スコアの比較記録](reference/trustworthiness_reference.md)

## 独立実装と数値互換性

実装側は公開論文と自前の仕様を参照し、設計判断の根拠を記録します。上流の実装コード・コメント・テストは閲覧や転用の対象から外します。クリーンルームで実装を担当する場合は、上流ソースの断片を含む過去の調査履歴も共有対象から外します。[UMAP 原論文](https://arxiv.org/abs/1802.03426)

検証側は `reference/` の専用環境で `umap-learn` 0.6.0 の公開 Wheel を使います。自作の入力を公開 API に渡し、入力、設定、期待出力、実行環境を JSON に記録して実装側に共有します。`reference/uv.lock` は NumPy、SciPy、Numba などの参照実行に使う依存を固定します。[PyPI の参照 Wheel](https://pypi.org/project/umap-learn/0.6.0/)

検証側で次のコマンドを実行すると、同じ固定シードでの出力がビット単位で再現することを確認し、期待値を `float32` のビット列として記録します。`reference/fixtures/` にある JSON は実装側で使う比較データです。

```sh
uv run --project reference --locked python -I reference/generate_fixture.py --output umap-reference.json
```

## ライセンス

このリポジトリは MIT License で公開します。[ライセンス本文](LICENSE)
