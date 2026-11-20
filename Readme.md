# umap-alternative

UMAP を Rust で独立実装するリポジトリです。`umap-learn` との数値の完全一致を検証したうえで高速化し、Rust ライブラリ、C ABI、PyO3 + Maturin による Python Wheel として提供します。

`crates/umap-core` が Rust 本体の置き場で、`crates/umap-python` が Python 拡張です。Python Wheel は `UMAP(transform_mode="graph")` のグラフ構築と変換、および固定シードを使った密な入力の埋め込みを提供します。`crates/umap-capi` は密入力のグラフと固定シードの埋め込みを返す C ABI とヘッダーを提供します。数値のビット単位の一致は、記録済みの fixture で検証しています。

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

```sh
uv run --locked maturin build --release --locked --interpreter python
```

Wheel は `target/wheels/` に生成されます。Python パッケージ名は `umap-alternative`、インポート名は `umap_alternative` です。Windows と Linux の CI では Wheel を隔離環境にインストールし、インポートまで確認します。

`UMAP(init="random", random_state=42, n_jobs=1, compatibility_layout=True).fit_transform(X)` は、密な特徴量から `float32` の `embedding_` を作り、学習時の CSR graph を `graph_` に格納します。初期座標として、各行が異なり各列に幅がある NumPy 配列も渡せます。`n_epochs` には非負の整数を指定でき、小規模データでは `None` が 500 エポックになります。学習データとビット列が一致する入力の `transform` は `embedding_` を返します。

埋め込みではスペクトル初期化、教師あり入力、疎行列、事前計算済み距離、Adam と momentum、未知の入力の変換、逆変換は未実装です。固定シードの検証条件は [埋め込みの参照記録](reference/seeded_fit_reference.md) を参照してください。

`UMAP(transform_mode="graph").transform(X)` は、学習データと同じ特徴数の密な入力を受け取ります。学習データと `float32` 変換後のビット列が一致すれば `graph_` をそのまま返します。新しい入力に対しては、行が入力サンプル、列が学習サンプルに対応する `float32` の SciPy COO 行列を返し、近傍順と明示的なゼロ要素を保持します。

変換は Euclidean、Manhattan、Chebyshev と既存の別名に対応し、`local_connectivity=1.0` を使用します。距離が等しい近傍は学習行の添字順で並べます。ビット単位の一致は記録済みの Euclidean fixture で検証しています。変換の対応範囲と公開 API の境界ケースは [変換の参照記録](reference/transform_reference.md) を参照してください。

## 独立実装と数値互換性

実装側は公開論文と自前の仕様を参照し、設計判断の根拠を記録します。上流の実装コード・コメント・テストは閲覧や転用の対象から外します。クリーンルームで実装を担当する場合は、上流ソースの断片を含む過去の調査履歴も共有対象から外します。[UMAP 原論文](https://arxiv.org/abs/1802.03426)

検証側は `reference/` の専用環境で `umap-learn` 0.6.0 の公開 Wheel を使います。自作の入力を公開 API に渡し、入力、設定、期待出力、実行環境を JSON に記録して実装側に共有します。`reference/uv.lock` は NumPy、SciPy、Numba などの参照実行に使う依存を固定します。[PyPI の参照 Wheel](https://pypi.org/project/umap-learn/0.6.0/)

検証側で次のコマンドを実行すると、同じ固定シードでの出力がビット単位で再現することを確認し、期待値を `float32` のビット列として記録します。`reference/fixtures/` にある JSON は実装側で使う比較データです。

```sh
uv run --project reference --locked python -I reference/generate_fixture.py --output umap-reference.json
```

## ライセンス

このリポジトリは MIT License で公開します。[ライセンス本文](LICENSE)
