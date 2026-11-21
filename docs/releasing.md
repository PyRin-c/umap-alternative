# GitHub Releases の準備

初回は `v0.1.0` のプレビューを GitHub Releases で配布します。Python wheel は Windows x64 と Linux x86-64 の CPython 3.10〜3.15 向けです。C ABI は Windows x64 の DLL を ZIP にまとめます。

1. `Cargo.toml` のワークスペース版と、`docs/releases/v<version>.md` のリリースノートを揃えます。
2. 配布するコミットを GitHub に反映し、そのコミットの CI が通ったことを確認します。
3. コミットに `v<version>` のタグを付けて GitHub に送ると、[Prepare GitHub Release](../.github/workflows/release.yml) が起動します。
4. 全ビルドとテストが完了すると、wheel 12本、Windows C ABI ZIP、`SHA256SUMS` を添付したドラフトが作成されます。Releases でファイルと説明を確認し、プレビューとして公開します。

ZIP は展開した状態で同梱サンプルと C ABI のスモークテスト2本をリンクして実行します。wheel は対応する各 Python でインストールし、学習・変換・更新・保存復元を確認します。

公開用ビルドでは Rust のビルドパスを置き換え、MSVC の PDB 参照をファイル名にします。Maturin の Rust SBOM 生成は、ローカルワークスペースのファイル URL を含むため無効にしています。添付前に ZIP と wheel の中身を照合し、ビルド端末のパスが検出された場合はドラフトの作成を中止します。

ビルドが失敗した場合は Actions で該当ジョブを確認します。CI の設定を修正した場合は `main` の **Prepare GitHub Release** を手動実行し、既存タグを指定すると、タグのソースをそのまま再ビルドできます。添付の途中でドラフトが残った場合は、同じ実行の成果物から不足分を補うか、未公開のドラフトを整理してから再実行します。同じタグのリリースが存在するとエラーになります。

固定フィクスチャとの比較を行う CI では、採取環境と同じ OpenBLAS の Haswell カーネルを単一スレッドで使います。配布用 wheel の動作確認では、各ランナーが選ぶ通常のカーネルを使います。

Windows で ZIP の作成と検証を再現するには、リポジトリのルートから次を実行します。

```powershell
python scripts/release_build.py cargo build --release --locked -p umap-capi
./scripts/package-windows-cabi.ps1
```

ZIP は `target/cabi-release/` に生成されます。
