# Windows x64 C ABI

GitHub Releases の `umap-alternative-cabi-<version>-win-x64.zip` に、DLL、MSVC 用インポートライブラリ、C ヘッダー、サンプルとライセンスを同梱しています。[GitHub Releases](https://github.com/PyRin-c/umap-alternative/releases)

1. ZIP を展開し、x64 Native Tools Command Prompt for Visual Studio でそのディレクトリを開きます。
2. 同梱のサンプルをビルドして実行します。

```bat
cl /nologo /std:c11 /W4 fit_graph.c umap_capi.lib
fit_graph.exe
```

アプリから使うときは、`umap_capi.h` をインクルードし、`umap_capi.lib` をリンクします。実行時は `umap_capi.dll` をアプリの実行ファイルと同じディレクトリに配置します。DLL の利用に Python は不要です。

DLL は Microsoft Visual C++ ランタイムの `VCRUNTIME140.dll` を使います。実行環境にランタイムがない場合は、Microsoft の x64 再頒布可能パッケージをインストールしてください。[Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist/)

密行列の入力は行優先の `float` 配列です。結果は所有ハンドルで返ります。ビューの配列は、そのハンドルを対応する `*_release` 関数で解放するまで読み取れます。各関数の引数・制約は同梱の `umap_capi.h` を参照してください。

逆変換で距離が同じ候補が並ぶ場合、C ABI は探索で発見した順序を使います。Python 版は NumPy の順位を使うため、同じ入力でも選ばれる候補が変わる場合があります。
