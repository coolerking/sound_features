# インストール手順

## pip

librosa を最も簡単にインストールしたいのであれば、Python Package Index (PyPI) を利用してください。これにより、必要なすべての依存関係が確実に満たされます。これは、次のコマンドを実行することで実現できます：

```bash
pip install librosa
```

システム全体にインストールする場合：

``bash
sudo pip install librosa
```


自ユーザのみで使用する場合：

```bash
pip install -u librosa
```

## conda

conda/Anaconda 環境を使用する場合、librosaを`conda-forge`チャネルからインストールできます：

```bash
conda install -c conda-forge librosa
```

## ソースコード

リリースページから手動でアーカイブをダウンロードした場合、`setuptools` スクリプトを使ってインストールできます：

```bash
tar xzf librosa-VERSION.tar.gz
cd librosa-VERSION/
python setup.py install
```

または、最新の開発バージョンを pip 経由でインストールできます：

```bash
pip install git+https://github.com/librosa/librosa
```

## ffmpeg

より多くの音声でコーディング機能を使って `audioread` を活用するには、様々な音声デコーダに同梱されている ffmpeg をインストールします。Linux や OSX の conda ユーザは、デフォルトでインストール済みであることに注意してください。Windows ユーザは、ffmpegを個別にインストールする必要があります。

OSXユーザは、homebrew を使い `brew install ffmgeg` にてインストールするか [https://www.ffmpeg.org](https://www.ffmpeg.org) からバイナリバージョンを入手します。

