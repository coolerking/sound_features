# Sound Features

音声ファイルを読み込み、librosaパッケージに備わっている機能を使って各種音声特徴量に変換しグラフ表示する。

- 音声ファイルの形式は wav のみ

# セットアップ

## ローカルPC

Python3 がインストールされている環境にて、以下のコマンドを実行する。

```bash
pip install -r requirements.txt
streamlit run view.py
```

## Google App Engine

Google CloudShellを起動し、以下のコマンドを実行する。

```bash
gcloud projects create your-project-name --set-as-default

gcloud app create --project=your-project-name
# リージョンを選択（東京なら asia-northeast1

git clone https://github.com/coolerking/sound_features.git
cd sound_features

gcloud app deploy
```

プロジェクトを削除する場合：

```bash
gcloud projects delete your-project-name
```

# 使い方

ブラウザから参照する。

- ローカルPCの場合は、http://localhost:8501/ を開く(`streamlit config show` でデフォルト設定確認可能)
- Google App Engine の場合CloudShell上で `gcloud app browse` を実行
