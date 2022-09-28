# チュートリアル

本セクションでは、パッケージの概要、基本的な使用方法と高度な使用方法、scikit-learn パッケージとの統合など、 librosa を使った開発の基本について説明します。
Python と NumPy/SciPy の基本的な知識があることを前提としています。

## 概要

librosa パッケージは、以下のような構成となっています。

- librosa
  - [librosa.beat](https://librosa.org/doc/latest/beat.html#beat)
    テンポの推定とビート イベントの検出のための関数。
  - [librosa.core](https://librosa.org/doc/latest/core.html#core)
    コア機能には、ディスクから音声をロードする関数、さまざまなスペクトログラム表現を計算する関数、音楽分析に一般的に使用されるさまざまなツールが含まれます。便宜上、このサブモジュールのすべての機能は、最上位の librosa.* 名前空間から直接アクセスできます。
  - [librosa.decompose](https://librosa.org/doc/latest/decompose.html#decompose)
    scikit-learn に実装されている行列分解法を使用した、ハーモニック-パーカッシブ音源分離 (HPSS[^1]) や汎用スペクトログラム[^2]分解の関数 。
  - [librosa.display](https://librosa.org/doc/latest/display.html#display)
    matplotlib を使った可視化・表示ルーチン。
  - [librosa.effects](https://librosa.org/doc/latest/display.html#display)
    ピッチシフトやタイムストレッチなどのタイムドメインオーディオ処理。このサブモジュールでは、decompose サブモジュールの時間領域ラッパも提供。
  - [librosa.feature](https://librosa.org/doc/latest/feature.html#feature)
    特徴の抽出と操作。クロマグラム、メルスペクトログラム、MFCC、その他のさまざまなスペクトルおよびリズム機能などの低レベルの特徴抽出が含まれます。また、デルタ機能やメモリ埋め込みなどの特徴操作方法も提供されます。
  - [librosa.filters](https://librosa.org/doc/latest/filters.html#filters)
    フィルタバンクの生成 (彩度、疑似 CQT、CQT など)。主にlibrosa の他の部分で使用される内部関数です。
  - [librosa.onset](https://librosa.org/doc/latest/onset.html#onset)
    オンセット(発音)検出およびオンセット強度計算。
  - [librosa.segment](https://librosa.org/doc/latest/segment.html#segment)
    再帰行列の構築、タイムラグ表現、逐次制約クラスタリングなど、構造的セグメンテーションに役立つ機能。
  - [librosa.sequence](https://librosa.org/doc/latest/sequence.html#sequence)
    シーケンシャル モデリングの関数。さまざまな形式のビタビ [^3] デコーディング、および遷移行列を構築するためのヘルパ関数。
  - [librosa.util](https://librosa.org/doc/latest/util.html#util)
    ヘルパー ユーティリティ (正規化、パディング、センタリングなど)。

[^1]:HPSS とは，スペクトログラムにおいて、調波音成分は時間軸方向に、非調波音成分は周波数軸方向に連続性が強
いということに着目し、それらを分離する手法（引用元：[Harmonic/Percussive Sound Separation を前処理とした和音認識の性能調査](https://ipsj.ixsq.nii.ac.jp/ej/?action=repository_action_common_download&item_id=146329&item_no=1&attribute_id=1&file_no=1)）
[^2]: スペクトログラムとは、周波数がX軸、時間がY軸で表される周波数対時間対振幅を表示したグラフ。パワー・レベルは色別で表示する。
[^3]: Viterbi デコーダ：Viterbi アルゴリズムを使用して、畳み込みコードまたはトレリスコードを使用してエンコードされます。畳み込みエンコードされたストリームをデコードするための他のアルゴリズムがあります。（引用元：[Viterbi decoder](https://en.wikipedia.org/wiki/Viterbi_decoder)）

## クイックスタート

詳細説明の前に、簡単なサンプルプログラムを解説します。

```python
# ビートトラッキングの例
#   ビートトラッキングとは、 人間が音楽に合わせて手拍子を打つように、 
#   曲のビート(4分音符)の位置を認識する技術である
import librosa

# 1. 音声サンプルを含むファイルパスの取得
filename = librosa.example('nutcracker')

# 2. 音声を波形 `y` としてロードする
#    サンプリング周波数を `sr` に格納する
y, sr = librosa.load(filename)

# 3. デフォルトビートトラッカを実行する
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# 4. ビートイベントのフレームインデックスをタイムスタンプに変換する
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
```

プログラムの最初のステップ：

```python
filename = librosa.example('nutcracker')
```

librosa に含まれている音声サンプルファイルへのパスを取得します。
このステップ実行後、 `filename` はサンプル音声ファイルへのパスを含む文字列が格納された変数となります。

２番めのステップ：

```python
y, sr = librosa.load(filename)
```

1次元の NumPy 浮動小数点配列として表される [時系列](https://librosa.org/doc/latest/glossary.html#term-time-series) `y` として音声をロードしデコードします。
変数 `sr` には `y` の [サンプリング周波数](https://librosa.org/doc/latest/glossary.html#term-sampling-rate)、 つまり音声の 1 秒あたりのサンプル数が含まれます。
デフォルトでは、すべての音声がモノラルにミックスされ、ロード時に 22050 Hz にリサンプリングされます。
この動作は、 `librosa.load` に追加の引数を指定することでオーバーライドできます。

次に、ビートトラッカを実行します：

```python
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
```

ビートトラッカの戻り値は、テンポ（１分あたりの拍数）の推定値と、検出されたビートイベントに対応するフレーム番号の配列です。


[フレーム](https://librosa.org/doc/latest/glossary.html#term-frame) は信号（ `y` ）の短いウィンドウに対応し、それぞれは `hop_length = 512` サンプルで区切られます。
librosa は、 k 番目のフレームが `k * hop_length` を中心とするように、中心化されたフレームを使用します。

次の操作は、フレーム番号 `beat_frames` をタイミングに変換します：

```python
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
```

上の式を実行することで、 `beat_times` は検出されたビートイベントに対応するタイムスタンプ(単位：秒)の配列になります。

`beat_times` の中は次のようになります：

```shell
7.43
8.29
9.218
10.124
...
```

## 高度な使い方

ここではより高度な例として、ハーモニック・パーカッシブ分離、複数のスペクトル特徴、およびビート同期特徴集約を統合したものを取り上げます。

```python
# 特徴抽出例
import numpy as np
import librosa

# サンプルクリップのロード
y, sr = librosa.load(librosa.ex('nutcracker'))

# ホップ長（切り出されるウィンドウに入るフレーム数）の設定
# サンプリングレート 22050 Hz、サンプル数512 ~= 23ミリ秒
hop_length = 512

# ハーモニクスとパーカッシブを2つの波形に分離
#   ハーモニクス：
#     ある周波数成分をもつ波動に対して、
#     その整数倍の高次の周波数成分のこと。倍音ともよばれる。
#   パーカッシブ：
#      パーカッション（打楽器）もしくは減衰音を多用した楽曲、
#      もしくはそのような楽音をさす。
#      ある程度強く、衝撃的で、しかも打撃音のような
#      歯切れのよい音をパーカッシブなサウンドという。
y_harmonic, y_percussive = librosa.effects.hpss(y)

# パーカッシブ信号のビートトラック
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                             sr=sr)

# 生信号からMFCC特徴を計算
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

# そして一次差分（デルタ特徴量）の計算
mfcc_delta = librosa.feature.delta(mfcc)

# ビートイベント間のスタックと同期
# 今回は、中央値ではなく、平均値（デフォルト）を使用
beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                    beat_frames)

# ハーモニック信号からクロマ特徴を計算
chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                        sr=sr)

# ビートイベント間のクロマ特徴の集計
# ここではビートフレーム間の各特徴の中央値を使用する
beat_chroma = librosa.util.sync(chromagram,
                                beat_frames,
                                aggregate=np.median)

# 最後にすべてのビートシンクロナス特徴を
# 配列の縦方向に結合
beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
```

> [クロマベクトル](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.docomo.ne.jp/binary/pdf/corporate/technology/rd/technical_journal/bn/vol25_2/vol25_2_004jp.pdf)：
> 演奏音の音程認識において、単音であれば f0抽出の技術（自己相関法など）を用いて、簡単に認識で見るものもあるが、ギターやピアノなどの複数の音程が混ざった演奏音となると、f0抽出では、単音んお音程しか認識できない場合や誤った音程を認識する場合が多いため困難である。そこで複数の音程の音や和音（コード）を解析するためにはクロマベクトルという特徴量を用いることが多い。
> クロマベクトルとは、12音階及びオクターブ違いも含めた音階ごとの周波数の振幅強度を特徴量としたベクトルである。ここでの振幅強度とは、ある区間に、特定の周波数の信号がどれくらい多く含まれているかを示すものである。たとえばA（ラ）の音は440Hz、E（ミ）の音は660Hzとなる。周波数が２倍になると1オクターブ上の音程となり、1/2になると1オクターブ下の音程となる。このように周波数の比で音程が決まるため、12音階の音程は、周波数が $ \sqrt[12]{2} $ 倍ごとに半音上がることになる。この各音階の周波数ごとに振幅強度を計算したものがクロマベクトルであり、これによりどの音階のどの音が強いかがわかるため、強い音階の組み合わせからコードを推定することなどが可能になる。

この例は、クイックスタートのサンプルですでに説明したツールをベースにしているので、ここでは新規登場部分だけに焦点を当てます。

最初の違いは、時系列のハーモニック・パーカッシブ分離のためのエフェクト・モジュールの使用です：

```python
y_harmonic, y_percussive = librosa.effects.hpss(y)
```

この行を実行すると、時系列 `y` が信号のハーモニック（音色）部分とパーカッシブ（過渡）部分の2つの時系列に分離されます。
`y_harmonic`と `y_percussive` はそれぞれ、`y` と同じ形式と継続時間を持ちます。

パーカッシブな要素はリズムの内容をより強く示す傾向があり、より安定したビートトラッキングの結果を提供するのに役立ちます。

次に、[特徴モジュール](https://librosa.org/doc/latest/feature.html#feature) の紹介と、生信号 `y` からメル周波数ケプストル係数の抽出を行います：

```python
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
```

この関数の戻り値は行列 `mfcc` です。これは、 `(n_mfcc, T)` 形式の `numpy.ndarray` です（ここで `T` はフレーム単位でトラックの継続時間を表します）。
ここではビートトラッカと同じ `hop_length` を使用しているため、検出された `beat_frames` の値は `mfcc` の列に対応することに注意してください。

特徴操作の最初のタイプとして `delta` の紹介です。
これは、入力の列間の一次差を計算（平滑化）しています：

```python
mfcc_delta = librosa.feature.delta(mfcc)
```

実行して得られる行列 `mfcc_delta` は、入力 `mfcc` と同じ形状になります。

特徴操作の 2 番目のタイプは `sync` です。これは、入力の列をサンプルインデックス（例えば、ビートフレーム）間で集約するものです：

```python
beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                    beat_frames)
```

ここでは、行列 `mfcc` と行列 `mfcc_delta` を垂直に結合しています。実行した結果として得られるのが、入力と同じ数の行を持つ行列 `beat_mfcc_delta` ですが、列の数は `beat_frames` に依存します。
各列 `beat_mfcc_delta[:, k]` は `beat_frames[k]` と `beat_frames[k+1]` の間の入力列の平均となります( `beat_frames` は `beat_frames[k]` と `beat_frames[k]` の間の入力列の平均となります( `beat_frames` は全データを考慮するため、全範囲である `[0, T]` に拡張されます)。

次に、ハーモニック成分のみを用いたクロマグラムを計算します：

```python
chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                        sr=sr)
```

この行を実行すると、`chromagram` は `(12, T)` 形式の `numpy.ndarray` となり、その各行はピッチクラス（例：C、C#など）に対応しています。
`chromagram` の各列はそのピーク値で正規化されますが、この動作は `norm` パラメータを設定することで上書きすることができます。

クロマグラムとビートフレームのリストが得られたら、再びビートイベント間のクロマを同期させます：

```python
beat_chroma = librosa.util.sync(chromagram,
                                beat_frames,
                                aggregate=np.median)
```

今回は、デフォルトの集計演算（平均値、MFCCで使用していたもの）を中央値に置き換えています。
一般的には、`np.max()` 、 `np.min()` 、 `np.std()` などの任意の統計的要約関数を使用する。

最後に、すべての特徴を再び縦に結合します：

```python
beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
```

結果、`(12 + 13 + 13, # 拍間隔)` の特徴行列 `beat_features` が得られます。

## その他の例

これら以上のサンプルは、[高度な例](https://librosa.org/doc/latest/advanced.html#advanced) セクションにて紹介しています。