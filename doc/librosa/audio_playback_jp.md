# Audio Playback

> **注意**
> 全サンプルコードは [こちら](https://librosa.org/doc/latest/auto_examples/plot_audio_playback.html#sphx-glr-download-auto-examples-plot-audio-playback-py) からダウンロードできます。

```python
# Code source: Brian McFee
# License: ISC
```

> ISC ライセンス：
> 
> Internet Systems Consortium(ISC)によって作成されたパーミッシブライセンスの一つ。
> 「ベルヌ条約によって不要となる言い回し」を取り除いた２条項BSDライセンスと機能上同等とされている。

このサンプルでは、 `numpy` と `matplotlib` が必要です：

```python
import numpy as np
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_path = '_static/playback-thumbnail.png'

import librosa
import librosa.display

# この例ではIPython.displayのAudioウィジェットが必要
from IPython.display import Audio

# またこの例では `mir_eval` を使って
# 信号の合成を行っている
import mir_eval.sonify
```

## 合成音をならす

IPython Audio ウィジェットは、生の `numpy` データをオーディオ信号として受け入れます。
これは、信号を直接合成してブラウザで再生できることを意味します。

たとえば、C3 から C5 へのサイン スイープを作成できます。

```python
sr = 22050

y_sweep = librosa.chirp(fmin=librosa.note_to_hz('C3'),
                        fmax=librosa.note_to_hz('C5'),
                        sr=sr,
                        duration=1)

Audio(data=y_sweep, rate=sr)
```

## もとの音声をならす

もちろん、実際に録音した音声も同様に再生できます。

```python
y, sr = librosa.load(librosa.ex('trumpet'))

Audio(data=y, rate=sr)
```

## ピッチ推定値のソニフィケーション

もう少し高度な例として、基本周波数推定器の出力を直接観察するために、音波化を使うことができます。

この例では、解析に `librosa.pyin` を、合成に `mir_eval.sonify.pitch_contour` を使用します。

```python
# fill_na=None を使用すると、
# 非ボイスフレームでのベストゲストのf0が保持される
f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                             sr=sr,
                                             fmin=librosa.note_to_hz('C2'),
                                             fmax=librosa.note_to_hz('C7'),
                                             fill_na=None)

# f0を合成するためには、サンプル時間が必要になる
times = librosa.times_like(f0)
```

mir_eval のシンセサイザは、負の `f0` 値を使用して無声領域を示します。

有声フレームの場合は `1` 、無声フレームの場合は `-1` の配列 `vneg` を作成します。
このように、`f0 * vneg` は有声推定を変更せずに残し、無声フレームの周波数を無効にします。

```python
vneg = (-1)**(~voiced_flag)

# そして mir_eval を使って f0 をソニファイ
y_f0 = mir_eval.sonify.pitch_contour(times, f0 * vneg, sr)

Audio(data=y_f0, rate=sr)
```

## 混合データの音波化

最後に、音声ウィジェットを使って、混合した信号を聴くこともできます。

この例では、オリジナルのテストクリップに対してオンセット検出器を実行し、検出ごとにクリック音を合成します。

クリックのトラックを元の信号の上に重ねて、両方を聞くことができます。

この機能を実現するには、合成されたクリックトラックと元の信号が同じ長さであることを確認する必要があります。

```python
# 5つの周波数ビンの最大フィルタを使用 偽陽性を減らすために
# オンセット強度エンベロープを計算
onset_env = librosa.onset.onset_strength(y=y, sr=sr, max_size=5)

# 強度エンベロープからオンセットタイムを検出する
onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')

# オンセットタイムをクリック数でソニファイ
y_clicks = librosa.clicks(times=onset_times, length=len(y), sr=sr)

Audio(data=y+y_clicks, rate=sr)
```

## 注意事項

最後に、インタラクティブ再生を使用する際に注意すべき点をいくつか挙げます。

- `IPython.display.Audio` は、音声信号全体をシリアライズし、`UUEncoded` ストリームとしてブラウザに送信することで動作します。これは長い信号では非効率的かもしれません。
- `IPython.display.Audio` は、ファイル名やURLを直接扱うこともできます。長いシグナルを扱う場合や、シグナルを直接pythonにロードしたくない場合は、これらのモードのいずれかを使用する方がよいかもしれません。
- 音声の再生は、デフォルトでは再生される信号の振幅を正規化します。ほとんどの場合、これはあなたが望むことですが、時にはそうでないかもしれませんので、正規化を無効にすることができることを認識しておいてください。
- Jupyter notebook で作業していて、同じセルに複数の Audio ウィジェットを表示したい場合、`IPython.display.display(IPython.display.Audio(...))` を使って明示的に各ウィジェットをレンダリングすることが可能です。これは関連する複数の信号を再生するときに便利です。

## ダウンロード

- [plot_audio_playback.py](./src/plot_audio_playback.py)
- [plot_audio_playback.ipynb](./src/plot_audio_playback.ipynb)
