# -*- coding: utf-8 -*-
"""
音声ファイル特徴量可視化Web UI

1. 以下のコマンドで実行
```bash
$ streamlit run view.py
```
2. ブラウザで http://localhost:8501 を開く
3. Browse files を押下し、wavファイルを選択
"""
import io
import sklearn
import librosa
import librosa.display
import librosa.feature
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def _show_feature(uploaded_file:any, num_bins:int=1024, mel_dims:int=24)-> None:
    """
    対象のwavファイルを読み込み、メタ情報や各種特徴量グラフを
    表示する。

    Parameters
    -----
    uploaded_file   アップロードファイルオブジェクト(streamlit)
    num_bins        解像度 分割数
    mel_dims        MELフィルタバンク処理後の次元数

    Returns
    -----
    None
    """
    # サンプル数 分割数の倍
    sample_count = num_bins * 2
    # ファイルパス
    path = uploaded_file.name


    # ヘッダ（ファイル名）
    st.header(f'{uploaded_file.name}')

    # 音声再生ガジェット
    bytes_data = uploaded_file.getvalue()
    st.audio(bytes_data)

    # 音声データ読み込み
    data, sr =sf.read(io.BytesIO(bytes_data))
    y = librosa.core.resample(data, orig_sr=sr, target_sr=sr)

    # フレームサイズ(20mSec)
    frame_size = int(0.02 * sr)
    # フレームシフトサイズ
    frame_shift = int(0.01 * sr)
    # テンポ
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # X軸をまたいだか
    zero_crossings = librosa.zero_crossings(y=y, pad=False)

    # メタ情報表示
    st.write('## Raw Signal')
    st.write(f'- Filename: {path}')
    st.write(f'- Size: {str(len(bytes_data))} bytes')
    st.write(f'- Sampling rate: {str(sr)} Hz')
    st.write('- Audio length: %.2f sec' % (len(y) / sr))
    st.write('- Tempo %.5f bpm' % tempo)
    st.write(f'- Zero Crossing Rate(sum): {str(sum(zero_crossings))}')
    st.write(f'- {sample_count} samples/frame, {num_bins} bins')
    st.write(f'- Frame Size: {frame_size}')
    st.write(f'- Frame Shift: {frame_shift}')
    st.write(f'- Mel Dimensions: {mel_dims}')

    # ヒストグラム
    # Raw Signal ヒストグラム
    # 正規化済み音声信号データの確率分布を可視化
    fig0 = plt.figure(figsize=(6,6))
    plt.title(f'Raw Signal Histgram / {path} (bins:{num_bins})')
    _, _, _ = plt.hist(y, bins = num_bins)
    st.write('### Histgram')
    st.write(f'- X: num_bins:{num_bins}')
    st.write('- Y: raw signal data count')
    st.pyplot(fig0)

    # 振幅グラフ Raw Signal Graph
    #  X座標: 時間
    #  Y座標: 音の大きさ(正規化済み)
    fig1 = plt.figure(figsize=(16,6))
    plt.title(f'Raw Signal / {path} (shape:{y.shape}/{y.dtype}, mean:{np.mean(y)}, var:{np.var(y)})')
    librosa.display.waveshow(y=y, sr=sr)
    st.write('### Raw Signal Graph')
    st.write('- X: time (Sec)')
    st.write('- Y: raw signal')
    st.pyplot(fig1)

    # 短時間フーリエ変換により、周波数ごとの振幅・位相に変換
    # 離散信号（複素フーリエ展開結果なので複素数）
    # 1次元の長さ：num_bins + 1 (0Hz:直流成分)
    # 2次元の長さ：Windowの数
    # librosa.util.exceptions.ParameterError: Target size (512) must be at least input size (882)
    D = librosa.stft(y, n_fft=sample_count, win_length=frame_size, hop_length=frame_shift)
    # 振幅
    magnitude = np.abs(D)
    # 振幅をスペクトルとする
    spectrum = magnitude
    # 位相
    phase = np.angle(D)
    # パワースペクトル
    power_spectrum = spectrum ** 2
    # パワースペクトルをデシベル単位(dB)に変換
    log_spectrum = librosa.power_to_db(power_spectrum)

    # STFT 結果行列
    st.write('## STFT')
    st.write(f'- D shape: {D.shape} / {D.dtype}')
    st.write(f'- magnitude shape: {magnitude.shape} / {magnitude.dtype}')
    st.write(f'- phase shape: {phase.shape} / {phase.dtype}')
    st.write(f'- power_spectrum shape: {power_spectrum.shape} / {power_spectrum.dtype}')
    st.write(f'- log_spectrum shape: {log_spectrum.shape} / {log_spectrum.dtype}')

    # 短時間フーリエ変換　振幅グラフ
    # STFT 結果：振幅の可視化
    fig2 = plt.figure(figsize=(16, 6))
    plt.title(f'STFT -> Magnitude of Frequency / {path}' + 
    ' (frame_size:{frame_size}, frame_shift:{frame_shift}, samples:{sample_count}, {spectrum.shape[1]} lines)')
    plt.plot(spectrum)
    plt.grid()
    st.write('### Magnitude of Frequency Graph')
    st.write(f'- X: num_bins(half of samples):{num_bins} + 1 (0Hz)')
    st.write('- Y: Magnitude of Frequency')
    st.write(f'- {spectrum.shape[1]} lines: distributed spectrums by STFT')
    st.pyplot(fig2)

    # 音声のパワースペクトログラム
    # X座標：時間 (Sec)
    # Y座標：周波数の振幅(Hz)
    # 値：パワースペクトログラム（振幅を２乗しlogスケール化）
    fig4 = plt.figure(figsize=(16, 6))
    librosa.display.specshow(log_spectrum, sr=sr, hop_length=frame_shift, x_axis='time', y_axis='hz')
    plt.title(f'Power Spectrogram(log scale) / {path} (frame_size:{frame_size}, frame_shift:{frame_shift}, samples:{sample_count})')
    plt.colorbar()
    st.write('### Power Spectrogram (log-scale)')
    st.write('- X: time (Sec)')
    st.write('- Y: frequency (Hz)')
    st.write('- value: power spectrum (log scale)')
    st.pyplot(fig4)

    # メルフィルタバンクにより256(257)次元を24次元に
    # hkt=True Hidden Markov Model Toolkit を使うかどうか
    mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=sample_count, n_mels=mel_dims, htk=True)
    # メルフィルタバンクの積を取りmel_dim次元にする        
    mel_spectrum = np.dot(mel_filter_bank, power_spectrum)
    # メル尺度でpowerスペクトルをdB単位化
    log_mel_spectrum = librosa.power_to_db(mel_spectrum)
    # メルフィルタバンク処理結果表示
    st.write('## Mel Filterbank')
    st.write(f'- log mel spectrum shape: {log_mel_spectrum.shape}')
    st.write(f'- mel_filter_bank: {mel_filter_bank.shape}')
    st.write(f'power spectrum shape: {power_spectrum.shape}')

    # 音声の対数メルスペクトログラム log Mel-Spectrogram
    #  X座標: 時間 (Sec)
    #  Y座標: mel スケール化された周波数の振幅(Hz)
    #  値: 座標点該当周波数の強度（強度２乗のlog）
    fig5 = plt.figure(figsize=(16, 6))
    img = librosa.display.specshow(log_mel_spectrum, sr=sr, hop_length=frame_shift, x_axis='time', y_axis='mel')
    plt.title(f'Log mel spectrum / {path} (mel_dims:{mel_dims} frame_size:{frame_size}, frame_shift:{frame_shift}, samples:{sample_count})')
    plt.colorbar()
    st.write('### Log Mel Spectrogram')
    st.write('- X: time (Sec)')
    st.write('- Y: Frequency (Hz)')
    st.write('- value: amplitude (log mel scale)')
    st.pyplot(fig5)

    # スペクトラルセントロイド(スペクトル重心) Spectral Centroid
    #  X座標: 時間 (Sec)
    #  Y座標(青): 周波数(Hz:1秒間に何回の生起が発生するか)
    #  Y座標(赤): スペクトル重心(周波数の加重平均値)

    # スペクトル重心の算出
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    # 可視化のためのフレームカウント(時間変数)を計算
    frames = range(len(spectral_centroids))
    # フレームカウントを時間（秒）に変換
    t = librosa.frames_to_time(frames)
    # アルファ
    alpha = 0.5

    # 振幅グラフ(青線)の描画
    fig6 = plt.figure(figsize=(16, 6))
    librosa.display.waveshow(y, sr=sr, alpha=alpha, color='b')
    # スペクトル重心(赤線)の描画
    plt.plot(t, sklearn.preprocessing.minmax_scale(spectral_centroids, axis=0), color='r')
    plt.title(f'Spectral Centroid / {path} ({len(spectral_centroids)} frames, num_bins:{num_bins}, {sample_count} samples)')

    st.write('### Spectral Centroid')
    st.write(f'- spectral centroids shape: {spectral_centroids.shape} / {spectral_centroids.dtype}')
    st.write(f'- {len(spectral_centroids)} frames')
    st.write('- X: time (Sec)')
    st.write(f'- Y (blue): magnitude of frequency {num_bins}bins / alpha: {alpha}')
    st.write(f'- Y (red):  spectral centroid {sample_count} samples')
    st.pyplot(fig6)

   # 周波数ロールオフ Spectral Rolloff
    #  X座標: 時間 (Sec)
    #  Y座標(青): 周波数(Hz:1秒間に何回の生起が発生するか)
    #  Y座標(赤): Spectral Rolloff(周波数の加重平均値、音響特徴量の一種)

    # ロールオフ（roll-off）とは、フィルタの「切れ」を表す特性。
    # フィルタの帯域の端における通過特性の変化の急峻さで表され、
    # 大きい値ほど切れがよいフィルタとなる。
    # 単位はdB/octave（周波数が2倍変化した時の通過特性の変化）
    # またはdB/decade（周波数が10倍変化した時の通過特性の変化）。

    # 周波数ロールオフの計算
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    # 振幅グラフ(青線)の描画
    fig7 = plt.figure(figsize=(16, 6))
    librosa.display.waveshow(y, sr=sr, alpha=alpha, color='b')
    # 周波数ロールオフ(赤線)の描画
    plt.plot(t, sklearn.preprocessing.minmax_scale(spectral_rolloff, axis=0), color='r')
    plt.title(f'Spectral Rolloff / {path} ({len(spectral_rolloff)} frames, num_bins:{num_bins}, {sample_count} samples)')
    st.write('### Spectral Rolloff')
    st.write('- X: time (Sec)')
    st.write(f'- Y (blue): magnitude of frequency / alpha: {alpha}')
    st.write('- Y (red):  spectral rolloff')
    st.pyplot(fig7)

    # メル周波数ケプストラム係数
    # Mel-Frequency Cepstral Coefficients (MFCC)

    # MFCC 算出
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs = sklearn.preprocessing.minmax_scale(mfccs, axis=1)

    fig8 = plt.figure(figsize=(16, 6))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.title(f'MFCC / {path} (shape:{mfccs.shape} mean:{mfccs.mean()}, var:{mfccs.var()})')
    plt.colorbar(format='%+2.0f')
    st.write('### MFCC')
    st.write('> MFCC(Mel-Frequency Cepstram Coefficient) is unusualy used with deep learning model.')
    st.write('- X: time (Sec)')
    st.write('- Y: Frequency (Hz)')
    st.write('- value: MFCC')
    st.write('  - mean: %.2f' % mfccs.mean())
    st.write('  - var:  %.2f' % mfccs.var())
    st.pyplot(fig8)


def main()->None:
    """
    Streamlit を使ってWeb UIを表示する。

    Parameters
    -----
    None

    Returns
    -----
    None
    """
    # ページ設定
    st.set_page_config(layout='wide')

    # タイトル
    st.title('Sound Feartures')
    st.write('You can evaluate sound features of selected wav files.')

    # ファイル選択(サイドバー)
    st.sidebar.header('Choose wav files')
    st.sidebar.write('You can select multi files.')
    uploaded_files = st.sidebar.file_uploader('Choose wav files', accept_multiple_files=True)
    if uploaded_files is not None and len(uploaded_files) >0:
        index = 0
        for col in st.columns(len(uploaded_files)):
            with col:
                _show_feature(uploaded_files[index])
                index = index + 1

if __name__ == '__main__':
    main()