from flask import Flask, request, jsonify
import vosk
from vosk import Model, KaldiRecognizer
import wave
import os
import requests
import threading
import time
import librosa
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import norm
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue
import json
import traceback
import re
import asyncio
import tempfile
from numpy.typing import NDArray


# Flaskアプリケーション
vosk.SetLogLevel(-1)

# Voskモデルのロード
MODEL_NAME_LARGE = "vosk-model-ja-0.22"
model = Model(model_name=MODEL_NAME_LARGE)  # 修正: model_nameに変更
vosk_sr:int = 16000
VOSK_HIRAKANA = [ ]
VOSK_IGNORE_WARDS = {
    'えー', 'えええ', 'あっ'
}
VOSK_NOIZE='<|noize|>'

def moving_ave( array:NDArray[np.float32], window_size ) ->tuple[NDArray[np.float32],NDArray[np.float32]]:
    padding = window_size // 2
    window_size = padding*2 + 1

    padding_np1 = np.zeros( padding+1, dtype=array.dtype )
    padding_np1[:] = array[0]
    padding_np2 = np.zeros( padding, dtype=array.dtype )
    padding_np2[:] = array[-1]
    padd_array = np.concatenate( (padding_np1, array, padding_np2 ) )
    cumsum = np.cumsum(padd_array)
    np_sum = cumsum[window_size:] - cumsum[:-window_size] 
    np_ave = np_sum / window_size
    return np_sum,np_ave

def audio_aaa( audio:NDArray[np.float32], sr:int, w:float=0.2, threshold:float=1e-3 ):
    audio_abs = np.abs(audio)
    # 音の開始点
    zero_up_idx = np.where( (audio_abs[:-1] <= threshold) & (audio_abs[1:]>threshold))[0]
    zero_up_idx += 1

def nomlize_audio( audio:NDArray[np.float32], sr:int, *, target_lv:float=0.8, signal_threshold:float=1e-9 ) ->NDArray[np.float32]:
    coef, _,_ = generate_peak_coefficients( audio, sr, signal_threshold=signal_threshold )
    a = audio *coef
    max_lv = np.max(np.abs(a))
    r = target_lv / max_lv
    return a*r

def generate_peak_coefficients(audio: NDArray[np.float32], sr: int, seg: float = 0.02, signal_threshold:float=1e-9, dbg:bool=False) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    audio_length: int = len(audio)
    seg_width: int = int(sr * seg)  # サンプリングレートと区間の長さに基づいて、区間の幅をサンプル数で計算

    # 音声信号の絶対値を取得して振幅を扱う
    audio_abs = np.abs(audio)

    # 各区間内の最大値を見つける (ループをnumpy演算に変更して高速化)
    l2 = audio_length - (audio_length % seg_width)
    reshaped_audio_abs = audio_abs[:l2].reshape(-1, seg_width) # 区間で２次元配列にする
    max_values = np.max(reshaped_audio_abs, axis=1)
    max_indices = np.argmax(reshaped_audio_abs, axis=1) + np.arange(0, l2, seg_width)

    # 音声全体の最大値を見つける
    audio_max = np.max(max_values)

    # 係数の配列を初期化
    coef = np.ones(audio_length, dtype=np.float32)  # 係数配列を1で初期化

    # デバッグ用に区間ごとの最大値を保存する配列を作成
    if dbg:
        audio_lvs = np.repeat(max_values, seg_width)  # numpyのrepeatを使って各区間の最大値を全体に広げる
        audio_lvs = np.concatenate((audio_lvs, np.full(audio_length - len(audio_lvs), max_values[-1])))  # 残りの部分に最大値を埋める
    else:
        audio_lvs = coef

    # ピーク検出のしきい値としてピークリミットを決定
    peak_limit = audio_max / 8

    # ピークリミットを超える最大値の中でピークを見つける
    peaks, props = find_peaks(max_values, height=peak_limit)

    if dbg:
        # マーカーの配列を初期化
        marker = np.full(audio_length, peak_limit, dtype=np.float32)  # マーカー配列をピークリミットの値で初期化
        for i in peaks:
            marker[max_indices[i]] = audio_max  # 検出されたピークをマーカー配列にマーク
    else:
        marker = coef

    # 有意なピークがない場合、デフォルトの係数配列を返す
    if len(peaks) <= 1:
        return coef, audio_lvs, marker

    # 検出されたピークを繰り返し処理して係数を計算
    before_idx = 0
    before_rate = audio_max / max_values[peaks[0]]  # 最初のピークに基づく初期レート
    for i in peaks:
        idx = max_indices[i]  # 元の音声でのピークのインデックスを取得
        peak_lv = max_values[i]  # ピークのレベルを取得
        rate = audio_max / peak_lv  # 正規化のためのレートを計算
        u = np.linspace(before_rate, rate, idx - before_idx)  # 前のレートと現在のレートの間を線形補間
        coef[before_idx:idx] = u  # 補間値を係数配列に割り当て
        coef[idx] = rate  # ピークインデックスに現在のレートを設定
        before_idx = idx  # 前のインデックスを現在のインデックスに更新
        before_rate = rate  # 前のレートを現在のレートに更新

    # 最後のピークの後に残っているサンプルがあれば、その係数を設定
    if before_idx < audio_length:
        coef[before_idx:] = before_rate

    return coef, audio_lvs, marker

def load_wave_file( file ) ->tuple[NDArray[np.float32],int]:
    with wave.open(file,'rb') as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        cname = wf.getcompname()
        ctype = wf.getcomptype()
        nf = wf.getnframes()
        sec = round( nf/sr, 3 )
        print(f"wave:{file} {sec}sec {sr}Hz {ch}ch {sw}byte {cname}")
        if ctype != 'NONE':
            raise Exception(f"can not read {cname}: {file}")
        audio_f32 = np.zeros( nf, dtype=np.float32 )
        block_size:int = 8192
        for i in range(0,nf,block_size):
            bz = min( nf-i, block_size)
            audio_bytes = wf.readframes(bz)
            seg = np.frombuffer(audio_bytes,dtype=np.int16).reshape( (-1,ch) )[:,0].astype(np.float32) / 32768.0
            np.copyto( audio_f32[i:i+bz], seg )
    return audio_f32,sr

def save_wave_file( file, audio:NDArray[np.float32], sr ):
    with wave.open(file,'wb') as wf:
        wf.setframerate(sr)
        wf.setsampwidth(2)
        wf.setnchannels(1)
        wf.setnframes(len(audio))
        by = (audio*32767).astype(np.int16).tobytes()
        wf.writeframes(by)

class VoskProcessor:

    @staticmethod
    def process_wave_in_process(input_data:bytes):
        global processor
        return processor.process_wave(input_data)

    @staticmethod
    def process_audio_in_process(input_data:NDArray[np.float32],sr):
        global processor
        return processor.process_audiof(input_data,sr)

    @staticmethod
    def initializer( threshold:float, audio_log:str|None=None):
        global processor
        processor = VoskProcessor(threshold,audio_log=audio_log)
        print(f"VoskProcessor initialized audio_log={audio_log}")

    def __init__(self, threshold:float, audio_log:str|None=None):
        vosk.SetLogLevel(-1)
        self.sec_threshold:float = 0.8
        self.signal_threshold:float = threshold
        self.audio_log:str|None = audio_log
        if audio_log:
            lock_dir = os.path.join(audio_log,'.lock')
            os.makedirs(lock_dir,exist_ok=True)
            os.removedirs(lock_dir)
        self.recognizer = KaldiRecognizer(model, vosk_sr)
        self.audio_norm = 2

    def process_audiof(self,input_data:NDArray[np.float32],sr:int) ->dict:
        sec = round( len(input_data)/sr,3)
        if sec>self.sec_threshold:
            self.saveto(input_data,sr)
        if sr != vosk_sr:
            input_data2 = librosa.resample( input_data, orig_sr=sr, target_sr=vosk_sr)
        else:
            input_data2 = input_data
        return self._process_audio(input_data2)

    def process_wave(self,wave_data:bytes) ->dict:
        # ログ出力
        try:
            with wave.open(BytesIO(wave_data), "rb") as wf:
                input_ch = wf.getnchannels()
                input_sr = wf.getframerate()
                input_sz = wf.getnframes()
            sec = round( input_sz/input_sr,3)
            print(f"Request sr:{input_sr}, ch:{input_ch}, sz:{input_sz}")
            if sec>self.sec_threshold:
                self.saveto(wave_data)
            # 音声データのリサンプリングとモノラル変換
            audio_f32, sr = librosa.load(BytesIO(wave_data), sr=vosk_sr, mono=True)
            return self._process_audio(audio_f32)
        except Exception as e:
            return {"error": f"Failed to read audio: {str(e)}"}

    def _get_text_from_result(self,result_text) ->str|None:
        if result_text:
            res_dict = json.loads(result_text)
            text = res_dict.get('text')
            if text is not None and len(text)>0:
                return text
        return None

    def _validate_text(self,text) ->str:
        if not text:
            return ''
        if text in VOSK_IGNORE_WARDS:
            return VOSK_NOIZE
        not_hirakana = re.compile('[^\u3041-\u309F]+')
        if not_hirakana.search(text) and len(text)>1:
            return text
        text2 = text.replace('っ','').replace('ー','')
        if len(text2)<=2:
            return VOSK_NOIZE
        return text

    def _process_audio(self,audio_f32:NDArray[np.float32]) ->dict:
        """
        音声を処理して認識結果を返す
        """
        resp = {}
        try:

            sec = round( len(audio_f32) / vosk_sr , 3 )
            resp['audio_sec'] = sec
            if sec<self.sec_threshold:
                resp["error"]="Audio is short"
                return resp
            audio_abs = np.abs(audio_f32)
            lv_max = np.max(audio_abs)
            resp['lv_max'] = round( float(lv_max),4)
            if lv_max<self.signal_threshold:
                resp["error"]="Audio is silent"
                return resp
            # ゼロでない要素のインデックスを取得
            non_zero_indices = np.where(audio_abs>1e-9)[0]
            # スライス範囲を取得
            if non_zero_indices.size == 0:
                resp["error"]="Audio is silent"
                return resp
            start, end = non_zero_indices[0], non_zero_indices[-1] + 1
            resp['voice_sec'] = round( (end-start)/vosk_sr, 4 )
            audio_f32 = audio_f32[start:end]

            target_lv = 32767 * 0.8
            audio_f32 = nomlize_audio( audio_f32, vosk_sr, target_lv=target_lv, signal_threshold=self.signal_threshold )
            audio_bytes = audio_f32.astype(np.int16).tobytes()
            audio_bz = len(audio_bytes)
            block_size = 4000
            # Voskで音声認識
            raw_results = []

            for st in range(0, len(audio_bytes), block_size):
                ed = min(st+block_size,audio_bz)
                if self.recognizer.AcceptWaveform(audio_bytes[st:ed]):
                    text = self._get_text_from_result( self.recognizer.Result())
                    if text:
                        raw_results.append(text)
            text = self._get_text_from_result(self.recognizer.FinalResult())
            if text:
                raw_results.append(text)
            raw_text = ' '.join(raw_results)
            resp['raw'] = raw_text
            resp['text'] = raw_text
            res = []
            nz:bool = False
            for t in raw_results:
                t = self._validate_text(t)
                if t == VOSK_NOIZE:
                    if not nz:
                        nz=True
                        res.append(t)
                else:
                    nz = False
                    res.append(t)
            if len(res)==0:
                text = ''
            elif len(res)==1 and nz:
                text = VOSK_NOIZE
            else:
                text = ''.join(res).replace(' ','')

            resp['text'] = text
            return resp
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}
        finally:
            self.recognizer.Reset()

    def saveto(self, data, sr:int|None=None):
        if self.audio_log is None or not os.path.isdir(self.audio_log):
            return

        lock_dir = os.path.join(self.audio_log,'.lock')
        prefix='audio'
        ext='.wav'
        for i in range(20):
            try:
                os.makedirs(lock_dir,exist_ok=False)
                try:
                    last_num = 1
                    for f in os.listdir(self.audio_log):
                        if f.startswith(prefix) and f.endswith(ext):
                            num = f[len(prefix):-len(ext)]
                            try:
                                inum = int(num)
                                if last_num<inum:
                                    last_num = inum
                            except:
                                pass
                    last_num += 1
                    wave_path = os.path.join(self.audio_log, f"{prefix}{last_num:04d}{ext}")
                    print(f"[vosk]save {wave_path}")
                    if isinstance(data,bytes):
                        with open(wave_path,'wb') as wf:
                            wf.write(data)
                    elif isinstance(data,np.ndarray) and data.dtype==np.float32:
                        save_wave_file( wave_path, data, sr if isinstance(sr,int) else vosk_sr)

                finally:
                    if os.path.exists(lock_dir):
                        os.removedirs(lock_dir)
                break
            except:
                pass
            time.sleep(0.2)

class VoskExecutor():

    def __init__(self, max_workers:int=4, threshold:float=0.02, audio_log:str|None=None):
        self.threshold:float = threshold
        self.executor = ProcessPoolExecutor(max_workers=max_workers,initializer=VoskProcessor.initializer, initargs=(threshold,audio_log,))

    def submit(self,input_data):
        future = self.executor.submit(VoskProcessor.process_wave_in_process, input_data)
        result = future.result()
        return result

    async def asubmit_wave(self, wave_data):
        # 非同期にProcessPoolExecutorを使ってタスクを実行
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self.executor, VoskProcessor.process_wave_in_process, wave_data)
        # 非同期に結果を待つ
        result = await future
        return result

    async def asubmit_audio(self, audio_data,sr):
        # 非同期にProcessPoolExecutorを使ってタスクを実行
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self.executor, VoskProcessor.process_audio_in_process, audio_data,sr)
        # 非同期に結果を待つ
        result = await future
        return result

class SttServer(Flask):
    def __init__(self, import_name:str, max_workers:int=4, threshold:float=0.02, audio_log:str|None=None, *args, **kwargs):
        super().__init__( import_name, *args, **kwargs)
        self.executor = VoskExecutor( max_workers=max_workers, threshold=threshold, audio_log=audio_log )
        self.add_url_rule('/recognize', 'recognize_audio', self.recognize_audio, methods=['POST'])

    def recognize_audio(self):
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        input_data = audio_file.read()

        result = self.executor.submit(input_data)
        return jsonify(result)

def start_server( *, host:str="0.0.0.0", port:int=5007):
    """
    Flaskサーバーをバックグラウンドで起動
    """
    app = SttServer(__name__, threshold=0.02)
    server_thread = threading.Thread(target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False))
    server_thread.daemon = True  # プロセス終了時にスレッドも終了
    server_thread.start()
    return server_thread

def test_recognition( *, url:str ):
    """
    Flaskサーバーに音声ファイルを送信して認識結果を取得
    """
    try:

        for w in os.listdir('tmp/audio'):
            if not w.endswith('.wav'):
                continue
            wav_path = os.path.join('tmp/audio',w)
            with open(wav_path, "rb") as f:
                wav_bytes = f.read()

            files = {"audio": BytesIO(wav_bytes)}
            st = time.time()
            response = requests.post(url, files=files)
            elaps_time = time.time() - st

            if response.status_code == 200:
                print(f"Recognition results: {w} elaps:{elaps_time:.3f}(sec)")
                print(response.json())
            else:
                print(f"Error: {w} {response.status_code}")
                print(response.json())
    except Exception as e:
        print(f"An error occurred during the test: {e}")

def test_main():

    host='127.0.0.1'
    port=5007
    SERVER_URL = f"http://{host}:{port}/recognize"

    # サーバー起動
    server_thread = start_server( host=host, port=port )

    # サーバーが完全に起動するまで少し待つ
    time.sleep(2)

    # テストを実行
    test_recognition( url=SERVER_URL )

    # サーバーを停止
    if server_thread.is_alive():
        print("Server thread is still running.")

def dbg_plot( pngfile, *, audio, norm_audio=None, levels=None, marker=None, coef=None, title=None, ylim:bool=False  ):
    import matplotlib.pyplot as plt
    plt.clf()
    fig, ax1 = plt.subplots( figsize=(13,7),dpi=300)
    if title:
        plt.title(title)
    if levels is not None:
        x = range(len(levels))
        ax1.fill_between( x, -levels, levels, alpha=0.2, color=None, edgecolor="black", label='signal level')
    if audio is not None:
        ax1.plot(audio, alpha=0.2, color='green', label='audio')
    if norm_audio is not None:
        ax1.plot(norm_audio, alpha=0.2, color='blue', label='norm audio')
    if marker is not None:
        ax1.plot(marker, alpha=0.2, color='red', label='marker')
    ax1.legend(loc='upper left')
    # ax1.plot(audio_lv)
    # ax1.plot(lv_ave)
    if ylim:
        ax1.set_ylim(-1,1)
    if coef is not None:
        ax2 = ax1.twinx()
        ax2.plot( coef, alpha=0.6, color='orange', label='coefficient' )
        ax2.legend(loc='upper right')
    plt.savefig( pngfile )
    plt.close()

def test_anti_fadein():

    file = 'tmp/audio/audio0002.wav'
    file = 'tmp/audio/audio0005.wav'
    #file = 'tmp/audio/audio0010.wav'
    #file = 'tmp/audio/audio0002.wav'
    audio_raw, sr = load_wave_file(file)
    dbg_plot( 'tmp/dbgaudio00.png', title='raw signal', audio=audio_raw, ylim=True)
    raw_lv = np.max( np.abs(audio_raw) )
    rate = round( 0.99/raw_lv, 2 )
    audio_x5 = audio_raw * rate
    save_wave_file( 'tmp/dbgaudio1.wav', audio_x5, sr )
    dbg_plot( 'tmp/dbgaudio01.png', title='raw signal', audio=audio_raw )

    st=time.time()
    coef, lvs, marker = generate_peak_coefficients( audio_raw, sr, dbg=True )
    t = time.time()-st
    print(f"Time {t:.3f} sec")
    dbg_plot( 'tmp/dbgaudio02.png', title=f'raw signal with signal level', audio=audio_raw, levels=lvs )
    dbg_plot( 'tmp/dbgaudio03.png', title=f'raw signal with peak point', audio=audio_raw, marker=marker, levels=lvs )
    dbg_plot( 'tmp/dbgaudio04.png', title=f'raw signal with coefficient', audio=audio_raw, levels=lvs, coef=coef )

    norm_audio = audio_raw*coef
    r = 1.0 / np.max(np.abs(norm_audio))
    norm_audio *= r
    save_wave_file( 'tmp/dbgaudio5.wav', norm_audio, sr )
    dbg_plot( 'tmp/dbgaudio05.png', title=f'normalized audio', audio=audio_x5, norm_audio=norm_audio, marker=marker )

if __name__ == "__main__":
    #test_main()
    test_anti_fadein()