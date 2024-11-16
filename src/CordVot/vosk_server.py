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
from scipy.stats import norm
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue
import json
import traceback
import re
import asyncio
import tempfile


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
class VoskProcessor:

    @staticmethod
    def process_wave_in_process(input_data):
        global processor
        return processor.process_wave(input_data)

    @staticmethod
    def process_audio_in_process(input_data,sr):
        global processor
        return processor.process_audiof(input_data,sr)

    @staticmethod
    def initializer( threshold:float, audio_log:str|None=None):
        global processor
        processor = VoskProcessor(threshold,audio_log=audio_log)
        print(f"VoskProcessor initialized audio_log={audio_log}")

    def __init__(self, threshold:float, audio_log:str|None=None):
        vosk.SetLogLevel(-1)
        self.threshold:float = threshold
        self.audio_log:str|None = audio_log
        if audio_log:
            lock_dir = os.path.join(audio_log,'.lock')
            os.makedirs(lock_dir,exist_ok=True)
            os.removedirs(lock_dir)
        self.recognizer = KaldiRecognizer(model, vosk_sr)

    def process_audiof(self,input_data:np.ndarray,sr:int) ->dict:
        self.saveto(input_data,sr)
        if sr != vosk_sr:
            input_data2 = librosa.resample( input_data, orig_sr=sr, target_sr=vosk_sr)
        else:
            input_data2 = input_data
        return self._process_audio(input_data2)

    def process_wave(self,wave_data) ->dict:
        # ログ出力
        try:
            with wave.open(BytesIO(wave_data), "rb") as wf:
                input_ch = wf.getnchannels()
                input_sr = wf.getframerate()
                input_sz = wf.getnframes()
            print(f"Request sr:{input_sr}, ch:{input_ch}, sz:{input_sz}")
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

    def _process_audio(self,audio_f32) ->dict:
        """
        音声を処理して認識結果を返す
        """
        try:
            audio_abs = np.abs(audio_f32)
            lv_max = np.max(audio_abs)
            if lv_max<self.threshold:
                lv_max = round( float(lv_max),4)
                return {"error": "Audio is silent","lv_max":lv_max}
            # ゼロでない要素のインデックスを取得
            non_zero_indices = np.where(audio_abs>1e-9)[0]
            # スライス範囲を取得
            if non_zero_indices.size == 0:
                return {"error": "Audio is silent","lv_max":lv_max}
            start, end = non_zero_indices[0], non_zero_indices[-1] + 1
            audio_f32 = audio_f32[start:end]
            audio_abs = audio_abs[start:end]
            # 音量を計算
            x = lv_max*0.2
            non_zero = audio_abs[audio_abs>= 1e-9]
            audio_abs2 = non_zero[non_zero<x]
            lv_mean = np.mean(audio_abs2)
            lv_std_dev = np.std(audio_abs2)
            lv = norm.ppf(0.8, loc=lv_mean, scale=lv_std_dev)
            r1 = 0.1/lv
            r2 = 0.8/lv_max
            rr = max(r1,r2)
            print(f"audio range {lv_max:.3f}-{lv_std_dev:.3f} mean:{lv_mean:.3f} lv:{lv:.3f}")
            rr = 32767 * rr
            audio_f32 *= rr
            audio_f32 = np.clip(-32767,32767,audio_f32)

            audio_bytes = audio_f32.astype(np.int16).tobytes()

            # Voskで音声認識
            raw_results = []

            for i in range(0, len(audio_bytes), 4000):
                data = audio_bytes[i:i+4000]
                if self.recognizer.AcceptWaveform(data):
                    text = self._get_text_from_result( self.recognizer.Result())
                    if text:
                        raw_results.append(text)
            text = self._get_text_from_result(self.recognizer.FinalResult())
            if text:
                raw_results.append(text)
            raw_text = ' '.join(raw_results)
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

            lv_max = round( float(lv_max),4)
            lv = round( float(lv),4 )
            return {"text": text, "raw":raw_text, "lv_max": lv_max, "lv_mean": lv}
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
                    elif isinstance(data,np.ndarray):
                        with wave.open(wave_path, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(sr if isinstance(sr,int) else vosk_sr)
                            i16 = (data*32767).astype(np.int16).tobytes()
                            wf.writeframes(i16)
                finally:
                    if os.path.exists(lock_dir):
                        os.removedirs(lock_dir)
                break
            except:
                pass
            time.sleep(0.2)

class VoskExecutor():

    def __init__(self, max_workers:int=4, threshold:float=0.2, audio_log:str|None=None):
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = VoskExecutor()
        self.add_url_rule('/recognize', 'recognize_audio', self.recognize_audio, methods=['POST'])

    def recognize_audio(self):
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        input_data = audio_file.read()

        result = self.executor.submit(input_data)
        return jsonify(result)

# サーバーのURL
SERVER_URL = "http://127.0.0.1:5007/recognize"
AUDIO_FILE_PATH = "tmp/audio0001.wav"

def start_server():
    """
    Flaskサーバーをバックグラウンドで起動
    """
    app = SttServer(__name__)
    server_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5007, debug=False, use_reloader=False))
    server_thread.daemon = True  # プロセス終了時にスレッドも終了
    server_thread.start()
    return server_thread

def test_recognition():
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
            response = requests.post(SERVER_URL, files=files)
            elaps_time = time.time() - st

            if response.status_code == 200:
                print(f"Recognition results: {w} {elaps_time:.3f}(sec)")
                print(response.json())
            else:
                print(f"Error: {w} {response.status_code}")
                print(response.json())
    except Exception as e:
        print(f"An error occurred during the test: {e}")

if __name__ == "__main__":
    # サーバー起動
    server_thread = start_server()

    # サーバーが完全に起動するまで少し待つ
    time.sleep(2)

    # テストを実行
    test_recognition()

    # サーバーを停止
    if server_thread.is_alive():
        print("Server thread is still running.")