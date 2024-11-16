from flask import Flask, request, jsonify
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


# Flaskアプリケーション


# Voskモデルのロード
MODEL_NAME_LARGE = "vosk-model-ja-0.22"
model = Model(model_name=MODEL_NAME_LARGE)  # 修正: model_nameに変更
vosk_sr:int = 16000
VOSK_IGNORE_WARDS = {
    'えー', 'えええ', 'あっ'
}
class VoskProcessor:

    @staticmethod
    def process_audio_in_process(input_data):
        global processor
        return processor.process_audio(input_data)

    @staticmethod
    def initializer():
        global processor
        processor = VoskProcessor()
        print("VoskProcessor initialized")

    def __init__(self):
        self.recognizer = KaldiRecognizer(model, vosk_sr)

    def text_strip(self,text) ->str:
        if not text:
            return ''
        xx = json.loads(text)
        text = xx.get('text')
        if not text:
            return ''
        ret = ''
        for tk in text.split(' '):
            if tk in VOSK_IGNORE_WARDS:
                ret += '<|noize|>'
            else:
                ret += tk
        return ret

    def process_audio(self,input_data) ->dict:
        """
        音声を処理して認識結果を返す
        """
        # ログ出力
        try:
            with wave.open(BytesIO(input_data), "rb") as wf:
                input_ch = wf.getnchannels()
                input_sr = wf.getframerate()
                input_sz = wf.getnframes()
            print(f"Request sr:{input_sr}, ch:{input_ch}, sz:{input_sz}")
        except Exception as e:
            return {"error": f"Failed to read audio: {str(e)}"}
        try:
            # 音声データのリサンプリングとモノラル変換
            audio_f32, sr = librosa.load(BytesIO(input_data), sr=vosk_sr, mono=True)
            audio_abs = np.abs(audio_f32)
            lv_max = np.max(audio_abs)
            if lv_max<1e-6:
                lv_max = round( float(lv_max),4)
                return {"error": "Audio is silent","lv_max":lv_max}
            x = lv_max*0.2
            non_zero = audio_abs[audio_abs>= 1e-9]
            audio_abs2 = non_zero[non_zero<x]
            lv_mean = np.mean(audio_abs2)
            lv_std_dev = np.std(audio_abs2)
            lv = norm.ppf(0.8, loc=lv_mean, scale=lv_std_dev)
            print(f"audio range {lv_max:.3f}-{lv_std_dev:.3f} mean:{lv_mean:.3f} lv:{lv:.3f}")
            rr = 32767 * (0.1 / lv)
            audio_f32 *= rr
            audio_f32 = np.clip(-32767,32767,audio_f32)

            audio_bytes = audio_f32.astype(np.int16).tobytes()

            # Voskで音声認識
            results = []

            text = ''
            for i in range(0, len(audio_bytes), 4000):
                data = audio_bytes[i:i+4000]
                if self.recognizer.AcceptWaveform(data):
                    text += self.text_strip( self.recognizer.Result())

            text += self.text_strip(self.recognizer.FinalResult())
            lv_max = round( float(lv_max),4)
            lv = round( float(lv),4)
            return {"results": text, "lv_max": lv_max, "lv_mean": lv}
        except Exception as e:
            return {"error": str(e)}
        finally:
            self.recognizer.Reset()

class SttServer(Flask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ProcessPoolExecutor(max_workers=4,initializer=VoskProcessor.initializer)
        self.add_url_rule('/recognize', 'recognize_audio', self.recognize_audio, methods=['POST'])

    def recognize_audio(self):
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        input_data = audio_file.read()

        future = self.executor.submit(VoskProcessor.process_audio_in_process, input_data)
        result = future.result()
        if "error" in result:
            return jsonify(result), 400
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
        with open(AUDIO_FILE_PATH, "rb") as f:
            wav_bytes = f.read()

        for i in range(4):
            files = {"audio": BytesIO(wav_bytes)}
            st = time.time()
            response = requests.post(SERVER_URL, files=files)
            elaps_time = time.time() - st

            if response.status_code == 200:
                print(f"Recognition results: {elaps_time:.3f}(sec)")
                print(response.json())
            else:
                print(f"Error: {response.status_code}")
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