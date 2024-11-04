import sys,os
from typing import TypeVar
from io import BytesIO
import wave
import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample as scipy_resample

# 型エイリアス
AudioF32 = NDArray[np.float32]
AudioF16 = NDArray[np.float16]
AudioI16 = NDArray[np.int16]
AudioI8 = NDArray[np.int8]
# 定数
EmptyF32:AudioF32 = np.zeros(0,dtype=np.float32)
EmptyI16:AudioI16 = np.zeros(0,dtype=np.int16)

def audio_info( audio:AudioF32, sample_rate:int ) ->str:
    size:int = len(audio)
    sec:float = size/sample_rate
    lo:float = min(audio)
    hi:float = max(audio)
    ave:float = signal_ave(audio)
    return f"[fr:{size},sec:{sec:.3f},lo:{lo:.2f},hi:{hi:.2f},ave:{ave:.2f}]"

def as_str( value, default:str='') ->str:
    if isinstance(value,str):
        return value
    return default

def as_list( value, default:list=[]) ->list:
    if isinstance(value,list):
        return value
    return default

def as_int( value, default:int=0) ->int:
    if isinstance(value,int|float):
        return int(value)
    return default

def as_float( value, default:float=0) ->float:
    if isinstance(value,int|float):
        return float(value)
    return default

def np_shiftL( a:np.ndarray, n:int=1 ):
    if 0<n and n<len(a)-1:
        a[:-n] = a[n:]

def np_append( buf:AudioF32, x:AudioF32 ):
    n:int = len(x)
    if n>=len(buf):
        buf = x[:-len(buf)]
    else:
        buf[:-n] = buf[n:]
        buf[-n:] = x

def is_f32(data:np.ndarray) ->bool:
    return isinstance(data,np.ndarray) and data.dtype==np.float32

def is_i16(data:np.ndarray) ->bool:
    return isinstance(data,np.ndarray) and data.dtype==np.int16

def from_f32( data:AudioF32, *, dtype ):
    if is_f32(data):
        if dtype == np.int8:
            return (data*126).astype(np.int8)
        elif dtype == np.int16:
            return (data*32767).astype(np.int16)
        elif dtype == np.float16:
            return data.astype(np.float16)
        elif dtype == np.float32:
            return data
    return np.zeros(0,dtype=dtype)

def to_f32( data ) ->AudioF32:
    if is_f32(data):
        return data
    if is_i16(data):
        return i16_to_f32(data)
    return np.zeros(0,dtype=np.float32)

def f32_to_i8( data:AudioF32 ) -> AudioI8:
    if is_f32(data):
        return (data*126).astype(np.int8)
    else:
        return np.zeros(0,dtype=np.int8)

def f32_to_i16( data:AudioF32 ) -> AudioI16:
    if is_f32(data):
        return (data*32767).astype(np.int16)
    else:
        return np.zeros(0,dtype=np.int16)

def i16_to_f32( data:AudioI16 ) -> AudioF32:
    if is_i16(data):
        return data.astype(np.float32)/32767.0
    else:
        return np.zeros(0,dtype=np.float32)

def resample( audio_f32:AudioF32, orig:int, target:int ):
    if orig == target:
        return audio_f32
    x = scipy_resample(audio_f32, int(len(audio_f32) * target / orig ))
    if not isinstance(x,np.ndarray):
        raise wave.Error("リサンプリングエラー")
    return x

# WAVファイルとして保存
def save_wave(filename:str, data:AudioF32, *, sampling_rate:int, ch:int):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(sampling_rate)
        if is_i16(data):
            wf.writeframes(data.tobytes())
        else:
            wf.writeframes( (data*32767).astype(np.int16).tobytes())

def load_wave(data:str|bytes, sampling_rate:int) ->AudioF32:
    if isinstance(data,bytes):
        input_file = BytesIO(data)
    elif isinstance(data,str):
        input_file = data
    else:
        return EmptyF32
    with wave.open(input_file,'rb') as iw:
        # データ読み出し
        wave_bytes:bytes = iw.readframes(iw.getnframes())
        if iw.getsampwidth()!=2:
            raise wave.Error('int16のwaveじゃない')
        audio_i16:AudioI16 = np.frombuffer(wave_bytes, dtype=np.int16)
        ch:int = iw.getnchannels()
        if ch>1:
            # ステレオデータの場合は片側だけにする
            audio_i16 = audio_i16[::ch]
        audio_f32:AudioF32 = audio_i16.astype(np.float32)/32768.0
        # リサンプリング（必要ならば）
        if iw.getframerate() != sampling_rate:
            audio_f32 = resample( audio_f32, iw.getframerate(), sampling_rate )
    return audio_f32

def signal_ave( signal:AudioF32 ) ->float:
    if not isinstance(signal, np.ndarray) or signal.dtype != np.float32 or len(signal.shape)!=1:
        raise TypeError("Invalid signal")
    # 絶対値が0.001以上の要素をフィルタリングするためのブール配列
    boolean_array = np.abs(signal) >= 0.001
    # 条件を満たす要素を抽出
    filtered_array = signal[boolean_array]
    if len(filtered_array)>0:
        # 平均を計算
        ave = np.mean(np.abs(filtered_array))
        return float(ave)
    else:
        return 0.0

def sin_signal( *, freq:int=220, duration:float=3.0, vol:float=0.5,sample_rate:int=16000, chunk:int|None=None) ->AudioF32:
    #frequency # 生成する音声の周波数 100Hz
    chunk_len:int = chunk if isinstance(chunk,int) and chunk>0 else int(sample_rate*0.2)
    chunk_sec:float = chunk_len / sample_rate  # 生成する音声の長さ（秒）
    t = np.linspace(0, chunk_sec, chunk_len, endpoint=False) # 時間軸
    chunk_f32:AudioF32 = np.sin(2 * np.pi * freq * t).astype(np.float32) # サイン波の生成
    # 音量調整
    chunk_f32 = chunk_f32 * vol
    # フェードin/out
    fw_half_len:int = int(chunk_len/5)
    fw:AudioF32 = np.hanning(fw_half_len*2)
    chunk_f32[:fw_half_len] *= fw[:fw_half_len]
    chunk_f32[-fw_half_len:] *= fw[-fw_half_len:]
    #print(f"signal len{len(chunk_f32)}")
    # 指定長さにする
    data_len:int = int( sample_rate * duration)
    n:int = (data_len+chunk_len-1)//chunk_len
    aaa = [ chunk_f32 for i in range(n) ]
    audio_f32 = np.concatenate( aaa )
    audio_f32 = audio_f32[:data_len]
    #result:AudioF32 = np.repeat( signal_f32, n )[:data_len]
    #print(f"result len{len(result)} {data_len} {chunk_len}x{n}")
    return audio_f32


def generate_mixed_tone( duration_sec: float, tone_hz1: int, tone_hz2: int, sample_rate: int, vol: float = 0.3) -> np.ndarray:
    """
    2つの周波数のトーンをエネルギー比50:50で混ぜ合わせ、音声データ全体にわたって同じ音を生成する関数。

    Args:
        tone_hz1 (int): 第一のトーンの周波数（Hz）。
        tone_hz2 (int): 第二のトーンの周波数（Hz）。
        duration_sec (float): 音声データの全体の長さ（秒）。
        sample_rate (int): サンプリングレート（Hz）。
        vol (float, optional): 総合振幅。デフォルトは0.3。

    Returns:
        np.ndarray: 生成された音声データ（float32）。
    """
    # 全体のサンプル数を計算
    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    # 各トーンの振幅を調整（エネルギー比50:50）
    # エネルギー E = (A^2)/2 + (B^2)/2 = vol^2 / 2 + vol^2 / 2 = vol^2
    # 各トーンの振幅は vol / sqrt(2) に設定
    amplitude = vol / np.sqrt(2)

    # 各トーンを生成
    tone1 = amplitude * np.sin(2 * np.pi * tone_hz1 * t)
    tone2 = amplitude * np.sin(2 * np.pi * tone_hz2 * t)

    # トーンを混合
    mixed_tone = tone1 + tone2

    # ノーマライズ
    hi:float = max(abs(mixed_tone))
    rate:float = vol / hi
    mixed_tone *= rate

    # データ型をfloat32に変換
    audio = mixed_tone.astype(np.float32)

    return audio

# 174Hz（ヘルツ）	安定の周波数と呼ばれ、人の内面に働きかけて心を安定させます。
# 285Hz（ヘルツ）	促進の周波数。スピリチュアル性の高い周波数で、自然治癒力を高めて心身を整えてくれます。
# 396Hz（ヘルツ）	解放の周波数といわれ、負の感情に働きかけて自己の解放を促します。
# 417Hz（ヘルツ）	変化の周波数。意識だけではなく無意識にも働きかけ、マイナス思考からの回復を促します。
# 528Hz（ヘルツ）   基本の周波数。副交感神経が優位になり、リラックス効果が期待できます。
# 639Hz（ヘルツ）	調和の周波数。人とのつながりをもたらし、人間関係の向上を助けてくれます。
# 741Hz（ヘルツ）	自由の周波数。表現力を高めてくれ、コミュニケーション力の向上を助けます。
# 852Hz（ヘルツ）	直感の周波数といわれ、脳の松果を活性化させて、洞察力や直感力を高めてくれます。

def add_tone(audio_data: AudioF32, sample_rate: int, frequency: int = 852, level: float = 0.01) -> AudioF32:
    """
    音声データに指定された周波数のトーン（サイン波）を追加し、元の音声データとのバランスを調整する関数。

    Parameters:
    - audio_data (np.ndarray): 元の音声データ（-1.0 から 1.0 の範囲が推奨される）。
    - sample_rate (int): 音声データのサンプルレート（例: 16000）。
    - frequency (int): 追加するトーンの周波数（デフォルトは 852Hz）。
    - level (float): トーンの音量レベルを指定（デフォルトは 0.1）。この値が大きいほどトーンの音量が大きくなる。

    Returns:
    - np.ndarray: トーンが追加された音声データ。
    """
    # 元の音声データの長さに対応する時間軸を生成
    time_axis = np.arange(len(audio_data)) / sample_rate
    
    # 指定された周波数のサイン波を生成
    sine_wave = np.sin(2 * np.pi * frequency * time_axis)
    
    # 元の音声とトーンのバランスを調整
    mixed_audio = (1 - level) * audio_data + level * sine_wave.astype(np.float32)
    
    # 音声データが -1.0 から 1.0 の範囲に収まるようにクリップ
    mixed_audio = np.clip(mixed_audio, -1.0, 1.0)
    
    return mixed_audio

def add_white_noise(audio_data: AudioF32, level: float = 0.01) -> AudioF32:
    """
    音声データにホワイトノイズを追加する関数。

    Parameters:
    - audio_data (np.ndarray): 元の音声データ（-1.0 から 1.0 の範囲が推奨される）。
    - level (float): ノイズの強さを指定。0.0 から 1.0 の間で調整可能（デフォルトは 0.05）。

    Returns:
    - np.ndarray: ホワイトノイズが追加された音声データ。
    """
    # 音声データと同じ長さのホワイトノイズを、標準偏差0.5で生成。
    # 約95％が±1の範囲内に収まる正規分布に基づいたノイズ。
    white_noise = np.random.normal(0, 0.5, len(audio_data)).astype(np.float32)
    
    # ノイズを追加する際のバランス調整
    # 元の音声データの影響を (1 - noise_level) で残し、ノイズは noise_level で加算
    adjusted_audio = (1 - level) * audio_data + level * white_noise
    
    # 最終的に音声データが -1.0 から 1.0 の範囲内に収まるようにクリップ
    noisy_audio = np.clip(adjusted_audio, -1.0, 1.0)
    
    return noisy_audio