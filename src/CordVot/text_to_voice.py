import sys,os,traceback,re
import json
import wave
from io import BytesIO
import time
import numpy as np
import requests
from requests.adapters import HTTPAdapter
import httpx
import asyncio
from typing import Optional, Tuple

sys.path.append(os.getcwd())
from rec_util import AudioF32, load_wave
from net_utils import a_find_first_responsive_host, find_first_responsive_host
# from ..translate import convert_to_katakana, convert_kuten

from logging import getLogger
logger = getLogger(__name__)

def _to_int( value, default:int ) -> int:
    try:
        val = int(value)
        if val>0:
            return val
    except:
        pass
    return default

class TtsEngine:
    EOT:str = "<|EOT|>"
    # 男性はM 女性はF
    VoiceList:list[tuple[str,int,str,str]] = [
        ( "VOICEVOX:四国めたん[ノーマル]",  2, 'ja_JP', 'F' ),
        ( "VOICEVOX:四国めたん[あまあま]",  0, 'ja_JP', 'F' ),
        ( "VOICEVOX:四国めたん[ツンツン]",  6, 'ja_JP', 'F' ),
        ( "VOICEVOX:四国めたん[セクシー]",  4, 'ja_JP', 'F' ),
        ( "VOICEVOX:四国めたん[ささやき]", 36, 'ja_JP', 'F' ),
        ( "VOICEVOX:四国めたん[ヒソヒソ]", 37, 'ja_JP', 'F' ),
        ( "VOICEVOX:ずんだもん[ノーマル]",  3, 'ja_JP', 'F' ),
        ( "VOICEVOX:ずんだもん[あまあま]",  1, 'ja_JP', 'F' ),
        ( "VOICEVOX:ずんだもん[ツンツン]",  7, 'ja_JP', 'F' ),
        ( "VOICEVOX:ずんだもん[セクシー]",  5, 'ja_JP', 'F' ),
        ( "VOICEVOX:ずんだもん[ささやき]", 22, 'ja_JP', 'F' ),
        ( "VOICEVOX:ずんだもん[ヒソヒソ]", 38, 'ja_JP', 'F' ),
        ( "VOICEVOX:ずんだもん[ヘロヘロ]", 75, 'ja_JP', 'F' ),
        ( "VOICEVOX:ずんだもん[なみだめ]", 76, 'ja_JP', 'F' ),
        ( "VOICEVOX:春日部つむぎ[ノーマル]",  8, 'ja_JP', 'F' ),
        ( "VOICEVOX:雨晴はう[ノーマル]",  10, 'ja_JP', 'F' ),
        ( "VOICEVOX:波音リツ[ノーマル]",   9, 'ja_JP', 'F' ),
        ( "VOICEVOX:波音リツ[クイーン]",  65, 'ja_JP', 'F' ),
        ( "VOICEVOX:玄野武宏[ノーマル]",  11, 'ja_JP', 'M' ),
        ( "VOICEVOX:玄野武宏[喜び]",     39, 'ja_JP', 'M' ),
        ( "VOICEVOX:玄野武宏[ツンギレ]",  40, 'ja_JP', 'M' ),
        ( "VOICEVOX:玄野武宏[悲しみ]",   41, 'ja_JP', 'M' ),
        ( "VOICEVOX:白上虎太郎[ふつう]",  12, 'ja_JP', 'M' ),
        ( "VOICEVOX:白上虎太郎[わーい]",  32, 'ja_JP', 'M' ),
        ( "VOICEVOX:白上虎太郎[びくびく]", 33, 'ja_JP', 'M' ),
        ( "VOICEVOX:白上虎太郎[おこ]",   34, 'ja_JP', 'M' ),
        ( "VOICEVOX:白上虎太郎[びえーん]", 35, 'ja_JP', 'M' ),
        ( "VOICEVOX:青山龍星[ノーマル]",  13, 'ja_JP', 'M' ),
        ( "VOICEVOX:青山龍星[熱血]",    81, 'ja_JP', 'M' ),
        ( "VOICEVOX:青山龍星[不機嫌]",   82, 'ja_JP', 'M' ),
        ( "VOICEVOX:青山龍星[喜び]",    83, 'ja_JP', 'M' ),
        ( "VOICEVOX:青山龍星[しっとり]",   84, 'ja_JP', 'M' ),
        ( "VOICEVOX:青山龍星[かなしみ]",   85, 'ja_JP', 'M' ),
        ( "VOICEVOX:青山龍星[囁き]",    86, 'ja_JP', 'M' ),
        ( "VOICEVOX:冥鳴ひまり[ノーマル]", 14, 'ja_JP', 'F' ),
        ( "VOICEVOX:九州そら[ノーマル]", 16, 'ja_JP', 'F' ),
        ( "VOICEVOX:九州そら[あまあま]", 15, 'ja_JP', 'F' ),
        ( "VOICEVOX:九州そら[ツンツン]", 18, 'ja_JP', 'F' ),
        ( "VOICEVOX:九州そら[セクシー]", 17, 'ja_JP', 'F' ),
        ( "VOICEVOX:九州そら[ささやき]", 19, 'ja_JP', 'F' ),
        ( "VOICEVOX:もち子さん[ノーマル]", 20, 'ja_JP', 'F' ),
        ( "VOICEVOX:もち子さん[セクシー／あん子]", 66, 'ja_JP', 'F' ),
        ( "VOICEVOX:もち子さん[泣き]", 77, 'ja_JP', 'F' ),
        ( "VOICEVOX:もち子さん[怒り]", 78, 'ja_JP', 'F' ),
        ( "VOICEVOX:もち子さん[喜び]", 79, 'ja_JP', 'F' ),
        ( "VOICEVOX:もち子さん[のんびり]", 80, 'ja_JP', 'F' ),
        ( "VOICEVOX:剣崎雌雄[ノーマル]", 21, 'ja_JP', 'M' ),
        ( "VOICEVOX:WhiteCUL[ノーマル]", 23, 'ja_JP', 'F' ),
        ( "VOICEVOX:WhiteCUL[たのしい]", 24, 'ja_JP', 'F' ),
        ( "VOICEVOX:WhiteCUL[かなしい]", 25, 'ja_JP', 'F' ),
        ( "VOICEVOX:WhiteCUL[びえーん]", 26, 'ja_JP', 'F' ),
        ( "VOICEVOX:後鬼[人間ver.]", 27, 'ja_JP', 'M' ),
        ( "VOICEVOX:後鬼[ぬいぐるみver.]", 28, 'ja_JP', 'M' ),
        ( "VOICEVOX:No.7[ノーマル]",  29, 'ja_JP', 'F' ),
        ( "VOICEVOX:No.7[アナウンス]", 30, 'ja_JP', 'F' ),
        ( "VOICEVOX:No.7[読み聞かせ]", 31, 'ja_JP', 'F' ),
        ( "VOICEVOX:ちび式じい[ノーマル]", 42, 'ja_JP', 'M' ),
        ( "VOICEVOX:櫻歌ミコ[ノーマル]", 43, 'ja_JP', 'F' ),
        ( "VOICEVOX:櫻歌ミコ[第二形態]", 44, 'ja_JP', 'F' ),
        ( "VOICEVOX:櫻歌ミコ[ロリ]",    45, 'ja_JP', 'F' ),
        ( "VOICEVOX:小夜/SAYO[ノーマル]", 46, 'ja_JP', 'F' ),
        ( "VOICEVOX:ナースロボ＿タイプＴ[ノーマル]", 47, 'ja_JP', 'F' ),
        ( "VOICEVOX:ナースロボ＿タイプＴ[楽々]", 48, 'ja_JP', 'F' ),
        ( "VOICEVOX:ナースロボ＿タイプＴ[恐怖]", 49, 'ja_JP', 'F' ),
        ( "VOICEVOX:ナースロボ＿タイプＴ[内緒話]", 50, 'ja_JP', 'F' ),
        ( "VOICEVOX:†聖騎士 紅桜†[ノーマル]", 51, 'ja_JP', 'M' ),
        ( "VOICEVOX:雀松朱司[ノーマル]", 52, 'ja_JP', 'M' ),
        ( "VOICEVOX:麒ヶ島宗麟[ノーマル]", 53, 'ja_JP', 'M' ),
        ( "VOICEVOX:春歌ナナ[ノーマル]", 54, 'ja_JP', 'F' ),
        ( "VOICEVOX:猫使アル[ノーマル]", 55, 'ja_JP', 'F' ),
        ( "VOICEVOX:猫使アル[おちつき]", 56, 'ja_JP', 'F' ),
        ( "VOICEVOX:猫使アル[うきうき]", 57, 'ja_JP', 'F' ),
        ( "VOICEVOX:猫使ビィ[ノーマル]", 58, 'ja_JP', 'F' ),
        ( "VOICEVOX:猫使ビィ[おちつき]", 59, 'ja_JP', 'F' ),
        ( "VOICEVOX:猫使ビィ[人見知り]", 60, 'ja_JP', 'F' ),
        ( "VOICEVOX:中国うさぎ[ノーマル]", 61, 'ja_JP', 'F' ),
        ( "VOICEVOX:中国うさぎ[おどろき]", 62, 'ja_JP', 'F' ),
        ( "VOICEVOX:中国うさぎ[こわがり]", 63, 'ja_JP', 'F' ),
        ( "VOICEVOX:中国うさぎ[へろへろ]", 64, 'ja_JP', 'F' ),
        ( "VOICEVOX:栗田まろん[ノーマル]", 67, 'ja_JP', 'F' ),
        ( "VOICEVOX:あいえるたん[ノーマル]", 68, 'ja_JP', 'F' ),
        ( "VOICEVOX:満別花丸[ノーマル]", 69, 'ja_JP', 'F' ),
        ( "VOICEVOX:満別花丸[元気]", 70, 'ja_JP', 'F' ),
        ( "VOICEVOX:満別花丸[ささやき]", 71, 'ja_JP', 'F' ),
        ( "VOICEVOX:満別花丸[ぶりっ子]", 72, 'ja_JP', 'F' ),
        ( "VOICEVOX:満別花丸[ボーイ]", 73, 'ja_JP', 'F' ),
        ( "VOICEVOX:琴詠ニア[ノーマル]", 74, 'ja_JP', 'F' ),
    ]

    @staticmethod
    def id_to_model( idx:int ) -> tuple[str,int,str,str]|None:
        return next((voice for voice in TtsEngine.VoiceList if voice[1] == idx), None )

    @staticmethod
    def id_to_name( idx:int ) -> str:
        voice = TtsEngine.id_to_model( idx )
        name = voice[0] if voice else None
        return name if name else '???'

    @staticmethod
    def id_to_lang( idx:int ) -> str:
        voice = TtsEngine.id_to_model( idx )
        lang = voice[2] if voice else None
        return lang if lang else 'ja_JP'

    @staticmethod
    def id_to_gender( idx:int ) -> str:
        voice = TtsEngine.id_to_model( idx )
        gender = voice[3] if voice else None
        return gender if gender else 'X'

    def __init__(self, *, config:dict={}, katakana_dir='tmp/katakana' ):
        # 発声中のセリフのID
        self._talk_id: int = 0
        self._talk_seq: int = 0
        # 音声エンジン選択
        self.speaker = 8
        self.speed = 1.0
        self.pitch = 0.0
        VOICEVOX_SPAKER = config.get('voicevox_speaker')
        if isinstance(VOICEVOX_SPAKER,int):
            self.speaker = VOICEVOX_SPAKER
        VOICEVOX_SPEED = config.get('voicevox_speed')
        if isinstance(VOICEVOX_SPEED,int|float):
            self.speed = float(VOICEVOX_SPEED)
        VOICEVOX_PITCH = config.get('voicevox_pitch')
        if isinstance(VOICEVOX_PITCH,int|float):
            self.pitch = float(VOICEVOX_PITCH)
        # VOICEVOXサーバURL
        self._voicevox_url:str|None = None
        self._voicevox_port = _to_int( os.getenv('VOICEVOX_PORT' ), 50021)
        self._voicevox_list = list(set([os.getenv('VOICEVOX_HOST','127.0.0.1'),'127.0.0.1','192.168.0.104','chickennanban.ddns.net','chickennanban1.ddns.net','chickennanban2.ddns.net','chickennanban3.ddns.net']))
        self._katakana_dir = katakana_dir

    async def a_get_voicevox_url( self ) ->str|None:
        if self._voicevox_url is None:
            self._voicevox_url = await a_find_first_responsive_host(self._voicevox_list,self._voicevox_port)
        return self._voicevox_url

    def _get_voicevox_url( self ) ->str|None:
        if self._voicevox_url is None:
            self._voicevox_url = find_first_responsive_host(self._voicevox_list,self._voicevox_port)
        return self._voicevox_url

    @staticmethod
    def remove_code_blocksRE(markdown_text):
        # 正規表現を使用してコードブロックを検出し、それらを改行に置き換えます
        # ```（コードブロックの開始と終了）に囲まれた部分を検出します
        # 正規表現のパターンは、```で始まり、任意の文字（改行を含む）にマッチし、最後に```で終わるものです
        # re.DOTALLは、`.`が改行にもマッチするようにするフラグです
        pattern = r'```.*?```'
        return re.sub(pattern, '\n', markdown_text, flags=re.DOTALL)

    @staticmethod
    def split_talk_text( text):
        sz = len(text)
        st = 0
        lines = []
        while st<sz:
            block_start = text.find("```",st)
            newline_pos = text.find('\n',st)
            if block_start>=0 and ( newline_pos<0 or block_start<newline_pos ):
                if st<block_start:
                    lines.append( text[st:block_start] )
                block_end = text.find( "```", block_start+3)
                if (block_start+3)<block_end:
                    block_end += 3
                else:
                    block_end = sz
                lines.append( text[block_start:block_end])
                st = block_end
            else:
                if newline_pos<0:
                    newline_pos = sz
                if st<newline_pos:
                    lines.append( text[st:newline_pos] )
                st = newline_pos+1
        return lines
    
    @staticmethod
    def __penpenpen( text, default=" " ) ->str:
        if text is None or text.startswith("```"):
            return default # VOICEVOX,OpenAI,gTTSで、エラーにならない無音文字列
        else:
            return text

    async def a_text_to_audio_by_voicevox(self, text: str, *, sampling_rate: int) -> tuple[AudioF32|None, str|None]:
        sv_url: Optional[str] = await self.a_get_voicevox_url()
        if sv_url is None:
            return None, None
        try:
            # textの変換処理
            text = TtsEngine.__penpenpen(text, ' ')
            timeout = httpx.Timeout(5.0, read=180.0)
            params = {'text': text, 'speaker': self.speaker}
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                # audio_queryエンドポイントへのPOSTリクエスト
                res1 = await client.post(f'{sv_url}/audio_query', params=params)
                res1.raise_for_status()
                res1_json = res1.json()

                # パラメータの調整
                ss: float = res1_json.get('speedScale', 1.0)
                res1_json['speedScale'] = ss * self.speed
                ps: float = res1_json.get('pitchScale', 0.0)
                res1_json['pitchScale'] = ps + self.pitch

                # synthesisエンドポイントへのPOSTリクエスト
                data = json.dumps(res1_json, ensure_ascii=False)
                headers = {'content-type': 'application/json'}
                res = await client.post(
                    f'{sv_url}/synthesis',
                    json=res1_json,
                    params={'speaker': self.speaker},
                    headers=headers
                )
                res.raise_for_status()

                # レスポンスを処理
                if res.status_code == 200:
                    model: str = TtsEngine.id_to_name(self.speaker)
                    f32 = load_wave(res.content, sampling_rate=sampling_rate)
                    return f32, model

                logger.error(f"[VOICEVOX] code:{res.status_code} {res.text}")

        except httpx.ConnectTimeout as ex:
            logger.error(f"[VOICEVOX] {type(ex)} {ex}")
        except httpx.RequestError as ex:
            logger.error(f"[VOICEVOX] {type(ex)} {ex}")
        except Exception as ex:
            logger.error(f"[VOICEVOX] {type(ex)} {ex}")
            logger.exception('')
        
        self._disable_voicevox = time.time()
        return None, None

    def _text_to_audio_by_voicevox(self, text:str, *, sampling_rate:int) -> tuple[AudioF32|None,str|None]:
        sv_url: str|None = self._get_voicevox_url()
        if sv_url is None:
            return None,None
        try:
            # text = convert_to_katakana(text,cache_dir=self._katakana_dir)
            # text = convert_kuten(text)
            text = TtsEngine.__penpenpen(text, ' ')
            timeout = (5.0,180.0)
            params = {'text': text, 'speaker': self.speaker, 'timeout': timeout }
            s:requests.Session = requests.Session()
            s.mount(f'{sv_url}/audio_query', HTTPAdapter(max_retries=1))
            res1 : requests.Response = requests.post( f'{sv_url}/audio_query', params=params)
            data = res1.content
            res1_json:dict = json.loads(data)
            ss:float = res1_json.get('speedScale',1.0)
            res1_json['speedScale'] = ss*1.1
            ps:float = res1_json.get('pitchScale',0.0)
            res1_json['pitchScale'] = ps-0.1
            data = json.dumps(res1_json,ensure_ascii=False)
            params = {'speaker': self.speaker, 'timeout': timeout }
            headers = {'content-type': 'application/json'}
            res = requests.post(
                f'{sv_url}/synthesis',
                data=data,
                params=params,
                headers=headers
            )
            if res.status_code == 200:
                model:str = TtsEngine.id_to_name(self.speaker)
                # wave形式 デフォルトは24kHz
                f32 = load_wave( res.content, sampling_rate=sampling_rate )
                return f32, model
            logger.error( f"[VOICEVOX] code:{res.status_code} {res.text}")
        except requests.exceptions.ConnectTimeout as ex:
            logger.error( f"[VOICEVOX] {type(ex)} {ex}")
        except requests.exceptions.ConnectionError as ex:
            logger.error( f"[VOICEVOX] {type(ex)} {ex}")
        except Exception as ex:
            logger.error( f"[VOICEVOX] {type(ex)} {ex}")
            logger.exception('')
        self._disable_voicevox = time.time()
        return None,None

    @staticmethod
    def convert_blank( text:str ) ->str:
        text = re.sub( r'[「」・、。]+',' ',text)
        return text.strip()

def main():
    config={ 'voicevox_speaker': 81 }
    tts:TtsEngine = TtsEngine(config=config)
    a,model = tts._text_to_audio_by_voicevox('あいうえお',sampling_rate=16000)
    if a is not None:
        print( min(a) )

if __name__ == "__main__":
    main()




