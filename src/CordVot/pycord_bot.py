import sys,os,io
import asyncio
from asyncio import Task
from queue import Queue
from typing import NamedTuple
import json

from dotenv import load_dotenv
import discord
from discord.commands.context import ApplicationContext
from discord.commands import Option
from discord.sinks import Sink, Filters
import numpy as np
from numpy.typing import NDArray
import vosk
import librosa

sys.path.append(os.getcwd())
from rec_util import AudioF32, load_wave
from text_to_voice import TtsEngine

class Ukey(NamedTuple):
    gid:int
    uid:int

class AudioSeg(NamedTuple):
    ukey:Ukey
    ctx:ApplicationContext
    data:NDArray[np.float32]

class BufSink(Sink):
    def __init__(self, ctx:ApplicationContext, *, filters=None):
        super().__init__( filters=filters)
        self.ctx:ApplicationContext = ctx
        self.sample_rate:int = 48000
        self.ch:int = 2
        self.sz:int = 2
        self.data_q:Queue[AudioSeg] = Queue()

    def init(self, vc:discord.VoiceClient):  # called under listen
        super().init(vc)
        if vc.decoder:
            self.sample_rate = vc.decoder.SAMPLING_RATE
            self.ch = vc.decoder.CHANNELS
            self.sz = vc.decoder.SAMPLE_SIZE // vc.decoder.CHANNELS
            print(f"buf {self.sample_rate} {self.ch} {self.sz}")
        else:
            print(f"ERROR: buf init, vc is None")

    @Filters.container
    def write(self, data:bytes, user:int):
        i16:np.ndarray = np.frombuffer( data, dtype=np.int16 )
        frames = len(i16)//self.ch
        i2 = i16.reshape((frames,self.ch))
        i1 = i2[:,0]
        f32 = i1.astype(np.float32) / 32768.0
        self.data_q.put(AudioSeg(Ukey(self.ctx.guild_id,user),self.ctx,f32))

    def cleanup(self):
        super().cleanup()

    def qsize(self) ->int:
        return self.data_q.qsize()

    def get_nowait(self) ->AudioSeg|None:
        try:
            if self.data_q.qsize()>0:
                return self.data_q.get_nowait()
        except:
            pass
        return None

    # def get_all_audio(self):
    #     return None

    # def get_user_audio(self, user: snowflake.Snowflake):
    #     return None
MODEL_NAME_SMALL = "vosk-model-small-ja-0.22"
IGNORE2="{\n  \"text\" : \"\"\n}"
IGNORE1="{\n  \"partial\" : \"\"\n}"
IGNORE3="{\"text\": \"\"}"

def strip_vosk_text( text:str ) ->str|None:
    if not text:
        return None
    text = text.strip() if isinstance(text,str) else ''
    if not text or len(text)==1:
        return None
    return text.replace(' ','')

class UserVosk:
    def __init__(self,ctx:ApplicationContext, model,sr):
        self.ctx:ApplicationContext = ctx
        self.recog = vosk.KaldiRecognizer( model, sr )
        self._drty:bool = False

    def AcceptWaveform(self,data):
        self._drty = True
        return self.recog.AcceptWaveform(data)
    
    def FinalResult(self):
        if self._drty:
            txt = self.recog.FinalResult()
            self._drty = False
            self.recog.Reset()
            if txt == IGNORE2:
                return IGNORE3
            return txt
        else:
            return IGNORE3

    def Result(self):
        return self.recog.Result()

    def PartialResult(self):
        return self.recog.PartialResult()

class MyBot(discord.Bot):
    def __init__(self):
        super().__init__()
        self._add_commands()
        self.vosk_task:Task|None = None
        self.buf:BufSink|None = None
        self.vc:discord.VoiceClient|None = None
        # vosk
        self.vosk_model_name:str = MODEL_NAME_SMALL
        self.vosk_model:vosk.Model|None = None
        self.vosk_map:dict = {}
        self.load_vosk_model()
        #
        self.tts:TtsEngine = TtsEngine()

    def load_vosk_model(self) ->vosk.Model:
        # VOSKモデルの読み込み
        model = self.vosk_model
        if model is None:
            print(f"vosk load {self.vosk_model_name}")
            model = vosk.Model(model_name=self.vosk_model_name)
            self.vosk_model = model
        return model

    async def uid_to_name(self,ukey:Ukey) ->str:
        guild = self.get_guild(ukey.gid)
        if guild is None:
            try:
                guild = await self.fetch_guild(ukey.gid)
            except:
                guild = None
        if guild is not None:
            member = guild.get_member(ukey.uid)
            if member is None:
                try:
                    member = await guild.fetch_member(ukey.uid)
                except:
                    member = None
            if member is not None:
                return member.display_name

        user = self.get_user(ukey.uid)
        if user is not None:
            return user.name
        try:
            user = await self.fetch_user(ukey.uid)
            return user.name
        except:
            return f"@{ukey.uid}"

    async def on_ready(self):
        print(f"on_ready")
        if self.vosk_task is None:
            self.vosk_task = asyncio.create_task(self.vosk_loop())
            print(f"vosk started")
    
    async def on_disconnect(self):
        print(f"on_disconnect")
        if self.vosk_task is not None:
            self.vosk_task.cancel()
            try:
                await self.vosk_task
            except asyncio.CancelledError:
                print(f"vosk stopped")
            self.vosk_task = None

    async def vosk_loop(self):
        user_data:dict[Ukey,list[NDArray[np.float32]]] = {}
        recog_map:dict[Ukey,UserVosk] = {}
        while True:
            for ii in range(3):
                seg:AudioSeg|None = self.buf.get_nowait() if self.buf is not None else None
                if seg is not None:
                    break
                await asyncio.sleep(0.1)
            
            if seg is None or self.buf is None:
                for ukey,recog in recog_map.items():
                    txt = recog.FinalResult()
                    if txt != IGNORE3:
                        j = json.loads(txt)
                        msg = strip_vosk_text( j.get('text') )
                        if msg:
                            await self.abc_mesg( recog.ctx, ukey, msg )
                continue

            ukey=seg.ukey
            data_list = user_data.get(ukey)
            if data_list is None:
                user_data[ukey] = data_list = []
            data_list.append(seg.data)
            total = sum( [len(a) for a in data_list])
            sample_rate = self.buf.sample_rate
            total_sec = total/sample_rate
            if total_sec<0.2:
                continue
            user_data[ukey] = []
            orig_f32:NDArray[np.float32] = np.concatenate(data_list)
            target_f32 = librosa.resample( orig_f32, orig_sr = sample_rate, target_sr=16000)
            audio_bytes = (target_f32*32767).astype(np.int16).tobytes()

            if ukey in recog_map:
                recog:UserVosk = recog_map[ukey]
            else:
                model = self.load_vosk_model()
                print(f"vosk create KaldiRecognizer")
                recog = UserVosk( seg.ctx, model, 16000 )
                recog_map[ukey] = recog

            if recog.AcceptWaveform( audio_bytes ):
                txt = recog.Result()
                if txt != IGNORE2:
                    j = json.loads(txt)
                    msg = strip_vosk_text( j.get('text') )
                    if msg:
                        await self.abc_mesg( recog.ctx, ukey, msg )
            # else:
            #     txt = recog.PartialResult()
            #     if txt != IGNORE1:
            #         print(f"VOSK partial {txt}")

    def _add_commands(self):
        @self.slash_command(description="ボイスチャンネルに参加します。")
        async def join( ctx:ApplicationContext):
            print(f"join {type(ctx)}")
            if ctx.author.voice is None:
                embed = discord.Embed(title="エラー",description="あなたがボイスチャンネルに参加していません。",color=discord.Colour.red())
                await ctx.respond(embed=embed)
                return
            if ctx.guild.voice_client is None or ctx.voice_client is None:
                try:
                    await ctx.author.voice.channel.connect()
                except:
                    embed = discord.Embed(title="エラー",description="ボイスチャンネルに接続できません。\nボイスチャンネルの権限を確認してください。",color=discord.Colour.red())
                    await ctx.respond(embed=embed)
                    return
            if ctx.guild.voice_client is None or ctx.voice_client is None:
                embed = discord.Embed(title="エラー",description="ボイスチャンネルに接続していません。",color=discord.Colour.red())
                await ctx.respond(embed=embed)
                return
            ctx.guild_id
            self.buf = BufSink(ctx)
            ctx.voice_client.start_recording(self.buf, self.vosk_finished_callback, ctx)
            embed = discord.Embed(title="成功",description="ボイスチャンネルに参加しました。",color=discord.Colour.green())
            await ctx.respond(embed=embed)

        @self.slash_command(description="ボイスチャンネルから切断します。")
        async def leave(ctx:ApplicationContext):
            if ctx.guild.voice_client is None:
                embed = discord.Embed(title="エラー",description="ボイスチャンネルに接続していません。",color=discord.Colour.red())
                await ctx.respond(embed=embed)
                return
            await ctx.guild.voice_client.disconnect()
            embed = discord.Embed(title="成功",description="ボイスチャンネルから切断しました。",color=discord.Colour.green())
            await ctx.respond(embed=embed)

    async def vosk_finished_callback(self, sink, ctx, *args):
        pass

    async def finished_callback(self,sink, ctx, *args):
        recorded_users = [f"<@{user_id}>" for user_id, audio in sink.audio_data.items()]
        files = [discord.File(audio.file, f"{user_id}.{sink.encoding}") for user_id, audio in sink.audio_data.items()]
        await ctx.respond(f"録音が完了しました！\n録音されたユーザー: {', '.join(recorded_users)}.", files=files)

    async def abc_mesg(self, ctx:ApplicationContext, uid:Ukey, mesg ):
        uname = await self.uid_to_name(uid)
        print(f"mesg {uid.gid} {uid.uid} {uname} {mesg}")
        vc:discord.VoiceClient = ctx.guild.voice_client
        if vc is None:
            await ctx.respond('ボイスチャンネルに未接続')
            return
        

        # 正弦波の音声データをBytesIOオブジェクトで生成
        #sine_wave_buffer = generate_sine_wave_bytes()
        f32q, model = self.tts._text_to_audio_by_voicevox( f"あのね、{mesg}ってなんだよ", sampling_rate=48000 )
        if f32q is None:
            return
        # ステレオに変換（左右チャンネルに同じデータをコピー）
        f32 = np.stack((f32q, f32q), axis=-1)
        b_i16 = (f32*32767.0).astype(np.int16)
        buffer = io.BytesIO()
        buffer.write(b_i16.tobytes())
        buffer.seek(0)  # 読み込み用にポインタを先頭に戻す

        # BytesIOを直接PCMAudioで再生
        audio_source = discord.PCMAudio(buffer)
        if not vc.is_playing():
            vc.play(audio_source, after=lambda e: print("再生終了:", e))

        # 再生中メッセージを送信
        #await ctx.respond("1秒間の正弦波を再生しています。")

def generate_sine_wave_bytes(duration=1):
    SAMPLE_RATE = 48000
    FREQUENCY = 440
    # 正弦波を生成してBytesIOオブジェクトに保存
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * FREQUENCY * t)
    waveform_integers = np.int16(waveform * 32767)  # int16に変換

    # BytesIOにバイナリデータとして書き込み
    buffer = io.BytesIO()
    buffer.write(waveform_integers.tobytes())
    buffer.seek(0)  # 読み込み用にポインタを先頭に戻す
    return buffer

def main():
    # .env ファイルから環境変数を読み込む
    load_dotenv("discord.env")

    # 環境変数からトークンを取得
    TOKEN = os.getenv('DISCORD_BOT_TOKEN')

    if TOKEN is None:
        print("Error: DISCORD_BOT_TOKEN is not set in discord.env")

    bot = MyBot()
    bot.run(TOKEN)

if __name__ == "__main__":
    main()