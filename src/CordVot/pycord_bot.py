import sys,os,io
import time
import traceback
import asyncio
from asyncio import Task, Queue as Aqueue
from queue import Queue
from typing import Type, Optional, NamedTuple
import json

from dotenv import load_dotenv
import discord
from discord import Message, Member, User
from discord.channel import TextChannel, VoiceChannel
from discord.commands.context import ApplicationContext
from discord.commands import Option
from discord.sinks import Sink, Filters
import numpy as np
from numpy.typing import NDArray
import vosk
import librosa

sys.path.append(os.getcwd())
from rec_util import AudioF32, load_wave, reverb, compressor
from text_to_voice import TtsEngine
from llm import LLM

class sessionId(NamedTuple):
    gid:int # guild id
    cid:int # channel id

    @staticmethod
    def from_ctx(ctx:ApplicationContext|None) ->Optional["sessionId"]:
        if isinstance(ctx,ApplicationContext):
            gid = ctx.guild_id
            cid = ctx.channel_id
            if gid and cid:
                return sessionId(gid,cid)
        return None

    @staticmethod
    def from_message(message:Message|None) ->Optional["sessionId"]:
        if isinstance(message,Message):
            gid = message.guild.id if message.guild else None
            cid = message.channel.id
            if gid and cid:
                return sessionId(gid,cid)
        return None

    @staticmethod
    def from_channel(ch) ->Optional["sessionId"]:
        try:
            gid = ch.guild.id if ch.guild else None
            cid = ch.id
            if gid and cid:
                return sessionId(gid,cid)
        except:
            pass
        return None

class Ukey(NamedTuple):
    sid:sessionId
    uid:int

class AudioSeg(NamedTuple):
    ukey:Ukey
    data:NDArray[np.float32]

class BufSink(Sink):
    def __init__(self, sid:sessionId, *, filters=None):
        super().__init__( filters=filters)
        self.sid:sessionId = sid
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
        self.data_q.put(AudioSeg(Ukey(self.sid,user),f32))

    def cleanup(self):
        super().cleanup()

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
VOSK_IGNORE_WARDS = {
    'えー', 'えええ', 'あっ'
}

def strip_vosk_text( text:str ) ->str|None:
    text = text.strip() if isinstance(text,str) else ''
    if len(text)==0:
        return None
    results = ''
    for w in text.split():
        if len(w)>1 and not w in VOSK_IGNORE_WARDS:
            results += w
    return results if len(results)>0 else None

class UserRecognizer:
    def __init__(self,uid:int, model,sr):
        self.ukey:int = uid
        self.recog = vosk.KaldiRecognizer( model, sr )
        self._drty:bool = False
        self._last_use:float = time.time()

    def AcceptWaveform(self,data) ->bool:
        self._last_use:float = time.time()
        self._drty = True
        return self.recog.AcceptWaveform(data)
    
    def FinalResult(self) ->str|None:
        try:
            if self._drty:
                self._drty = False
                self._last_use:float = time.time()
                dat = json.loads(self.recog.FinalResult())
                self.recog.Reset()
                return strip_vosk_text( dat.get('text') )
        except:
            pass
        return None

    def Result(self) ->str|None:
        try:
            dat = json.loads(self.recog.Result())
            return strip_vosk_text( dat.get('text') )
        except:
            return None

    def PartialResult(self):
        try:
            dat = json.loads(self.recog.PartialResult())
            return strip_vosk_text( dat.get('text') )
        except:
            return None

class MyBot(discord.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self._add_commands()
        # session
        self._session_map:dict[sessionId,BotSession] = {}
        #
        self.vc:discord.VoiceClient|None = None
        # vosk
        self.vosk_model_name:str = MODEL_NAME_SMALL
        self.vosk_model:vosk.Model|None = None
        self.vosk_map:dict = {}
        self.load_vosk_model()
        #
        self.tts:TtsEngine = TtsEngine()

    def _get_session(self, ctx:ApplicationContext|sessionId|None) -> Optional["BotSession"]:
        if isinstance(ctx,sessionId):
            return self._session_map.get(ctx)
        if not isinstance(ctx,ApplicationContext):
            return None
        sid:sessionId|None = sessionId.from_ctx(ctx)
        if sid is None:
            return None
        session:BotSession|None = self._session_map.get(sid)
        if session is not None:
            return session
        session = BotSession(self, ctx.channel )
        self._session_map[sid] = session
        return session

    def _get_session2(self, ch ) -> Optional["BotSession"]:
        sid:sessionId|None = sessionId.from_channel(ch)
        if sid is None:
            return None
        session:BotSession|None = self._session_map.get(sid)
        if session is not None:
            return session
        session = BotSession(self, ch, None )
        self._session_map[sid] = session
        return session

    def load_vosk_model(self) ->vosk.Model:
        # VOSKモデルの読み込み
        model = self.vosk_model
        if model is None:
            print(f"vosk load {self.vosk_model_name}")
            vosk.SetLogLevel(-1)
            model = vosk.Model(model_name=self.vosk_model_name)
            self.vosk_model = model
        return model

    async def uid_to_name(self,gid:int,uid:int) ->str:
        guild = self.get_guild(gid)
        if guild is None:
            try:
                guild = await self.fetch_guild(gid)
            except:
                guild = None
        if guild is not None:
            member = guild.get_member(uid)
            if member is None:
                try:
                    member = await guild.fetch_member(uid)
                except:
                    member = None
            if member is not None:
                return member.display_name

        user = self.get_user(uid)
        if user is not None:
            return user.name
        try:
            user = await self.fetch_user(uid)
            return user.name
        except:
            return f"@{uid}"

    async def on_ready(self):
        print(f"on_ready")
    
    async def on_disconnect(self):
        print(f"on_disconnect")
        for k,v in self._session_map.items():
            await v.stop()

    async def on_message(self, mesg:Message ):
        try:
            if mesg.author == self.user:
                return
            uid = mesg.author.id
            content = mesg.content
            session:BotSession|None = self._get_session2(mesg.channel)
            if session:
                if uid and content:
                    await session.on_message( uid, content, False )

        except Exception as ex:
            print(f"ERROR {ex}")

    def _add_commands(self):
        @self.slash_command(description="ボイスチャンネルに参加します。")
        async def join( ctx:ApplicationContext):
            try:
                session:BotSession|None = self._get_session(ctx)
                if session is None:
                    embed = discord.Embed(title="エラー",description="あなたがボイスチャンネルに参加していません。",color=discord.Colour.red())
                    await ctx.respond(embed=embed)
                    return
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
                session.vc = ctx.voice_client
                await session.start()
                session.buf = BufSink(session.sid)
                ctx.voice_client.start_recording(session.buf, self.vosk_finished_callback, ctx)
                embed = discord.Embed(title="成功",description="ボイスチャンネルに参加しました。",color=discord.Colour.green())
                await ctx.respond(embed=embed)
            except Exception as ex:
                print(f"ERROR {ex}")

        @self.slash_command(description="ボイスチャンネルから切断します。")
        async def leave(ctx:ApplicationContext):
            try:
                if ctx.guild.voice_client is None:
                    embed = discord.Embed(title="エラー",description="ボイスチャンネルに接続していません。",color=discord.Colour.red())
                    await ctx.respond(embed=embed)
                    return
                await ctx.guild.voice_client.disconnect()
                embed = discord.Embed(title="成功",description="ボイスチャンネルから切断しました。",color=discord.Colour.green())
                await ctx.respond(embed=embed)
            except Exception as ex:
                print(f"ERROR {ex}")

    async def vosk_finished_callback(self, sink, ctx, *args):
        pass

class VoiceRes(NamedTuple):
    mesg:str
    audio:bytes|None

class BotSession:

    def __init__(self, bot:MyBot, ch, vc:discord.VoiceClient|None=None):
        self.bot:MyBot = bot
        sid:sessionId|None = sessionId.from_channel(ch)
        if sid is None:
            raise ValueError("can not get sessionId from context")
        self.sid:sessionId = sid
        self.ch = ch
        self.vc:discord.VoiceClient|None = vc
        self._response_q:Aqueue[VoiceRes] = Aqueue()
        self._task:Task|None = None
        self.vosk_task:Task|None = None
        self.buf:BufSink|None = None

    async def start(self):
        if self.vosk_task is None:
            self.vosk_task = asyncio.create_task( self._th_vosk_loop() )

    async def stop(self):
        if self.vosk_task is not None:
            self.vosk_task.cancel()
            try:
                await self.vosk_task
            except asyncio.CancelledError:
                print(f"vosk stopped")
            self.vosk_task = None

    def is_voice(self):
        try:
            if self.vc is not None and self.vc.is_connected():
                return True
        except:
            pass
        return False

    async def _th_vosk_loop(self):
        user_data:dict[int,list[NDArray[np.float32]]] = {}
        recog_map:dict[int,UserRecognizer] = {}
        try:
            while True:
                seg:AudioSeg|None
                for ii in range(3):
                    seg = self.buf.get_nowait() if self.buf is not None else None
                    if seg is not None:
                        break
                    await asyncio.sleep(0.1)
                
                if seg is None or self.buf is None:
                    for uid,recog in recog_map.items():
                        txt = recog.FinalResult()
                        if txt:
                            await self.abc_mesg( uid, txt )
                        else:
                            await self.ghi(uid,False)
                    continue

                uid:int=seg.ukey.uid
                data_list = user_data.get(uid)
                if data_list is None:
                    user_data[uid] = data_list = []
                data_list.append(seg.data)
                total = sum( [len(a) for a in data_list])
                sample_rate = self.buf.sample_rate
                total_sec = total/sample_rate
                if total_sec<0.2:
                    continue
                user_data[uid] = []
                orig_f32:NDArray[np.float32] = np.concatenate(data_list)
                target_f32 = librosa.resample( orig_f32, orig_sr = sample_rate, target_sr=16000)
                audio_bytes = (target_f32*32767).astype(np.int16).tobytes()

                if uid in recog_map:
                    recog:UserRecognizer = recog_map[uid]
                else:
                    model = self.bot.load_vosk_model()
                    print(f"vosk create KaldiRecognizer")
                    recog = UserRecognizer( uid, model, 16000 )
                    recog_map[uid] = recog

                if recog.AcceptWaveform( audio_bytes ):
                    txt = recog.Result()
                    if txt:
                        await self.abc_mesg( uid, txt )
                    else:
                        await self.ghi(uid,False)
                else:
                    txt = recog.PartialResult()
                    if txt:
                        await self.ghi(uid,True)
                #     if txt != IGNORE1:
                #         print(f"VOSK partial {txt}")
        except:
            traceback.print_exc()
        finally:
            self.vosk_task = None
            # for uid,recog in recog_map.items():
            #     recog.

    async def get_username(self, uid:int) ->str:
        guild = self.ch.guild
        member = guild.get_member(uid)
        if member is None:
            try:
                member = await guild.fetch_member(uid)
            except:
                member = None
        if member is not None:
            return member.display_name
        return f"@{uid}"

    async def abc_mesg(self, uid:int, mesg ):
        gid = self.ch.guild.id
        uname = await self.bot.uid_to_name(gid,uid)
        print(f"mesg {gid} {uid} {uname} {mesg}")

        await self.on_message( uid, mesg, True )

    async def on_message(self, uid:int, mesg:str, echo:bool ):

        if echo:
            username = await self.get_username(uid)
            await self.ch.send(f"User: {username} {mesg}")

        global_messages:list[dict] = []
        llm:LLM = LLM()
        async for ans in llm.th_get_response_from_openai( global_messages, mesg ):
            await self.add_response_queue( ans )

    async def ghi(self, uid:int, st:bool ):
        pass

    async def add_response_queue(self, ans:str ):
        audio_bytes:bytes|None
        if self.is_voice():
            # 正弦波の音声データをBytesIOオブジェクトで生成
            #sine_wave_buffer = generate_sine_wave_bytes()
            voice_f32, model = await self.bot.tts.a_text_to_audio_by_voicevox( ans, sampling_rate=48000 )
            if voice_f32 is not None:
                audio_f32 = reverb( compressor(voice_f32) )
                # ステレオに変換（左右チャンネルに同じデータをコピー）
                f32 = np.stack((audio_f32, audio_f32), axis=-1)
                b_i16 = (f32*32767.0).astype(np.int16)
                audio_bytes = b_i16.tobytes()
        else:
            audio_bytes = None
        
        buffer = VoiceRes( ans, audio_bytes )
        await self._response_q.put( buffer )
        await asyncio.sleep(0.1)
        if self._task is None:
            self._task = asyncio.create_task( self._th_talk_task() )

    async def _th_talk_task(self):
        try:
            x = self._response_q.get_nowait()
            if x.audio is None:
                await self.talk( x.mesg )
            elif self.vc is not None:
                buffer = io.BytesIO()
                buffer.write(x.audio)
                buffer.seek(0)
                audio_source = discord.PCMAudio(buffer)
                while self.vc.is_playing():
                    await asyncio.sleep(0.1)
                await self.talk( x.mesg )
                self.vc.play(audio_source, after=lambda e: print("再生終了:", e))
        finally:
            self._task = None

    async def talk(self,mesg):
        await self.ch.send( mesg )

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