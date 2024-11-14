import sys,os,io
import time
import copy
import traceback
from io import BytesIO
import wave
from enum import Enum
import asyncio
from asyncio import Task, Queue as Aqueue
from queue import Queue
from typing import Type, Optional, NamedTuple
import json

from dotenv import load_dotenv
import discord.opus
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
from scipy.signal import resample as scipy_resample

sys.path.append(os.getcwd())
from rec_util import AudioF32, AudioI16, EmptyF32, EmptyI16, load_wave, reverb, compressor
from text_to_voice import TtsEngine
from llm import LLM,LLM2

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

class AudioSeg(NamedTuple):
    uid:int
    rcvt:float
    data:NDArray[np.float32]

class AudioBuffer:

    def __init__(self,uid):
        self.uid = uid
        self.rcvt:float = 0
        self.aaa:list[AudioI16] = []

    def put(self, rcvt:float, b:AudioI16 ):
        self.rcvt = rcvt
        self.aaa.append( b )

    def get(self) ->tuple[float,AudioI16|None]:
        now = time.time()
        if len(self.aaa)==0:
            return 0.0, None
        if (now-self.rcvt)<0.8:
            return self.rcvt, EmptyI16
        print(f"<B{len(self.aaa)}>")
        mono_i16:AudioI16 = np.concatenate(self.aaa)
        self.aaa = []
        # ゼロでない要素のインデックスを取得
        non_zero_indices = np.nonzero(mono_i16)[0]
        # スライス範囲を取得
        if non_zero_indices.size == 0:
            return self.rcvt, EmptyI16
        start, end = non_zero_indices[0], non_zero_indices[-1] + 1
        return self.rcvt, mono_i16[start:end]

class BufSink(Sink):
    def __init__(self, *, filters=None):
        super().__init__( filters=filters)
        self.sampling_rate:int = 48000
        self.ch:int = 2
        self.width:int = 2
        self.recvq:Queue = Queue()
        self.data_q:Queue[AudioSeg] = Queue()
        self.vol:float = 1.0
        self._audio_buf:list[AudioF32] = []
        self._zero_count:int = 0
        self._audo_active:bool = True
        self._save_idx:int=1
        self._buf_map:dict[int,AudioBuffer] = {}
        self._savebuf:dict[int,BytesIO] = {}

    def init(self, vc:discord.VoiceClient):  # called under listen
        super().init(vc)
        if vc.decoder:
            self.sampling_rate = vc.decoder.SAMPLING_RATE
            self.ch = vc.decoder.CHANNELS
            self.width = vc.decoder.SAMPLE_SIZE // vc.decoder.CHANNELS
            print(f"buf {self.sampling_rate} {self.ch} {self.width}")
        else:
            print(f"ERROR: buf init, vc is None")

    @Filters.container
    def write(self, audio_bytes:bytes, uid:int):
        try:
            self.recvq.put_nowait( (uid,time.time(),copy.copy(audio_bytes)) )
        except:
            pass

    async def get_nowait(self) ->list[AudioSeg]:
        ret:list[AudioSeg] = []
        try:
            while self.recvq.qsize()>0:
                uid,rcvt,audio_bytes = self.recvq.get_nowait()
                audio_i16 = np.frombuffer( audio_bytes, dtype=np.int16 )
                stereo_i16 = audio_i16.reshape(-1,self.ch).copy()
                mono_i16 = stereo_i16[:,0]
                buf = self._buf_map.get(uid)
                if buf is None:
                    buf = AudioBuffer(uid)
                    self._buf_map[uid] = buf
                buf.put(rcvt,mono_i16)
            
            await asyncio.sleep(0.001)
            for uid,buf in self._buf_map.items():
                rcvt, mono_i16 = buf.get()
                if mono_i16 is not None:
                    audio_f32 = mono_i16.astype(np.float32) / 32768.0
                    if len(mono_i16)>0:
                        self.saveto( uid, mono_i16.tobytes() )
                        amp = 0.8 / np.max(np.abs(audio_f32))
                        audio_f32 *= amp
                    seg = AudioSeg(uid,rcvt,audio_f32)
                    ret.append(seg)
            await asyncio.sleep(0.001)
        except:
            traceback.print_exc()
        return ret

    def saveto(self,uid, b):
        wave_path = f"tmp/audio{self._save_idx:04d}.wav"
        self._save_idx += 1
        print(f"[SINK]save {wave_path}")
        with wave.open(wave_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.width)
            wf.setframerate(self.sampling_rate)
            wf.writeframes(b)

    def cleanup(self):
        super().cleanup()

    # def get_all_audio(self):
    #     return None

    # def get_user_audio(self, user: snowflake.Snowflake):
    #     return None


MODEL_NAME_SMALL = "vosk-model-small-ja-0.22"
MODEL_NAME_LARGE = "vosk-model-ja-0.22"
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
        self.uid:int = uid
        self.recog = vosk.KaldiRecognizer( model, sr )
        self._drty:bool = False
        self._last_use:float = time.time()
        self._pre:int = 4000

    def AcceptWaveform(self,data) ->bool:
        self._last_use:float = time.time()
        if not self._drty:
            self._drty = True
            if len(data)>self._pre:
                data = data[:self._pre] + data
            else:
                data = b'\0'*self._pre + data + data
        return self.recog.AcceptWaveform(data)
    
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

    def FinalResult(self) ->str|None:
        try:
            if self._drty:
                self._drty = False
                self._last_use:float = time.time()
                dat = json.loads(self.recog.FinalResult())
                return strip_vosk_text( dat.get('text') )
        except:
            pass
        finally:
            self.recog.Reset()
        return None

    def Reset(self):
        self._drty = False
        self.recog.Reset()

class MyBot(discord.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self._add_commands()
        # session
        self._session_map:dict[sessionId,BotSession] = {}
        self._text_ch = None
        #
        self.vc:discord.VoiceClient|None = None
        # vosk
        self.vosk_model_name:str = MODEL_NAME_LARGE
        self.vosk_model:vosk.Model|None = None
        self.vosk_map:dict = {}
        self._load_task = None
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

    async def _th_load_vosk_model(self):
        # VOSKモデルの読み込み
        model = self.vosk_model
        if model is None:
            print(f"vosk load {self.vosk_model_name}")
            vosk.SetLogLevel(-1)
            model = vosk.Model(model_name=self.vosk_model_name)
            self.vosk_model = model

    async def get_vosk_model(self) ->vosk.Model:
        if self.vosk_model is None and self._load_task is None:
            self._load_task = asyncio.create_task(self._th_load_vosk_model())
        while self.vosk_model is None:
            await asyncio.sleep(0.1)
        return self.vosk_model

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
        self._load_task = asyncio.create_task(self._th_load_vosk_model())
    
    async def on_disconnect(self):
        print(f"on_disconnect")
        for k,v in self._session_map.items():
            await v.stop()

    async def on_message(self, mesg:Message ):
        try:
            if mesg.author == self.user:
                return
            if self._text_ch is None or self._text_ch != mesg.channel.id:
                return
            uid = mesg.author.id
            content = mesg.content
            session:BotSession|None = self._get_session2(mesg.channel)
            if session:
                if uid and content:
                    await session.on_message( uid, content )

        except Exception as ex:
            print(f"ERROR {ex}")

    def _add_commands(self):
        @self.slash_command(description="テキストチャンネルにBOTを参加")
        async def join( ctx:ApplicationContext):
            self._text_ch = ctx.channel_id
            embed = discord.Embed(title="成功",description="テキストチャンネルに参加しました。",color=discord.Colour.green())
            await ctx.respond(embed=embed)
            return
        @self.slash_command(description="テキストチャンネルにBOTを除外")
        async def leave( ctx:ApplicationContext):
            self._text_ch = None
            embed = discord.Embed(title="成功",description="テキストチャンネルにから抜けました",color=discord.Colour.green())
            await ctx.respond(embed=embed)
            return
        @self.slash_command(description="ボイスチャンネルに参加します。")
        async def voice_on( ctx:ApplicationContext):
            try:
                self._text_ch = ctx.channel_id
                session:BotSession|None = self._get_session(ctx)
                if session is None:
                    embed = discord.Embed(title="エラー",description="事前に、あなたがボイスチャンネルに参加て下さい",color=discord.Colour.red())
                    await ctx.respond(embed=embed)
                    return
                if not isinstance(ctx.author,Member) or ctx.author.voice is None or ctx.author.voice.channel is None:
                    embed = discord.Embed(title="エラー",description="事前に、あなたがボイスチャンネルに参加て下さい",color=discord.Colour.red())
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
                session.voice_client = ctx.voice_client
                await session.start()
                session.sink = BufSink()
                ctx.voice_client.start_recording(session.sink, session.sink_finished_callback )
                embed = discord.Embed(title="成功",description="ボイスチャンネルに参加しました。",color=discord.Colour.green())
                await ctx.respond(embed=embed)
            except Exception as ex:
                print(f"ERROR {ex}")

        @self.slash_command(description="ボイスチャンネルから切断します。")
        async def voice_off(ctx:ApplicationContext):
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

class VoiceStat:
    def __init__(self,uid:int):
        self.uid:int = uid
        self._partial:str = ''
        self._texts:list[str] = []
        self._audio:list[NDArray[np.float32]] = []
        self._rcvt:float = 0
        self.total:int = 0

    def add_audio(self,rcvt:float,audio:NDArray[np.float32]):
        self._audio.append(audio)
        self._rcvt = rcvt
        self.total += len(audio)

    def get_audio(self) ->NDArray[np.float32]:
        a = np.concatenate(self._audio)
        self._audio = []
        self.total = 0
        return a

    def partial(self,msg:str):
        if self._partial or msg:
            print(f"[VOICE] partial {msg}")
        self._partial = msg

    def final(self,msg:str|None):
        if self._partial or (msg and msg.strip()):
            print(f"[VOICE] final {msg}")
        self._partial=''
        if not msg:
            return
        if msg.strip()=='' and len(self._texts)==0:
            return
        self._texts.append(msg)
    
    def have_data(self) ->bool:
        if self._partial or len(self._texts)>0:
            return True
        return False

    def get_data(self) ->str:
        if len(self._texts)>0: # and self._texts[-1]==' ':
            ret = ' '.join(self._texts).strip()
            self._texts = []
            return ret
        return ''

def convert_audio(xaudio,original_rate,target_rate) ->bytes:
    if len(xaudio)==0:
        return b''
    if target_rate == original_rate:
        target_f32 = xaudio
    else:
        num_samples = int(len(xaudio) * target_rate / original_rate)
        target_f32:NDArray[np.float32] = scipy_resample( xaudio, num_samples) # type: ignore
        #target_f32 = librosa.resample( xaudio, orig_sr = original_rate, target_sr=target_rate)
    audio_bytes = (target_f32*32767).astype(np.int16).tobytes()
    return audio_bytes

class VoiceRes(NamedTuple):
    uid:int
    mesg:str
    audio:bytes|None

class BotSession:

    def __init__(self, bot:MyBot, ch, voice_client:discord.VoiceClient|None=None):
        self.bot:MyBot = bot
        sid:sessionId|None = sessionId.from_channel(ch)
        if sid is None:
            raise ValueError("can not get sessionId from context")
        self.sid:sessionId = sid
        self.ch = ch
        self.voice_client:discord.VoiceClient|None = voice_client
        self._response_q:Aqueue[VoiceRes] = Aqueue()
        self._task:Task|None = None
        self._stt_task:Task|None = None
        self._notify_task:Task|None = None
        self.stt_stat:dict[int,VoiceStat] = {}
        self.sink:BufSink|None = None
        #
        self._res_task = None
        #
        self._global_messages:list[dict] = []

    async def start(self):
        if self._stt_task is None:
            self._stt_task = asyncio.create_task( self._th_stt_task() )
        if self._notify_task is None:
            self._notify_task = asyncio.create_task( self._th_notify_task() )

    async def stop(self):
        if self._stt_task is not None:
            self._stt_task.cancel()
            try:
                await self._stt_task
            except asyncio.CancelledError:
                print(f"vosk stopped")
            self._stt_task = None

    def is_voice(self):
        try:
            if self.voice_client is not None and self.voice_client.is_connected():
                return True
        except:
            pass
        return False

    async def _th_stt_task(self):
        target_rate = 48000
        recog_map:dict[int,UserRecognizer] = {}
        seg_sec:float = 1.2
        term_sec:float = 0.8
        try:
            is_connected:bool = True
            idx:int = 0
            while self.sink:
                if self.voice_client is not None and not self.voice_client.is_connected():
                    if is_connected:
                        print(f"voice client is disconnected")
                        is_connected=False
                original_rate:int = self.sink.sampling_rate
                segment_sz:int = int(original_rate*seg_sec) # int(0.2*original_rate)

                seg_list:list[AudioSeg] = await self.sink.get_nowait()
                if len(seg_list)>0:
                    idx = 0
                    for uid,rcvt,data in seg_list:
                        stat = self.stt_stat.get(uid)
                        if stat is None:
                            stat = self.stt_stat[uid] = VoiceStat(uid)
                        stat.add_audio(rcvt,data)
                        if stat.total<segment_sz:
                            continue
                        audio_bytes = convert_audio( stat.get_audio(), original_rate, target_rate )
                        if uid in recog_map:
                            recog:UserRecognizer = recog_map[uid]
                        else:
                            model = await self.bot.get_vosk_model()
                            print(f"vosk create KaldiRecognizer")
                            recog = UserRecognizer( uid, model, target_rate )
                            recog_map[uid] = recog
                        if recog.AcceptWaveform( audio_bytes ):
                            txt = recog.Result()
                            stat.final(txt)
                        else:
                            txt = recog.PartialResult()
                            if txt:
                                stat.partial(txt)
                    await asyncio.sleep(0.03)
                else:
                    idx+=1
                    now = time.time()
                    for uid,stat in self.stt_stat.items():
                        t = now - stat._rcvt
                        precog:UserRecognizer|None = recog_map.get(uid)
                        if stat.total == 0 and (precog is None or not precog._drty):
                            continue
                        if t<=term_sec:
                            continue
                        if stat.total>0:
                            print(f"|{t}|",end="")
                            audio_bytes = convert_audio( stat.get_audio(), original_rate, target_rate )
                            if precog is None:
                                model = await self.bot.get_vosk_model()
                                print(f"vosk create KaldiRecognizer")
                                precog = UserRecognizer( uid, model, target_rate )
                                recog_map[uid] = precog
                            precog.AcceptWaveform( audio_bytes )
                        if precog is not None:
                            print("$",end="")
                            txt = precog.FinalResult()
                            stat.final(txt if txt else ' ')
                    await asyncio.sleep(0.03)
        except:
            traceback.print_exc()
        finally:
            self._stt_task = None
            self.stt_stat = {}

    async def sink_finished_callback(self, sink, *args):
        pass

    async def _th_notify_task(self):
        try:
            xpause:bool = False
            msg_buffer = {}
            while True:
                await asyncio.sleep(0.01)
                # 音声認識に結果があるか？             
                pause:bool = False
                for uid,stat in self.stt_stat.items():
                    if stat.have_data():
                        pause = True
                if xpause != pause:
                    print(f"[notify]pause:{pause}")
                    xpause = pause
                # 結果があれば一時停止をかける
                if self._res_task is not None:
                    self._res_task.pause(pause)
                await asyncio.sleep(0.01)                
                for uid,stat in self.stt_stat.items():
                    mesg = stat.get_data()
                    if mesg:
                        print(f"[notify]uid:{uid} msg:{mesg}")
                        msg_buffer[uid] = ' '.join([msg_buffer.get(uid,''),mesg]).strip()
                accept:bool = False
                if len(msg_buffer)>0:
                    if self._res_task is not None:
                        self._res_task.pause(True)
                    accept = await self.is_accept(msg_buffer)
                if accept:
                    # 実行中ならキャンセルする
                    if self._res_task is not None:
                        await self._res_task.cancel()
                    asyncio.create_task( self._th_response_task(msg_buffer,echoback=True,speech=True) )
                    msg_buffer = {}
                    await asyncio.sleep(0.1)                
        except:
            traceback.print_exc()
        finally:
            self._notify_task = None

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

    async def is_accept(self,mesg:dict[int,str]) ->bool:
        return True
        # # とりあえず、ユーザひとりだけの前提で実装。
        # is_break = False
        # llm = LLM2()
        # for uid,input_text in mesg.items():
        #     s1,s2,update_text = await llm.th_is_accept_break(input_text)
        #     if s1<0:
        #         is_break = True
        #     if update_text and len(update_text)>0 and update_text!=input_text:
        #         mesg[uid] = f"{input_text} (推測:{update_text})"
        # return is_break

    async def on_message(self, uid:int, user_content:str, echoback:bool=False, speech:bool=False ):

        while self._res_task is not None:
            #print(f"[MSG]cancel")
            await self._res_task.cancel()
    
        asyncio.create_task( self._th_response_task( {uid:user_content},echoback=echoback,speech=speech) )
        await asyncio.sleep(0.1)

    async def _th_response_task(self, umsg:dict[int,str], echoback:bool=False, speech:bool=False):
        print(f"[#ResTask]start")
        try:
            if echoback:
                for uid,user_content in umsg.items():
                    username = await self.get_username(uid)
                    gid = self.ch.guild.id
                    print(f"mesg {gid} {uid} {username} {user_content}")
                    await self.ch.send(f"🎤 User: {username} {user_content}")
            for uid,user_content in umsg.items():
                pass
            dicord_msg:discord.message.Message = await self.ch.send( '......' )
            tts = self.bot.tts if speech else None
            vc = self.voice_client if speech else None
            self._res_task = t = ResponseTask( dicord_msg, self._global_messages, user_content, tts=tts, voice_client=vc )
            ai_content:str = await t.start()
            if user_content:
                self._global_messages.append( {'role':'user', 'content':user_content})
            if ai_content:
                self._global_messages.append( {'role':'assistant', 'content':ai_content})
        finally:
            self._res_task = None
        print(f"[#ResTask]done")

class ResponseTask:

    def __init__(self, msg:discord.message.Message, messages:list[dict], user_content:str, *, tts:TtsEngine|None, voice_client:discord.VoiceClient|None):
        self.dmsg:discord.message.Message = msg
        self._message:list[dict] = messages
        self._user_content = user_content
        #
        self._cancel:bool = False
        self._pause:bool = False
        #
        # LLM
        self._llm:LLM = LLM()
        self._llm_task = None
        self._llm_response_list:list[str] = []
        #
        self._tts:TtsEngine|None = tts
        self._tts_task = None
        self._tts_pos:int = 0
        self._audio_list:list[discord.PCMAudio|None] = []
        # play
        self._voice_client:discord.VoiceClient|None = voice_client
        self._play_task = None
        self._play_pos:int = 0

    async def start(self):
        print(f"[ResTask]start")
        try:
            self._llm_task = asyncio.create_task( self._th_llm() )
            self._tts_task = asyncio.create_task( self._th_tts() )
            self._play_task = asyncio.create_task( self._th_talk() )
            while self._llm_task is not None or self._tts_task is not None or self._play_task is not None:
                await asyncio.sleep(0.1)
        except:
            traceback.print_exc()
        print(f"[ResTask]done")
        return self.get_talk_text()

    def pause(self,b:bool):
        if self._pause != b:
            print(f"[ResTask]pause:{b}")
        self._pause = b

    async def cancel(self):
        print("[ResTask]cancel Start")
        try:
            self._llm.cancel()
            self._cancel = True
            await asyncio.sleep(0.1)
            while self._llm_task is not None or self._tts_task is not None or self._play_task is not None:
                self.stop_play()
                await asyncio.sleep(0.1)
        finally:
            print("[ResTask]cancel End")

    async def _th_llm(self):
        try:
            print(f"[LLM]start")
            async for talk_segment in self._llm.th_get_response_from_openai( self._message, self._user_content ):
                self._llm_response_list.append(talk_segment)
                if self._cancel:
                    break
                await asyncio.sleep(0.01)
        finally:
            print(f"[LLM]done")
            self._llm_task = None

    async def _th_tts(self):
        try:
            print(f"[TTS]start")
            while True:
                while self._tts_pos>=len(self._llm_response_list):
                    if self._cancel or self._llm_task is None:
                        break
                    await asyncio.sleep(0.2)
                if self._cancel or ( self._tts_pos>=len(self._llm_response_list) and self._llm_task is None ):
                    break
                ans = self._llm_response_list[self._tts_pos]
                self._tts_pos+=1
                audio_source:discord.PCMAudio|None
                if self._voice_client is not None and self._tts is not None:
                    voice_f32, model = await self._tts.a_text_to_audio_by_voicevox( ans, sampling_rate=48000 )
                    if voice_f32 is not None:
                        audio_f32 = voice_f32 #reverb( compressor(voice_f32) )
                        # ステレオに変換（左右チャンネルに同じデータをコピー）
                        f32 = np.stack((audio_f32, audio_f32), axis=-1)
                        b_i16 = (f32*32767.0).astype(np.int16)
                        audio_bytes = b_i16.tobytes()
                        buffer = io.BytesIO(audio_bytes)
                        audio_source = discord.PCMAudio(buffer)
                else:
                    audio_source = None
                self._audio_list.append(audio_source)
        finally:
            print(f"[TTS]done")
            self._tts_task = None

    def is_connected(self) ->bool:
        try:
            return self._voice_client is not None and self._voice_client.is_connected()
        except:
            return False

    def is_playing(self) ->bool:
        try:
            return self._voice_client is not None and self._voice_client.is_playing()
        except:
            return False

    def is_paused(self) ->bool:
        try:
            return self._voice_client is not None and self._voice_client.is_paused()
        except:
            return False

    def stop_play(self):
        try:
            if self._voice_client is not None:
                self._voice_client.stop()
        except:
            return False

    async def _th_talk(self):
        try:
            print(f"[PLAY]start")
            emoji=""
            while True:
                while self._play_pos>=len(self._audio_list) or self._pause or self.is_playing():
                    if self._cancel or self._tts_task is None:
                        break
                    await asyncio.sleep(0.2)
                content_str = self.get_talk_text()
                if self._cancel:
                    try:
                        if self._voice_client:
                            self._voice_client.stop()
                    except Exception as ex:
                        print(f"[PLAY]{ex}")
                    await self.dmsg.edit(content=f"{emoji}{content_str}🚫")
                    break
                if self._cancel or (self._play_pos>=len(self._audio_list) and self._tts_task is None ):
                    await self.dmsg.edit(content=f"{emoji}{content_str}↩️")
                    break
                src = self._audio_list[self._play_pos]
                self._play_pos+=1
                if src:
                    while self.is_playing():
                        await asyncio.sleep(0.2)
                    try:
                        if self._voice_client and self.is_connected():
                            self._voice_client.play(src)
                    except Exception as ex:
                        print(f"[PLAY]{ex}")
                    emoji="🔊"
                await self.dmsg.edit(content=f"{emoji}{content_str} ......")
        finally:
            print(f"[PLAY]done")
            self._play_task = None
            self._cancel = True

    def get_talk_text(self):
        return ''.join(self._llm_response_list[:self._play_pos+1])

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

    # MacOSの場合はbrew install opusが必要
    if not discord.opus.is_loaded():
        libopus_path = ['/opt/homebrew/lib/libopus.dylib']
        for l in libopus_path:
            if os.path.exists(l):
                try:
                    discord.opus.load_opus(l)
                except:
                    pass
    bot = MyBot()
    bot.run(TOKEN)

if __name__ == "__main__":
    main()