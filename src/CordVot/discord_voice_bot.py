import sys
import os
import traceback
import discord
from discord.ext import commands
import vosk
import wave
import json
import asyncio
from dotenv import load_dotenv

def main():
    # .env ファイルから環境変数を読み込む
    load_dotenv("discord.env")

    # 環境変数からトークンを取得
    TOKEN = os.getenv('DISCORD_BOT_TOKEN')

    if TOKEN is None:
        print("Error: DISCORD_BOT_TOKEN is not set in discord.env")
        return

    # VOSKモデルの読み込み
    MODEL_NAME = "vosk-model-small-ja-0.22"
    model = vosk.Model(model_name=MODEL_NAME)

    # Discord Botの設定
    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True
    intents.members = True
    intents.guilds = True

    bot = commands.Bot(command_prefix="!", intents=intents)

    # 音声データを一時的に保存するディレクトリ
    TEMP_DIR = "tmp/audio"
    os.makedirs(TEMP_DIR, exist_ok=True)

    @bot.event
    async def on_ready():
        print(f'Bot is ready as {bot.user}')

    # ボイスチャンネルに参加するコマンド
    @bot.command()
    async def join(ctx):
        try:
            if ctx.author.voice:
                channel = ctx.author.voice.channel
                await channel.connect()
                await ctx.send(f'{channel} に参加しました。録音を開始できます...')
            else:
                await ctx.send("ボイスチャンネルに入ってからコマンドを実行してください。")
        except Exception as ex:
            traceback.print_exc()
            await ctx.send(f"エラーです: {str(ex)}")

    # 録音を開始するコマンド
    @bot.command()
    async def record(ctx):
        if ctx.voice_client:
            audio_filename = os.path.join(TEMP_DIR, "audio_recording.pcm")
            ctx.voice_client.recording = True
            ctx.voice_client.audio_filename = audio_filename
            await ctx.send("録音を開始しました。")
        else:
            await ctx.send("まずボイスチャンネルに参加してください。")

    # 録音を停止し、音声チャンネルから退出するコマンド
    @bot.command()
    async def leave(ctx):
        if ctx.voice_client:
            if hasattr(ctx.voice_client, 'recording') and ctx.voice_client.recording:
                ctx.voice_client.recording = False
                await ctx.send("録音を停止しました。")
            await ctx.voice_client.disconnect()
            await ctx.send("ボイスチャンネルから退出しました。")
        else:
            await ctx.send("ボイスチャンネルに接続していません。")

    # 録音が終了した際の処理（音声解析）
    async def analyze_audio(audio_filename, ctx):
        await ctx.send("音声データの録音が完了しました。解析を開始します。")
        try:
            with wave.open(audio_filename, "rb") as wf:
                rec = vosk.KaldiRecognizer(model, wf.getframerate())
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        result = rec.Result()
                        result_dict = json.loads(result)
                        if 'text' in result_dict:
                            await ctx.send(f"認識結果: {result_dict['text']}")
            await ctx.send("解析が完了しました。")
        except Exception as ex:
            traceback.print_exc()
            await ctx.send(f"音声解析中にエラーが発生しました: {str(ex)}")

    # テキストメッセージに対する応答
    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return

        # コマンドの場合は返答しない
        if message.content.startswith(bot.command_prefix):
            await bot.process_commands(message)
            return

        # 受けたメッセージに対して「ふーん、{受けた会話} なんですね」と応答する
        response = f"ふーん、{message.content} なんですね"
        await message.channel.send(response)

        # コマンドも処理するために追加
        await bot.process_commands(message)

    bot.run(TOKEN)

if __name__ == "__main__":
    main()
