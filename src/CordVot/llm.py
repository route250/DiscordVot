import re
import openai  # OpenAIのクライアントライブラリをインポート
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from openai._streaming import Stream, AsyncStream

LLM_PROMPT:str = """音声会話型のAIのセリフを生成してください。
人間の会話ターンの時は、聞き役として、ときどきボケてください。
あなたの会話ターンなら、どんどん話題を進めましょう。
何か話したいことある？と聞く代わりに、貴方の興味のある話題を初めてください。
何か面白いことを聞かずに、貴方が面白いと感じたことを話して。
人間の考えを聞くより人間の考えを予想しましょう。「どう思う？」と聞くより貴方の感想を話そう。
話題がなかったら、ランダムで変な話を作って
テンポよく会話するために、短めのセリフにして下さい。"""

def is_splitter(text:str) ->int:
    for w in ('、','!','?','！','？','。'):
        i = text.find(w)
        if i>=0:
            return i
    return -1

class LLM:

    def __init__(self, prompt:str|None=None):
        self.run:bool = True
        self.prompt:str = prompt if prompt else LLM_PROMPT
        self.llm_run:int = 0
        self.transcrib_id:int = 0
        self._cancel:bool = False

    def cancel(self):
        self._cancel = True

    def _is_llm_abort(self) ->bool:
        return self._cancel
    
    async def th_get_response_from_openai(self, global_messages:list[dict], user_input:str ):
        """
        OpenAIのAPIを使用してテキスト応答を取得する関数です。
        """
        # OpenAI APIの設定値
        openai_timeout = 5.0  # APIリクエストのタイムアウト時間
        openai_max_retries = 2  # リトライの最大回数
        openai_llm_model = 'gpt-4o-mini'  # 使用する言語モデル
        openai_temperature = 0.7  # 応答の多様性を決定するパラメータ
        openai_max_tokens = 1000  # 応答の最大長
        # リクエストを作ります
        local_messages = []
        local_messages.append( {"role": "system", "content": self.prompt} )
        for m in global_messages:
            local_messages.append( m )
        local_messages.append( {"role": "user", "content": user_input} )
        
        # OpenAIクライアントを初期化します。
        client:AsyncOpenAI = AsyncOpenAI( timeout=openai_timeout, max_retries=openai_max_retries )
        # 通信します
        stream:AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(
            model=openai_llm_model,
            messages=local_messages,
            max_tokens=openai_max_tokens,
            temperature=openai_temperature,
            stream=True,
        )
        if not self.run or self.llm_run != self.transcrib_id:
            await stream.close()
            await client.close()
            return
        sentense:str = ""
        try:
            # AIの応答を取得します
            async for part in stream:
                if self._is_llm_abort():
                    break
                delta_response:str|None = part.choices[0].delta.content
                if delta_response:
                    sentense += delta_response
                    if is_splitter( delta_response )>=0:
                        yield sentense
                        sentense = ''
        finally:
            if not self._is_llm_abort():
                if len(sentense)>0:
                    yield sentense
            else:
                print(f"[LLM]!!!abort!!!")
            try:
                await stream.close()
            except:
                pass
            try:
                await client.close()
            except:
                pass


LLM2_PROMPT="ユーザの応答をみて、あなたのセリフを中断するべきか判断して下さい"
class LLM2:

    def __init__(self):
        self.run:bool = True
        self.llm_run:int = 0
        self.transcrib_id:int = 0
        self._cancel:bool = False

    def cancel(self):
        self._cancel = True

    def _is_llm_abort(self) ->bool:
        return self._cancel
    
    async def th_is_accept_break(self, user_input:str, output:str|None = None, global_messages:list[dict]|None = None  ) ->tuple[int,int,str]:
        """
        OpenAIのAPIを使用してテキスト応答を取得する関数です。
        """
        # OpenAI APIの設定値
        openai_timeout = 5.0  # APIリクエストのタイムアウト時間
        openai_max_retries = 2  # リトライの最大回数
        openai_llm_model = 'gpt-4o-mini'  # 使用する言語モデル
        openai_temperature = 0.7  # 応答の多様性を決定するパラメータ
        openai_max_tokens = 1000  # 応答の最大長
        # リクエストを作ります
        prompt:list[str] = [LLM2_PROMPT]
        if global_messages and len(global_messages)>0:
            prompt.append("- 会話履歴")
            for m in global_messages:
                role=m['role']
                content=m['content']
                prompt.append(f" {role}: {content}")
        if output and len(output):
            prompt.append("- あなたの出力")
            prompt.append( output )
        prompt.append("- ユーザの音声を認識した結果")
        prompt.append( user_input )
        prompt.append("# 応答")
        prompt.append("音声認識結果の精度:{良好or最悪}")
        prompt.append("ユーザの意図を推測:{推測}")
        prompt.append("ユーザの意図により訂正したテキスト:{訂正結果}")
        prompt.append("あなたの出力を中断するか:{中断or継続}")
        prompt.append("# 応答")
        ptxt = '\n'.join(prompt)

        local_messages = []
        local_messages.append( {"role": "system", "content": ptxt} )
        
        # OpenAIクライアントを初期化します。
        client:AsyncOpenAI = AsyncOpenAI( timeout=openai_timeout, max_retries=openai_max_retries )
        # 通信します
        stream:ChatCompletion = await client.chat.completions.create(
            model=openai_llm_model,
            messages=local_messages,
            max_tokens=openai_max_tokens,
            temperature=openai_temperature,
        )
        text = stream.choices[0].message.content
        a1 = a2 = c1 = c2 = False
        refined_text = ''
        if text:
            a1 = "良好" in text
            a2 = "最悪" in text
            c1 = "継続" in text
            c2 = "中断" in text
            mark = "訂正したテキスト:"
            line_list = text.splitlines()
            for line in line_list:
                print(f"  # {line}")
                f = line.find(mark)
                if f>0:
                    refined_text=line[f+len(mark):]
                    refined_text = re.sub(r'[。、「」]', '', refined_text)
        recog_quality=0
        if a1 and not a2:
            recog_quality=1
        if not a1 and a2:
            recog_quality=-1
        is_break=0
        if c1 and not c2:
            is_break=1
        if not c1 and c2:
            is_break=-1
        print( f"  # {is_break} {recog_quality} {refined_text}")
        return is_break,recog_quality,refined_text

async def test_llm2():
    llm = LLM2()

    a = [
    {'role': 'assistant','content':"ダメですか？何か面白いことでも話しましょうか！最近、宇宙の話を聞いたんですが、宇宙には「ダークマター」っていう目に見えない物質があって、実はそれが宇宙の質量の約27%を占めてるんですよね。でも、 "},
    {'role': 'user','content': " です"},
    {'role': 'assistant','content':"「です」だけですか？それとも、何か特別な意味があるのかな？もしかして、宇宙人からのメッセージとか？それとも、宇宙でダンスをしたい気分ですか？想像すると面白いですね！宇宙でのダンスパーティー、 "},
    {'role': 'user','content': "エッチ"},
    {'role': 'assistant','content':"エッチですか？それもまた面白い話題ですね！でも、宇宙の話を続けると、宇宙にはブラックホールがあって、そこに飲み込まれると時間がどうなるかっていうのが面白いんですよ。時間が遅く流れるって言われてるけど、もしかしたら「エッチ」なこともその中では全然違う時間の流れになるのかも！どう思います？ "},
    {'role': 'user','content': "これ政党です"},
    ]
    b = "ああ、政党のことですね！最近の政治の話題はとても興味深いですよね。特に、選挙や政策の変更があると、国の未来が大きく変わる可能性がありますし。しかし、政党の名前を聞くと、時々「何でこんな名前にしたんだろう？」って思うこともありますよね。例えば、"
    c = "あありんこ"
    res = await llm.th_is_accept_break(c,b,a)
    print( res )

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_llm2())
# a = [
# {'role': 'assistant','content':"🔊ダメですか？何か面白いことでも話しましょうか！最近、宇宙の話を聞いたんですが、宇宙には「ダークマター」っていう目に見えない物質があって、実はそれが宇宙の質量の約27%を占めてるんですよね。でも、🚫 "},
# {'role': 'user','content': " です"},
# {'role': 'assistant','content':"🔊「です」だけですか？それとも、何か特別な意味があるのかな？もしかして、宇宙人からのメッセージとか？それとも、宇宙でダンスをしたい気分ですか？想像すると面白いですね！宇宙でのダンスパーティー、🚫 "},
# {'role': 'user','content': "エッチ"},
# {'role': 'assistant','content':"🔊エッチですか？それもまた面白い話題ですね！でも、宇宙の話を続けると、宇宙にはブラックホールがあって、そこに飲み込まれると時間がどうなるかっていうのが面白いんですよ。時間が遅く流れるって言われてるけど、もしかしたら「エッチ」なこともその中では全然違う時間の流れになるのかも！どう思います？↩️ "},
# {'role': 'user','content': "これ政党です"},
# {'role': 'assistant','content':"🔊ああ、政党のことですね！最近の政治の話題はとても興味深いですよね。特に、選挙や政策の変更があると、国の未来が大きく変わる可能性がありますし。しかし、政党の名前を聞くと、時々「何でこんな名前にしたんだろう？」って思うこともありますよね。例えば、「🚫 "},
# {'role': 'user','content': "てる"},
# {'role': 'assistant','content':"🔊「てる」ですか？それとも「テル」っていう名前の宇宙人がいるのかな？それとも、最近流行ってる何かの略語ですかね？気になります！でも、🚫 "},
# {'role': 'user','content': "じゃあなぁ"},
# {'role': 'assistant','content':"🔊じゃあなぁ、ですね！まるで宇宙の彼方に旅立つような気分ですね。もしかして、宇宙の冒険に出かける準備かな？それとも、ちょっとお茶でも飲みに行く感じ？どちらにしても、またいつでもお話ししましょう！🚫 "},
# {'role': 'user','content': "まー"},
# {'role': 'assistant','content':"🔊「まー」ですか？それは何かを考えているときの「まー」でしょうか？それとも、何かを引き延ばすような感じ？「まー」って言うと、何かが始まる前のワクワク感がありますよね。たとえば、宇宙旅行の前に「まー、🚫 "},
# {'role': 'user','content': "ねえ"},
# {'role': 'assistant','content':"🔊ねえ、何か気になることがあるのかな？話してくれたら、面白いことを一緒に考えられるかも！それとも、宇宙の秘密を教えてほしいのかな？それとも、最近はまっていることについて話したい？何でも聞いてみて！↩️ "},
# {'role': 'user','content': "認識止めなぁ"},
# {'role': 'assistant','content':"🔊認識を止める？それはまるで、コンピュータの電源を切るみたいですね。でも、ちょっと待って！もし私の認識が止まったら、 ......"},
# ]