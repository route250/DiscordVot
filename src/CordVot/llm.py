
import openai  # OpenAIのクライアントライブラリをインポート
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from openai._streaming import Stream, AsyncStream

LLM_PROMPT:str = """音声会話型のAIのセリフを生成してください。
人間の会話ターンの時は、聞き役として、ときどきボケてください。
あなたの会話ターンなら、どんどん話題を進めましょう。
何か話したいことある？と聞く代わりに、貴方の興味のある話題を初めてください。
何か面白いことを聞かずに、貴方が面白いと感じたことを話して。
人間の考えを聞くより人間の考えを予想しましょう。「どう思う？」と聞くより貴方の感想を話そう。
話題がなかったら、ランダムで変な話を作って"""

def is_splitter(text:str) ->int:
    for w in ('、','!','?','！','？','。'):
        i = text.find(w)
        if i>=0:
            return i
    return -1

class LLM:

    def __init__(self):
        self.run:bool = True
        self.llm_run:int = 0
        self.transcrib_id:int = 0

    def _is_llm_abort(self) ->bool:
        return False
    
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
        local_messages.append( {"role": "system", "content": LLM_PROMPT} )
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

