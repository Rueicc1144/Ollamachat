import requests
import json
import time
import sounddevice as sd
import numpy as np
import soundfile as sf
import threading
import queue
import io

class OllamaGPTSoVITSSystem:
    def __init__(self, 
                ollama_model="llama3", 
                ollama_api_url="http://localhost:11434/api",
                gpt_sovits_url="http://localhost:9880",
                ref_audio_path="path/to/reference_audio.wav",
                gpt_weights_path=None,
                sovits_weights_path=None,
                system_prompt=None):
        """
        初始化整合系統
        
        參數:
        ollama_model: 要使用的Ollama模型名稱
        ollama_api_url: Ollama API的基礎URL
        gpt_sovits_url: GPT-SoVITS API的URL
        ref_audio_path: 參考音檔路徑，用於確定語音風格
        gpt_weights_path: GPT模型權重路徑
        sovits_weights_path: SoVITS模型權重路徑
        system_prompt: 設定LLM的system prompt
        """
        self.ollama_model = ollama_model
        self.ollama_api_url = ollama_api_url
        self.gpt_sovits_url = gpt_sovits_url
        self.ref_audio_path = ref_audio_path
        self.audio_queue = queue.Queue()
        self.playing_thread = None
        self.stop_playing = False
        self.system_prompt = system_prompt
        
        # 如果提供了模型權重路徑，設置模型權重
        if gpt_weights_path:
            self.set_gpt_weights(gpt_weights_path)
        if sovits_weights_path:
            self.set_sovits_weights(sovits_weights_path)
        
    def set_gpt_weights(self, weights_path):
        """設置GPT模型權重"""
        url = f"{self.gpt_sovits_url}/set_gpt_weights"
        params = {"weights_path": weights_path}
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                print("GPT權重設置成功")
            else:
                print(f"GPT權重設置失敗: {response.text}")
        except Exception as e:
            print(f"設置GPT權重時出錯: {e}")
    
    def set_sovits_weights(self, weights_path):
        """設置SoVITS模型權重"""
        url = f"{self.gpt_sovits_url}/set_sovits_weights"
        params = {"weights_path": weights_path}
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                print("SoVITS權重設置成功")
            else:
                print(f"SoVITS權重設置失敗: {response.text}")
        except Exception as e:
            print(f"設置SoVITS權重時出錯: {e}")
    
    def set_ref_audio(self, ref_audio_path):
        """設置參考音檔路徑"""
        self.ref_audio_path = ref_audio_path
        print(f"參考音檔已設置為: {ref_audio_path}")
        
    def query_ollama(self, prompt, stream=False):
        """向Ollama發送請求並獲取回應"""
        url = f"{self.ollama_api_url}/generate"
        
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": stream
        }
        
        # 如果提供了system prompt，將它添加到請求payload中
        if system:
            payload["system"] = system
        
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                print(f"Ollama API錯誤: {response.status_code}")
                print(response.text)
                return ""
        except Exception as e:
            print(f"調用Ollama API時出錯: {e}")
            return ""
            
    def text_to_speech(self, text):
        """使用GPT-SoVITS將文本轉換為語音"""
        url = f"{self.gpt_sovits_url}/tts"
        
        payload = {
            "text": text,
            "text_lang": "ja",  # 中文文本
            "ref_audio_path": self.ref_audio_path,
            "prompt_lang": "ja",  # 提示語言
            "prompt_text": "ちょっと頑張りすぎたかなー，トモリーは心配性だなー",  # 可選的提示文本
            "speed_factor": 1.0,  # 語速可調整
            "media_type": "wav"  # 輸出格式
        }
        
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                # 返回的是音檔數據
                audio_data = response.content
                return audio_data
            else:
                print(f"GPT-SoVITS API錯誤: {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"調用GPT-SoVITS時出錯: {e}")
            return None
    
    def play_audio(self, audio_data):
        """播放音檔數據"""
        try:
            # 將二進制音檔數據轉換為NumPy數組
            audio_np, sample_rate = sf.read(io.BytesIO(audio_data))
            
            # 播放音檔
            sd.play(audio_np, sample_rate)
            sd.wait()  # 等待音檔播放完成
        except Exception as e:
            print(f"播放音檔時出錯: {e}")
    
    def start_audio_player(self):
        """啟動音檔播放Thread"""
        self.stop_playing = False
        self.playing_thread = threading.Thread(target=self._audio_player_thread)
        self.playing_thread.daemon = True
        self.playing_thread.start()
    
    def _audio_player_thread(self):
        """音檔播放Thread的實現"""
        while not self.stop_playing:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                self.play_audio(audio_data)
                self.audio_queue.task_done()
            except queue.Empty:
                pass  # 隊列為空，繼續等待
            except Exception as e:
                print(f"音檔播放Thread錯誤: {e}")
    
    def stop_audio_player(self):
        """停止音檔播放Thread"""
        self.stop_playing = True
        if self.playing_thread:
            self.playing_thread.join(timeout=2)
    
    def process_response(self, llm_response):
        """處理LLM回應，分句轉換成語音並加入播放隊列"""
        # 根據常見標點符號分句
        sentences = []
        current_sentence = ""
        
        for char in llm_response:
            current_sentence += char
            if char in ["。", "！", "？", ".", "!", "?"]:
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # 添加最後可能的句子
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        for sentence in sentences:
            if sentence:  # 確保句子不為空
                audio_data = self.text_to_speech(sentence)
                if audio_data:
                    self.audio_queue.put(audio_data)
    
    def chat(self):
        """啟動互動式聊天會話"""
        print("開始與AI助手對話（輸入'退出'結束對話）")
        self.start_audio_player()
        
        try:
            while True:
                user_input = input("\n用戶: ")
                
                if user_input.lower() in ['退出', 'exit', 'quit', 'bye']:
                    break
                
                print("AI思考中...")
                response = self.query_ollama(user_input, system=self.system_prompt)
                print(f"AI助手: {response}")
                
                # 將回應分句處理並轉為語音
                self.process_response(response)
                
        finally:
            self.stop_audio_player()
            print("對話已結束")

# 使用範例
if __name__ == "__main__":
    # 定義system prompt，根據你的需求設定角色和行為
    system_prompt = """
    你是一個友善、有幫助的AI助手。你具有以下特點：
    1. 你喜歡用簡短、清晰的語言回答問題
    2. 你總是保持禮貌和尊重
    3. 你是[アノン]，你的個性有點吵鬧有點任性，但確實地用自己的方法維繫團體，面對他真正看重的要求，其他人當然會回應，但多少也總會抱著真是拿你沒辦法的想法。偶爾不慎流露出真心話或表現得過於厚臉皮，顯得粗心且具層次感。
    """
    
    # 初始化系統
    system = OllamaGPTSoVITSSystem(
        ollama_model="llama3.2:3b",  # 替換為你想使用的模型
        ollama_api_url="http://localhost:11434/api",
        gpt_sovits_url="http://localhost:9880",  # 替換為你的GPT-SoVITS API URL
        ref_audio_path="C:/GPT-SoVITS/GPT-SoVITS_MyGO/参考音频/Anon干声素材/参考音频/ちょっと頑張りすぎたかなートモリーは心配性だなー.wav",  # 替換為你的參考音檔路徑
        gpt_weights_path="C:/GPT-SoVITS/GPT-SoVITS-v3lora-20250228/GPT_weights/anon1-e15.ckpt",  # 可選，替換為你的GPT權重路徑
        sovits_weights_path="C:/GPT-SoVITS/GPT-SoVITS-v3lora-20250228/SoVITS_weights/anon1_e8_s2184.pth",  # 可選，替換為你的SoVITS權重路徑
        system_prompt=system_prompt  # 添加system prompt
    )
    
    # 開始聊天
    system.chat()