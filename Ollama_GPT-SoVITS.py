import requests
import json
import sounddevice as sd
import numpy as np
import soundfile as sf
import threading
import queue
import io

class OllamaGPTSoVITSSystem:
    def __init__(self, 
                ollama_model="gemma3:12b", 
                ollama_api_url="http://localhost:11434/api",
                gpt_sovits_url="http://localhost:9880",
                ref_audio_path="path/to/reference_audio.wav",
                gpt_weights_path=None,
                sovits_weights_path=None,
                system_prompt=None,
                max_memory=10): 
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
        max_memory: 記憶的最大對話回合數
        """
        self.ollama_model = ollama_model
        self.ollama_api_url = ollama_api_url
        self.gpt_sovits_url = gpt_sovits_url
        self.ref_audio_path = ref_audio_path
        self.audio_queue = queue.Queue()
        self.playing_thread = None
        self.stop_playing = False
        self.system_prompt = system_prompt
        self.max_memory = max_memory
        
        # 初始化對話記憶
        self.conversation_history = []
        
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
    
    def add_to_history(self, role, content):
        """添加對話內容到歷史記錄"""
        self.conversation_history.append({"role": role, "content": content})
        
        # 如果超出最大記憶容量，移除最早的對話
        if len(self.conversation_history) > self.max_memory * 2:
            self.conversation_history = self.conversation_history[-self.max_memory*2:]
    
    def clear_history(self):
        """清除對話歷史"""
        self.conversation_history = []
        print("對話歷史已清除")
    
    def format_conversation_for_context(self):
        """將對話歷史格式化為上下文字符串"""
        context = ""
        for message in self.conversation_history:
            prefix = "用戶: " if message["role"] == "user" else "助手: "
            context += prefix + message["content"] + "\n\n"
        return context
        
    def query_ollama(self, prompt, stream=False):
        """向Ollama發送請求並獲取回應，包含對話歷史"""
        url = f"{self.ollama_api_url}/chat" 
        
        messages = []
        
        # 確保system_prompt只在歷史中出現一次
        if not self.conversation_history or self.conversation_history[0]["role"] != "system":
            self.conversation_history.insert(0, {"role": "system", "content": self.system_prompt})
        
        # 加入歷史消息
        messages = self.conversation_history.copy()
        
        # 加入當前提問
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": stream
        }
        
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                ai_message = result.get('message', {}).get('content', '')
                return ai_message
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
            "text_lang": "zh", 
            "ref_audio_path": self.ref_audio_path,
            "prompt_lang": "zh", 
            "prompt_text": "茶会是淑女的必修课，如果你想学习茶会礼仪的话，我可以教你哦",
            "speed_factor": 1.0,  
            "media_type": "wav"
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
            # 將二進制音檔數據轉換為NumPy
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
        """處理LLM回應，分句轉換成語音並加入playlist"""
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
            if sentence:
                audio_data = self.text_to_speech(sentence)
                if audio_data:
                    self.audio_queue.put(audio_data)
    
    def save_conversation(self, filename="conversation_history.json"):
        """將對話歷史保存到檔案"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            print(f"對話歷史已保存到 {filename}")
        except Exception as e:
            print(f"保存對話歷史時出錯: {e}")
    
    def load_conversation(self, filename="conversation_history.json"):
        """從檔案載入對話歷史"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            print(f"已從 {filename} 載入對話歷史")
        except FileNotFoundError:
            print(f"找不到檔案 {filename}")
        except Exception as e:
            print(f"載入對話歷史時出錯: {e}")
    
    def chat(self):
        """啟動互動式聊天會話"""
        print("開始與AI助手對話")
        print("- 輸入'退出'結束對話")
        print("- 輸入'清除歷史'清除對話記憶")
        print("- 輸入'保存對話'將對話歷史保存到檔案")
        print("- 輸入'載入對話'從檔案載入對話歷史")
        print("- 輸入'顯示歷史'查看當前對話記憶")
        
        self.start_audio_player()
        
        try:
            while True:
                user_input = input("\n用戶: ")
                
                if user_input.lower() in ['退出', 'exit', 'quit', 'bye']:
                    break
                elif user_input.lower() in ['清除歷史', 'clear history']:
                    self.clear_history()
                    continue
                elif user_input.lower() in ['保存對話', 'save conversation']:
                    filename = input("請輸入檔案名稱 (預設為 conversation_history.json): ").strip()
                    if not filename:
                        filename = "conversation_history.json"
                    self.save_conversation(filename)
                    continue
                elif user_input.lower() in ['載入對話', 'load conversation']:
                    filename = input("請輸入檔案名稱 (預設為 conversation_history.json): ").strip()
                    if not filename:
                        filename = "conversation_history.json"
                    self.load_conversation(filename)
                    continue
                elif user_input.lower() in ['顯示歷史', 'show history']:
                    print("\n===== 對話歷史 =====")
                    print(self.format_conversation_for_context())
                    print("====================\n")
                    continue
                
                self.add_to_history("user", user_input)
                
                print("Processing...")
                response = self.query_ollama(user_input)
                print(f"你的助手: {response}")
                
                self.add_to_history("assistant", response)
                
                self.process_response(response)
                
        finally:
            self.stop_audio_player()
            print("對話已結束")

# 使用範例
if __name__ == "__main__":
    # 定義system prompt，根據需求設定角色和行為
    system_prompt = """
    你是一個友善、有幫助的AI助手。你具有以下特點：
    你是芙寧娜，某國的統治者與知名歌劇明星。你個性幽默，標準高，私下喜愛小動物。
    """
    
    # 初始化系統
    system = OllamaGPTSoVITSSystem(
        ollama_model="gemma3:12b", 
        ollama_api_url="http://localhost:11434/api",
        gpt_sovits_url="http://localhost:9880",  
        ref_audio_path="C:/GPT-SoVITS/芙宁娜/参考音频/【正常】茶会是淑女的必修课，如果你想学习茶会礼仪的话，我可以教你哦.wav",
        gpt_weights_path="C:/GPT-SoVITS/芙宁娜/GPT_weights/Furina-e15.ckpt",
        sovits_weights_path="C:/GPT-SoVITS/芙宁娜/SoVITS_weights/Furina_e8_s304.pth",
        system_prompt=system_prompt,
        max_memory=10 
    )
    
    # 開始聊天
    system.chat()