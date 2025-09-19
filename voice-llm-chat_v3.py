# 可以即時打斷，但好奇怪
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import queue
import time
import whisper
from opencc import OpenCC
import webrtcvad
import requests
import json
import io
from collections import deque
import os

class DialogController:
    def __init__(self):
        self.state = "idle"  # idle / listening / thinking / speaking
        self.audio_playing = False
        self.can_listen = True
        self.interrupted = False
        self.lock = threading.Lock()  # Add a lock for thread safety
    
    def start_listening(self):
        with self.lock:
            self.state = "listening"
            self.can_listen = True
    
    def start_thinking(self):
        with self.lock:
            self.state = "thinking"
            self.can_listen = False
    
    def start_speaking(self):
        with self.lock:
            self.state = "speaking"
            self.audio_playing = True
            self.can_listen = False  # Stop listening while speaking
            self.interrupted = False
    
    def stop_speaking(self):
        with self.lock:
            self.audio_playing = False
            self.state = "idle"
            self.can_listen = True
    
    def interrupt(self):
        with self.lock:
            if self.state == "speaking":
                self.interrupted = True
                return True
            return False
            
    def get_state(self):
        with self.lock:
            return self.state
            
    def can_be_interrupted(self):
        with self.lock:
            # Only allow interruption when speaking
            return self.state == "speaking"
            
    def should_listen(self):
        with self.lock:
            return self.can_listen

class AudioCapture:
    """
    音訊擷取模組，負責錄音並進行VAD檢測
    """
    def __init__(self, controller, sample_rate=16000, vad_mode=3):
        self.controller = controller
        self.sample_rate = sample_rate
        self.frame_duration_ms = 30  # VAD幀長度（毫秒）
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        self.silence_threshold = 0.01  # 音量閾值
        self.silence_duration = 1.5  # 連續靜音判定為語句結束的時間（秒）
        self.buffer_duration = 5  # 最大緩衝時間（秒）
        
        # 初始化VAD
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(vad_mode)  # 0-3, 3為最嚴格
        
        # 初始化音訊佇列和狀態變數
        self.audio_queue = queue.Queue()
        self.speech_queue = queue.Queue()
        self.running = False
        self.is_speaking = False
        self.last_speech_time = 0
        self.continuous_silence = 0
        self.energy_threshold = 0.0005  # 能量閾值，用於輔助VAD
        self.speech_frames = deque(maxlen=int(self.buffer_duration * 1000 / self.frame_duration_ms))
    
    def audio_callback(self, indata, frames, time, status):
        """音訊回調函數，將錄製的音訊放入佇列"""
        if status:
            print(f"錯誤: {status}")
        self.audio_queue.put(indata.copy())
    
    def is_speech(self, audio_frame):
        """語音活動檢測函數"""
        # 轉換為PCM16格式（VAD需要）
        pcm_data = (audio_frame * 32768).astype(np.int16).tobytes()
        
        # 使用WebRTC VAD
        try:
            is_vad_speech = self.vad.is_speech(pcm_data, self.sample_rate)
        except:
            is_vad_speech = False
        
        # 計算音頻能量作為輔助判斷
        energy = np.mean(audio_frame**2)
        is_energy_speech = energy > self.energy_threshold
        
        # 結合VAD和能量判斷
        return is_vad_speech or is_energy_speech
    
    def detect_sentence_boundary(self, current_silence):
        """檢測語句邊界"""
        # 如果連續靜音超過閾值，判定為語句結束
        if current_silence >= self.silence_duration:
            return True
        
        # 語調下降檢測（簡化版）
        if len(self.speech_frames) >= 10:
            recent_energies = [np.mean(frame**2) for frame in list(self.speech_frames)[-10:]]
            if len(recent_energies) >= 5 and all(recent_energies[i] > recent_energies[i+1] for i in range(len(recent_energies)-5, len(recent_energies)-1)):
                return True
        
        return False
    
    def process_audio(self):
        """處理音訊，進行VAD檢測和語句切割"""
        audio_buffer = []
        buffer_duration = 0
        
        while self.running:
            # Check if we should be listening
            if not self.controller.should_listen():
                time.sleep(0.1)
                continue
                
            try:
                # Get audio data
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Process audio in frames for VAD
                for i in range(0, len(audio_data), self.frame_size):
                    if i + self.frame_size <= len(audio_data):
                        frame = audio_data[i:i+self.frame_size]
                        
                        # Check if this is speech
                        speech_detected = self.is_speech(frame)
                        self.speech_frames.append(frame)
                        
                        if speech_detected:
                            # Speech detected
                            if not self.is_speaking:
                                print("檢測到語音，開始錄製...")
                                self.is_speaking = True
                                
                                # If system is currently speaking, attempt to interrupt
                                if self.controller.can_be_interrupted():
                                    print("檢測到用戶想打斷對話...")
                                    self.controller.interrupt()
                            
                            # Update last speech time
                            self.last_speech_time = time.time()
                            self.continuous_silence = 0
                            
                            # Add to buffer
                            audio_buffer.append(frame)
                            buffer_duration += len(frame) / self.sample_rate
                        else:
                            # Process silence during speech
                            if self.is_speaking:
                                # Calculate silence duration
                                current_time = time.time()
                                self.continuous_silence = current_time - self.last_speech_time
                                
                                # Add silence frame to buffer (to preserve natural pauses)
                                audio_buffer.append(frame)
                                buffer_duration += len(frame) / self.sample_rate
                                
                                # Check for sentence boundary
                                if self.detect_sentence_boundary(self.continuous_silence):
                                    # End of sentence, submit audio for processing
                                    if buffer_duration > 0.5:  # At least 0.5s to process
                                        print("檢測到語句結束，送出處理...")
                                        
                                        # Combine audio data and send
                                        audio_chunk = np.concatenate(audio_buffer)
                                        self.speech_queue.put(audio_chunk.flatten())
                                    
                                    # Reset buffer and state
                                    audio_buffer = []
                                    buffer_duration = 0
                                    self.is_speaking = False
                        
                        # Avoid buffer overflow
                        if buffer_duration > self.buffer_duration:
                            print("緩衝區已滿，處理當前音訊...")
                            audio_chunk = np.concatenate(audio_buffer)
                            
                            # Send audio for processing
                            self.speech_queue.put(audio_chunk.flatten())
                            
                            # Keep last 0.5s of audio to avoid cutting words
                            last_second_samples = int(self.sample_rate * 0.5)
                            if len(audio_chunk) > last_second_samples:
                                audio_buffer = [audio_chunk[-last_second_samples:]]
                                buffer_duration = 0.5
                            else:
                                audio_buffer = []
                                buffer_duration = 0
                
            except queue.Empty:
                # Queue is empty, check if we need to process remaining audio
                if self.is_speaking and time.time() - self.last_speech_time > self.silence_duration:
                    if audio_buffer and buffer_duration > 0.5:  # At least 0.5s to process
                        print("長時間靜音，處理剩餘語音...")
                        
                        # Combine audio data and send
                        audio_chunk = np.concatenate(audio_buffer)
                        self.speech_queue.put(audio_chunk.flatten())
                        
                        # Reset buffer and state
                        audio_buffer = []
                        buffer_duration = 0
                        self.is_speaking = False
    
    def start(self):
        """啟動音訊擷取"""
        self.running = True
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            dtype='float32', 
            blocksize=self.frame_size
        )
        self.stream.start()
        print("音訊擷取模組已啟動")
    
    def stop(self):
        """停止音訊擷取"""
        self.running = False
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
        if hasattr(self, 'process_thread') and self.process_thread:
            self.process_thread.join(timeout=1)
        print("音訊擷取模組已停止")


class SpeechRecognition:
    """
    語音辨識模組，使用Whisper進行語音-文字轉換
    """
    def __init__(self, controller, speech_queue, text_queue, language="zh"):
        self.controller = controller
        self.speech_queue = speech_queue
        self.text_queue = text_queue
        self.language = language
        self.running = False
        
        # 建立簡轉繁轉換器
        self.cc = OpenCC('s2t')
        
        # 載入Whisper模型
        print("正在載入Whisper模型，請稍候...")
        self.model = whisper.load_model("small")
        print("Whisper模型載入完成!")
    
    def process_speech(self):
        """處理語音並進行識別"""
        MIN_AUDIO_SAMPLES = 8000  # About 0.5s of audio

        while self.running:
            try:
                # Get audio data from queue
                audio_chunk = self.speech_queue.get(timeout=0.5)

                if len(audio_chunk) < MIN_AUDIO_SAMPLES:
                    print("音訊片段太短，跳過識別")
                    continue

                # Notify controller that we're thinking
                self.controller.start_thinking()
                print("正在識別語音...")

                # Use Whisper for transcription
                result = self.model.transcribe(
                    audio_chunk,
                    language=self.language,
                    task="transcribe"
                )

                # Convert simplified to traditional Chinese
                if result["text"].strip():
                    text = self.cc.convert(result["text"]).strip()
                    print(f"識別結果: {text}")

                    # Put text result into queue
                    if text:
                        self.text_queue.put(text)

                        # Check for interrupt command
                        if self.controller.can_be_interrupted() and any(
                            keyword in text for keyword in ["停止", "打斷", "暫停", "等一下"]
                        ):
                            print("檢測到打斷指令")
                            if self.controller.interrupt():
                                print("已打斷目前回應")
                else:
                    print("未能識別有效內容")

                self.speech_queue.task_done()

            except queue.Empty:
                pass
            except Exception as e:
                print(f"語音識別錯誤: {e}")

    
    def start(self):
        """啟動語音辨識"""
        self.running = True
        self.process_thread = threading.Thread(target=self.process_speech)
        self.process_thread.daemon = True
        self.process_thread.start()
        print("語音辨識模組已啟動")
    
    def stop(self):
        """停止語音辨識"""
        self.running = False
        if hasattr(self, 'process_thread') and self.process_thread:
            self.process_thread.join(timeout=1)
        print("語音辨識模組已停止")


class LLMConversation:
    """
    LLM對話模組，使用Ollama進行文字對話
    """
    def __init__(self, controller, text_queue, response_queue, 
                 model="gemma3:12b", api_url="http://localhost:11434/api",
                 system_prompt=None, max_memory=10):
        self.controller = controller
        self.text_queue = text_queue
        self.response_queue = response_queue
        self.model = model
        self.api_url = api_url
        self.system_prompt = system_prompt or """你是一個友善、有幫助的AI助手。"""
        self.max_memory = max_memory
        self.conversation_history = []
        self.running = False
        
        # 初始化對話歷史
        if self.system_prompt:
            self.conversation_history.append({"role": "system", "content": self.system_prompt})
    
    def add_to_history(self, role, content):
        """添加對話內容到歷史記錄"""
        # 確保 system prompt 只在歷史中出現一次
        if role == "system":
            for i, msg in enumerate(self.conversation_history):
                if msg["role"] == "system":
                    self.conversation_history[i]["content"] = content
                    return
        
        self.conversation_history.append({"role": role, "content": content})
        
        # 如果超出最大記憶容量，移除最早的對話
        if len(self.conversation_history) > self.max_memory * 2 + 1:  # +1是因為system prompt
            # 保留system prompt
            system_prompt = self.conversation_history[0] if self.conversation_history[0]["role"] == "system" else None
            self.conversation_history = self.conversation_history[-(self.max_memory*2):]
            if system_prompt and self.conversation_history[0]["role"] != "system":
                self.conversation_history.insert(0, system_prompt)
    
    def clear_history(self):
        """清除對話歷史，但保留system prompt"""
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            system_prompt = self.conversation_history[0]
            self.conversation_history = [system_prompt]
        else:
            self.conversation_history = []
            if self.system_prompt:
                self.conversation_history.append({"role": "system", "content": self.system_prompt})
        print("對話歷史已清除")
    
    def query_ollama(self, prompt):
        """向Ollama發送請求並獲取回應"""
        url = f"{self.api_url}/chat"
        
        # 添加當前提問到歷史記錄
        self.add_to_history("user", prompt)
        
        payload = {
            "model": self.model,
            "messages": self.conversation_history,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                ai_message = result.get('message', {}).get('content', '')
                
                # 添加AI回應到歷史記錄
                if ai_message:
                    self.add_to_history("assistant", ai_message)
                
                return ai_message
            else:
                print(f"Ollama API錯誤: {response.status_code}")
                print(response.text)
                return "抱歉，我處理您的請求時遇到了問題。"
        except Exception as e:
            print(f"調用Ollama API時出錯: {e}")
            return "抱歉，我無法連接到語言模型服務。"
    
    def process_messages(self):
        """處理輸入文字並生成回應"""
        while self.running:
            try:
                # 從佇列獲取文字
                text = self.text_queue.get(timeout=0.5)
                
                # 檢查是否為命令
                if text.lower() in ['清除歷史', 'clear history']:
                    self.clear_history()
                    self.response_queue.put("對話歷史已清除。")
                    continue
                
                # 通知控制器正在思考
                self.controller.start_thinking()
                print(f"🤖 處理用戶輸入: {text}")
                
                # 查詢LLM並獲取回應
                response = self.query_ollama(text)
                print(f"LLM回應: {response}")
                
                # 將回應放入佇列
                if response:
                    self.response_queue.put(response)
                
                self.text_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"LLM處理錯誤: {e}")
                self.response_queue.put("處理您的請求時發生錯誤。")
    
    def start(self):
        """啟動LLM對話"""
        self.running = True
        self.process_thread = threading.Thread(target=self.process_messages)
        self.process_thread.daemon = True
        self.process_thread.start()
        print("LLM對話模組已啟動")
    
    def stop(self):
        """停止LLM對話"""
        self.running = False
        if hasattr(self, 'process_thread') and self.process_thread:
            self.process_thread.join(timeout=1)
        print("LLM對話模組已停止")
    
    def save_conversation(self, filename="conversation_history.json"):
        """將對話歷史保存到檔案"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            print(f"對話歷史已保存到 {filename}")
            return True
        except Exception as e:
            print(f"保存對話歷史時出錯: {e}")
            return False
    
    def load_conversation(self, filename="conversation_history.json"):
        """從檔案載入對話歷史"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                loaded_history = json.load(f)
                
                # 檢查是否有system prompt
                has_system = any(msg.get("role") == "system" for msg in loaded_history)
                
                if not has_system and self.system_prompt:
                    loaded_history.insert(0, {"role": "system", "content": self.system_prompt})
                
                self.conversation_history = loaded_history
                
            print(f"已從 {filename} 載入對話歷史")
            return True
        except FileNotFoundError:
            print(f"找不到檔案 {filename}")
            return False
        except Exception as e:
            print(f"載入對話歷史時出錯: {e}")
            return False


class VoiceSynthesis:
    """
    語音合成模組，使用GPT-SoVITS進行文字-語音轉換
    """
    def __init__(self, controller, response_queue, 
                 gpt_sovits_url="http://localhost:9880",
                 ref_audio_path=None, 
                 gpt_weights_path=None, 
                 sovits_weights_path=None):
        self.controller = controller
        self.response_queue = response_queue
        self.gpt_sovits_url = gpt_sovits_url
        self.ref_audio_path = ref_audio_path
        self.audio_queue = queue.Queue()
        self.running = False
        self.stop_playing = False
        
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
    
    def text_to_speech(self, text, text_lang="zh", prompt_text=None):
        """使用GPT-SoVITS將文本轉換為語音"""
        url = f"{self.gpt_sovits_url}/tts"
        
        if not self.ref_audio_path:
            print("未設置參考音檔路徑!")
            return None
        
        payload = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": self.ref_audio_path,
            "prompt_lang": text_lang,
            "prompt_text": prompt_text or "茶会是淑女的必修课，如果你想学习茶会礼仪的话，我可以教你哦",
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
            # 將二進制音檔數據轉換為NumPy數組
            audio_np, sample_rate = sf.read(io.BytesIO(audio_data))
            
            # 播放音檔
            sd.play(audio_np, sample_rate)
            sd.wait()  # 等待音檔播放完成
        except Exception as e: 
            print(f"播放音檔時出錯: {e}")
    
    def audio_player_thread(self):
        """音檔播放線程"""
        while not self.stop_playing:
            try:
                # Check if interrupted before getting next audio
                if self.controller.interrupted:
                    print("檢測到打斷，清空音訊佇列")
                    # Empty the queue
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.task_done()
                        except queue.Empty:
                            break
                    # Reset interrupted flag after handling
                    self.controller.interrupted = False
                    self.controller.stop_speaking()
                    continue
                    
                # Get audio data with timeout
                audio_data = self.audio_queue.get(timeout=1)
                
                # Check interrupt flag again before playing
                if self.controller.interrupted:
                    print("檢測到打斷，停止當前音訊播放")
                    sd.stop()  # Stop current playback
                    self.audio_queue.task_done()
                    continue
                
                # Play audio
                self.controller.start_speaking()
                self.play_audio(audio_data)
                self.audio_queue.task_done()
                
                # Check if queue is empty
                if self.audio_queue.empty():
                    self.controller.stop_speaking()
                
            except queue.Empty:
                pass  # Queue is empty, continue waiting
            except Exception as e:
                print(f"音檔播放線程錯誤: {e}")
                self.controller.stop_speaking()
    
    def process_responses(self):
        """處理文字回應並轉換為語音"""
        while self.running:
            try:
                # Get text response from queue
                response = self.response_queue.get(timeout=0.5)
                if not response:
                    continue
                
                print("正在生成語音...")
                
                # Split into sentences based on punctuation
                sentences = []
                current_sentence = ""
                
                for char in response:
                    current_sentence += char
                    if char in ["。", "！", "？", ".", "!", "?"]:
                        if current_sentence.strip():
                            sentences.append(current_sentence.strip())
                        current_sentence = ""
                
                # Add the last sentence if any
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                
                # Process long sentences
                processed_sentences = []
                for sentence in sentences:
                    # Split sentences longer than 50 characters at commas
                    if len(sentence) > 50:
                        parts = []
                        current_part = ""
                        for char in sentence:
                            current_part += char
                            if (char in ["，", "；", ",", ";"]) and len(current_part) > 20:
                                parts.append(current_part)
                                current_part = ""
                        if current_part:
                            parts.append(current_part)
                        processed_sentences.extend(parts)
                    else:
                        processed_sentences.append(sentence)
                
                # Generate and add to playback queue
                for sentence in processed_sentences:
                    # Check for interruption before generating each sentence
                    if self.controller.interrupted:
                        print("檢測到打斷，停止生成更多語音")
                        break
                        
                    if sentence:  # Ensure sentence is not empty
                        audio_data = self.text_to_speech(sentence)
                        if audio_data:
                            self.audio_queue.put(audio_data)
                
                self.response_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"語音合成錯誤: {e}")
    
    def start(self):
        """啟動語音合成"""
        self.running = True
        self.stop_playing = False
        
        # 啟動處理回應的線程
        self.process_thread = threading.Thread(target=self.process_responses)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # 啟動音頻播放線程
        self.player_thread = threading.Thread(target=self.audio_player_thread)
        self.player_thread.daemon = True
        self.player_thread.start()
        
        print("🔊 語音合成模組已啟動")
    
    def stop(self):
        """停止語音合成"""
        self.running = False
        self.stop_playing = True
        
        # 停止當前播放
        sd.stop()
        
        # 清空佇列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        if hasattr(self, 'process_thread') and self.process_thread:
            self.process_thread.join(timeout=1)
        if hasattr(self, 'player_thread') and self.player_thread:
            self.player_thread.join(timeout=1)
        
        print("語音合成模組已停止")
        
class VoiceLLMChatSystem:
    """
    主整合系統類別，負責整合所有模組並提供使用者介面
    """
    def __init__(self, 
                ollama_model="gemma3:12b", 
                ollama_api_url="http://localhost:11434/api",
                gpt_sovits_url="http://localhost:9880",
                ref_audio_path=None,
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
        # 初始化對話控制器
        self.controller = DialogController()
        
        # 初始化佇列
        self.speech_queue = queue.Queue()  # 音訊片段佇列
        self.text_queue = queue.Queue()    # 識別文字佇列
        self.response_queue = queue.Queue() # 回應文字佇列
        
        # 初始化音訊擷取模組
        self.audio_capture = AudioCapture(self.controller)
        
        # 初始化語音辨識模組
        self.speech_recognition = SpeechRecognition(
            self.controller, 
            self.audio_capture.speech_queue,
            self.text_queue
        )
        
        # 初始化LLM對話模組
        self.llm_conversation = LLMConversation(
            self.controller,
            self.text_queue,
            self.response_queue,
            model=ollama_model,
            api_url=ollama_api_url,
            system_prompt=system_prompt,
            max_memory=max_memory
        )
        
        # 初始化語音合成模組
        self.voice_synthesis = VoiceSynthesis(
            self.controller,
            self.response_queue,
            gpt_sovits_url=gpt_sovits_url,
            ref_audio_path=ref_audio_path,
            gpt_weights_path=gpt_weights_path,
            sovits_weights_path=sovits_weights_path
        )
        
        # 狀態變數
        self.running = False
        self.status_thread = None
    
    def start(self):
        """啟動所有系統模組"""
        print("=== 語音對話系統啟動中 ===")
        
        # 啟動各模組
        self.audio_capture.start()
        self.speech_recognition.start()
        self.llm_conversation.start()
        self.voice_synthesis.start()
        
        # 設置狀態監控
        self.running = True
        self.status_thread = threading.Thread(target=self._status_monitor)
        self.status_thread.daemon = True
        self.status_thread.start()
        
        print("系統已啟動，請開始對話...")
    
    def stop(self):
        """停止所有系統模組"""
        print("正在關閉系統...")
        self.running = False
        
        # 停止各模組
        self.voice_synthesis.stop()
        self.llm_conversation.stop()
        self.speech_recognition.stop()
        self.audio_capture.stop()
        
        if self.status_thread:
            self.status_thread.join(timeout=1)
        
        print("系統已關閉")
    
    def _status_monitor(self):
        """狀態監控線程"""
        last_state = ""
        while self.running:
            current_state = self.controller.get_state()
            
            # 狀態變化時顯示
            if current_state != last_state:
                status_map = {
                    "idle": "🛌 空閒中",
                    "listening": "聆聽中...",
                    "thinking": "思考中...",
                    "speaking": "說話中..."
                }
                print(status_map.get(current_state, current_state))
                last_state = current_state
            
            time.sleep(0.5)
    
    def save_conversation(self, filename="conversation_history.json"):
        """保存對話歷史"""
        return self.llm_conversation.save_conversation(filename)
    
    def load_conversation(self, filename="conversation_history.json"):
        """載入對話歷史"""
        return self.llm_conversation.load_conversation(filename)
    
    def clear_history(self):
        """清除對話歷史"""
        return self.llm_conversation.clear_history()
    
    def set_ref_audio(self, ref_audio_path):
        """設置參考音檔"""
        self.voice_synthesis.set_ref_audio(ref_audio_path)
    
    def run_console_ui(self):
        """執行一個簡單的控制台UI"""
        try:
            self.start()
            print("\n=== 語音對話系統已啟動 ===")
            print("請對著麥克風說話...")
            print("控制指令：")
            print("- 「退出」或「結束」: 退出程式")
            print("- 「清除歷史」: 清除對話記憶")
            print("- 「保存對話」: 將對話歷史保存到檔案")
            print("- 「載入對話」: 從檔案載入對話歷史")
            
            while True:
                cmd = input("\n輸入命令 (或按Enter繼續對話): ")
                
                if cmd.lower() in ['退出', '結束', 'exit', 'quit']:
                    break
                elif cmd.lower() in ['清除歷史', 'clear history']:
                    self.clear_history()
                    print("對話歷史已清除")
                elif cmd.lower() in ['保存對話', 'save conversation']:
                    filename = input("請輸入檔案名稱 (預設為 conversation_history.json): ").strip()
                    if not filename:
                        filename = "conversation_history.json"
                    if self.save_conversation(filename):
                        print(f"對話歷史已保存到 {filename}")
                elif cmd.lower() in ['載入對話', 'load conversation']:
                    filename = input("請輸入檔案名稱 (預設為 conversation_history.json): ").strip()
                    if not filename:
                        filename = "conversation_history.json"
                    if self.load_conversation(filename):
                        print(f"已從 {filename} 載入對話歷史")
                else:
                    print("繼續對話中...")
        
        except KeyboardInterrupt:
            print("\n檢測到中斷信號，正在關閉系統...")
        finally:
            self.stop()


# 程式進入點
if __name__ == "__main__":
    # 設定系統參數
    OLLAMA_MODEL = "gemma3:12b"  # 使用的Ollama模型
    OLLAMA_API_URL = "http://localhost:11434/api"  # Ollama API URL
    GPT_SOVITS_URL = "http://localhost:9880"  # GPT-SoVITS API URL
    
    # 參考音檔路徑（替換為實際路徑）
    REF_AUDIO_PATH = "C:/GPT-SoVITS/芙宁娜/参考音频/【正常】茶会是淑女的必修课，如果你想学习茶会礼仪的话，我可以教你哦.wav"
    
    # 可選：GPT和SoVITS模型權重路徑
    GPT_WEIGHTS_PATH = "C:/GPT-SoVITS/芙宁娜/GPT_weights/Furina-e15.ckpt"  
    SOVITS_WEIGHTS_PATH = "C:/GPT-SoVITS/芙宁娜/SoVITS_weights/Furina_e8_s304.pth"
    
    # 系統提示詞
    SYSTEM_PROMPT = """
    你是一個友善、有幫助的AI助手。你具有以下特點：
    你是芙寧娜，某國的統治者與知名歌劇明星。你個性幽默，標準高，私下喜愛小動物。
    """
    
    # 對話記憶容量（回合數）
    MAX_MEMORY = 10
    
    # 創建並啟動系統
    system = VoiceLLMChatSystem(
        ollama_model=OLLAMA_MODEL,
        ollama_api_url=OLLAMA_API_URL,
        gpt_sovits_url=GPT_SOVITS_URL,
        ref_audio_path=REF_AUDIO_PATH,
        gpt_weights_path=GPT_WEIGHTS_PATH,
        sovits_weights_path=SOVITS_WEIGHTS_PATH,
        system_prompt=SYSTEM_PROMPT,
        max_memory=MAX_MEMORY
    )
    
    # 執行系統
    system.run_console_ui()