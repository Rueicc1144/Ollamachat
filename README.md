# 讓LLM說話(Ollama + GPT-SoVITS)

本專案整合開源的大型語言模型（LLM）與語音合成技術（TTS），實現可部署於本地端的語音互動 AI 助手。

---

## 專案功能

- 使用 [Ollama](https://ollama.com/) API 部署並串接本地 LLM（如 `gemma3:12b`）
- 整合 [GPT-SoVITS](https://github.com/RVC-Project/GPT-SoVITS) 模型，實現TTS
- 支援語音風格指定與語速控制
- 對話歷史儲存於 JSON，具備記憶上限機制
- 音訊播放自動排程，回應可自然斷句播放
- CLI 模式互動介面，可以保存與載入歷史

---
