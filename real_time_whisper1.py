import os
import numpy as np
import whisper
import torch
import speech_recognition as sr
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from typing import List
from datetime import datetime, timedelta
from queue import Queue
import signal
import sys
from pyngrok import ngrok

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.get("/")
async def get():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>WebSocket Audio Streaming</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f8f9fa;
                display: flex;
                flex-direction: row;
                height: 100vh;
                margin: 0;
                padding: 0;
            }
            .left-panel {
                background-color: #343a40;
                color: white;
                width: 200px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 20px;
                box-sizing: border-box;
            }
            .right-panel {
                flex-grow: 1;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                box-sizing: border-box;
            }
            h1 {
                color: white;
                margin-bottom: 20px;
                text-align: center;
            }
            button {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                margin: 10px;
                cursor: pointer;
                border-radius: 5px;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #0056b3;
            }
            pre {
                background: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                width: 600px; /* 고정된 폭 설정 */
                height: 400px; /* 고정된 높이 설정 */
                overflow: auto;
                font-size: 14px;
                line-height: 1.5;
            }
        </style>
    </head>
    <body>
        <div class="left-panel">
            <h1>캡스톤 불끈불끈 STT</h1>
            <button id="startBtn" onclick="startRecording()">녹음 시작</button>
        </div>
        <div class="right-panel">
            <pre id="transcription"></pre>
        </div>

        <script>
            let ws;
            let mediaRecorder;

            function startRecording() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    console.log("WebSocket is already open.");
                    return;
                }

                ws = new WebSocket("ws://127.0.0.1:8000/ws/transcribe/");

                ws.onopen = function() {
                    console.log("WebSocket connection opened");
                    navigator.mediaDevices.getUserMedia({ audio: true })
                        .then(stream => {
                            mediaRecorder = new MediaRecorder(stream);
                            mediaRecorder.ondataavailable = event => {
                                if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
                                    ws.send(event.data);
                                }
                            };
                            mediaRecorder.start(250);  // Collect 250ms of data chunks
                        });
                };

                ws.onmessage = function(event) {
                    const transcription = document.getElementById("transcription");
                    transcription.textContent += event.data + "\\n";
                };

                ws.onclose = function() {
                    console.log("WebSocket connection closed");
                    // Reconnect after a delay
                    setTimeout(startRecording, 1000);
                };

                ws.onerror = function(error) {
                    console.log("WebSocket error: " + error.message);
                };
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws/transcribe/")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # 큐에서 마지막으로 녹음이 검색된 시간.
        phrase_time = None
        # 스레드 안전 큐로, 스레드 녹음 콜백에서 데이터를 전달합니다.
        data_queue = Queue()
        # 음성이 끝났을 때를 감지할 수 있는 기능이 있는 SpeechRecognizer를 사용하여 오디오를 녹음합니다.
        recorder = sr.Recognizer()
        recorder.energy_threshold = 1000
        recorder.dynamic_energy_threshold = False

        source = sr.Microphone(sample_rate=16000)
        audio_model = whisper.load_model("base")

        record_timeout = 2.0
        phrase_timeout = 3.0

        transcription = ['']

        with source:
            recorder.adjust_for_ambient_noise(source)

        def record_callback(_, audio: sr.AudioData) -> None:
            """
            녹음이 끝날 때 오디오 데이터를 받는 스레드 콜백 함수
            audio: 녹음된 바이트를 포함하는 AudioData.
            """
            # 원시 바이트를 잡아서 스레드 안전 큐에 넣습니다.
            data = audio.get_raw_data()
            data_queue.put(data)

        # 백그라운드 스레드를 생성합니다.
        recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

        while True:
            now = datetime.utcnow()
            # 큐에서 녹음된 원시 오디오를 가져옴
            if not data_queue.empty():
                phrase_complete = False
                # 녹음 사이에 충분한 시간이 지났다면, 구문을 완성된 것으로 간주
                # 현재 작업 중인 오디오 버퍼를 지워서 새 데이터로 시작
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # 큐에서 새 오디오 데이터를 받은 마지막 시간.
                phrase_time = now
                
                # 큐에서 오디오 데이터를 결합
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # 램 버퍼에서 모델이 직접 사용할 수 있는 것으로 변환
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # 전사를 읽습니다.
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                # 녹음 사이에 일시 중지를 감지했다면, 전사에 새 항목을 추가합니다.
                # 그렇지 않다면 기존 것을 편집합니다.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # 클라이언트에 전송
                await manager.send_message(text, websocket)
            else:
                # 프로세스 쉬게 하기
                await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

#프로그램 종료 인터럽터
def signal_handler(sig, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(sig))
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Start ngrok
    public_url = ngrok.connect(8000)
    print("ngrok tunnel \"{}\" -> \"http://127.0.0.1:8000\"".format(public_url))

    # Start FastAPI server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
