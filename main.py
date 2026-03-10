import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import httpx

# Inisialisasi Aplikasi FastAPI
app = FastAPI(title="GenAI Proxy Backend", description="Backend untuk menjembatani Frontend dan Hugging Face Spaces")

# 1. MENGATASI MASALAH CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. KONFIGURASI ENVIRONMENT VARIABLE
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:7860/v1/chat/completions")

# Skema data yang diharapkan dari Frontend
class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    temperature: float = 0.7
    max_tokens: int = 512

# 3 & 4. MENGATASI EFEK MESIN KETIK (STREAMING) & COLD START
async def stream_generator(payload: dict):
    # Mengatasi Cold Start:
    timeout_config = httpx.Timeout(300.0, connect=60.0)
    
    # --- FIX ERROR 400 BAD REQUEST ---
    # Standar OpenAI mewajibkan parameter 'model' dan 'stream'.
    payload["stream"] = True
    # Ubah nama model persis sesuai path file yang di-download di start.sh
    payload["model"] = "./model/qwen1_5-0_5b-chat-q4_k_m.gguf" 
    # ---------------------------------

    # Gunakan httpx AsyncClient agar tidak memblokir server FastAPI
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        try:
            # Melakukan koneksi streaming (Server-Sent Events) ke server LLM siswa di Hugging Face
            async with client.stream("POST", LLM_API_URL, json=payload) as response:
                
                # Jika ada error dari Hugging Face (misal limit atau url salah)
                if response.status_code != 200:
                    await response.aread() # Membaca detail pesan error asli dari Hugging Face
                    error_detail = response.text
                    yield f"data: {json.dumps({'error': f'HF Server Error (Code {response.status_code}): {error_detail}'})}\n\n"
                    return

                # Membaca token demi token secara real-time dan meneruskannya ke Frontend
                async for chunk in response.aiter_lines():
                    if chunk:
                        yield f"{chunk}\n\n"
                        
        except httpx.ReadTimeout:
             yield f"data: {json.dumps({'error': 'Request Timeout. Hugging Face Space mungkin butuh waktu lebih lama untuk bangun dari mode Sleep.'})}\n\n"
        except Exception as exc:
             yield f"data: {json.dumps({'error': f'Koneksi Gagal: {str(exc)}'})}\n\n"

@app.get("/")
async def root():
    return {"message": "Server GenAI Proxy Berjalan Normal! Silakan tembak endpoint /api/chat dari Frontend Anda."}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint ini akan di-hit oleh antarmuka Frontend.
    Ia menerima JSON riwayat chat, lalu membukakan terowongan streaming ke Hugging Face.
    """
    # Ubah format request dari Pydantic menjadi Dictionary (JSON)
    payload = request.model_dump()
    
    # Kembalikan response berupa Server-Sent Events (SSE) stream
    return StreamingResponse(stream_generator(payload), media_type="text/event-stream")
