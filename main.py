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
# Mengizinkan request dari domain frontend mana saja (Render/Vercel/Localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Catatan untuk siswa: Di production sungguhan, ganti "*" dengan URL Frontend kalian
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. KONFIGURASI ENVIRONMENT VARIABLE
# Mengambil URL Hugging Face Spaces milik siswa dari Environment Variables Render/Railway.
# Jika tidak ada (misal saat testing lokal), gunakan URL default localhost.
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:7860/v1/chat/completions")

# Skema data yang diharapkan dari Frontend
class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    temperature: float = 0.7
    max_tokens: int = 512

# 3 & 4. MENGATASI EFEK MESIN KETIK (STREAMING) & COLD START
async def stream_generator(payload: dict):
    # Mengatasi Cold Start:
    # Set timeout yang panjang (5 menit / 300 detik) karena saat container Hugging Face tertidur,
    # ia butuh waktu sekitar 1-2 menit untuk bangun dan memuat model ke RAM.
    timeout_config = httpx.Timeout(300.0, connect=60.0)
    
    # Pastikan request diset agar meminta streaming dari llama.cpp
    payload["stream"] = True
    payload["model"] = "qwen-0.5b"

    # Gunakan httpx AsyncClient agar tidak memblokir server FastAPI
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        try:
            # Melakukan koneksi streaming (Server-Sent Events) ke server LLM siswa di Hugging Face
            async with client.stream("POST", LLM_API_URL, json=payload) as response:
                
                # Jika ada error dari Hugging Face (misal limit atau url salah)
                if response.status_code != 200:
                    yield f"data: {json.dumps({'error': f'HF Server Error (Code {response.status_code})'})}\n\n"
                    return

                # Membaca token demi token secara real-time dan meneruskannya ke Frontend
                async for chunk in response.aiter_lines():
                    if chunk:
                        # Chunk sudah berformat 'data: {"id":"...","choices":[...]}', 
                        # kita tinggal menambahkan double newline (\n\n) sesuai standar SSE
                        yield f"{chunk}\n\n"
                        
        except httpx.ReadTimeout:
             yield f"data: {json.dumps({'error': 'Request Timeout. Hugging Face Space mungkin butuh waktu lebih lama untuk bangun dari mode Sleep.'})}\n\n"
        except Exception as exc:
             yield f"data: {json.dumps({'error': f'Koneksi Gagal: {str(exc)}'})}\n\n"

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

# Catatan Tambahan: Server ini dijalankan dengan Uvicorn, misal:

# uvicorn main:app --host 0.0.0.0 --port 10000
