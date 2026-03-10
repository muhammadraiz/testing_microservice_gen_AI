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

# 3. FUNGSI STREAMING KE HUGGING FACE
async def stream_generator(payload: dict):
    timeout_config = httpx.Timeout(300.0, connect=60.0)
    
    # Standar OpenAI parameter
    payload["stream"] = True
    # WAJIB ADA: Server akan menolak jika parameter model kosong
    payload["model"] = "qwen"

    # Tambahkan User-Agent standar dan Accept Header khusus Streaming agar tidak diblokir proxy HF
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    async with httpx.AsyncClient(timeout=timeout_config) as client:
        try:
            async with client.stream("POST", LLM_API_URL, json=payload, headers=headers) as response:
                
                if response.status_code != 200:
                    await response.aread() 
                    error_detail = response.text.strip()
                    # Menambahkan pelacak URL untuk mendeteksi salah ketik
                    yield f"data: {json.dumps({'error': f'HF Error ({response.status_code}) di URL {LLM_API_URL}. Info: {error_detail}'})}\n\n"
                    return

                async for chunk in response.aiter_lines():
                    if chunk:
                        yield f"{chunk}\n\n"
                        
        except httpx.ReadTimeout:
             yield f"data: {json.dumps({'error': 'Request Timeout. Hugging Face sedang loading model.'})}\n\n"
        except Exception as exc:
             yield f"data: {json.dumps({'error': f'Koneksi Gagal: {str(exc)}'})}\n\n"

@app.get("/")
async def root():
    return {"message": "Server GenAI Proxy Berjalan Normal! Pastikan LLM_API_URL di Environment sudah benar."}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    payload = request.model_dump()
    return StreamingResponse(stream_generator(payload), media_type="text/event-stream")
