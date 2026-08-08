"""Minimal local web interface for phone uploads."""

import argparse
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from delaida_transcriber.config import Settings
from delaida_transcriber.service import TranscriptionService
from delaida_transcriber.transcriber import WhisperTranscriber

HTML = """<!doctype html>
<html lang="en"><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Delaida Transcriber</title>
<style>
body{font:16px system-ui,sans-serif;max-width:38rem;margin:2rem auto;padding:0 1rem;background:#f6f4ef;color:#222}
main{background:white;padding:1.5rem;border-radius:1rem;box-shadow:0 4px 20px #0001}
input,select,button{font:inherit;width:100%;box-sizing:border-box;margin:.5rem 0;padding:.75rem;border-radius:.5rem;border:1px solid #bbb}
button{background:#315c52;color:white;border:0;font-weight:700}button:disabled{opacity:.5}
pre{white-space:pre-wrap;background:#f0eee8;padding:1rem;border-radius:.5rem;min-height:3rem}
.muted{color:#666;font-size:.9rem}
</style></head><body><main>
<h1>Delaida Transcriber</h1>
<p class="muted">Private local transcription. Your file is processed on the computer running this page.</p>
<form id="form"><label>Audio or video<input name="file" type="file" accept="audio/ogg,audio/mpeg,audio/mp4,video/mp4,.ogg,.mp3,.mp4,.m4a" required></label>
<label>Language<select name="language"><option value="auto">Auto-detect</option><option value="bs">Bosnian</option><option value="en">English</option></select></label>
<button id="button">Transcribe</button></form><p id="status" class="muted"></p><pre id="result"></pre>
<a id="download" hidden download="transcription.json">Download JSON result</a>
</main><script>
const form=document.querySelector('#form'),button=document.querySelector('#button'),status=document.querySelector('#status'),result=document.querySelector('#result'),download=document.querySelector('#download');
form.addEventListener('submit',async(e)=>{e.preventDefault();button.disabled=true;status.textContent='Transcribing… first use may download the model.';result.textContent='';download.hidden=true;
try{const response=await fetch('/transcribe',{method:'POST',body:new FormData(form)});const data=await response.json();if(!response.ok)throw Error(data.detail||'Transcription failed');result.textContent=data.text||'(no speech detected)';const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});download.href=URL.createObjectURL(blob);download.hidden=false;status.textContent=`Detected: ${data.detected_language||'unknown'}`;}catch(error){status.textContent=error.message;}finally{button.disabled=false;}});
</script></body></html>"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    service = TranscriptionService(WhisperTranscriber(settings), settings.max_upload_bytes)
    app = FastAPI(title="Delaida Transcriber", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return HTML

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/transcribe")
    async def transcribe(
        file: UploadFile = File(...), language: str = Form("auto")
    ) -> JSONResponse:
        filename = file.filename or "recording.ogg"
        if Path(filename).suffix.lower() not in {".ogg", ".mp3", ".mp4"}:
            raise HTTPException(
                status_code=400,
                detail="Only .ogg, .mp3, .mp4, and .m4a files are supported.",
            )

        contents = await file.read(settings.max_upload_bytes + 1)
        if len(contents) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="The file exceeds the upload limit.")

        try:
            with tempfile.NamedTemporaryFile(suffix=".ogg") as temporary:
                temporary.write(contents)
                temporary.flush()
                result = await service.transcribe(Path(temporary.name), language)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {error}") from error

        return JSONResponse(
            result.to_dict() | {"filename": filename, "requested_language": language}
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the phone-friendly transcription server.")
    parser.add_argument("--host", default=None, help="Bind address; use 0.0.0.0 for phone access.")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force the base CPU model.")
    args = parser.parse_args()
    settings = Settings(
        device="cpu" if args.cpu else None,
        model="base" if args.cpu else None,
        host=args.host,
        port=args.port,
    )
    print(f"Open http://{settings.host}:{settings.port}")
    if settings.host == "0.0.0.0":
        print("Phone access: use this computer's LAN IP, for example http://192.168.1.20:8765")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
