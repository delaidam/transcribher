"""Minimal local web interface for phone uploads."""

import argparse
import asyncio
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from delaida_transcriber.backends import create_backend
from delaida_transcriber.config import Settings
from delaida_transcriber.local_llm import refine_transcript
from delaida_transcriber.service import SUPPORTED_SUFFIXES, TranscriptionService
from delaida_transcriber.subtitles import to_srt

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
<form id="form"><label>Language<select name="language"><option value="auto">Auto-detect (recommended)</option><option value="hr">Bosnian / Croatian</option><option value="en">English</option><option value="bs">Bosnian (bs code, less accurate)</option></select></label>
<button id="record" type="button">● Snimi i transkribuj</button>
<p id="recordHint" class="muted" hidden></p>
<p class="muted">…ili pošalji gotovu datoteku:</p>
<label>Audio or video<input name="file" type="file" accept="audio/ogg,audio/mpeg,audio/mp4,audio/wav,audio/webm,video/mp4,.ogg,.mp3,.mp4,.m4a,.wav,.webm" required></label>
<button id="button">Transcribe</button></form><p id="status" class="muted"></p><pre id="result"></pre>
<button id="refine" hidden>Pročisti lokalnim modelom i napravi TL;DR</button>
<section id="refined" hidden><h2>Uređeni transkript</h2><pre id="cleaned"></pre><h2>TL;DR</h2><pre id="summary"></pre><h2>Ključne tačke</h2><pre id="points"></pre><h2>Nejasni dijelovi</h2><pre id="unclear"></pre></section>
<a id="download" hidden download="transcription.json">Download JSON result</a>
<a id="downloadSrt" hidden download="transcription.srt">Download SRT subtitles</a>
</main><script>
const form=document.querySelector('#form'),button=document.querySelector('#button'),record=document.querySelector('#record'),recordHint=document.querySelector('#recordHint'),refine=document.querySelector('#refine'),status=document.querySelector('#status'),result=document.querySelector('#result'),refined=document.querySelector('#refined'),cleaned=document.querySelector('#cleaned'),summary=document.querySelector('#summary'),points=document.querySelector('#points'),unclear=document.querySelector('#unclear'),download=document.querySelector('#download'),downloadSrt=document.querySelector('#downloadSrt');let rawText='';
async function transcribe(body){button.disabled=true;record.disabled=true;refine.hidden=true;refined.hidden=true;status.textContent='Transcribing… first use may download the model.';result.textContent='';download.hidden=true;downloadSrt.hidden=true;
try{const response=await fetch('/transcribe',{method:'POST',body});const data=await readJson(response);if(!response.ok)throw Error(data.detail||'Transcription failed');rawText=data.text||'';result.textContent=rawText||'(no speech detected)';refine.hidden=!rawText;const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});download.href=URL.createObjectURL(blob);download.hidden=false;if(data.srt){const srtBlob=new Blob([data.srt],{type:'text/plain'});downloadSrt.href=URL.createObjectURL(srtBlob);downloadSrt.hidden=false;}status.textContent=`Detected: ${data.detected_language||'unknown'}`;}catch(error){status.textContent=error.message;}finally{button.disabled=false;record.disabled=false;}}
form.addEventListener('submit',(e)=>{e.preventDefault();transcribe(new FormData(form));});
async function readJson(response){const body=await response.text();try{return JSON.parse(body);}catch(_){throw Error(body||'Server je vratio neispravan odgovor.');}}
// Chrome and Firefox record Opus in WebM, Safari records audio/mp4. Both decode
// server-side; the extension has to match, because that is all the upload
// endpoint has to go on.
const RECORDING_TYPES=[['audio/webm;codecs=opus','webm'],['audio/webm','webm'],['audio/mp4','m4a'],['audio/ogg;codecs=opus','ogg']];
function pickRecordingType(){if(typeof MediaRecorder==='undefined'||!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)return null;for(const[type,extension]of RECORDING_TYPES)if(MediaRecorder.isTypeSupported(type))return{type,extension};return null;}
const recording=pickRecordingType();let recorder=null,ticker=null;
// getUserMedia exists only in a secure context, so over plain http from a phone
// it is not merely blocked, it is absent. Say why rather than offer a button
// that cannot work.
if(!recording){record.hidden=true;recordHint.hidden=false;recordHint.textContent='Snimanje traži HTTPS ili otvaranje stranice na samom računaru (localhost) — preko obične http adrese preglednik ne daje pristup mikrofonu. Pošalji datoteku ispod.';}
else record.addEventListener('click',async()=>{
if(recorder&&recorder.state==='recording'){recorder.stop();return;}
let stream;try{stream=await navigator.mediaDevices.getUserMedia({audio:true});}catch(error){status.textContent='Mikrofon nije dostupan: '+error.message;return;}
const chunks=[];recorder=new MediaRecorder(stream,{mimeType:recording.type});
recorder.addEventListener('dataavailable',(event)=>{if(event.data.size)chunks.push(event.data);});
recorder.addEventListener('stop',()=>{clearInterval(ticker);stream.getTracks().forEach((track)=>track.stop());record.textContent='● Snimi i transkribuj';const body=new FormData();body.append('file',new Blob(chunks,{type:recording.type}),'dictation.'+recording.extension);body.append('language',form.language.value);transcribe(body);});
recorder.start();const started=Date.now();record.textContent='■ Zaustavi (0s)';status.textContent='Snimam… pritisni ponovo kad završiš.';
ticker=setInterval(()=>{record.textContent='■ Zaustavi ('+Math.round((Date.now()-started)/1000)+'s)';},500);});
refine.addEventListener('click',async()=>{refine.disabled=true;status.textContent='Lokalni model uređuje tekst… kod dužih snimaka ovo može potrajati.';try{const response=await fetch('/refine',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:rawText})});const data=await readJson(response);if(!response.ok)throw Error(data.detail||'Refinement failed');cleaned.textContent=data.cleaned_text||'';summary.textContent=data.summary||'';points.textContent=(data.key_points||[]).map(x=>'• '+x).join('\\n')||'(nema)';unclear.textContent=(data.unclear_parts||[]).map(x=>'• '+x).join('\\n')||'(nema)';refined.hidden=false;status.textContent='Gotovo — obrada je urađena lokalno preko Ollame.';}catch(error){status.textContent=error.message;}finally{refine.disabled=false;}});
</script></body></html>"""


def create_app(
    settings: Settings | None = None, service: TranscriptionService | None = None
) -> FastAPI:
    settings = settings or Settings()
    service = service or TranscriptionService(create_backend(settings))
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
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Only these media formats are supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}.",
            )

        # The size limit guards this upload path only. A CLI run reads a local file
        # straight from disk, so there is nothing to cap and no reason to refuse it.
        contents = await file.read(settings.max_upload_bytes + 1)
        if len(contents) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="The file exceeds the upload limit.")

        try:
            # A directory rather than NamedTemporaryFile: that keeps its handle
            # open, and Windows then refuses every other attempt to open the
            # path, so faster-whisper -- which opens by name -- got "Permission
            # denied" for every upload on Windows.
            with tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory) / f"upload{suffix}"
                temporary.write_bytes(contents)
                result = await service.transcribe(temporary, language)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {error}") from error

        return JSONResponse(
            result.to_dict()
            | {"filename": filename, "requested_language": language, "srt": to_srt(result)}
        )

    @app.post("/refine")
    async def refine(payload: dict[str, str]) -> JSONResponse:
        text = payload.get("text", "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Nema teksta za uređivanje.")
        try:
            result = await asyncio.to_thread(
                refine_transcript, text, settings.ollama_model, settings.ollama_url
            )
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return JSONResponse(result)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the phone-friendly transcription server.")
    parser.add_argument("--host", default=None, help="Bind address; use 0.0.0.0 for phone access.")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU/int8 transcription.")
    args = parser.parse_args()
    settings = Settings(
        device="cpu" if args.cpu else None,
        compute_type="int8" if args.cpu else None,
        host=args.host,
        port=args.port,
    )
    print(f"Open http://{settings.host}:{settings.port}")
    if settings.host == "0.0.0.0":
        print("Phone access: use this computer's LAN IP, for example http://192.168.1.20:8765")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
