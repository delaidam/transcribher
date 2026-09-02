"""Minimal local web interface for phone uploads."""

import argparse
import asyncio
import tempfile
from collections.abc import Callable
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from delaida_transcriber import tasks
from delaida_transcriber.backends import create_backend
from delaida_transcriber.config import Settings
from delaida_transcriber.llm_backends import LLMBackend, create_llm_backend
from delaida_transcriber.local_llm import chat_budget_chars, validate_history
from delaida_transcriber.service import SUPPORTED_SUFFIXES, TranscriptionService
from delaida_transcriber.store import SessionStore
from delaida_transcriber.subtitles import to_srt

# The chat cap exists because Ollama truncates an over-long prompt instead of
# refusing it. A hosted model with a million-token window does not do that, so
# applying the local ceiling there would refuse work that would have succeeded.
# This is a sanity bound, not a context limit.
HOSTED_CHAT_BUDGET = 2_000_000

HTML = """<!doctype html>
<html lang="en"><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Delaida Transcriber</title>
<style>
body{font:16px system-ui,sans-serif;max-width:38rem;margin:2rem auto;padding:0 1rem;background:#f6f4ef;color:#222}
main{background:white;padding:1.5rem;border-radius:1rem;box-shadow:0 4px 20px #0001}
input,select,button,textarea{font:inherit;width:100%;box-sizing:border-box;margin:.5rem 0;padding:.75rem;border-radius:.5rem;border:1px solid #bbb}
textarea{resize:vertical}
.msg{padding:.6rem .85rem;border-radius:.6rem;margin:.4rem 0;white-space:pre-wrap}
.msg.user{background:#e6efe9;margin-left:2rem}
.msg.assistant{background:#f0eee8;margin-right:2rem}
.msg b{display:block;font-size:.8rem;color:#666;margin-bottom:.15rem}
.session{display:flex;gap:.5rem;align-items:baseline;padding:.5rem .25rem;border-bottom:1px solid #e4e1d8}
.session:last-child{border-bottom:0}
.session button{width:auto;margin:0;padding:.35rem .6rem;font-size:.85rem}
.session .open{flex:1;text-align:left;background:none;color:#315c52;border:0;font-weight:600;cursor:pointer}
.session .open small{display:block;color:#888;font-weight:400}
.session .remove{background:none;border:1px solid #ccc;color:#933}
.session.active .open{text-decoration:underline}
button{background:#315c52;color:white;border:0;font-weight:700}button:disabled{opacity:.5}
pre{white-space:pre-wrap;background:#f0eee8;padding:1rem;border-radius:.5rem;min-height:3rem}
.muted{color:#666;font-size:.9rem}
.warn{background:#fbeee0;border:1px solid #d9a441;color:#7a4b00;padding:.7rem .9rem;border-radius:.5rem;font-size:.9rem}
</style></head><body><main>
<h1>Delaida Transcriber</h1>
<p class="muted">Private local transcription. Your file is processed on the computer running this page.</p>
<section id="library" hidden><h2>Ranije</h2><div id="sessions"></div></section>
<form id="form"><label>Language<select name="language" data-default="__LANGUAGE__"><option value="auto">Auto-detect (recommended)</option><option value="hr">Bosnian / Croatian</option><option value="en">English</option><option value="no">Norsk</option><option value="bs">Bosnian (bs code, less accurate)</option></select></label>
<button id="record" type="button">● Snimi i transkribuj</button>
<p id="recordHint" class="muted" hidden></p>
<p class="muted">…ili pošalji gotovu datoteku:</p>
<label>Audio or video<input name="file" type="file" accept="audio/ogg,audio/mpeg,audio/mp4,audio/wav,audio/webm,video/mp4,.ogg,.mp3,.mp4,.m4a,.wav,.webm" required></label>
<button id="button">Transcribe</button></form><p id="status" class="muted"></p><pre id="result"></pre>
<button id="copyText" type="button" hidden>Kopiraj transkript</button>
<section id="tools" hidden>
<p id="hosted" class="warn" hidden></p>
<label>Šta uraditi s transkriptom<select id="task"></select></label>
<label id="instructionRow" hidden>Šta tačno želiš?<textarea id="instruction" rows="3" placeholder="npr. izvuci samo ono što se tiče rokova"></textarea></label>
<label>Jezik odgovora<select id="outputLanguage"></select></label>
<button id="run" type="button">Pokreni lokalni model</button>
</section>
<section id="output" hidden></section>
<button id="copyOutput" type="button" hidden>Kopiraj odgovor</button>
<section id="chatBox" hidden><h2>Pitaj o snimku</h2>
<div id="messages"></div>
<textarea id="question" rows="2" placeholder="npr. šta smo rekli o rokovima?"></textarea>
<button id="send" type="button">Pošalji</button>
<button id="resetChat" type="button" hidden>Počni novi razgovor</button></section>
<a id="download" hidden download="transcription.json">Download JSON result</a>
<a id="downloadSrt" hidden download="transcription.srt">Download SRT subtitles</a>
</main><script>
const $=(id)=>document.querySelector(id);
const form=$('#form'),button=$('#button'),record=$('#record'),recordHint=$('#recordHint'),status=$('#status'),result=$('#result'),download=$('#download'),downloadSrt=$('#downloadSrt');
const tools=$('#tools'),taskSelect=$('#task'),instructionRow=$('#instructionRow'),instruction=$('#instruction'),outputLanguage=$('#outputLanguage'),run=$('#run'),output=$('#output'),copyText=$('#copyText'),copyOutput=$('#copyOutput');
const chatBox=$('#chatBox'),messages=$('#messages'),question=$('#question'),send=$('#send'),resetChat=$('#resetChat');
const library=$('#library'),sessions=$('#sessions'),hosted=$('#hosted');
let rawText='',taskIndex={},history=[],currentSession=null;
form.language.value=form.language.dataset.default;
async function transcribe(body){button.disabled=true;record.disabled=true;tools.hidden=true;output.hidden=true;copyText.hidden=true;copyOutput.hidden=true;chatBox.hidden=true;clearChat();status.textContent='Transcribing… first use may download the model.';result.textContent='';download.hidden=true;downloadSrt.hidden=true;
try{const response=await fetch('/transcribe',{method:'POST',body});const data=await readJson(response);if(!response.ok)throw Error(data.detail||'Transcription failed');rawText=data.text||'';result.textContent=rawText||'(no speech detected)';tools.hidden=!rawText;copyText.hidden=!rawText;chatBox.hidden=!rawText;const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});download.href=URL.createObjectURL(blob);download.hidden=false;if(data.srt){const srtBlob=new Blob([data.srt],{type:'text/plain'});downloadSrt.href=URL.createObjectURL(srtBlob);downloadSrt.hidden=false;}preselectLanguage(data.detected_language);currentSession=data.id||null;if(currentSession)location.hash=currentSession;loadSessions();status.textContent=`Detected: ${data.detected_language||'unknown'}`;}catch(error){status.textContent=error.message;}finally{button.disabled=false;record.disabled=false;}}
// Naming the answer's language works; asking the model to infer it does not --
// left to infer it answered a Bosnian transcript in English. Whisper already
// knows, so start the dropdown there and let the user override it.
const DETECTED_TO_OUTPUT={bs:'bs',hr:'bs',sr:'bs',en:'en',no:'no',nn:'no',nb:'no'};
function preselectLanguage(detected){const code=DETECTED_TO_OUTPUT[detected];if(!code)return;if([...outputLanguage.options].some(o=>o.value===code))outputLanguage.value=code;}
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
// The task menu comes from the server so a new preset needs no change here.
(async()=>{try{const data=await(await fetch('/tasks')).json();for(const task of data.tasks){taskIndex[task.id]=task;taskSelect.add(new Option(task.label,task.id));}taskSelect.value=data.default_task;if(data.backend&&!data.backend.local){hosted.hidden=false;hosted.textContent='Obrada ide preko '+data.backend.name+' — transkript napušta ovaj računar. Transkripcija ostaje lokalna.';}for(const language of data.output_languages)outputLanguage.add(new Option(language.label,language.code));syncInstruction();}catch(_){tools.hidden=true;}})();
function syncInstruction(){instructionRow.hidden=!(taskIndex[taskSelect.value]||{}).needs_instruction;}
taskSelect.addEventListener('change',syncInstruction);
function renderText(value){return Array.isArray(value)?value.map(x=>'• '+x).join('\\n'):String(value??'');}
run.addEventListener('click',async()=>{
const task=taskIndex[taskSelect.value]||{};
if(task.needs_instruction&&!instruction.value.trim()){status.textContent='Napiši šta želiš da se uradi s transkriptom.';instruction.focus();return;}
run.disabled=true;output.hidden=true;copyOutput.hidden=true;status.textContent='Lokalni model radi… kod dužih snimaka ovo može potrajati.';
// A saved session refines against its stored transcript, so the result is
// filed under the recording it came from.
try{const url=currentSession?`/sessions/${currentSession}/refine`:'/refine';
const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:rawText,task:taskSelect.value,instruction:instruction.value,output_language:outputLanguage.value})});
const data=await readJson(response);if(!response.ok)throw Error(data.detail||'Obrada nije uspjela');
renderResult(data);status.textContent='Gotovo — obrada je urađena lokalno preko Ollame.';}
catch(error){status.textContent=error.message;}finally{run.disabled=false;}});
function renderResult(data){output.replaceChildren();
if(data.kind==='text'){const block=document.createElement('pre');block.textContent=data.output||'';output.append(block);}
else for(const field of data.fields||[]){const heading=document.createElement('h2');heading.textContent=field.label;const block=document.createElement('pre');block.textContent=renderText(field.value)||'(nema)';output.append(heading,block);}
output.hidden=false;copyOutput.hidden=false;}
// --- the library ---------------------------------------------------------
// Work is saved server-side, so a reload restores what you were looking at
// instead of losing the recording and everything made from it.
async function loadSessions(){
try{const data=await(await fetch('/sessions?limit=25')).json();
sessions.replaceChildren();library.hidden=!data.sessions.length;
for(const session of data.sessions){
const row=document.createElement('div');row.className='session'+(session.id===currentSession?' active':'');
const open=document.createElement('button');open.className='open';open.type='button';
open.append(session.title);const when=document.createElement('small');
when.textContent=new Date(session.created_at).toLocaleString()+(session.has_audio?' · sa zvukom':'');
open.append(when);open.addEventListener('click',()=>openSession(session.id));
const remove=document.createElement('button');remove.className='remove';remove.type='button';remove.textContent='Obriši';
remove.addEventListener('click',()=>deleteSession(session.id,session.title));
row.append(open,remove);sessions.append(row);}}
catch(_){library.hidden=true;}}
async function openSession(id){
status.textContent='Otvaram…';
try{const response=await fetch('/sessions/'+id);const data=await readJson(response);
if(!response.ok)throw Error(data.detail||'Nije moguće otvoriti snimak');
currentSession=id;location.hash=id;rawText=data.text||'';
result.textContent=rawText;tools.hidden=false;copyText.hidden=false;chatBox.hidden=false;
output.hidden=true;copyOutput.hidden=true;download.hidden=true;downloadSrt.hidden=true;
preselectLanguage(data.detected_language);
clearChat();for(const message of data.messages||[]){addMessage(message.role,message.content);history.push(message);}
resetChat.hidden=!history.length;
const last=(data.outputs||[]).slice(-1)[0];if(last)renderResult(last.payload);
loadSessions();status.textContent='';}
catch(error){status.textContent=error.message;}}
async function deleteSession(id,title){
if(!confirm(`Obrisati „${title}”? Transkript, obrade i razgovor se brišu trajno.`))return;
try{const response=await fetch('/sessions/'+id,{method:'DELETE'});
if(!response.ok)throw Error((await readJson(response)).detail||'Brisanje nije uspjelo');
if(currentSession===id){currentSession=null;location.hash='';rawText='';result.textContent='';
tools.hidden=true;chatBox.hidden=true;output.hidden=true;copyText.hidden=true;copyOutput.hidden=true;clearChat();}
loadSessions();status.textContent='Obrisano.';}
catch(error){status.textContent=error.message;}}
loadSessions();
if(location.hash.length>1)openSession(location.hash.slice(1));

// The whole conversation goes back every turn -- the server is stateless and
// keeps the transcript and the rules in a system message it builds itself.
function clearChat(){history=[];messages.replaceChildren();resetChat.hidden=true;}
function addMessage(role,content){const el=document.createElement('div');el.className='msg '+role;const who=document.createElement('b');who.textContent=role==='user'?'Ti':'Model';el.append(who,document.createTextNode(content));messages.append(el);el.scrollIntoView({block:'nearest'});}
async function ask(){
const text=question.value.trim();if(!text)return;
send.disabled=true;question.value='';addMessage('user',text);history.push({role:'user',content:text});
status.textContent='Model čita transkript i odgovara…';
try{const url=currentSession?`/sessions/${currentSession}/chat`:'/chat';
const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:rawText,messages:history,output_language:outputLanguage.value})});
const data=await readJson(response);
if(!response.ok){history.pop();throw Error(data.detail||'Razgovor nije uspio');}
addMessage('assistant',data.reply);history.push({role:'assistant',content:data.reply});
resetChat.hidden=false;status.textContent='';}
catch(error){status.textContent=error.message;}finally{send.disabled=false;question.focus();}}
send.addEventListener('click',ask);
// Enter sends, Shift+Enter is a newline -- the usual bargain in a chat box.
question.addEventListener('keydown',(event)=>{if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();ask();}});
resetChat.addEventListener('click',clearChat);
async function copy(text,btn,label){try{await navigator.clipboard.writeText(text);btn.textContent='Kopirano';setTimeout(()=>{btn.textContent=label;},1500);}catch(_){status.textContent='Preglednik nije dao pristup clipboardu.';}}
copyText.addEventListener('click',()=>copy(rawText,copyText,'Kopiraj transkript'));
copyOutput.addEventListener('click',()=>copy(output.innerText,copyOutput,'Kopiraj odgovor'));
</script></body></html>"""


def create_app(
    settings: Settings | None = None,
    service: TranscriptionService | None = None,
    llm: Callable[..., dict[str, object]] | None = None,
    chat: Callable[..., dict[str, object]] | None = None,
    store: SessionStore | None = None,
    backend: LLMBackend | None = None,
) -> FastAPI:
    """Build the app.

    ``llm``, ``chat`` and ``store`` are injectable for the same reason
    ``service`` is: without them every test of those paths would need a running
    Ollama, a loaded model, and the user's real data directory.
    """
    settings = settings or Settings()
    service = service or TranscriptionService(create_backend(settings))
    backend = backend or create_llm_backend(settings)
    llm = llm or backend.run_task
    chat = chat or backend.chat
    store = store or SessionStore(settings.data_dir)
    app = FastAPI(title="Delaida Transcriber", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        # The page opens on whatever STT_LANGUAGE says, so the phone agrees with
        # the CLI instead of quietly defaulting to auto-detect.
        return HTML.replace("__LANGUAGE__", settings.language)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/transcribe")
    async def transcribe(
        file: UploadFile = File(...), language: str | None = Form(None)
    ) -> JSONResponse:
        language = language or settings.language
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

        # Saved before the response goes out, so a reload cannot lose it. Audio
        # is kept only when STT_KEEP_AUDIO says so; the transcript always is.
        session_id = await asyncio.to_thread(
            store.create,
            text=result.text,
            segments=result.to_dict(),
            filename=filename,
            language=language,
            detected_language=result.language,
            audio=contents if settings.keep_audio else None,
            suffix=suffix,
        )

        return JSONResponse(
            result.to_dict()
            | {
                "id": session_id,
                "filename": filename,
                "requested_language": language,
                "srt": to_srt(result),
            }
        )

    @app.get("/tasks")
    async def list_tasks() -> JSONResponse:
        # The page builds its menu from this rather than hardcoding one, so a
        # new preset needs no change to the HTML.
        return JSONResponse(
            {
                "tasks": [
                    {
                        "id": task.id,
                        "label": task.label,
                        "structured": task.structured,
                        "needs_instruction": not task.instruction,
                    }
                    for task in tasks.TASKS
                ],
                "output_languages": [
                    {"code": code, "label": label} for code, label in tasks.OUTPUT_LANGUAGES
                ],
                "default_task": tasks.DEFAULT_TASK_ID,
                "backend": {"name": backend.name, "local": backend.local},
            }
        )

    def _load(session_id: str):
        session = store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Taj snimak ne postoji.")
        return session

    async def _refine(text: str, payload: dict) -> tuple[dict, object, str, str | None]:
        # Defaulting to "refine" keeps every client that predates the task
        # parameter working unchanged.
        try:
            task = tasks.get(payload.get("task") or tasks.DEFAULT_TASK_ID)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        instruction = (payload.get("instruction") or "").strip()
        if not task.instruction and not instruction:
            raise HTTPException(
                status_code=400,
                detail=f"Radnja „{task.label}” traži da napišeš šta želiš da se uradi.",
            )

        output_language = (payload.get("output_language") or "").strip() or None
        try:
            result = await asyncio.to_thread(
                llm,
                text,
                task,
                output_language=output_language,
                instruction=instruction or None,
            )
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return result, task, instruction, output_language

    @app.post("/refine")
    async def refine(payload: dict) -> JSONResponse:
        text = (payload.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Nema teksta za obradu.")
        result, _, _, _ = await _refine(text, payload)
        return JSONResponse(result)

    @app.post("/sessions/{session_id}/refine")
    async def refine_session(session_id: str, payload: dict) -> JSONResponse:
        """The same work, kept. The transcript comes from the store rather than
        the client, so a saved session cannot be refined against other text."""
        session = _load(session_id)
        result, task, instruction, output_language = await _refine(session.text, payload)
        await asyncio.to_thread(
            store.add_output,
            session_id,
            task=task.id,
            payload=result,
            instruction=instruction or None,
            output_language=output_language,
        )
        return JSONResponse(result)

    async def _chat(text: str, payload: dict) -> tuple[dict, list]:
        # The transcript is in the system message on every turn, so it spends the
        # window before the conversation does. Check it here, where both halves
        # are known -- the model will not complain, it will just stop seeing the
        # start of the recording.
        budget = (
            chat_budget_chars(settings.ollama_num_ctx) - len(text)
            if backend.local
            else HOSTED_CHAT_BUDGET
        )
        if budget <= 0:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Transkript sam po sebi prelazi ono što model može držati. "
                    "Povećaj OLLAMA_NUM_CTX ili koristi manji model s većim prozorom."
                ),
            )
        try:
            history = validate_history(payload.get("messages"), budget)
        except ValueError as error:
            # Too long is a 413 so the page can tell it apart from a malformed
            # turn; everything else here is the client's shape being wrong.
            status = 413 if "prelaze" in str(error) or "predugačak" in str(error) else 400
            raise HTTPException(status_code=status, detail=str(error)) from error

        try:
            result = await asyncio.to_thread(
                chat,
                text,
                history,
                output_language=(payload.get("output_language") or "").strip() or None,
            )
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return result, history

    @app.post("/chat")
    async def chat_turn(payload: dict) -> JSONResponse:
        text = (payload.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Nema transkripta za razgovor.")
        result, _ = await _chat(text, payload)
        return JSONResponse(result)

    @app.post("/sessions/{session_id}/chat")
    async def chat_session(session_id: str, payload: dict) -> JSONResponse:
        session = _load(session_id)
        result, history = await _chat(session.text, payload)
        # Only the turn that just happened is appended; the client sends the
        # whole conversation each time, and storing all of it again every turn
        # would multiply it.
        await asyncio.to_thread(
            store.add_messages,
            session_id,
            [history[-1], {"role": "assistant", "content": result["reply"]}],
        )
        return JSONResponse(result)

    @app.get("/sessions")
    async def list_sessions(limit: int = 50, offset: int = 0) -> JSONResponse:
        summaries = await asyncio.to_thread(store.list, limit, offset)
        total = await asyncio.to_thread(store.count)
        return JSONResponse(
            {"sessions": [summary.to_dict() for summary in summaries], "total": total}
        )

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> JSONResponse:
        session = _load(session_id)
        return JSONResponse(session.to_dict() | {"srt": None})

    @app.patch("/sessions/{session_id}")
    async def rename_session(session_id: str, payload: dict) -> JSONResponse:
        try:
            renamed = await asyncio.to_thread(
                store.rename, session_id, payload.get("title") or ""
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if not renamed:
            raise HTTPException(status_code=404, detail="Taj snimak ne postoji.")
        return JSONResponse({"id": session_id, "title": payload["title"].strip()})

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str) -> JSONResponse:
        # The only destructive action in the app. The page confirms first.
        if not await asyncio.to_thread(store.delete, session_id):
            raise HTTPException(status_code=404, detail="Taj snimak ne postoji.")
        return JSONResponse({"deleted": session_id})

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
