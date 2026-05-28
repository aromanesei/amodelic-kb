"""
AMODELIC Knowledge Base — Railway Microservice
Responsabilitate: PDF/DOCX/XLSX/PPTX → Markdown + chunks logice
NU face embeddings (acelea se fac în Edge Function cu OpenAI direct)
"""

import os, re, tempfile, logging
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from markitdown import MarkItDown
from pydantic import BaseModel

API_KEY   = os.environ.get("AMODELIC_API_KEY", "schimba-cheia")
MAX_MB    = int(os.environ.get("MAX_FILE_MB", "50"))
ORIGINS   = os.environ.get("ALLOWED_ORIGINS", "https://app.amodelic.com").split(",")

ALLOWED_EXT = {".pdf",".docx",".doc",".xlsx",".xls",".pptx",".ppt",
               ".png",".jpg",".jpeg",".msg",".html",".htm",".epub"}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("amodelic-kb")

app = FastAPI(title="AMODELIC KB Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS,
                   allow_methods=["POST","GET"], allow_headers=["*"])
converter = MarkItDown()


# ── Auth ──────────────────────────────────────────────────────────────────────
def auth(key: str):
    if key != API_KEY:
        raise HTTPException(401, "API key invalid")


# ── Chunking logic ─────────────────────────────────────────────────────────────
def chunk_markdown(markdown: str, max_tokens: int = 800) -> list[dict]:
    """
    Împarte Markdown în secțiuni logice respectând structura documentului.
    Prioritate: headere (##, ###) → paragrafe → fallback la ~800 tokeni.
    Returnează: [{chunk_index, section_title, chunk_text, token_count}]
    """
    chunks = []
    
    # Split pe headere de nivel 2+ (##, ###, ####)
    # Păstrăm headerul în chunk-ul lui
    header_pattern = re.compile(r'^(#{2,6}\s+.+)$', re.MULTILINE)
    parts = header_pattern.split(markdown)
    
    # parts = [text_before_first_header, header1, content1, header2, content2, ...]
    # Reconstruim perechi (header, content)
    sections = []
    
    # Text înainte de primul header (intro, metadata)
    preamble = parts[0].strip()
    if preamble:
        sections.append(("Document", preamble))
    
    # Restul: header + conținut
    i = 1
    while i < len(parts) - 1:
        header  = parts[i].strip()
        content = parts[i+1].strip() if i+1 < len(parts) else ""
        if content:
            sections.append((header, content))
        elif header:
            # Header fără conținut — îl lipim la secțiunea anterioară
            if sections:
                prev_h, prev_c = sections[-1]
                sections[-1] = (prev_h, prev_c + "\n\n" + header)
        i += 2
    
    # Dacă nu avem headere (document plat), facem split pe paragrafe
    if not sections:
        paragraphs = [p.strip() for p in markdown.split("\n\n") if p.strip()]
        current_title = "Document"
        current_text  = ""
        for para in paragraphs:
            est = len((current_text + para).split()) * 1.3  # rough token estimate
            if est > max_tokens and current_text:
                sections.append((current_title, current_text.strip()))
                current_title = "Continuare"
                current_text  = para + "\n\n"
            else:
                current_text += para + "\n\n"
        if current_text.strip():
            sections.append((current_title, current_text.strip()))
    
    # Acum verificăm dacă vreun chunk e prea mare și îl spargem
    final_chunks = []
    for title, text in sections:
        token_est = int(len(text.split()) * 1.3)
        if token_est <= max_tokens:
            final_chunks.append((title, text, token_est))
        else:
            # Split pe paragrafe
            paras = [p.strip() for p in text.split("\n\n") if p.strip()]
            sub_text, sub_tokens = "", 0
            sub_index = 0
            for para in paras:
                pt = int(len(para.split()) * 1.3)
                if sub_tokens + pt > max_tokens and sub_text:
                    lbl = f"{title} [{sub_index+1}]" if sub_index else title
                    final_chunks.append((lbl, sub_text.strip(), sub_tokens))
                    sub_text, sub_tokens, sub_index = "", 0, sub_index + 1
                sub_text   += para + "\n\n"
                sub_tokens += pt
            if sub_text.strip():
                lbl = f"{title} [{sub_index+1}]" if sub_index else title
                final_chunks.append((lbl, sub_text.strip(), sub_tokens))
    
    # Normalizare titluri — scoatem prefixul ## din headere
    result = []
    for idx, (title, text, tokens) in enumerate(final_chunks):
        clean_title = re.sub(r'^#{2,6}\s*', '', title).strip()
        result.append({
            "chunk_index":   idx,
            "section_title": clean_title or f"Secțiunea {idx+1}",
            "chunk_text":    text,
            "token_count":   tokens,
        })
    
    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "amodelic-kb"}


@app.post("/process")
async def process_file(
    file: UploadFile = File(...),
    x_api_key: str = Header(...),
):
    """
    Primește fișier → returnează markdown_text + chunks.
    Embeddings NU se fac aici — le face Edge Function cu OpenAI.
    """
    auth(x_api_key)

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(400, f"Format nesuportat: {suffix}")

    content = await file.read()
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Fișier prea mare. Limită: {MAX_MB}MB")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result   = converter.convert(tmp_path)
        markdown = result.text_content or ""

        if not markdown.strip():
            raise HTTPException(422, "Document gol sau nescanabil (PDF imagine fără OCR)")

        chunks = chunk_markdown(markdown)

        log.info(f"Procesat: {file.filename} → {len(markdown)} chars, {len(chunks)} chunks")

        return {
            "markdown_text": markdown,
            "chunks":        chunks,
            "chunk_count":   len(chunks),
            "char_count":    len(markdown),
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Eroare procesare {file.filename}: {e}")
        raise HTTPException(500, f"Eroare procesare: {str(e)}")
    finally:
        os.unlink(tmp_path)


class UrlPayload(BaseModel):
    url:      str
    filename: Optional[str] = None


@app.post("/process-url")
async def process_from_url(
    payload:   UrlPayload,
    x_api_key: str = Header(...),
):
    """
    Descarcă fișier de la URL Supabase Storage și returnează markdown + chunks.
    Apelat din Edge Function cu URL semnat din Storage.
    """
    auth(x_api_key)

    url_path = payload.url.split("?")[0]
    suffix   = Path(payload.filename or url_path).suffix.lower() or ".pdf"

    if suffix not in ALLOWED_EXT:
        raise HTTPException(400, f"Format nesuportat: {suffix}")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(payload.url)
            r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(400, f"Download eșuat: {str(e)}")

    content = r.content
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Fișier prea mare. Limită: {MAX_MB}MB")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result   = converter.convert(tmp_path)
        markdown = result.text_content or ""

        if not markdown.strip():
            raise HTTPException(422, "Document gol sau nescanabil")

        chunks = chunk_markdown(markdown)

        log.info(f"Procesat URL: ...{url_path[-40:]} → {len(chunks)} chunks")

        return {
            "markdown_text": markdown,
            "chunks":        chunks,
            "chunk_count":   len(chunks),
            "char_count":    len(markdown),
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Eroare procesare URL: {e}")
        raise HTTPException(500, f"Eroare: {str(e)}")
    finally:
        os.unlink(tmp_path)
