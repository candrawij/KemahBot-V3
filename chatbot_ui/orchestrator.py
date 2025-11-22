# orchestrator_final.py
# Versi final aman untuk production / lokal
# - Mapping field disesuaikan dengan API HF
# - Batasi konteks ke top 3
# - Path .env absolut
# - Debug logs terperinci
# - Fallback jika Gemini tidak tersedia
# - Proteksi terhadap input/response yang tidak valid

import os
import sys
import json
import traceback
import httpx
from dotenv import load_dotenv

# Optional: jika Anda menggunakan google generative api
try:
    import google.generativeai as genai
except Exception:
    genai = None

# -----------------------
# CONFIG
# -----------------------
DEBUG = True
# URL retrieval API (HuggingFace)
RETRIEVAL_API_URL = os.getenv("RETRIEVAL_API_URL", "https://Kaira21-campground-api.hf.space/search")
# limit jumlah hasil yang dikirim ke LLM
TOP_K = 3

def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# -----------------------
# Load .env (absolute path)
# -----------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dotenv_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    log(f"[ENV] Loaded .env from {dotenv_path}")
else:
    log(f"[ENV] .env not found at {dotenv_path} (continuing without it)")

# -----------------------
# Initialize LLM (Gemini) if available
# -----------------------
model = None
if genai is not None:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            generation_config = {
                "temperature": 0.6,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 1024,
            }
            model = genai.GenerativeModel(
                model_name=os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash"),
                generation_config=generation_config
            )
            log("[LLM] Gemini initialized.")
        except Exception as e:
            log("[LLM] Failed to init Gemini:", e)
            log(traceback.format_exc())
            model = None
    else:
        log("[LLM] GEMINI_API_KEY not set in .env")
else:
    log("[LLM] google.generativeai not installed; running without LLM (fallback only).")

# -----------------------
# Helper: sanitize/prune API context
# -----------------------
def simplify_context(api_list):
    """
    Normalize the API response items to a small set of fields:
    - nama (from 'name')
    - lokasi (from 'location')
    - rating (from 'avg_rating')
    - fasilitas (from 'facilities' if exists)
    - harga_termurah (min of price_items[*]['harga'])
    """
    cleaned = []
    for item in api_list[:TOP_K]:
        try:
            # Support both snake_case and camelCase if needed
            name = item.get("name") or item.get("nama") or item.get("Nama_Tempat")
            location = item.get("location") or item.get("lokasi") or item.get("Lokasi")
            rating = item.get("avg_rating") or item.get("Avg_Rating") or item.get("rating")
            facilities = item.get("facilities") or item.get("Facilities") or ""
            # price_items may be list of dicts with keys 'item' and 'harga'
            harga_termurah = None
            price_items = item.get("price_items") or item.get("Price_Items") or []
            if isinstance(price_items, list) and price_items:
                numeric_prices = []
                for p in price_items:
                    if isinstance(p, dict):
                        h = p.get("harga") or p.get("price") or p.get("harga_rupiah")
                        try:
                            numeric_prices.append(float(h))
                        except Exception:
                            # skip non-numeric
                            continue
                if numeric_prices:
                    harga_termurah = min(numeric_prices)
            cleaned.append({
                "nama": name,
                "lokasi": location,
                "rating": rating,
                "fasilitas": facilities,
                "harga_termurah": harga_termurah
            })
        except Exception as e:
            log("[simplify_context] item skip due to:", e)
            log(traceback.format_exc())
            continue
    return cleaned

# -----------------------
# Helper: safe LLM call wrappers
# -----------------------
def call_llm(prompt: str, max_tokens: int = 1024):
    """
    Wrapper aman untuk Gemini — tidak lagi crash meskipun response.text tidak tersedia.
    """
    if model is None:
        raise RuntimeError("LLM model not configured")

    try:
        resp = model.generate_content(prompt)

        # =============== PATCH AMAN ===================
        # Gemini baru kadang mengembalikan kandidat tanpa parts
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()

        # Coba akses parts secara manual
        try:
            if resp.candidates:
                cand = resp.candidates[0]
                if hasattr(cand, "content") and cand.content and cand.content.parts:
                    # Ambil teks dari parts
                    text_blocks = []
                    for p in cand.content.parts:
                        if hasattr(p, "text") and p.text:
                            text_blocks.append(p.text)
                    if text_blocks:
                        return "\n".join(text_blocks).strip()
        except:
            pass

        # Jika tetap tidak ada, ini error "NO_OUTPUT"
        return ""
        # ==============================================

    except Exception as e:
        log("[call_llm] error:", e)
        log(traceback.format_exc())
        raise

# -----------------------
# Step 1: extract keywords via LLM (fallback naive)
# -----------------------
def extract_keywords_from_query(user_query: str) -> str:
    """
    Use LLM to extract simple keywords; fallback to naive tokenization.
    """
    user_query = (user_query or "").strip()
    if not user_query:
        return "kemah"

    # If LLM not available, simple fallback: take lowercase words > 2 chars
    if model is None:
        tokens = [t.lower() for t in user_query.split() if len(t) > 2]
        return " ".join(tokens[:8]) or "kemah"

    prompt = (
        "Ekstrak HANYA keyword penting (lokasi, fasilitas, suasana) "
        "dari pertanyaan berikut. Jawab HANYA keyword dipisah spasi.\n\n"
        f"Pertanyaan: {user_query}\n\nKeyword:"
    )
    try:
        kws = call_llm(prompt)
        kws = (kws or "").strip().lower()
        # sanitize: remove punctuation, keep up to 12 tokens
        import re
        kws = re.sub(r"[^0-9a-z\s\-]", " ", kws)
        kws = " ".join(kws.split()[:12])
        if not kws:
            return "kemah"
        return kws
    except Exception:
        # fallback simple tokenization
        tokens = [t.lower() for t in user_query.split() if len(t) > 2]
        return " ".join(tokens[:8]) or "kemah"

# -----------------------
# Step 2: call retrieval API
# -----------------------
def get_retrieval_context(keywords: str):
    payload = {"query": keywords}
    try:
        with httpx.Client(timeout=25.0) as client:
            r = client.post(RETRIEVAL_API_URL, json=payload)
        log(f"[API] status={r.status_code}")
        # Log raw snippet (avoid huge output)
        log(f"[API] raw snippet: {r.text[:800]}")
        if r.status_code != 200:
            log("[API] non-200 response")
            return []
        data = r.json()
        if not isinstance(data, list):
            log("[API] unexpected response type; expected list")
            return []
        if not data:
            log("[API] empty list returned")
            return []
        return data
    except Exception as e:
        log("[get_retrieval_context] exception:", e)
        log(traceback.format_exc())
        return []

# -----------------------
# Step 3: generate augmented response (RAG)
# -----------------------
def generate_augmented_response(user_query: str, raw_context: list) -> str:
    # simple guards
    if not raw_context:
        return "Maaf, saya tidak menemukan tempat kemah yang cocok dengan kriteria Anda."
    # simplify and limit
    cleaned = simplify_context(raw_context)
    if not cleaned:
        return "Maaf, data konteks tidak valid."
    # Build compact context string
    try:
        # Only include non-null fields to reduce noise
        items_lines = []
        for i, it in enumerate(cleaned, start=1):
            parts = []
            if it.get("nama"):
                parts.append(f"{it['nama']}")
            if it.get("lokasi"):
                parts.append(f"({it['lokasi']})")
            if it.get("rating") is not None:
                parts.append(f"rating: {it['rating']}")
            if it.get("fasilitas"):
                parts.append(f"fasilitas: {it['fasilitas']}")
            if it.get("harga_termurah") is not None:
                parts.append(f"harga_termurah: {it['harga_termurah']}")
            items_lines.append(f"{i}. " + " • ".join(parts))
        context_text = "\n".join(items_lines)
    except Exception as e:
        log("[generate_augmented_response] failed building context_text:", e)
        log(traceback.format_exc())
        context_text = json.dumps(cleaned, ensure_ascii=False)[:2000]

    # Build prompt
    prompt = (
        "Anda adalah KemahBot, asisten rekomendasi tempat kemah di Jawa Tengah & DIY.\n\n"
        "Gunakan HANYA data berikut dan jawab dalam bahasa Indonesia.\n\n"
        f"DATA SINGKAT:\n{context_text}\n\n"
        "Instruksi:\n"
        "- Sebutkan 1-3 rekomendasi terbaik dari data di atas dan berikan alasan singkat (rating, fasilitas, atau harga).\n"
        "- Jika user menanyakan filter spesifik (mis. WiFi), tunjukkan apakah fasilitas itu tersedia berdasarkan data.\n"
        "- Jangan menambahkan informasi di luar data.\n\n"
        f"Pertanyaan: {user_query}\n\nJawaban:\n"
    )

    log("[LLM_PROMPT] preview:", prompt[:2000])

    # If no LLM, fallback to templated reply
    if model is None:
        # simple templated summary
        lines = ["Halo! Saya menemukan beberapa tempat yang mungkin cocok:"]
        for it in cleaned[:TOP_K]:
            line = f"- {it.get('nama') or 'Nama tidak tersedia'}"
            if it.get('rating') is not None:
                line += f" (rating: {it['rating']})"
            if it.get('fasilitas'):
                line += f", fasilitas: {it['fasilitas']}"
            lines.append(line)
        lines.append("Mau lihat detail atau link maps?")
        return "\n".join(lines)

    # call LLM
    try:
        ans = call_llm(prompt)
        ans = (ans or "").strip()
        if not ans:
            return "Maaf, model tidak menghasilkan jawaban."
        return ans
    except Exception as e:
        log("[generate_augmented_response] LLM call failed:", e)
        log(traceback.format_exc())
        # fallback templated
        lines = ["Halo! Saya menemukan beberapa tempat yang mungkin cocok:"]
        for it in cleaned[:TOP_K]:
            line = f"- {it.get('nama') or 'Nama tidak tersedia'}"
            if it.get('rating') is not None:
                line += f" (rating: {it['rating']})"
            if it.get('fasilitas'):
                line += f", fasilitas: {it['fasilitas']}"
            lines.append(line)
        lines.append("Mau lihat detail atau link maps?")
        return "\n".join(lines)

# -----------------------
# Orchestrator main
# -----------------------
def get_chatbot_reply(user_input: str) -> str:
    log("[ORCH] user_input:", user_input)
    keywords = extract_keywords_from_query(user_input)
    log("[ORCH] keywords:", keywords)
    raw_context = get_retrieval_context(keywords)
    log(f"[ORCH] raw_context length: {len(raw_context) if raw_context is not None else 0}")
    answer = generate_augmented_response(user_input, raw_context)
    log("[ORCH] answer preview:", (answer or "")[:200])
    return answer

# -----------------------
# Self-test (if run directly)
# -----------------------
if __name__ == "__main__":
    q = "Halo, saya mau cari tempat kemah yang sejuk di Jogja dan ada wifi"
    print(get_chatbot_reply(q))
