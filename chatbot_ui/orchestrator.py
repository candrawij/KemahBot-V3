import os
import httpx
import json
import google.generativeai as genai
from dotenv import load_dotenv

# =====================================================================
# 1. KONFIGURASI MODEL LLM (GEMINI)
# =====================================================================

try:
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path)

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY tidak ditemukan di .env")

    genai.configure(api_key=GEMINI_API_KEY)

    generation_config = {
        "temperature": 0.6,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    model = genai.GenerativeModel(
        model_name="models/gemini-2.5-flash",
        generation_config=generation_config
    )

    print("✅ Orchestrator: Gemini siap digunakan.")

except Exception as e:
    print(f"❌ ERROR LLM: {e}")
    model = None

# API HuggingFace Anda
RETRIEVAL_API_URL = "https://Kaira21-campground-api.hf.space/search"


# =====================================================================
# 2. NORMALISASI / FILTER DATA KONTEXT
# =====================================================================
def simplify_context(context_list: list) -> list:
    """
    Mengubah raw JSON hasil API (yang sangat besar)
    menjadi format ringkas agar aman diberikan ke LLM.
    """
    cleaned = []

    for item in context_list:
        try:
            harga_termurah = None
            if item.get("Price_Items"):
                harga_termurah = min(
                    p.get("harga", 0) for p in item["Price_Items"]
                )

            cleaned.append({
                "nama": item.get("Nama_Tempat"),
                "lokasi": item.get("Lokasi"),
                "rating": item.get("Avg_Rating"),
                "fasilitas": item.get("Facilities"),
                "harga_termurah": harga_termurah,
            })
        except Exception:
            continue

    return cleaned


# =====================================================================
# 3. LLM – Ekstraksi keyword dari user_query
# =====================================================================
def extract_keywords_from_query(user_query: str) -> str:
    if not model:
        return user_query.lower()

    prompt = f"""
    Ekstrak HANYA keyword penting dari pertanyaan berikut.
    Jangan tambahkan kata lain.
    Jawab hanya keyword dipisahkan spasi.

    Pertanyaan: "{user_query}"
    Keyword:
    """

    try:
        response = model.generate_content(prompt)
        keywords = response.text.strip().lower()
        print(f"[LLM-1] Keyword diekstrak → {keywords}")
        return keywords
    except Exception as e:
        print(f"[LLM-1] ERROR: {e}")
        return user_query.lower()


# =====================================================================
# 4. CALL API – Ambil data context hasil VSM
# =====================================================================
def get_retrieval_context(keywords: str) -> list:
    payload = {"query": keywords}

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(RETRIEVAL_API_URL, json=payload)

        if response.status_code == 200:
            context = response.json()

            # Context harus list
            if not isinstance(context, list):
                print(f"[API] WARNING: Response bukan list → {context}")
                return []

            print(f"[API] Berhasil ambil {len(context)} data")
            return context
        else:
            print(f"[API] ERROR {response.status_code}: {response.text}")
            return []
    except Exception as e:
        print(f"[API] ERROR koneksi: {e}")
        return []


# =====================================================================
# 5. LLM – Membuat jawaban final (RAG)
# =====================================================================
def generate_augmented_response(user_query: str, raw_context: list) -> str:
    if not model:
        return "Maaf, model AI tidak tersedia."

    if not raw_context:
        return (
            "Maaf, saya tidak menemukan tempat kemah yang cocok "
            "dengan kriteria Anda. Coba ubah kata kunci lain."
        )

    # Ringankan konteks
    context = simplify_context(raw_context)

    if not context:
        return "Maaf, data konteks tidak valid."

    context_json = json.dumps(context, indent=2, ensure_ascii=False)

    prompt = f"""
    Anda adalah KemahBot, asisten rekomendasi tempat kemah di Jawa Tengah & DIY.

    Gunakan HANYA data berikut (sudah diringkas aman):

    DATA:
    {context_json}

    Instruksi:
    - Jawab secara ramah dan langsung.
    - Berikan 1–3 rekomendasi terbaik dengan alasan (rating, fasilitas, harga).
    - Jangan gunakan pengetahuan di luar data.

    Pertanyaan Pengguna:
    {user_query}

    Jawaban:
    """

    try:
        response = model.generate_content(prompt)
        print("[LLM-2] Jawaban berhasil dibuat.")
        return response.text.strip()
    except Exception as e:
        print(f"[LLM-2] ERROR: {e}")
        return "Maaf, terjadi kesalahan saat membuat jawaban."


# =====================================================================
# 6. FUNGSI UTAMA UNTUK STREAMLIT
# =====================================================================
def get_chatbot_reply(user_input: str) -> str:
    print("\n==============================")
    print(f"🔍 Query masuk → {user_input}")

    keywords = extract_keywords_from_query(user_input)
    context = get_retrieval_context(keywords)
    answer = generate_augmented_response(user_input, context)

    print(f"💬 Final Answer: {answer[:80]}...")
    return answer


# =====================================================================
# 7. MODE DEBUG / TESTING
# =====================================================================
if __name__ == "__main__":
    print("--- MODE TEST ORCHESTRATOR ---")

    tanya = "Halo, saya mau cari tempat kemah yang sejuk di Jogja dan ada wifi"
    print(get_chatbot_reply(tanya))
