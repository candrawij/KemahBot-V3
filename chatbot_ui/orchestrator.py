import os
import httpx  # Library modern untuk memanggil API, pengganti 'requests'
import json
import google.generativeai as genai
from dotenv import load_dotenv

# =====================================================================
# 1. KONFIGURASI MODEL LLM (GEMINI)
# =====================================================================

# Muat API key dari file .env yang ada di folder *utama* proyek
# Kita perlu .. untuk "naik" satu level dari chatbot_ui/ ke CampGround Bot/
try:
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path)
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY tidak ditemukan. Pastikan ada di file .env di folder utama.")
        
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Atur model Gemini yang akan kita gunakan
    generation_config = {
      "temperature": 0.7,
      "top_p": 1,
      "top_k": 1,
      "max_output_tokens": 2048,
    }
    model = genai.GenerativeModel(model_name="models/gemini-2.5-flash",
                                  generation_config=generation_config)
    print("✅ Orchestrator: Model Generatif Gemini (LLM) siap.")

except Exception as e:
    print(f"❌ ERROR: Gagal mengkonfigurasi Gemini. {e}")
    model = None

# URL tempat API FastAPI (Fase 1) Anda berjalan
RETRIEVAL_API_URL = "https://kaira21-campground-api.hf.space/search"

# =====================================================================
# FUNGSI 1: PANGGIL LLM (Ekstraksi Keyword)
# =====================================================================
def extract_keywords_from_query(user_query: str) -> str:
    """
    Menggunakan LLM (Call #1) untuk mengekstrak keyword VSM 
    dari pertanyaan bahasa alami pengguna.
    """
    if not model: return user_query # Fallback jika LLM gagal

    prompt = f"""
    Anda adalah asisten SEO yang sangat efisien. 
    Ekstrak HANYA keyword PENTING (kata kunci) untuk mesin pencari VSM dari pertanyaan pengguna berikut.
    Fokus pada lokasi, fasilitas, atau suasana (misal: 'sejuk', 'kamar mandi bersih', 'anak').
    JANGAN JAWAB pertanyaannya.
    JANGAN tambahkan kata-kata pembuka atau "keyword:".
    Jawab HANYA dengan keyword yang dipisahkan spasi.
    Jika tidak ada keyword relevan, kembalikan 'kemah'.

    Contoh:
    Pertanyaan: "Halo, saya mau cari tempat kemah yang sejuk di Jogja dan ada WiFi-nya."
    Jawaban: "sejuk jogja wifi"
    
    Pertanyaan: "tempat camping di semarang yang boleh bawa anjing"
    Jawaban: "semarang bawa anjing"
    
    Pertanyaan: "ada rekomendasi?"
    Jawaban: "kemah"

    ---
    Pertanyaan: "{user_query}"
    Jawaban:
    """
    
    try:
        response = model.generate_content(prompt)
        keywords = response.text.strip().lower()
        print(f"INFO (LLM-1): Keyword diekstrak: '{keywords}'")
        return keywords
    except Exception as e:
        print(f"ERROR (LLM-1): Gagal ekstraksi keyword: {e}")
        return user_query.lower() # Fallback: gunakan query asli

# =====================================================================
# FUNGSI 2: PANGGIL API FASE 1 (Retrieval & Augmentation)
# =====================================================================
def get_retrieval_context(keywords: str) -> list:
    """
    Memanggil API FastAPI (Fase 1) kita untuk mendapatkan data 
    kontekstual (hasil VSM + metadata).
    """
    payload = {"query": keywords}
    
    try:
        # Kita gunakan httpx.post untuk memanggil API kita
        with httpx.Client(timeout=30.0) as client:
            response = client.post(RETRIEVAL_API_URL, json=payload)
            
            # Cek jika API merespons dengan sukses (200 OK)
            if response.status_code == 200:
                context_data = response.json()
                print(f"INFO (API-Call): Berhasil mengambil {len(context_data)} data konteks.")
                return context_data
            else:
                print(f"ERROR (API-Call): API Fase 1 mengembalikan error {response.status_code} - {response.text}")
                return []
                
    except httpx.ConnectError as e:
        print(f"❌ ERROR (API-Call): Gagal terhubung ke API Fase 1 di {RETRIEVAL_API_URL}.")
        print("   Pastikan server FastAPI (Fase 1) Anda sedang berjalan di terminal lain!")
        return []
    except Exception as e:
        print(f"ERROR (API-Call): Error tidak diketahui: {e}")
        return []

# =====================================================================
# FUNGSI 3: PANGGIL LLM (Generasi Jawaban RAG)
# =====================================================================
def generate_augmented_response(user_query: str, context: list) -> str:
    """
    Menggunakan LLM (Call #2) untuk menghasilkan jawaban bahasa alami
    berdasarkan data konteks (dari API) dan pertanyaan asli.
    """
    if not model: return "Maaf, model AI sedang bermasalah."
    if not context:
        return "Maaf, saya sudah mencari tapi tidak menemukan tempat kemah yang cocok dengan kriteria Anda. Coba gunakan kata kunci lain."

    # Ubah data konteks (list of dicts) menjadi string JSON
    # Ini cara termudah agar LLM bisa membacanya
    try:
        context_json_string = json.dumps(context, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"ERROR: Gagal mengubah konteks ke JSON: {e}")
        return "Maaf, terjadi error saat memproses data."

    prompt = f"""
    Anda adalah "KemahBot", asisten travel ahli tempat kemah di Jawa Tengah & DIY.
    Tugas Anda adalah menjawab pertanyaan pengguna secara ramah dan membantu, HANYA berdasarkan KONTEKS DATA di bawah ini.
    JANGAN gunakan pengetahuan lain di luar konteks.
    JANGAN mengarang informasi.
    
    Saat menjawab:
    - Sapa pengguna dan jawab pertanyaannya secara langsung.
    - Sebutkan 1-3 tempat kemah terbaik dari konteks.
    - Berikan alasan singkat mengapa tempat itu direkomendasikan (misal: rating, fasilitas utama, atau harga jika relevan).
    - Jawab dalam bahasa Indonesia yang alami dan informatif.

    ---
    KONTEKS DATA (dari API VSM):
    {context_json_string}
    ---

    PERTANYAAN PENGGUNA:
    {user_query}
    ---

    JAWABAN ANDA:
    """

    try:
        response = model.generate_content(prompt)
        print(f"INFO (LLM-2): Jawaban RAG berhasil dibuat.")
        return response.text
    except Exception as e:
        print(f"ERROR (LLM-2): Gagal generasi jawaban: {e}")
        return "Maaf, terjadi kesalahan saat saya mencoba merangkum jawaban."

# =====================================================================
# FUNGSI 4: ORKESTRATOR UTAMA
# =====================================================================
def get_chatbot_reply(user_input: str) -> str:
    """
    Fungsi utama yang mengelola seluruh alur RAG.
    Ini yang akan dipanggil oleh Streamlit (Fase 3).
    """
    print(f"\n==========================\n🔄 Memproses query: '{user_input}'")
    
    # 1. Ekstraksi Keyword
    keywords = extract_keywords_from_query(user_input)
    
    # 2. Retrieval (Panggil API Fase 1)
    context_data = get_retrieval_context(keywords)
    
    # 3. Generation (Panggil LLM Fase 2)
    final_answer = generate_augmented_response(user_input, context_data)
    
    print(f"💬 Jawaban Final: {final_answer[:100]}...")
    return final_answer

# =====================================================================
# BLOK UJI COBA LANGSUNG
# =====================================================================
if __name__ == "__main__":
    """
    Ini memungkinkan kita menguji file orchestrator.py ini langsung 
    dari terminal, SEBELUM membuat UI Streamlit.
    
    Cara menjalankan:
    1. Pastikan API Fase 1 (uvicorn) berjalan di terminal 1.
    2. Buka terminal 2, aktifkan venv, masuk ke folder 'chatbot_ui'.
    3. Jalankan: python orchestrator.py
    """
    
    print("--- [Mode Uji Coba Orchestrator] ---")
    print("   Pastikan API Fase 1 Anda (uvicorn) sedang berjalan!")
    
    # Contoh pertanyaan untuk diuji
    test_query_1 = "Halo, saya mau cari tempat kemah yang sejuk di Jogja dan ada WiFi-nya."
    test_query_2 = "cariin kemah di semarang dong"
    test_query_3 = "yang ada kolam renangnya"

    # --- Uji Coba 1 ---
    balasan = get_chatbot_reply(test_query_1)
    print("\n--- HASIL UJI COBA 1 ---")
    print(balasan)
    
    # --- Uji Coba 2 ---
    balasan_2 = get_chatbot_reply(test_query_2)
    print("\n--- HASIL UJI COBA 2 ---")
    print(balasan_2)