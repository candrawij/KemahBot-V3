import google.generativeai as genai
import os
from dotenv import load_dotenv

def check_my_models():
    """
    Skrip ini akan memuat kunci API Anda dan mencetak
    daftar model yang BISA Anda gunakan.
    """
    try:
        # Muat API key dari file .env
        dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        load_dotenv(dotenv_path)

        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY tidak ditemukan. Pastikan ada di file .env di folder utama.")

        genai.configure(api_key=GEMINI_API_KEY)
        print("==========================================================")
        print("✅ Kunci API berhasil dimuat. Mencari model yang tersedia...")
        print("==========================================================")

        found_models = 0
        # Loop dan cetak model
        for m in genai.list_models():
          # Kita hanya peduli model yang mendukung 'generateContent' (membuat teks)
          if 'generateContent' in m.supported_generation_methods:
            print(f"✔️ Model Name: {m.name}")
            print(f"   Description: {m.description}\n")
            found_models += 1

        if found_models == 0:
             print("\n❌ Tidak ada model 'generateContent' yang ditemukan untuk kunci API ini.")

        print("--- SELESAI ---")
        print("Silakan salin daftar 'Model Name' di atas (misal: 'models/gemini-1.5-flash') dan kirimkan.")

    except Exception as e:
        print(f"❌ ERROR: Gagal memuat kunci API atau model: {e}")
        print("   Pastikan GEMINI_API_KEY Anda di file .env sudah benar.")

if __name__ == "__main__":
    check_my_models()