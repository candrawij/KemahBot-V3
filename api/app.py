# 1. Import library yang dibutuhkan
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import uvicorn # Untuk menjalankan server

# 2. Import skrip VSM Anda yang sudah ada
# Pastikan file-file ini ada di folder /api yang sama
try:
    import mesin_pencari
    import preprocessing
    import utils
except ImportError as e:
    print(f"❌ FATAL ERROR: Gagal mengimpor modul VSM. Pastikan file .py ada di folder /api: {e}")
    # Hentikan program jika modul inti tidak ada
    exit()

# =====================================================================
# 3. DEFINISI MODEL DATA (PYDANTIC)
# =====================================================================
# Ini adalah "cetakan" untuk input dan output API Anda.
# Ini memastikan data yang masuk dan keluar sesuai format.

class SearchQuery(BaseModel):
    """Model untuk data yang dikirim oleh KLIEN ke API kita."""
    query: str = Field(..., example="kamar mandi bersih di jogja")

class PriceItem(BaseModel):
    """Model untuk nested object di dalam 'price_items'."""
    item: str
    harga: int | float # Harga bisa integer (10000) atau float

class KemahResponse(BaseModel):
    """
    Model untuk data yang dikirim oleh API kita ke KLIEN.
    Struktur ini HARUS SAMA dengan dictionary yang di-return 
    oleh fungsi mesin_pencari.search()
    """
    name: str
    location: str
    avg_rating: float
    top_vsm_score: float
    photo_url: str
    gmaps_link: str
    price_items: List[PriceItem] = [] # Default list kosong jika tidak ada
    facilities: str = ""             # Default string kosong jika tidak ada
    
    # Konfigurasi Pydantic untuk contoh di dokumentasi /docs
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Kuncen Camp Ground",
                "location": "Kab. Semarang, Jawa Tengah",
                "avg_rating": 4.8,
                "top_vsm_score": 0.75,
                "photo_url": "https://example.com/foto.jpg",
                "gmaps_link": "https://maps.app.goo.gl/example",
                "price_items": [{"item": "Tiket Masuk", "harga": 20000}],
                "facilities": "Kolam Renang|WiFi|Toilet"
            }
        }


# =====================================================================
# 4. INISIALISASI APLIKASI FASTAPI
# =====================================================================

app = FastAPI(
    title="🏕️ CampGround Bot Retrieval API",
    description="API untuk mengambil data tempat kemah menggunakan VSM (TF-IDF) "
                "dan data terstruktur (metadata).",
    version="1.0.0"
)

# =====================================================================
# 5. EVENT STARTUP: MUAT ASET VSM
# =====================================================================
# Ini adalah bagian terpenting.
# Kode di sini hanya berjalan SATU KALI saat server API dinyalakan.
# Kita memuat file .pkl ke memori agar pencarian super cepat.

@app.on_event("startup")
async def startup_event():
    """
    Memuat semua aset VSM (.pkl) ke memori saat server dimulai.
    """
    print("==================================================")
    print("🚀 Server FastAPI mulai berjalan...")
    print("--- Memanggil initialize_mesin() dari mesin_pencari.py ---")
    
    # Panggil fungsi inisialisasi dari skrip Anda
    mesin_pencari.initialize_mesin()
    
    print("--- Mesin Pencari VSM dan Aset .pkl SIAP. ---")
    print("==================================================")


# =====================================================================
# 6. ENDPOINT API UTAMA
# =====================================================================

@app.post("/search", response_model=List[KemahResponse])
async def search_kemah(query: SearchQuery):
    """
    Endpoint utama untuk menjalankan pencarian VSM (Retrieval) 
    dan Augmentasi Data (Pengambilan Metadata).
    
    - **Menerima**: JSON berisi 'query' (string).
    - **Mengembalikan**: List JSON berisi data lengkap tempat kemah.
    """
    print(f"\nINFO: Menerima permintaan pencarian baru: '{query.query}'")
    
    # --- LANGKAH 1: Analisis Query (dari mesin_pencari.py) ---
    # Mengubah "kamar mandi bersih jogja" -> (['bersih', 'mandi'], None, 'jogja')
    print(f"INFO: Memanggil analyze_full_query...")
    
    query_tokens, special_intent, region_filter = mesin_pencari.analyze_full_query(
        query_text=query.query
    )
    
    print(f"INFO: Hasil Analisis: Tokens={query_tokens}, Intent={special_intent}, Region={region_filter}")

    # --- LANGKAH 2: Panggil Fungsi Pencarian VSM (dari mesin_pencari.py) ---
    # Sekarang kita memanggil fungsi dengan argumen yang TEPAT
    
    results = mesin_pencari.search_by_keyword(
        query_tokens=query_tokens,
        special_intent=special_intent,
        region_filter=region_filter
    )
    
    # Fungsi search_by_keyword() sudah mengembalikan list of dicts
    # FastAPI akan otomatis mengubahnya menjadi JSON
    # dan memvalidasinya sesuai model KemahResponse.
    
    print(f"INFO: Pencarian selesai. Mengembalikan {len(results)} hasil.")
    return results

@app.get("/", include_in_schema=False)
async def root():
    """Endpoint 'health check' sederhana untuk memastikan API berjalan."""
    return {"message": "CampGround Bot Retrieval API sedang berjalan. Buka /docs untuk dokumentasi."}

# =====================================================================
# 7. (Opsional) Untuk Menjalankan Langsung
# =====================================================================
if __name__ == "__main__":
    """
    Ini memungkinkan Anda menjalankan server langsung dari terminal 
    dengan 'python main.py' (berguna untuk debugging).
    """
    print("--- Menjalankan server Uvicorn (mode debug) ---")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)