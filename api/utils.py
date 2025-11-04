import pandas as pd
import os
import joblib
# Import ini mungkin diperlukan oleh joblib untuk memuat .pkl yang berisi class kustom
from vsm_structures import Node, SlinkedList 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_map_from_csv(filename):
    """
    Memuat file CSV (kolom A: key, kolom B: value) ke dalam dictionary.
    Fungsi ini mengabaikan baris yang diawali '#' (untuk komentar)
    dan mengabaikan kolom ekstra (seperti kolom 'kategori').
    """

    # Tentukan path file absolut di dalam folder 'api/Kamus/'
    filepath = os.path.join(BASE_DIR, 'Kamus', filename)

    try:
        # Cek apakah file ada
        if not os.path.exists(filepath):
             print(f"!!! GAGAL: File Kamus tidak ditemukan di: {filepath}")
             return {}

        # 'comment=#' memberi tahu pandas untuk mengabaikan baris yang diawali #
        df = pd.read_csv(filepath, comment='#', dtype=str).fillna('')
        
        # Ambil nama kolom pertama (A) dan kedua (B)
        key_col = df.columns[0]
        value_col = df.columns[1]
        
        # Hapus baris yang mungkin kosong di kolom A atau B
        df = df.dropna(subset=[key_col, value_col])
        
        # Buat dictionary: Kolom A jadi key, Kolom B jadi value
        mapper_dict = pd.Series(df[value_col].values, index=df[key_col]).to_dict()
        
        print(f"✅ Berhasil memuat {len(mapper_dict)} aturan dari {filepath}")
        return mapper_dict
        
    except FileNotFoundError:
        print(f"!!! PERINGATAN: File konfigurasi {filepath} tidak ditemukan. Menggunakan dictionary kosong.")
        return {}
    except Exception as e:
        print(f"!!! ERROR saat memuat {filepath}: {e}")
        return {}
    
def load_assets():
    """Memuat aset VSM dari folder 'api/Assets/' menggunakan path absolut."""
    
    # Tentukan path file absolut di dalam folder 'api/Assets/'
    assets_dir = os.path.join(BASE_DIR, 'Assets')
    
    try:
        # Memuat tiga aset utama
        IDF_SCORES = joblib.load(os.path.join(assets_dir, 'idf_scores.pkl'))
        LINKED_LIST_DATA = joblib.load(os.path.join(assets_dir, 'linked_list_data.pkl'))
        DF_METADATA = joblib.load(os.path.join(assets_dir, 'df_metadata.pkl'))
        
        print(f"✅ Aset VSM (idf, linked_list, metadata) berhasil dimuat dari: {assets_dir}")
        return IDF_SCORES, LINKED_LIST_DATA, DF_METADATA
    
    except FileNotFoundError:
        print(f"❌ ERROR: File aset .pkl tidak ditemukan di '{assets_dir}'.")
        print("   Pastikan Anda sudah menjalankan skrip indexing dan menyimpan file .pkl di folder 'api/Assets'.")
        return None, None, None
    except Exception as e:
        print(f"❌ ERROR saat memuat aset VSM: {e}")
        return None, None, None