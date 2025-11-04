import pandas as pd
import math
import joblib
import os
import json 

try:
    from preprocessing import full_preprocessing
    # Impor Class dari file barunya
    from vsm_structures import SlinkedList, Node
except ImportError as e:
    print(f"❌ FATAL ERROR: Gagal mengimpor modul. Pastikan semua file .py ada: {e}")
    exit()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KORPUS_FOLDER = 'Documents'
KORPUS_FILENAME = 'corpus_master.csv'
DATASET_PATH = os.path.join(BASE_DIR, KORPUS_FOLDER, KORPUS_FILENAME)

try:
    print(f"🔄 Memuat korpus dari: {DATASET_PATH} ...")
    df_corpus = pd.read_csv(DATASET_PATH)
    print(f"✅ Berhasil memuat korpus dari: {DATASET_PATH}")
except FileNotFoundError:
    print(f"❌ FATAL ERROR: File korpus tidak ditemukan di: {DATASET_PATH}")
    print("   Pastikan folder 'korpus' dan file 'corpus_master.csv' ada.")
    exit() # Hentikan skrip jika korpus tidak ada
except Exception as e:
    print(f"❌ FATAL ERROR saat memuat korpus: {e}")
    exit()

# --- 3. APLIKASI PREPROCESSING & HITUNG DF & IDF (INDEXING PHASE 1) ---
# Pastikan dataset sudah dimuat di df_corpus
print("🔄 Memulai preprocessing dan perhitungan DF/IDF...")
df_corpus['Teks_Mentah'] = df_corpus['Teks_Mentah'].fillna('')
df_corpus['Clean_Tokens'] = df_corpus['Teks_Mentah'].apply(full_preprocessing)

N = len(df_corpus)
df_counts = {} # Document Frequency

for tokens in df_corpus['Clean_Tokens']:
    for word in set(tokens): 
        df_counts[word] = df_counts.get(word, 0) + 1

idf_scores = {}
for term, count in df_counts.items():
    idf_scores[term] = math.log10(N / count)
print("✅ Selesai preprocessing dan perhitungan DF/IDF.")

# --- 4. BUILDING THE INVERTED INDEX WITH TF-IDF (INDEXING PHASE 2) ---
print("🔄 Membangun inverted index dengan TF-IDF...")
linked_list_data = {}
unique_words_all = set(df_counts.keys())

for word in unique_words_all:
    linked_list_data[word] = SlinkedList()
    linked_list_data[word].head = Node(docId=0, freq=None) 

for index, row in df_corpus.iterrows():
    doc_id = row['Doc_ID']
    tokens = row['Clean_Tokens']
    
    tf_in_doc = {word: tokens.count(word) for word in set(tokens)}

    for term, tf in tf_in_doc.items():
        if term in idf_scores: # Pastikan term ada di IDF
            tfidf = tf * idf_scores[term]
            
            # Cari ujung linked list untuk term ini
            current_node = linked_list_data[term].head
            while current_node.nextval is not None:
                current_node = current_node.nextval
            
            # Tambahkan node baru di ujung
            current_node.nextval = Node(docId=doc_id, freq=tfidf)

# Mapping Doc ID to Name and Rating for final result
df_metadata = df_corpus[['Doc_ID', 'Nama_Tempat', 'Lokasi', 'Rating']].copy()
avg_rating_per_place = df_metadata.groupby('Nama_Tempat')['Rating'].mean().reset_index()
avg_rating_per_place.rename(columns={'Rating': 'Avg_Rating'}, inplace=True)
df_metadata = df_metadata.merge(avg_rating_per_place, on='Nama_Tempat', how='left')

# Kita gabungkan data statis (foto, harga, dll) dari info_tempat.csv
try:
    print("🔄 Memuat data statis (foto, harga, dll)...")
    INFO_STATIS_PATH = os.path.join(BASE_DIR, KORPUS_FOLDER, 'info_tempat.csv')
    df_info_statis = pd.read_csv(INFO_STATIS_PATH)
    print(f"✅ Berhasil memuat data statis dari: {INFO_STATIS_PATH}")

    # Definisikan parser JSON yang aman HANYA UNTUK HARGA
    def parse_price_json(json_str):
        if pd.isna(json_str) or not isinstance(json_str, str) or not json_str.startswith('['):
            return [] # Fallback: list kosong
        try:
            # Membaca string JSON dari CSV
            return json.loads(json_str) 
        except (json.JSONDecodeError, TypeError):
            return [] # Fallback jika JSON rusak

    # 1. Proses Price_Items (HARUSNYA Tipe List) -> BENAR
    df_info_statis['Price_Items'] = df_info_statis['Price_Items'].apply(parse_price_json)
    
    # 2. Proses Facilities (HARUSNYA Tipe String) -> BENAR
    df_info_statis['Facilities'] = df_info_statis['Facilities'].fillna("").astype(str)
    
    # 3. Proses kolom lain (Jaga-jaga) -> BENAR
    df_info_statis['Photo_URL'] = df_info_statis['Photo_URL'].fillna("")
    df_info_statis['Gmaps_Link'] = df_info_statis['Gmaps_Link'].fillna("")

    # Gabungkan data statis ke metadata utama berdasarkan 'Nama_Tempat'
    df_metadata = df_metadata.merge(df_info_statis, on='Nama_Tempat', how='left')
    print("✅ Berhasil menggabungkan data statis (foto, harga, dll).")

    # 4. FINAL FALLBACK -> BENAR
    df_metadata['Photo_URL'] = df_metadata['Photo_URL'].fillna("")
    df_metadata['Gmaps_Link'] = df_metadata['Gmaps_Link'].fillna("")
    df_metadata['Facilities'] = df_metadata['Facilities'].fillna("")
    df_metadata['Price_Items'] = df_metadata['Price_Items'].apply(lambda x: [] if isinstance(x, float) and pd.isna(x) else x)

except FileNotFoundError:
    print(f"⚠️ PERINGATAN: {INFO_STATIS_PATH} tidak ditemukan.")
    print("   Melanjutkan tanpa data foto/harga/fasilitas.")
    # Buat kolom placeholder KONSISTEN
    df_metadata['Photo_URL'] = ""
    df_metadata['Gmaps_Link'] = ""
    df_metadata['Price_Items'] = [[] for _ in range(len(df_metadata))] # Tipe List
    df_metadata['Facilities'] = "" # Tipe String

# Jadikan Doc_ID sebagai index
df_metadata.set_index('Doc_ID', inplace=True)

# --- SIMPAN HASIL INDEXING KE FILE ASET ---

OUTPUT_DIR = 'Assets'
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    print("🔄 Menyimpan file aset (.pkl)...")
    joblib.dump(idf_scores, os.path.join(OUTPUT_DIR, 'idf_scores.pkl'))
    joblib.dump(linked_list_data, os.path.join(OUTPUT_DIR, 'linked_list_data.pkl'))
    joblib.dump(df_metadata, os.path.join(OUTPUT_DIR, 'df_metadata.pkl'))

    print(f"✅ SUKSES: Semua file aset (.pkl) telah dibuat dan disimpan di folder '{OUTPUT_DIR}'.")

except Exception as e:
    print(f"❌ GAGAL menyimpan file aset: {e}")