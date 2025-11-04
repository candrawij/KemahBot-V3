import math
import os
import joblib
import utils
import preprocessing
import pandas as pd
from vsm_structures import Node, SlinkedList
import urllib.parse
import json # Pastikan json diimpor

# ======================================================================
# 1. VARIABEL GLOBAL ASET VSM
# ======================================================================
IDF_SCORES = None
LINKED_LIST_DATA = None
DF_METADATA = None

# ======================================================================
# 2. FUNGSI INISIALISASI (Dipanggil oleh app.py)
# ======================================================================
def initialize_mesin():
    """Memuat semua aset VSM (.pkl) ke dalam variabel global."""
    global IDF_SCORES, LINKED_LIST_DATA, DF_METADATA 
    
    print("--- Memuat Aset VSM (Indeks, IDF, Metadata)... ---")
    IDF_SCORES, LINKED_LIST_DATA, DF_METADATA = utils.load_assets()
    
    if IDF_SCORES is None or LINKED_LIST_DATA is None or DF_METADATA is None:
        print("❌ FATAL ERROR: Gagal memuat aset VSM. Mesin pencari tidak akan berfungsi.")
    else:
        print("✅ Mesin Pencari (VSM) Siap.")

# ======================================================================
# 3. FUNGSI ANALISIS KASTEM (Wrapper untuk preprocessing)
# ======================================================================
def analyze_full_query(query_text):
    """Fungsi "Otak" (dari Sel 8) yang memanggil semua fungsi preprocessing."""
    query_after_intent, special_intent = preprocessing.detect_intent(query_text)
    final_vsm_text, region_filter = preprocessing.detect_region_and_filter_query(query_after_intent)
    vsm_tokens = preprocessing.full_preprocessing(final_vsm_text)

    if region_filter:
        generic_fluff_words = {'cari', 'tampil', 'lihat', 'berikan', 'saran', 'rekomendasikan'} 
        if vsm_tokens and all(token in generic_fluff_words for token in vsm_tokens):
            vsm_tokens = [] 

    if not vsm_tokens and (special_intent or region_filter):
        vsm_tokens = ['kemah'] 
        if not special_intent and region_filter:
            special_intent = 'ALL'
            
    return vsm_tokens, special_intent, region_filter

# ======================================================================
# 4. FUNGSI PENCARIAN UTAMA (diperbarui dengan fallback kuat)
# ======================================================================
def search_by_keyword(query_tokens, special_intent, region_filter):
    """
    Melakukan pencarian VSM atau bypass jika intent 'ALL'.
    Menggunakan ASET GLOBAL (IDF_SCORES, LINKED_LIST_DATA, DF_METADATA).
    """
    if DF_METADATA is None or IDF_SCORES is None or LINKED_LIST_DATA is None:
        print("!!! ERROR: Aset VSM tidak dimuat. Pencarian dibatalkan.")
        return []

    # --- Jalur 1: Logika 'ALL' (Tanpa VSM) ---
    if special_intent == 'ALL':
        df_unique_places = DF_METADATA.drop_duplicates(subset='Nama_Tempat').copy()
        
        if region_filter:
            df_unique_places = df_unique_places[df_unique_places['Lokasi'].str.lower().str.contains(region_filter, na=False)]
        
        df_unique_places = df_unique_places.sort_values(by='Avg_Rating', ascending=False)
        
        final_recommendations = []
        for _, row in df_unique_places.iterrows():
            
            photo_url = row.get('Photo_URL') 
            if not photo_url or pd.isna(photo_url): # pd.isna() aman di sini
                photo_url = f"https://placehold.co/400x200/556B2F/FFFFFF?text={urllib.parse.quote(str(row['Nama_Tempat']))}&font=poppins"

            gmaps_link = row.get('Gmaps_Link')
            if not gmaps_link or pd.isna(gmaps_link):
                gmaps_link = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(str(row['Nama_Tempat']) + ' ' + str(row['Lokasi']))}"

            # --- PERBAIKAN ValueError: Ganti pd.isna(list) ---
            facilities = row.get('Facilities')
            if pd.isna(facilities) or not isinstance(facilities, str):
                facilities = "" # Fallback ke string kosong 

            price_items = row.get('Price_Items')
            if not isinstance(price_items, list): # Cek jika bukan list
                price_items = []

            final_recommendations.append({
                'name': row['Nama_Tempat'],
                'location': row['Lokasi'],
                'avg_rating': row['Avg_Rating'],
                'top_vsm_score': 0.0,
                'photo_url': photo_url,
                'gmaps_link': gmaps_link,
                'price_items': price_items, 
                'facilities': facilities 
            })
        return final_recommendations

    # --- Jalur 2: Logika VSM (Jika bukan 'ALL') ---
    if not query_tokens: return [] 

    query_tf = {word: query_tokens.count(word) for word in set(query_tokens)}
    query_weights = {}
    involved_docs = set() 

    for term, tf in query_tf.items():
        if term in IDF_SCORES: 
            query_weights[term] = tf * IDF_SCORES[term]
            current_node = LINKED_LIST_DATA[term].head.nextval 
            while current_node is not None:
                involved_docs.add(current_node.doc)
                current_node = current_node.nextval
    if not involved_docs: return [] 

    doc_scores = {doc_id: 0 for doc_id in involved_docs}
    for term, W_q in query_weights.items():
        current_node = LINKED_LIST_DATA[term].head.nextval 
        while current_node is not None:
            doc_id, W_d = current_node.doc, current_node.freq
            if doc_id in doc_scores: 
                doc_scores[doc_id] += W_d * W_q 
            current_node = current_node.nextval

    ranked_results_by_doc = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)

    final_recommendations, unique_names = [], set()
    for doc_id, vsm_score in ranked_results_by_doc:
        try:
            meta = DF_METADATA.loc[doc_id] 
        except KeyError: continue 

        if region_filter and region_filter not in meta['Lokasi'].lower(): continue

        name = meta['Nama_Tempat']
        if name not in unique_names:
            unique_names.add(name)
            
            photo_url = meta.get('Photo_URL') 
            if not photo_url or pd.isna(photo_url):
                photo_url = f"https://placehold.co/400x200/556B2F/FFFFFF?text={urllib.parse.quote(str(name))}&font=poppins"

            gmaps_link = meta.get('Gmaps_Link')
            if not gmaps_link or pd.isna(gmaps_link):
                gmaps_link = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(str(name) + ' ' + str(meta['Lokasi']))}"

            facilities = meta.get('Facilities')
            if pd.isna(facilities) or not isinstance(facilities, str):
                facilities = "" # Fallback ke string kosong

            price_items = meta.get('Price_Items')
            if not isinstance(price_items, list): # Cek jika bukan list
                price_items = []
            
            final_recommendations.append({
                'name': name,
                'location': meta['Lokasi'],
                'avg_rating': meta['Avg_Rating'],
                'top_vsm_score': vsm_score,
                'photo_url': photo_url,
                'gmaps_link': gmaps_link,
                'price_items': price_items, # Ini adalah LIST
                'facilities': facilities    # Ini adalah STRING
            })

    if special_intent == 'RATING_TOP':
        final_recommendations.sort(key=lambda x: x['avg_rating'], reverse=True)
    elif special_intent == 'RATING_BOTTOM':
        final_recommendations.sort(key=lambda x: x['avg_rating'], reverse=False)

    return final_recommendations