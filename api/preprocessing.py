import re
import os
import utils
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# ======================================================================
# 1. INISIALISASI ALAT BANTU NLP
# ======================================================================

try:
    stopwords_id = set(stopwords.words('indonesian')) 
    negation_words = ['tidak', 'kurang', 'jangan', 'bukan', 'tanpa', 'enggak', 'gak', 'nggak', 'ndak', 'tak', 'kecuali']
    for word in negation_words:
        if word in stopwords_id:
            stopwords_id.remove(word)
    print("✅ Stopwords (termasuk kustomisasi negasi) siap.")
except LookupError:
    print("⚠️ Gagal memuat stopwords NLTK. Harap download dulu: nltk.download('stopwords')")
    stopwords_id = {"yang", "dan", "di", "ke", "ini"}

# Inisialisasi Stemmer Sastrawi
try:
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    print("✅ Stemmer Sastrawi siap.")
except:
    print("⚠️ Gagal memuat Stemmer Sastrawi.")
    # Fallback ke dummy stemmer jika Sastrawi gagal
    class DummyStemmer:
        def stem(self, text):
            return text
    stemmer = DummyStemmer()

# ======================================================================
# 2. MEMUAT SEMUA KAMUS (Kode Anda sudah benar)
# ======================================================================
# Memanggil fungsi 'load_map_from_csv' dari file 'utils.py'

print("--- Memuat Kamus dari folder 'Kamus/' ---")
PHRASE_MAP = utils.load_map_from_csv('config_phrase_map.csv')
REGION_MAP = utils.load_map_from_csv('config_region_map.csv')
SPECIAL_INTENT_MAP = utils.load_map_from_csv('config_special_intent.csv')
print("✅ Semua kamus (Phrase, Region, Intent) berhasil dimuat.")

# ======================================================================
# 3. SEMUA FUNGSI PREPROCESSING
# (Ini adalah gabungan dari Sel 5, 6, 7, 9 dari notebook Anda)
# ======================================================================

def remove_special_characters(text):
    """Fungsi helper dari Sel 9"""
    if not isinstance(text, str):
        return "" 
    regex = re.compile(r'[^a-zA-Z0-9\s]')
    return re.sub(regex, '', text)

def substitute_complex_phrases(text):
    """
    Fungsi dari Sel 5 (Versi aman dengan Regex & Word Boundary).
    Fungsi ini sekarang otomatis menggunakan PHRASE_MAP global.
    """
    text_lower = text.lower()
    
    # Urutkan kamus dari frasa TERPANJANG ke TERPENDEK
    sorted_phrases = sorted(PHRASE_MAP.items(), key=lambda item: len(str(item[0])), reverse=True)
    
    for phrase, token in sorted_phrases:
        try:
            # Gunakan regex \b (word boundary) agar "ga" tidak merusak "dengan"
            regex_phrase = r'\b' + re.escape(str(phrase)) + r'\b'
            text_lower = re.sub(regex_phrase, str(token), text_lower)
        except re.error:
            # Fallback jika regex error
            text_lower = text_lower.replace(str(phrase), str(token))
        
    return text_lower

def full_preprocessing(text):
    """Fungsi utama preprocessing dari Sel 9."""
    if not isinstance(text, str):
        return []
        
    cleaned_text = remove_special_characters(text)
    cleaned_text = re.sub(r'\d', '', cleaned_text)

    # 1. Jalankan kamus (slang, typo, frasa)
    text_with_phrases = substitute_complex_phrases(cleaned_text)
    
    # 2. Tokenisasi
    words = text_with_phrases.lower().split()
    
    # 3. Hapus Stopwords (yang sudah dikustomisasi)
    words = [w for w in words if w not in stopwords_id]
    
    # 4. Stemming
    stemmed_words = [stemmer.stem(w) for w in words]
    
    # 5. Hapus token sisa yang terlalu pendek
    final_words = [w for w in stemmed_words if len(w) > 1]
    return final_words

# --- Fungsi dari Sel 6 & 7 ---

def detect_region_and_filter_query(query_text):
    """
    Menganalisis query untuk region.
    Otomatis menggunakan REGION_MAP global.
    """
    query_text_lower = query_text.lower()
    detected_region = None
    
    # Urutkan dari region terpanjang (misal: "jawa tengah" sebelum "jawa")
    sorted_regions = sorted(REGION_MAP.items(), key=lambda item: len(str(item[0])), reverse=True)
    
    for term, region in sorted_regions:
        if term in query_text_lower:
            detected_region = region
            query_text_lower = query_text_lower.replace(term, '') 
            break 
            
    filtered_query_text = " ".join([word for word in query_text_lower.split() if word])
    return filtered_query_text, detected_region

def detect_intent(query_text):
    """
    Menganalisis query untuk intent khusus.
    Otomatis menggunakan SPECIAL_INTENT_MAP global.
    """
    query_text_lower = query_text.lower()
    special_intent = None
    
    # Urutkan dari intent terpanjang
    sorted_intents = sorted(SPECIAL_INTENT_MAP.items(), key=lambda item: len(str(item[0])), reverse=True)

    for term, intent in sorted_intents:
        if term in query_text_lower:
            special_intent = intent
            query_text_lower = query_text_lower.replace(term, '')
            break
            
    filtered_query_text = " ".join([word for word in query_text_lower.split() if word])
    return filtered_query_text, special_intent