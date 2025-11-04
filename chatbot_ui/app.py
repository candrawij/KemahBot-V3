import streamlit as st
import os

# --- Impor "Otak" (Orchestrator) ---
# Kita asumsikan orchestrator.py ada di folder yang sama
try:
    from orchestrator import get_chatbot_reply
except ImportError:
    st.error("❌ Gagal memuat 'orchestrator.py'. Pastikan file tersebut ada di folder 'chatbot_ui'.")
    st.stop()

# =====================================================================
# 1. KONFIGURASI HALAMAN & CSS
# =====================================================================

st.set_page_config(
    page_title="KemahBot 🏕️",
    page_icon="🏕️",
    layout="centered" # 'centered' lebih cocok untuk UI chat
)

def load_css(file_name):
    """Fungsi untuk memuat file CSS eksternal."""
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    try:
        with open(file_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        print(f"⚠️ Peringatan: File CSS '{file_name}' tidak ditemukan.")

# Muat CSS kustom Anda (opsional, tapi disarankan)
load_css("style.css")

# --- Judul Halaman ---
st.title("🏕️ KemahBot")
st.markdown("<p class='sub-judul'>Asisten AI untuk rencana kemah Anda di Jawa Tengah & DIY</p>", unsafe_allow_html=True)


# =====================================================================
# 2. INISIALISASI RIWAYAT OBROLAN (Session State)
# =====================================================================

# Kita gunakan st.session_state agar riwayat chat tidak hilang
# setiap kali pengguna berinteraksi.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya KemahBot, siap membantu Anda merencanakan kemah. Mau cari tempat seperti apa?"}
    ]

# =====================================================================
# 3. TAMPILKAN RIWAYAT OBROLAN
# =====================================================================

# Loop semua pesan di riwayat dan tampilkan ke layar
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================================================
# 4. TANGANI INPUT PENGGUNA (Chat Input)
# =====================================================================

# 'st.chat_input' akan "menempel" di bagian bawah layar
if prompt := st.chat_input("Cari kemah sejuk, ada WiFi, dll..."):
    
    # 1. Tambahkan pesan pengguna ke riwayat & tampilkan
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. Tampilkan status "berpikir" & panggil "Otak" (Orchestrator)
    with st.chat_message("assistant"):
        with st.spinner("KemahBot sedang berpikir..."):
            
            # --- INI ADALAH INTI PROSES ---
            # Memanggil Fase 2 (Orchestrator), yang kemudian
            # akan memanggil Fase 1 (API VSM)
            response = get_chatbot_reply(prompt)
            # -----------------------------
            
            # Tampilkan jawaban dari "Otak"
            st.markdown(response)
    
    # 3. Tambahkan jawaban asisten ke riwayat
    st.session_state.messages.append({"role": "assistant", "content": response})