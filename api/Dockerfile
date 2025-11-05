# 1. Mulai dengan image Python
FROM python:3.10-slim

# 2. Atur folder kerja di dalam container
WORKDIR /code

# 3. Salin file requirements & install (langkah ini di-cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Salin SEMUA file proyek Anda (skrip .py, folder Assets, folder Kamus)
COPY . .

# 5. Perintah untuk menjalankan aplikasi saat container dimulai
# Kita gunakan gunicorn (server produksi) untuk menjalankan app.py (yang dulu main.py)
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:7860"]