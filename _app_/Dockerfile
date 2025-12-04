# Base Image Python yang ringan
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies sistem
# Menambahkan 'libmagic-dev' dan 'poppler-utils' (wajib untuk library 'unstructured' & pdf processing)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libmagic-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install library Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code aplikasi
COPY app_ui.py .

# Buat folder untuk mounting
RUN mkdir -p /app/target_project
RUN mkdir -p /app/db_chroma_ui

# Expose port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command jalankan aplikasi
CMD ["streamlit", "run", "app_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]