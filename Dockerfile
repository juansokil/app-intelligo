# Use slim Python; 3.11 is fine with Streamlit/OpenAI/Pydantic
FROM python:3.11-slim

# Helpful defaults and a writable HOME for Streamlit
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/app \
    PORT=8501

WORKDIR /app

# Minimal system tools (curl for the healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY src/ ./src/

# Streamlit configuration & writable .streamlit dir
RUN mkdir -p $HOME/.streamlit \
 && printf "[server]\nheadless = true\naddress = \"0.0.0.0\"\nport = 8501\n\n[browser]\ngatherUsageStats = false\n" > $HOME/.streamlit/config.toml

# If you expect big PDFs, you can add a limit (uncomment and adjust):
# RUN printf "maxUploadSize = 200\n" >> $HOME/.streamlit/config.toml

EXPOSE 8501

# Healthcheck (Streamlit exposes /_stcore/health)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# Start Streamlit. Many platforms inject $PORT; default stays 8501.
ENTRYPOINT ["sh", "-c", "streamlit run src/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT}"]
