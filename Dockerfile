# Интерфейс на уже подготовленных артефактах: сначала на машине выполняются make build и make eval.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Версия torch для процессора: сборка с GPU не нужна и намного больше.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Модель эмбеддингов скачивается при сборке и остаётся в образе.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small', device='cpu')"

COPY src src
COPY app app
COPY configs configs
COPY kb kb
COPY .streamlit .streamlit

ENV HF_HUB_OFFLINE=1
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
