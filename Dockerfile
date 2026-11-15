FROM python:3.12-slim

WORKDIR /app

# Fewer glibc malloc arenas: stops threaded workers (aiosqlite, to_thread) from hoarding freed memory
ENV MALLOC_ARENA_MAX=2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data

CMD uvicorn src.server:app --host 0.0.0.0 --port ${PORT:-8000}
