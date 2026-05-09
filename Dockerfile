FROM python:3.11-slim

WORKDIR /app

# System deps for pymupdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY src/ src/
COPY README.md .

EXPOSE 8080

CMD ["uvicorn", "scientis.main:app", "--host", "0.0.0.0", "--port", "8080"]
