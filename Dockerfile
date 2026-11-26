FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY streamlit_app.py .

RUN python3 scripts/expand_global_atlas.py

EXPOSE 8000 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
