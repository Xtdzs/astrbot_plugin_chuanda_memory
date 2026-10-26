FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY chuanda_memory ./chuanda_memory
EXPOSE 8765
VOLUME ["/data"]
CMD ["python", "-m", "chuanda_memory", "serve", "--data-dir", "/data", "--host", "0.0.0.0"]
