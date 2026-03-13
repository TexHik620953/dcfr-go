FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-u", "server.py"]
