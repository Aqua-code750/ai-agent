FROM python:3.10-slim

WORKDIR /app

COPY my_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "python main.py"]
