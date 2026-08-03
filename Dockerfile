FROM python:3.11-slim

WORKDIR /app

# ติดตั้ง dependency ก่อน เพื่อใช้ Docker layer cache ให้ build เร็วขึ้นในครั้งถัดไป
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกซอร์สโค้ดทั้งหมด
COPY . .

# Render จะกำหนด PORT มาให้ผ่าน environment variable โดยอัตโนมัติ
# bot.py จะอ่านค่านี้เองตอนเปิด Health Check Server
ENV PYTHONUNBUFFERED=1

CMD ["python", "bot.py"]
