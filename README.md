# Tumni Trading Scanner Bot

บอท Telegram สำหรับสแกนหาสัญญาณเทรดด้วยแนวคิด Fibo + แท่งเทียนตำหนิ (Rejection Candle)
บนกราฟ H1 รองรับทั้ง Crypto (Binance) และ Forex/Gold (Yahoo Finance)

## โครงสร้างไฟล์

```
.
├── bot.py                   # ไฟล์หลักของบอท (Health check server + Telegram handlers)
├── fibo_tumni_scanner.py    # คลาส TumniScanner สำหรับหาแท่งตำหนิและคำนวณ Fibo
├── requirements.txt         # Python dependencies
├── Dockerfile               # สำหรับ deploy แบบ Docker (ทางเลือก)
└── README.md
```

## วิธี Deploy บน Render.com (Web Service, Free Plan)

### ตัวเลือกที่ 1: Native Python (ไม่ใช้ Docker) — แนะนำเพราะ build เร็วกว่า

1. Push โค้ดทั้งหมดขึ้น GitHub repository
2. ไปที่ Render Dashboard -> **New** -> **Web Service**
3. เชื่อมต่อกับ repository ของคุณ
4. ตั้งค่าดังนี้:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python bot.py`
   - **Instance Type:** Free
5. ไปที่แท็บ **Environment** แล้วเพิ่มตัวแปรต่อไปนี้:
   - `TELEGRAM_BOT_TOKEN` = โทเคนที่ได้จาก BotFather (**จำเป็นต้องตั้งค่า**)
   - `AI_CONFIDENCE_THRESHOLD` = `0.6` (ทางเลือก ปรับได้ตามต้องการ)

   > หมายเหตุ: ไม่ต้องตั้งค่า `PORT` เอง — Render จะกำหนดให้อัตโนมัติ และ `bot.py` จะอ่านค่านี้เอง

6. กด **Create Web Service** แล้วรอ deploy

### ตัวเลือกที่ 2: ใช้ Dockerfile

1. เลือก **Environment: Docker** ตอนสร้าง Web Service แทน
2. Render จะ build จาก `Dockerfile` ที่แนบมาให้อัตโนมัติ (ไม่ต้องตั้ง Build/Start Command เอง)
3. ตั้งค่า Environment Variable `TELEGRAM_BOT_TOKEN` เหมือนตัวเลือกที่ 1

## ทำไมต้องมี Health Check Server?

Render Web Service (ทั้ง Free และ Paid) คาดหวังให้แอปพลิเคชัน bind HTTP port (จาก
environment variable `PORT`) ให้สำเร็จภายในเวลาไม่นานหลัง deploy มิเช่นนั้น Render จะ
มองว่า service ไม่ ready และ restart/kill process (Exit Code 1)

เนื่องจาก Telegram Bot ทำงานแบบ polling (ไม่ได้เปิด HTTP server เอง) โค้ดใน `bot.py`
จึงเปิด `http.server.HTTPServer` แบบง่ายๆ ใน background thread ขึ้นมาก่อนเสมอ เพื่อ
"หลอก" ให้ Render เห็นว่า port ถูก bind สำเร็จ จากนั้นจึงค่อยเริ่ม Telegram polling

## AI Confidence Filter (ทางเลือก)

หากต้องการใช้ AI ช่วยกรองสัญญาณเพิ่มเติม ให้วางไฟล์โมเดลที่เทรนไว้แล้ว (scikit-learn,
บันทึกด้วย `joblib.dump`) ไว้ที่ root ของโปรเจกต์:

- `model.pkl` — โมเดลหลัก (ต้องมีเมธอด `predict_proba` หรือ `predict`)
- `scaler.pkl` — ตัว scaler สำหรับ normalize feature (ถ้ามี)

ถ้าไม่มีไฟล์เหล่านี้ บอทจะข้ามการกรองด้วย AI โดยอัตโนมัติ และแสดงเฉพาะผลจาก
TumniScanner เท่านั้น

## คำเตือน

บอทนี้เป็นเครื่องมือช่วยสแกนหาสัญญาณเบื้องต้นเท่านั้น ไม่ใช่คำแนะนำการลงทุน
ผู้ใช้ควรตรวจสอบและบริหารความเสี่ยงด้วยตนเองก่อนตัดสินใจเทรดจริงทุกครั้ง
