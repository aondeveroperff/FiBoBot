import os
import logging
import asyncio
from datetime import datetime
from typing import Optional, Tuple
import requests
import pandas as pd
import numpy as np
import zoneinfo
import joblib  # สำหรับโหลดไฟล์ AI model.pkl

import uvicorn
from fastapi import FastAPI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# ---------------------------------------------------------
# 1. Configuration & Logging
# ---------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("tumni-bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "YOUR_TWELVEDATA_API_KEY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "") # Chat ID ของคุณสำหรับรับการแจ้งเตือนอัตโนมัติ

HTTP_TIMEOUT_SECONDS = 15
AI_CONFIDENCE_THRESHOLD = 0.60  # เกณฑ์ AI ยืนยันความมั่นใจ (60% ขึ้นไป)

# โหลด AI Model (ถ้ามีไฟล์ model.pkl ในโฟลเดอร์โปรเจกต์)
MODEL_PATH = "model.pkl"
ai_model = None
if os.path.exists(MODEL_PATH):
    try:
        ai_model = joblib.load(MODEL_PATH)
        logger.info("โหลด AI Model (model.pkl) สำเร็จเรียบร้อย!")
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการโหลด AI Model: {e}")
else:
    logger.warning("ไม่พบไฟล์ model.pkl! ระบบจะใช้ Rule-Based กรองแท่งตำหนิเป็นหลัก")

YAHOO_ALIASES = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X",
}

# ---------------------------------------------------------
# 2. Data Structures & Memory
# ---------------------------------------------------------
class DataFetchError(Exception):
    pass

class ScanResult:
    def __init__(
        self,
        symbol: str,
        signal: str,
        pattern_time: datetime,
        current_time: datetime,
        current_price: float,
        KRH2: float,
        KRH1: float,
        RUN: float,
    ):
        self.symbol = symbol
        self.signal = signal
        self.pattern_time = pattern_time
        self.current_time = current_time
        self.current_price = current_price
        self.KRH2 = KRH2
        self.KRH1 = KRH1
        self.RUN = RUN

# บันทึกเวลาแท่งสัญญาณล่าสุดที่ส่งเตือนไปแล้ว เพื่อป้องกันส่งซ้ำ
last_alerted_pattern_time = {}

# ---------------------------------------------------------
# 3. Data Fetching Logic
# ---------------------------------------------------------
def fetch_twelvedata_chart(symbol: str) -> pd.DataFrame:
    formatted_symbol = "XAU/USD" if symbol == "XAUUSD" else "XAG/USD"
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={formatted_symbol}&interval=1h&outputsize=150&apikey={TWELVEDATA_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        
        if "values" not in data:
            error_msg = data.get("message", "ไม่พบข้อมูลจาก Twelve Data")
            raise DataFetchError(f"Twelve Data Error: {error_msg}")
            
        df = pd.DataFrame(data["values"])
        df["time"] = pd.to_datetime(df["datetime"], utc=True)
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype(float)
            
        df = df.sort_values("time").set_index("time")
        return df[["open", "high", "low", "close"]]
    except Exception as e:
        raise DataFetchError(f"เรียก Twelve Data API ไม่สำเร็จ: {e}")

def fetch_yahoo_chart(symbol: str) -> pd.DataFrame:
    ticker = YAHOO_ALIASES.get(symbol.upper(), f"{symbol.upper()}=X")
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval=60m&range=1mo&includePrePost=false"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        
        df = pd.DataFrame({
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"]
        }, index=pd.to_datetime(timestamps, unit="s", utc=True))
        
        df.index.name = "time"
        return df.dropna()
    except Exception as e:
        raise DataFetchError(f"เรียก Yahoo Finance API ไม่สำเร็จ ({ticker}): {e}")

def fetch_chart_data(symbol: str) -> pd.DataFrame:
    sym = symbol.upper()
    if sym in ("XAUUSD", "GOLD", "XAGUSD", "SILVER"):
        return fetch_twelvedata_chart("XAUUSD" if sym in ("XAUUSD", "GOLD") else "XAGUSD")
    else:
        return fetch_yahoo_chart(sym)

# ---------------------------------------------------------
# 4. Flaw Bar Filtering & AI Prediction
# ---------------------------------------------------------
def is_valid_flaw_bar(prev_row: pd.Series, df: pd.DataFrame) -> bool:
    """กรองแท่งตำหนิ (Flaw Bar): ต้องมี Rejection ชัดเจน และขนาดใหญ่พอ"""
    high, low, open_p, close_p = prev_row["high"], prev_row["low"], prev_row["open"], prev_row["close"]
    total_range = high - low
    
    if total_range == 0:
        return False

    body_size = abs(close_p - open_p)
    upper_wick = high - max(open_p, close_p)
    lower_wick = min(open_p, close_p) - low

    # 1. เช็คไส้เทียนปฏิเสธราคา (Rejection Wick > 45% ของความยาวแท่ง)
    has_rejection = (upper_wick / total_range > 0.45) or (lower_wick / total_range > 0.45)
    
    # 2. เช็คขนาดแท่งเทียบกับค่าเฉลี่ย 14 แท่งก่อนหน้า (ไม่เอาโดจิเล็กๆ)
    avg_range = (df["high"] - df["low"]).tail(14).mean()
    is_significant = total_range >= (avg_range * 0.75)

    return has_rejection and is_significant

def predict_ai_confidence(df: pd.DataFrame) -> Optional[float]:
    """คำนวณ Technical Indicators และส่งให้ AI Predict หาค่า Confidence"""
    if ai_model is None or len(df) < 30:
        return None

    try:
        # คำนวณ Features
        df_feat = df.copy()
        
        # RSI 14
        delta = df_feat["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df_feat["rsi"] = 100 - (100 / (1 + rs))

        # ATR 14
        tr = np.maximum(
            df_feat["high"] - df_feat["low"],
            np.maximum(
                abs(df_feat["high"] - df_feat["close"].shift()),
                abs(df_feat["low"] - df_feat["close"].shift())
            )
        )
        df_feat["atr"] = tr.rolling(14).mean()

        # EMA 20 & 50
        df_feat["ema20"] = df_feat["close"].ewm(span=20, adjust=False).mean()
        df_feat["ema50"] = df_feat["close"].ewm(span=50, adjust=False).mean()
        
        # Body Ratio
        df_feat["body_ratio"] = abs(df_feat["close"] - df_feat["open"]) / (df_feat["high"] - df_feat["low"])

        # ดึงแถบล่าสุด
        latest = df_feat.iloc[-2]  # แท่งตำหนิ
        features = np.array([[
            latest["rsi"],
            latest["atr"],
            latest["ema20"],
            latest["ema50"],
            latest["body_ratio"]
        ]])

        # ทำการทายผลด้วย AI
        probabilities = ai_model.predict_proba(features)
        confidence = float(probabilities[0][1])  # คืนค่าความมั่นใจฝั่ง Win (Class 1)
        return confidence
    except Exception as e:
        logger.error(f"AI Prediction Error: {e}")
        return None

# ---------------------------------------------------------
# 5. Calculation Strategy Logic
# ---------------------------------------------------------
def calculate_fibo_levels(df: pd.DataFrame, symbol: str) -> Tuple[Optional[ScanResult], Optional[float]]:
    if len(df) < 15:
        raise ValueError("ข้อมูลราคาไม่เพียงพอในการคำนวณ")

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    # 1. กรองแท่งตำหนิ
    if not is_valid_flaw_bar(prev_row, df):
        return None, None

    # 2. ทำการพยากรณ์ด้วย AI
    confidence = predict_ai_confidence(df)
    if confidence is not None and confidence < AI_CONFIDENCE_THRESHOLD:
        return None, confidence  # AI ประเมินว่าความน่าจะเป็นต่ำ ให้ข้าม

    current_price = last_row["close"]
    current_time = df.index[-1]
    pattern_time = df.index[-2]

    high = prev_row["high"]
    low = prev_row["low"]
    diff = high - low

    krh2 = high - (diff * 0.236)  # Entry
    krh1 = low                    # Stop Loss
    run = high + (diff * 0.618)   # Take Profit

    result = ScanResult(
        symbol=symbol.upper(),
        signal="CONFIRMED",
        pattern_time=pattern_time,
        current_time=current_time,
        current_price=current_price,
        KRH2=krh2,
        KRH1=krh1,
        RUN=run,
    )
    return result, confidence

# ---------------------------------------------------------
# 6. Formatting & Handlers
# ---------------------------------------------------------
def format_signal_reply(result: ScanResult, confidence: Optional[float]) -> str:
    tz_th = zoneinfo.ZoneInfo("Asia/Bangkok")
    pattern_th = result.pattern_time.astimezone(tz_th).strftime("%Y-%m-%d %H:%M:%S")
    current_th = result.current_time.astimezone(tz_th).strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"🎯 *[{result.symbol}] แท่งตำหนิ CONFIRMED*",
        f"สถานะ: `{result.signal}`",
        "",
        f"เวลาแท่งสัญญาณ: `{pattern_th}` (ไทย)",
        f"เวลาราคาล่าสุด: `{current_th}` (ไทย)",
        f"ราคาปัจจุบัน: `{result.current_price:.2f}`",
        "",
        f"📍 Entry (KRH2): `{result.KRH2:.2f}`",
        f"🛑 Stop Loss (KRH1): `{result.KRH1:.2f}`",
        f"🟢 Take Profit (RUN): `{result.RUN:.2f}`",
    ]
    if confidence is not None:
        lines.append("")
        lines.append(f"🤖 AI Confidence: `{confidence:.2%}`")
        
    return "\n".join(lines)

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """คำสั่ง /scan <symbol>"""
    if not context.args:
        await update.message.reply_text("กรุณาระบุคู่เงิน เช่น `/scan XAUUSD`", parse_mode="Markdown")
        return

    symbol = context.args[0].upper()
    try:
        df = fetch_chart_data(symbol)
        result, confidence = calculate_fibo_levels(df, symbol)
        
        if result is None:
            await update.message.reply_text(f"⚠️ {symbol}: ไม่พบโครงสร้างแท่งตำหนิ หรือ AI ประเมินว่าความมั่นใจต่ำกว่าเกณฑ์")
            return

        msg = format_signal_reply(result, confidence)
        await update.message.reply_text(msg, parse_mode="Markdown")
        
    except DataFetchError as e:
        logger.warning(f"Data fetch error for {symbol}: {e}")
        await update.message.reply_text(f"❌ ดึงข้อมูลราคาไม่สำเร็จ: {e}")
    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}", exc_info=True)
        await update.message.reply_text(f"❌ เกิดข้อผิดพลาดในการประมวลผล: {e}")

# ---------------------------------------------------------
# 7. Automated Background Alert Task
# ---------------------------------------------------------
async def auto_scan_and_alert(bot):
    """ระบบเบื้องหลังสแกนราคาส่งแจ้งเตือนอัตโนมัติ"""
    symbols = ["XAUUSD", "EURUSD", "GBPUSD"]
    
    if not TELEGRAM_CHAT_ID:
        logger.warning("ยังไม่ได้ใส่ TELEGRAM_CHAT_ID ระบบแจ้งเตือนอัตโนมัติจะไม่ส่งข้อความ")
        return

    while True:
        for symbol in symbols:
            try:
                df = fetch_chart_data(symbol)
                result, confidence = calculate_fibo_levels(df, symbol)
                
                if result is not None:
                    # เช็คป้องกันการส่งแจ้งเตือนแท่งเดิมซ้ำ
                    last_time = last_alerted_pattern_time.get(symbol)
                    if last_time != result.pattern_time:
                        last_alerted_pattern_time[symbol] = result.pattern_time
                        
                        msg = "🔔 *[แจ้งเตือนสัญญาณอัตโนมัติ]*\n\n" + format_signal_reply(result, confidence)
                        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown")
                        logger.info(f"ส่งแจ้งเตือนอัตโนมัติ {symbol} เรียบร้อยแล้ว")
            except Exception as e:
                logger.error(f"Auto Alert Error ({symbol}): {e}")
                
        # พัก 5 นาที (300 วินาที) แล้วค่อยสแกนรอบถัดไป
        await asyncio.sleep(300)

# ---------------------------------------------------------
# 8. Main Entry Point & Web Server
# ---------------------------------------------------------
app_web = FastAPI()

@app_web.get("/")
def health_check():
    return {"status": "ok", "bot": "running_with_ai"}

async def run_bot_and_web():
    telegram_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    telegram_app.add_handler(CommandHandler("scan", scan_command))
    
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling()

    # เรียกใช้ระบบแจ้งเตือนอัตโนมัติใน Background
    asyncio.create_task(auto_scan_and_alert(telegram_app.bot))

    port = int(os.environ.get("PORT", 10000))
    config = uvicorn.Config(app=app_web, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    
    logger.info(f"เริ่มต้นเปิด Web Server บน Port {port}...")
    await server.serve()

def main():
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.error("โปรดตั้งค่า TELEGRAM_BOT_TOKEN ใน Environment Variables")
        return

    asyncio.run(run_bot_and_web())

if __name__ == "__main__":
    main()
