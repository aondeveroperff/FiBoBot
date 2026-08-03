import os
import logging
from datetime import datetime
from typing import Optional, Tuple
import requests
import pandas as pd
import zoneinfo

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

HTTP_TIMEOUT_SECONDS = 15

# รายชื่อคู่เงินสำหรับ Yahoo Finance
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
# 2. Data Structures
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

# ---------------------------------------------------------
# 3. Data Fetching Logic (Twelve Data & Yahoo)
# ---------------------------------------------------------
def fetch_twelvedata_chart(symbol: str) -> pd.DataFrame:
    """ดึงข้อมูลราคา Real-time Spot (เช่น XAU/USD) จาก Twelve Data API"""
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
    """ดึงข้อมูลราคา Forex ทั่วไปจาก Yahoo Finance"""
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
    """เลือกใช้ API ให้เหมาะสมกับแต่ละ Asset"""
    sym = symbol.upper()
    if sym in ("XAUUSD", "GOLD", "XAGUSD", "SILVER"):
        return fetch_twelvedata_chart("XAUUSD" if sym in ("XAUUSD", "GOLD") else "XAGUSD")
    else:
        return fetch_yahoo_chart(sym)

# ---------------------------------------------------------
# 4. Calculation & Strategy Logic
# ---------------------------------------------------------
def calculate_fibo_levels(df: pd.DataFrame, symbol: str) -> ScanResult:
    """คำนวณสัญญาณ Fibonacci / Pattern Strategy"""
    if len(df) < 10:
        raise ValueError("ข้อมูลราคาไม่เพียงพอในการคำนวณ")

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    current_price = last_row["close"]
    current_time = df.index[-1]
    pattern_time = df.index[-2]

    high = prev_row["high"]
    low = prev_row["low"]
    diff = high - low

    # คำนวณระดับราคา Fibo
    krh2 = high - (diff * 0.236)  # Entry
    krh1 = low                    # Stop Loss
    run = high + (diff * 0.618)   # Take Profit

    return ScanResult(
        symbol=symbol.upper(),
        signal="CONFIRMED",
        pattern_time=pattern_time,
        current_time=current_time,
        current_price=current_price,
        KRH2=krh2,
        KRH1=krh1,
        RUN=run,
    )

# ---------------------------------------------------------
# 5. Formatting & Telegram Handlers
# ---------------------------------------------------------
def format_signal_reply(result: ScanResult, confidence: Optional[float]) -> str:
    """จัดรูปแบบข้อความตอบกลับ พร้อมแปลงเวลาเป็น UTC+7 (เวลาไทย)"""
    tz_th = zoneinfo.ZoneInfo("Asia/Bangkok")
    
    # แปลงเวลาเป็น UTC+7
    pattern_th = result.pattern_time.astimezone(tz_th).strftime("%Y-%m-%d %H:%M:%S")
    current_th = result.current_time.astimezone(tz_th).strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"*{result.symbol}*",
        f"สถานะ: `{result.signal}`",
        "",
        f"เวลาแท่งสัญญาณ: `{pattern_th}` (ไทย)",
        f"เวลาราคาล่าสุด: `{current_th}` (ไทย)",
        f"ราคาปัจจุบัน: `{result.current_price:.2f}`",
        "",
        f"Entry (KRH2): `{result.KRH2:.2f}`",
        f"Stop Loss (KRH1): `{result.KRH1:.2f}`",
        f"Take Profit (RUN): `{result.RUN:.2f}`",
    ]
    if confidence is not None:
        lines.append("")
        lines.append(f"AI Confidence: `{confidence:.2%}`")
        
    return "\n".join(lines)

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """คำสั่ง /scan <symbol>"""
    if not context.args:
        await update.message.reply_text("กรุณาระบุคู่เงิน เช่น `/scan XAUUSD`", parse_mode="Markdown")
        return

    symbol = context.args[0].upper()
    try:
        df = fetch_chart_data(symbol)
        result = calculate_fibo_levels(df, symbol)
        
        # สมมติการตรวจจับโมเดล AI (ถ้ามีไฟล์ model.pkl)
        confidence = None
        
        msg = format_signal_reply(result, confidence)
        await update.message.reply_text(msg, parse_mode="Markdown")
        
    except DataFetchError as e:
        logger.warning(f"Data fetch error for {symbol}: {e}")
        await update.message.reply_text(f"❌ ดึงข้อมูลราคาไม่สำเร็จ: {e}")
    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}", exc_info=True)
        await update.message.reply_text(f"❌ เกิดข้อผิดพลาดในการประมวลผล: {e}")

# ---------------------------------------------------------
# 6. Main Application Entry Point
# ---------------------------------------------------------
import uvicorn
from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler

# ... (โค้ดส่วนดึงข้อมูลราคาเหมือนเดิม) ...

# สร้าง FastAPI app เพื่อ Bind Port ให้ Render
api_app = FastAPI()
telegram_app = None

@api_app.get("/")
async def health_check():
    return {"status": "ok", "message": "Bot is running"}

@api_app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"status": "ok"}

def main():
    global telegram_app
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.error("โปรดตั้งค่า TELEGRAM_BOT_TOKEN ใน Environment Variables")
        return

    telegram_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    telegram_app.add_handler(CommandHandler("scan", scan_command))

    # ดึง PORT จาก Render (ถ้าไม่มีจะใช้ 10000)
    port = int(os.environ.get("PORT", 10000))
    
    # รัน Web Server
    uvicorn.run(api_app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
