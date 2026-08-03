"""
bot.py
------
Telegram Trading Scanner Bot
สำหรับ Deploy บน Render.com (Web Service - Free Plan)
"""

from __future__ import annotations

import os
import sys
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timezone
from typing import Optional

import requests
import pandas as pd

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

from fibo_tumni_scanner import TumniScanner, ScanResult

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("tumni-bot")

# --------------------------------------------------------------------------- #
# Config (จาก Environment Variables ของ Render)
# --------------------------------------------------------------------------- #
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
PORT = int(os.environ.get("PORT", "8080"))
AI_MODEL_PATH = os.environ.get("AI_MODEL_PATH", "model.pkl")
AI_SCALER_PATH = os.environ.get("AI_SCALER_PATH", "scaler.pkl")
AI_CONFIDENCE_THRESHOLD = float(os.environ.get("AI_CONFIDENCE_THRESHOLD", "0.60"))
HTTP_TIMEOUT_SECONDS = float(os.environ.get("HTTP_TIMEOUT_SECONDS", "10"))

if not TELEGRAM_BOT_TOKEN:
    logger.critical(
        "ไม่พบ TELEGRAM_BOT_TOKEN ใน Environment Variables — "
        "กรุณาตั้งค่าใน Render Dashboard > Environment ก่อนรัน"
    )

# --------------------------------------------------------------------------- #
# 1) Dummy HTTP Health Check Server
# --------------------------------------------------------------------------- #
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Tumni Trading Scanner Bot is running.")

    def log_message(self, format, *args):  # noqa: A002
        pass


def start_health_server(port: int) -> None:
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    logger.info("Health check server listening on 0.0.0.0:%s", port)
    server.serve_forever()


def run_health_server_in_background(port: int) -> threading.Thread:
    thread = threading.Thread(target=start_health_server, args=(port,), daemon=True)
    thread.start()
    return thread


# --------------------------------------------------------------------------- #
# 2) Data Fetching
# --------------------------------------------------------------------------- #
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
YAHOO_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"

# 🔥 จุดที่แก้ไข: กำหนดให้ XAUUSD และ GOLD ดึงจาก GC=F (Gold Futures)
YAHOO_ALIASES = {
    "XAUUSD": "XAUUSD=X",
    "GOLD": "XAUUSD=X",
    "XAGUSD": "XAGUSD=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X",
}

USER_AGENT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


class DataFetchError(Exception):
    """เกิดข้อผิดพลาดระหว่างดึงข้อมูลราคา"""


def is_crypto_symbol(symbol: str) -> bool:
    return symbol.upper().endswith("USDT")


def fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 150) -> pd.DataFrame:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    try:
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        resp.raise_for_status()
        raw = resp.json()
    except requests.RequestException as exc:
        raise DataFetchError(f"เรียก Binance API ไม่สำเร็จ: {exc}") from exc

    if not isinstance(raw, list) or len(raw) == 0:
        raise DataFetchError(f"Binance ไม่พบข้อมูลสำหรับสัญลักษณ์ {symbol}")

    df = pd.DataFrame(
        raw,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    df = df[["time", "open", "high", "low", "close", "volume"]].set_index("time")
    return df


def fetch_yahoo_chart(symbol: str, interval: str = "60m", range_: str = "1mo") -> pd.DataFrame:
    symbol_clean = symbol.upper().strip()
    
    # ดึงสัญลักษณ์จาก Alias ก่อน ถ้าไม่มีค่อยเติม =X
    if symbol_clean in YAHOO_ALIASES:
        yahoo_symbol = YAHOO_ALIASES[symbol_clean]
    elif not symbol_clean.endswith("=X"):
        yahoo_symbol = f"{symbol_clean}=X"
    else:
        yahoo_symbol = symbol_clean

    url = YAHOO_CHART_URL.format(symbol=yahoo_symbol)
    params = {"interval": interval, "range": range_, "includePrePost": "false"}

    try:
        resp = requests.get(
            url, params=params, headers=USER_AGENT_HEADERS, timeout=HTTP_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise DataFetchError(f"เรียก Yahoo Finance API ไม่สำเร็จ ({yahoo_symbol}): {exc}") from exc

    try:
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
    except (KeyError, IndexError, TypeError) as exc:
        error_desc = None
        if isinstance(data, dict):
            error_desc = (data.get("chart") or {}).get("error")
        raise DataFetchError(
            f"Yahoo Finance ไม่พบข้อมูลสำหรับสัญลักษณ์ {symbol} ({error_desc})"
        ) from exc

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(timestamps, unit="s", utc=True),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    df = df.dropna(subset=["open", "high", "low", "close"]).set_index("time")

    if df.empty:
        raise DataFetchError(f"Yahoo Finance คืนข้อมูลว่างเปล่าสำหรับสัญลักษณ์ {symbol}")

    return df


def fetch_h1_candles(symbol: str) -> pd.DataFrame:
    symbol = symbol.upper().strip()
    if is_crypto_symbol(symbol):
        return fetch_binance_klines(symbol, interval="1h", limit=150)
    return fetch_yahoo_chart(symbol, interval="60m", range_="1mo")


# --------------------------------------------------------------------------- #
# 3) AI Confidence Filter
# --------------------------------------------------------------------------- #
_ai_model = None
_ai_scaler = None
_ai_load_attempted = False


def _try_load_ai_model() -> None:
    global _ai_model, _ai_scaler, _ai_load_attempted
    if _ai_load_attempted:
        return
    _ai_load_attempted = True

    if not os.path.exists(AI_MODEL_PATH):
        logger.info("ไม่พบไฟล์โมเดล AI (%s) — จะข้ามการกรองด้วย AI Confidence", AI_MODEL_PATH)
        return

    try:
        import joblib

        _ai_model = joblib.load(AI_MODEL_PATH)
        if os.path.exists(AI_SCALER_PATH):
            _ai_scaler = joblib.load(AI_SCALER_PATH)
        logger.info("โหลดโมเดล AI สำเร็จจาก %s", AI_MODEL_PATH)
    except Exception:
        logger.exception("โหลดโมเดล AI ล้มเหลว จะข้ามการกรองด้วย AI Confidence")
        _ai_model = None
        _ai_scaler = None


def compute_ai_confidence(df: pd.DataFrame, result: ScanResult) -> Optional[float]:
    _try_load_ai_model()
    if _ai_model is None:
        return None

    try:
        last = df.iloc[-1]
        candle_range = (result.candle_high or 0) - (result.candle_low or 0)
        features = [[
            float(last["close"]),
            float(last["open"]),
            float(last["high"]),
            float(last["low"]),
            candle_range,
            (result.current_price or 0) - (result.KRH2 or 0),
        ]]
        if _ai_scaler is not None:
            features = _ai_scaler.transform(features)

        if hasattr(_ai_model, "predict_proba"):
            proba = _ai_model.predict_proba(features)[0]
            confidence = float(max(proba))
        else:
            confidence = float(_ai_model.predict(features)[0])
        return confidence
    except Exception:
        logger.exception("คำนวณ AI Confidence ล้มเหลว จะข้ามการกรองรอบนี้")
        return None


# --------------------------------------------------------------------------- #
# 4) Telegram Handlers
# --------------------------------------------------------------------------- #
scanner = TumniScanner()


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "*Tumni Trading Scanner Bot*\n\n"
        "สวัสดีครับ ผมเป็นบอทสแกนหาสัญญาณเทรดด้วยแนวคิด Fibo + แท่งเทียนตำหนิ (Rejection Candle)\n\n"
        "*วิธีใช้งาน:*\n"
        "`/scan XAUUSD` — สแกนทองคำ\n"
        "`/scan BTCUSDT` — สแกน Bitcoin\n"
        "`/scan EURUSD` — สแกนคู่เงิน EUR/USD\n\n"
        "บอทจะแสดงเฉพาะสัญญาณสถานะ `WAIT_BUY` หรือ `CONFIRMED` เท่านั้น "
        "หากราคาหลุด Stop Loss ไปแล้ว จะรายงานว่าไม่พบสัญญาณ"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "กรุณาระบุสัญลักษณ์ที่ต้องการสแกน เช่น `/scan XAUUSD`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    symbol = context.args[0].upper().strip()
    processing_msg = await update.message.reply_text(f"⏳ กำลังสแกน `{symbol}` ...", parse_mode=ParseMode.MARKDOWN)

    try:
        df = fetch_h1_candles(symbol)
    except DataFetchError as exc:
        logger.warning("Data fetch error for %s: %s", symbol, exc)
        await processing_msg.edit_text(
            f"❌ ไม่สามารถดึงข้อมูลราคาสำหรับ `{symbol}` ได้\n`{exc}`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    except Exception:
        logger.exception("Unexpected error fetching data for %s", symbol)
        await processing_msg.edit_text(
            f"❌ เกิดข้อผิดพลาดที่ไม่คาดคิดขณะดึงข้อมูล `{symbol}`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    try:
        result = scanner.scan(df, symbol=symbol)
    except Exception:
        logger.exception("Unexpected error scanning %s", symbol)
        await processing_msg.edit_text(
            f"❌ เกิดข้อผิดพลาดที่ไม่คาดคิดขณะวิเคราะห์ `{symbol}`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    if result.signal not in ("WAIT_BUY", "CONFIRMED"):
        reason = result.reason or "ไม่พบเงื่อนไขสัญญาณที่ตรงตามเกณฑ์"
        await processing_msg.edit_text(
            f"*{symbol}*\nสถานะ: `NONE`\nเหตุผล: {reason}",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    confidence = compute_ai_confidence(df, result)
    if confidence is not None and confidence < AI_CONFIDENCE_THRESHOLD:
        await processing_msg.edit_text(
            f"*{symbol}*\n"
            f"สถานะ: `REJECTED BY AI`\n"
            f"AI Confidence: `{confidence:.2%}` (ต่ำกว่าเกณฑ์ {AI_CONFIDENCE_THRESHOLD:.0%})\n"
            f"ระบบตรวจพบสัญญาณ `{result.signal}` แต่ความมั่นใจของ AI ไม่ถึงเกณฑ์ จึงไม่แนะนำให้เข้าเทรด",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    reply = format_signal_reply(result, confidence)
    await processing_msg.edit_text(reply, parse_mode=ParseMode.MARKDOWN)


import zoneinfo # เพิ่มไว้ด้านบนสุดของไฟล์ถ้ายังไม่มี

def format_signal_reply(result: ScanResult, confidence: Optional[float]) -> str:
    # แปลงเวลาเป็นเวลาประเทศไทย (UTC+7)
    tz_th = zoneinfo.ZoneInfo("Asia/Bangkok")
    
    pattern_time_str = str(result.pattern_time)
    current_time_str = str(result.current_time)
    
    if isinstance(result.pattern_time, datetime):
        pattern_time_str = result.pattern_time.astimezone(tz_th).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(result.current_time, datetime):
        current_time_str = result.current_time.astimezone(tz_th).strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"*{result.symbol}*",
        f"สถานะ: `{result.signal}`",
        "",
        f"เวลาแท่งสัญญาณ: `{pattern_time_str}` (ไทย)",
        f"เวลาราคาล่าสุด: `{current_time_str}` (ไทย)",
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



async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception while processing update: %s", context.error, exc_info=context.error)


# --------------------------------------------------------------------------- #
# 5) Entry point
# --------------------------------------------------------------------------- #
def build_application() -> Application:
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("scan", scan_command))
    application.add_error_handler(error_handler)
    return application


def main() -> None:
    run_health_server_in_background(PORT)

    if not TELEGRAM_BOT_TOKEN:
        logger.critical(
            "หยุดการทำงานของ Telegram Bot เนื่องจากไม่มี TELEGRAM_BOT_TOKEN "
            "(Health Check Server ยังคงทำงานต่อไปเพื่อไม่ให้ Render mark เป็น failed)"
        )
        threading.Event().wait()
        return

    logger.info("กำลังเริ่มต้น Telegram Bot Polling ...")
    application = build_application()
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        close_loop=False,
        stop_signals=None,
    )


if __name__ == "__main__":
    main()
