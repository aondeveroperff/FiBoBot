"""
fibo_tumni_scanner.py
----------------------
สแกนหา "แท่งเทียนตำหนิ" (rejection candle) บนกราฟ H1 แล้วคำนวณระดับราคาตาม
แนวคิด Fibonacci retracement / extension เพื่อกำหนด:

    KRH1  = ระดับ Stop Loss   (จุดต่ำสุดของแท่งสัญญาณ)
    KRH2  = ระดับ Entry       (จุด Fibo retracement สำหรับเข้าออเดอร์)
    RUN   = ระดับ Take Profit (จุด Fibo extension เป็นเป้าหมายทำกำไร)

สถานะสัญญาณที่คืนค่า:
    WAIT_BUY   -> เจอแท่งตำหนิแล้ว แต่ราคายังไม่ย่อลงมาถึงโซนเข้า (KRH2)
    CONFIRMED  -> ราคาได้ย่อลงมาแตะ/เข้าโซน KRH2 แล้ว และยังไม่หลุด SL (KRH1)
    NONE       -> ไม่พบแท่งสัญญาณที่ถูกต้อง หรือราคาหลุด SL ไปแล้ว
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    signal: str                      # "WAIT_BUY" | "CONFIRMED" | "NONE"
    symbol: str
    pattern_time: Optional[str]
    candle_high: Optional[float]
    candle_low: Optional[float]
    KRH1: Optional[float]            # Stop Loss
    KRH2: Optional[float]            # Entry
    RUN: Optional[float]             # Take Profit target
    current_price: Optional[float]
    current_time: Optional[str]
    reason: Optional[str] = None     # เหตุผลเวลาสัญญาณเป็น NONE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TumniScanner:
    """
    สแกนหาแท่งเทียนตำหนิ (rejection candle) สำหรับฝั่ง Buy เท่านั้น
    (ตามสเปกของบอทที่กรองเฉพาะ WAIT_BUY / CONFIRMED)
    """

    def __init__(
        self,
        lookback: int = 40,
        min_wick_body_ratio: float = 2.0,
        entry_fib_ratio: float = 0.5,
        target_fib_ratio: float = 1.618,
        sl_buffer_pct: float = 0.0005,
    ) -> None:
        """
        Parameters
        ----------
        lookback : จำนวนแท่งเทียนล่าสุดที่จะย้อนหาสัญญาณ (ตัวล่าสุดก่อน)
        min_wick_body_ratio : ไส้ล่างต้องยาวกว่าตัวแท่งอย่างน้อยกี่เท่า จึงนับเป็นแท่งตำหนิ
        entry_fib_ratio : สัดส่วน Fibo retracement ที่ใช้คำนวณ KRH2 (Entry)
        target_fib_ratio : สัดส่วน Fibo extension ที่ใช้คำนวณ RUN (Take Profit)
        sl_buffer_pct : บัฟเฟอร์เพิ่มใต้ low ของแท่งสัญญาณ สำหรับ KRH1 (กันโดนล่าหลอก)
        """
        self.lookback = lookback
        self.min_wick_body_ratio = min_wick_body_ratio
        self.entry_fib_ratio = entry_fib_ratio
        self.target_fib_ratio = target_fib_ratio
        self.sl_buffer_pct = sl_buffer_pct

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def scan(self, df: pd.DataFrame, symbol: str = "") -> ScanResult:
        """
        df ต้องมีคอลัมน์: ['open', 'high', 'low', 'close'] อย่างน้อย
        และเรียงจากเก่า -> ใหม่ (แท่งสุดท้าย = แท่งปัจจุบัน/ล่าสุด)
        Index ควรเป็น DatetimeIndex (ไม่บังคับ)
        """
        if df is None or len(df) < 5:
            return ScanResult(
                signal="NONE",
                symbol=symbol,
                pattern_time=None,
                candle_high=None,
                candle_low=None,
                KRH1=None,
                KRH2=None,
                RUN=None,
                current_price=None,
                current_time=None,
                reason="ข้อมูลแท่งเทียนไม่เพียงพอสำหรับการสแกน",
            )

        df = df.copy().reset_index(drop=False)
        if "time" not in df.columns:
            # เผื่อไม่มีคอลัมน์เวลาชัดเจน ให้ตั้งชื่อ index เดิมเป็น time
            df.rename(columns={df.columns[0]: "time"}, inplace=True)

        current = df.iloc[-1]
        current_price = float(current["close"])
        current_time = str(current["time"])

        # ค้นหาแท่งตำหนิที่ล่าสุดที่สุดภายใน lookback แท่ง (ไม่รวมแท่งปัจจุบัน)
        search_start = max(0, len(df) - 1 - self.lookback)
        candidate_idx = None
        for i in range(len(df) - 2, search_start - 1, -1):
            row = df.iloc[i]
            if self._is_flaw_candle(row):
                candidate_idx = i
                break

        if candidate_idx is None:
            return ScanResult(
                signal="NONE",
                symbol=symbol,
                pattern_time=None,
                candle_high=None,
                candle_low=None,
                KRH1=None,
                KRH2=None,
                RUN=None,
                current_price=current_price,
                current_time=current_time,
                reason="ไม่พบแท่งเทียนตำหนิภายในช่วงที่กำหนด",
            )

        signal_candle = df.iloc[candidate_idx]
        high = float(signal_candle["high"])
        low = float(signal_candle["low"])
        candle_range = high - low

        if candle_range <= 0:
            return ScanResult(
                signal="NONE",
                symbol=symbol,
                pattern_time=str(signal_candle["time"]),
                candle_high=high,
                candle_low=low,
                KRH1=None,
                KRH2=None,
                RUN=None,
                current_price=current_price,
                current_time=current_time,
                reason="แท่งสัญญาณมี range เป็นศูนย์ ไม่สามารถคำนวณ Fibo ได้",
            )

        KRH1 = low * (1 - self.sl_buffer_pct)                      # Stop Loss
        KRH2 = low + (candle_range * self.entry_fib_ratio)         # Entry
        RUN = high + (candle_range * (self.target_fib_ratio - 1))  # Take Profit

        # --- Guard: เช็คว่าราคาปัจจุบันหลุด SL ไปแล้วหรือยัง ---
        latest_low = float(current["low"])
        if current_price <= KRH1 or latest_low <= KRH1:
            return ScanResult(
                signal="NONE",
                symbol=symbol,
                pattern_time=str(signal_candle["time"]),
                candle_high=high,
                candle_low=low,
                KRH1=KRH1,
                KRH2=KRH2,
                RUN=RUN,
                current_price=current_price,
                current_time=current_time,
                reason="ราคาปัจจุบันหลุดระดับ Stop Loss (KRH1) แล้ว สัญญาณหมดอายุ",
            )

        # --- ตัดสินสถานะ WAIT_BUY vs CONFIRMED ---
        if current_price <= KRH2:
            signal = "CONFIRMED"
        else:
            signal = "WAIT_BUY"

        return ScanResult(
            signal=signal,
            symbol=symbol,
            pattern_time=str(signal_candle["time"]),
            candle_high=high,
            candle_low=low,
            KRH1=KRH1,
            KRH2=KRH2,
            RUN=RUN,
            current_price=current_price,
            current_time=current_time,
            reason=None,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _is_flaw_candle(self, row: pd.Series) -> bool:
        """
        นิยาม 'แท่งเทียนตำหนิ' (bullish rejection candle) สำหรับฝั่ง Buy:
          - มีไส้ล่าง (lower wick) ยาวกว่าตัวแท่ง (body) อย่างน้อย min_wick_body_ratio เท่า
          - ราคาปิดอยู่ในครึ่งบนของ range แท่งเทียน (แสดงถึงแรงซื้อกลับ)
        """
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        body = abs(c - o)
        candle_range = h - l
        if candle_range <= 0:
            return False

        lower_wick = min(o, c) - l
        if body == 0:
            # แท่ง Doji: ใช้สัดส่วนไส้ล่างเทียบกับ range แทน
            wick_ok = lower_wick >= candle_range * 0.6
        else:
            wick_ok = lower_wick >= body * self.min_wick_body_ratio

        close_position = (c - l) / candle_range  # 0 = ปิดที่ low, 1 = ปิดที่ high
        close_upper_half = close_position >= 0.5

        return wick_ok and close_upper_half
