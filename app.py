# app.py — DBF → JSON (daily close), multi-file, robust parsing, code-length filter
import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---- Dependency check for DBF parsing ----
try:
    from dbfread import DBF
    DBF_AVAILABLE = True
except Exception:
    DBF_AVAILABLE = False

# ------------------------ UI Setup ------------------------
st.set_page_config(page_title="DBF → JSON (Daily Close Prices)", layout="wide")
st.title("📦➡️🧾 DBF (CPyyyymmdd / CPyymmdd) → JSON Converter")

st.markdown(
    """
Upload **one or more** `.dbf` files and/or `.zip` files that contain `.dbf`.
The app parses your **daily closing prices** and exports **JSON**.

**Assumptions:**
- Filenames are `CPyyyymmdd.dbf` **or** `CPyymmdd.dbf` (date from filename)
- Each DBF has exactly **two columns**: stock **code**, **close**
- Codes are normalized (trim + uppercase)
- JSON will include only codes with **length 4–7**
"""
)

if not DBF_AVAILABLE:
    st.warning(
        "Python package **dbfread** is not installed. "
        "Install it first:\n\n"
        "```bash\npip install dbfread\n```"
    )

# ------------------------ Helpers ------------------------
def parse_cp_date_from_name(name: str) -> pd.Timestamp:
    """
    Parse trading date from filenames like:
      - CP20250925.dbf  (YYYYMMDD, 8 digits)
      - CP250925.dbf    (YYMMDD,   6 digits)
    Returns pandas.Timestamp (date only).
    """
    base = Path(name).name.upper()
    m8 = re.search(r'CP(\d{8})', base)
    if m8:  # YYYYMMDD
        return pd.Timestamp(datetime.strptime(m8.group(1), "%Y%m%d").date())
    m6 = re.search(r'CP(\d{6})', base)
    if m6:  # YYMMDD  (Python %y: 00–68 => 2000–2068; 69–99 => 1969–1999)
        return pd.Timestamp(datetime.strptime(m6.group(1), "%y%m%d").date())
    raise ValueError("No CPyymmdd/CPyyyymmdd date pattern found")

def _clean_code(v) -> str:
    if isinstance(v, (bytes, bytearray)):
        v = bytes(v).decode("latin-1", errors="ignore")
    return str(v).replace("\x00", "").strip().upper()

def _clean_close(v):
    """
    Convert DBF field to float, tolerating:
    - bytes padded with NULs
    - thousand separators
    - parentheses for negatives
    - stray non-numeric characters
    """
    from decimal import Decimal
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (bytes, bytearray)):
        v = bytes(v).decode("latin-1", errors="ignore")

    s = str(v)
    neg = "(" in s and ")" in s
    s = s.replace("\x00", "").replace(",", "").strip()
    s = re.sub(r"[^0-9\.\-]", "", s)  # keep only digits/dot/minus
    if s in ("", ".", "-", "-."):
        return np.nan
    try:
        val = float(s)
        if neg and val > 0:
            val = -val
        return val
    except Exception:
        return np.nan

def read_dbf_bytes(dbf_bytes: bytes) -> pd.DataFrame:
    """
    Read a DBF (raw bytes) into DataFrame with columns ['code','close'].
    Detect which of the two columns is price by numeric success rate after cleaning.
    """
    if not DBF_AVAILABLE:
        raise RuntimeError("dbfread is required. Install with: pip install dbfread")

    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dbf") as tmp:
        tmp.write(dbf_bytes)
        tmp_path = tmp.name

    try:
        rows = list(DBF(tmp_path, load=True, ignore_missing_memofile=True))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if not rows:
        return pd.DataFrame(columns=["code", "close"])

    df = pd.DataFrame(rows)
    cols = df.columns.tolist()
    if len(cols) != 2:
        raise ValueError(f"Expected 2 columns, found {len(cols)}: {cols}")

    c0, c1 = cols[0], cols[1]
    col0_num = df[c0].map(_clean_close)
    col1_num = df[c1].map(_clean_close)
    c0_valid = col0_num.notna().sum()
    c1_valid = col1_num.notna().sum()

    if c0_valid > c1_valid:
        price_series = col0_num
        code_series = df[c1].map(_clean_code)
    elif c1_valid > c0_valid:
        price_series = col1_num
        code_series = df[c0].map(_clean_code)
    else:
        # tie-breaker: try raw numeric coercion as a hint
        t0 = pd.to_numeric(df[c0], errors="coerce").notna().sum()
        t1 = pd.to_numeric(df[c1], errors="coerce").notna().sum()
        if t0 >= t1:
            price_series = col0_num; code_series = df[c1].map(_clean_code)
        else:
            price_series = col1_num; code_series = df[c0].map(_clean_code)

    out = pd.DataFrame({"code": code_series, "close": price_series}).dropna(subset=["close"])
    # Deduplicate codes per file (keep last)
    return out.drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)

def read_zip_of_dbfs(zip_bytes: bytes):
    """
    Read a ZIP (bytes) of DBFs.
    Returns (triplets, skipped) where:
      - triplets: [(date: Timestamp, df(code,close): DataFrame, member_name: str), ...]
      - skipped:  [{"member": str, "reason": str}, ...]
    """
    results, skipped = [], []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        members = [
            m for m in z.namelist()
            if m.lower().endswith(".dbf")
            and "__macosx/" not in m.lower()
            and not Path(m).name.startswith("._")
        ]
        if not members:
            return results, [{"member": "<zip>", "reason": "No .dbf files in ZIP"}]

        for m in sorted(members):
            try:
                date = parse_cp_date_from_name(m)
            except Exception as e:
                skipped.append({"member": m, "reason": f"Bad filename/date: {e}"})
                continue
            try:
                with z.open(m) as f:
                    dbf_bytes = f.read()
                df = read_dbf_bytes(dbf_bytes)
                results.append((date, df, m))
            except Exception as e:
                skipped.append({"member": m, "reason": f"DBF read error: {e}"})
    return results, skipped

def ingest_uploaded_files(uploaded_files) -> Tuple[List[Tuple[pd.Timestamp, pd.DataFrame]], List[Dict[str, str]]]:
    """
    Accept many UploadedFile objects (.dbf or .zip).
    Returns:
      pairs:   [(date, df(code,close)), ...]
      skipped: [{"member": str, "reason": str}, ...]
    """
    pairs, skipped = [], []
    for uf in uploaded_files:
        name = uf.name
        try:
            if name.lower().endswith(".zip"):
                triplets, sk = read_zip_of_dbfs(uf.read())
                pairs.extend([(d, df) for (d, df, _m) in triplets])
                skipped.extend(sk)
            elif name.lower().endswith(".dbf"):
                try:
                    date = parse_cp_date_from_name(name)
                except Exception as e:
                    skipped.append({"member": name, "reason": f"Bad filename/date: {e}"})
                    continue
                df = read_dbf_bytes(uf.read())
                pairs.append((date, df))
            else:
                skipped.append({"member": name, "reason": "Unsupported file type (use .dbf or .zip)"})
        except Exception as e:
            skipped.append({"member": name, "reason": f"Processing error: {e}"})
    return pairs, skipped

def build_price_matrix(pairs: List[Tuple[pd.Timestamp, pd.DataFrame]]) -> pd.DataFrame:
    """
    Pivot (date, df[code,close]) pairs to wide matrix: index=date, columns=code, values=close.
    """
    if not pairs:
        return pd.DataFrame()
    frames = []
    for date, df in pairs:
        if df.empty:
            continue
        w = df.set_index("code")[["close"]].T
        w.index = [date]
        frames.append(w)
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, axis=0).sort_index()
    return prices.apply(pd.to_numeric, errors="coerce")

def filter_stock_codes(prices: pd.DataFrame, min_len: int = 4, max_len: int = 7) -> pd.DataFrame:
    """
    Keep only stock codes whose name length is between min_len and max_len.
    """
    valid_cols = [c for c in prices.columns if min_len <= len(str(c)) <= max_len]
    return prices[valid_cols]

def to_json_nested_by_date(prices: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    { "YYYY-MM-DD": { "CODE": close, ... }, ... }
    """
    nested = {}
    for d, row in prices.iterrows():
        nested[str(pd.to_datetime(d).date())] = {c: float(v) for c, v in row.dropna().to_dict().items()}
    return nested

def to_json_records(prices: pd.DataFrame) -> List[Dict[str, object]]:
    """
    [ {"date":"YYYY-MM-DD","code":"XXX","close":123.0}, ... ]
    """
    records = []
    for d, row in prices.iterrows():
        ds = str(pd.to_datetime(d).date())
        for code, val in row.dropna().items():
            records.append({"date": ds, "code": code, "close": float(val)})
    return records

def to_json_wide(prices: pd.DataFrame) -> Dict[str, object]:
    """
    { "meta": {...}, "data": [ {"date":"YYYY-MM-DD", "BBCA":..., ...}, ... ] }
    """
    data_rows = prices.copy()
    data_rows.insert(0, "date", data_rows.index.date.astype(str))
    data_rows = data_rows.reset_index(drop=True)
    return {
        "meta": {"index": "date", "columns": list(prices.columns),
                 "notes": "Rows are dates; columns are stock codes; values are closing prices."},
        "data": data_rows.to_dict(orient="records")
    }

# ------------------------ UI Controls ------------------------
colL, colR = st.columns([2, 1])
with colL:
    uploaded_files = st.file_uploader(
        "Upload one or more files (.dbf and/or .zip)",
        type=["dbf", "zip"],
        accept_multiple_files=True
    )
with colR:
    layout = st.selectbox("JSON layout", ["Nested by date (default)", "Row records (long)", "Wide table with meta"], index=0)
    indent = st.number_input("JSON indent", min_value=0, max_value=8, value=2, step=1)
    drop_empty_cols = st.checkbox("Drop columns that are entirely empty", value=True)

st.divider()

if not uploaded_files:
    st.info("Upload `.dbf` and/or `.zip` files (e.g., `CP250925.dbf`, `CP20250925.dbf`).")
    st.stop()

with st.expander("Uploaded filenames"):
    st.write([f.name for f in uploaded_files])

# ------------------------ Processing ------------------------
try:
    pairs, skipped = ingest_uploaded_files(uploaded_files)
    if not pairs and skipped:
        st.error("No valid DBFs found in the uploaded files.")
        with st.expander("Skipped files / reasons"):
            st.write(pd.DataFrame(skipped))
        st.stop()

    prices = build_price_matrix(pairs)
    if prices.empty:
        st.error("No data parsed from the provided file(s).")
        if skipped:
            with st.expander("Skipped files / reasons"):
                st.write(pd.DataFrame(skipped))
        st.stop()

    if drop_empty_cols:
        prices = prices.dropna(axis=1, how="all")

    # Filter codes to length 4–7 (your requirement)
    prices = filter_stock_codes(prices, min_len=4, max_len=7)

    # Summary
    n_days, n_codes = prices.shape
    date_min = str(prices.index.min().date())
    date_max = str(prices.index.max().date())

    m1, m2, m3 = st.columns(3)
    m1.metric("Trading days", f"{n_days}")
    m2.metric("Codes (4–7 chars)", f"{n_codes}")
    m3.metric("Date range", f"{date_min} → {date_max}")

    with st.expander("Processed (by date)"):
        st.write(pd.DataFrame([{"date": str(d.date()), "cols": prices.shape[1]} for d in prices.index]).drop_duplicates())

    if skipped:
        with st.expander("Skipped files / reasons"):
            st.write(pd.DataFrame(skipped))

    with st.expander("Preview (tail) of wide price table"):
        st.dataframe(prices.tail())

    # Build JSON
    if layout.startswith("Nested by date"):
        json_obj = to_json_nested_by_date(prices)
        fname = f"close_prices_nested_{date_min}_{date_max}.json"
    elif layout.startswith("Row records"):
        json_obj = to_json_records(prices)
        fname = f"close_prices_records_{date_min}_{date_max}.json"
    else:
        json_obj = to_json_wide(prices)
        fname = f"close_prices_wide_{date_min}_{date_max}.json"

    json_bytes = json.dumps(json_obj, indent=indent, ensure_ascii=False).encode("utf-8")
    st.download_button(
        "⬇️ Download JSON",
        data=json_bytes,
        file_name=fname,
        mime="application/json",
        use_container_width=True
    )

    st.success("JSON is ready.")
except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
