import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# -------------- Optional dependency: dbfread --------------
# We use dbfread for robust DBF parsing.
# If not installed, show a friendly message.
try:
    from dbfread import DBF
    DBF_AVAILABLE = True
except Exception:
    DBF_AVAILABLE = False

st.set_page_config(page_title="DBF → JSON (Daily Close Prices)", layout="wide")
st.title("📦➡️🧾 DBF (CPyyyymmdd) → JSON Converter")

st.markdown(
    """
This app converts your **daily close price** files (DBF) into **JSON**.

**Assumptions (per your spec):**
- Each file is named `CPyyyymmdd.dbf` (date from filename)
- Each file contains **two columns**:  
  1) stock **code**  
  2) **closing price**  
- Files are usually provided in a **.zip** (multiple DBFs).  
"""
)

if not DBF_AVAILABLE:
    st.warning(
        "Python package **dbfread** is not installed. "
        "Please install it first:\n\n"
        "```bash\npip install dbfread\n```"
    )

# ----------------- Helpers -----------------
def parse_cp_date_from_name(name: str) -> pd.Timestamp:
    """
    Parse date from filename like 'CP20250925.dbf' (case-insensitive).
    Returns pandas Timestamp (date only).
    """
    stem = Path(name).stem.upper()
    if not (stem.startswith("CP") and len(stem) >= 10):
        raise ValueError(f"Filename '{name}' does not match 'CPyyyymmdd' pattern.")
    try:
        dt = datetime.strptime(stem[2:10], "%Y%m%d")
    except ValueError:
        raise ValueError(f"Filename '{name}' has invalid date portion.")
    return pd.Timestamp(dt.date())

def read_dbf_bytes(dbf_bytes: bytes) -> pd.DataFrame:
    """
    Read a DBF from raw bytes into a DataFrame.
    Expect exactly two columns: code and close (in any order).
    """
    if not DBF_AVAILABLE:
        raise RuntimeError("dbfread is required. Install with: pip install dbfread")

    # dbfread accepts file-like objects only via temporary workaround:
    # We write to a BytesIO-backed temporary file-like interface by using DBF on BytesIO
    # However dbfread expects a filesystem path. So write to a temp file.
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
        raise ValueError(f"DBF expected 2 columns but found {len(cols)}: {cols}")

    # Heuristic: numeric column is price, non-numeric is code
    c0_is_num = pd.api.types.is_numeric_dtype(df[cols[0]])
    if c0_is_num:
        price_col, code_col = cols[0], cols[1]
    else:
        price_col, code_col = cols[1], cols[0]

    out = df[[code_col, price_col]].copy()
    out.columns = ["code", "close"]
    # normalize
    out["code"] = out["code"].astype(str).str.strip().str.upper()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["close"])
    # keep last duplicate code (if present)
    out = out.drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)
    return out

def read_zip_of_dbfs(zip_bytes: bytes) -> List[Tuple[pd.Timestamp, pd.DataFrame, str]]:
    """
    Reads a ZIP (bytes) of DBF files and returns a list of (date, df, member_name).
    df has columns: code, close
    """
    results = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        members = [m for m in z.namelist() if m.lower().endswith(".dbf")]
        if not members:
            raise ValueError("No .dbf files found inside the ZIP.")

        for m in sorted(members):
            # parse date from filename
            date = parse_cp_date_from_name(Path(m).name)
            with z.open(m) as f:
                dbf_bytes = f.read()
            df = read_dbf_bytes(dbf_bytes)
            results.append((date, df, m))
    return results

def build_price_matrix(pairs: List[Tuple[pd.Timestamp, pd.DataFrame]]) -> pd.DataFrame:
    """
    Given list of (date, df[code, close]), pivot into wide matrix:
    index=date, columns=code, values=close
    """
    frames = []
    for date, df in pairs:
        if df.empty:
            continue
        wide = df.set_index("code")[["close"]].T
        wide.index = [date]
        frames.append(wide)
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, axis=0).sort_index()
    # enforce numeric
    prices = prices.apply(pd.to_numeric, errors="coerce")
    return prices

def to_json_nested_by_date(prices: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    { "YYYY-MM-DD": { "CODE": close, ... }, ... }
    """
    nested = {}
    for d, row in prices.iterrows():
        # drop nans
        data = {c: float(v) for c, v in row.dropna().to_dict().items()}
        nested[str(pd.to_datetime(d).date())] = data
    return nested

def to_json_records(prices: pd.DataFrame) -> List[Dict[str, object]]:
    """
    [ {"date": "YYYY-MM-DD", "code": "XXX", "close": 123.0}, ... ]  (long format)
    """
    records = []
    for d, row in prices.iterrows():
        date_str = str(pd.to_datetime(d).date())
        for code, val in row.dropna().items():
            records.append({"date": date_str, "code": code, "close": float(val)})
    return records

def to_json_wide(prices: pd.DataFrame) -> Dict[str, object]:
    """
    { "meta": {...}, "data": [ {"date":"YYYY-MM-DD", "BBCA":..., "TLKM":...}, ... ] }
    """
    data_rows = prices.copy()
    data_rows.insert(0, "date", data_rows.index.date.astype(str))
    data_rows = data_rows.reset_index(drop=True)
    return {
        "meta": {
            "index": "date",
            "columns": list(prices.columns),
            "notes": "Rows are dates; columns are stock codes; values are closing prices."
        },
        "data": data_rows.to_dict(orient="records")
    }

# ----------------- UI: upload & options -----------------
colL, colR = st.columns([2, 1])

with colL:
    uploaded = st.file_uploader("Upload a ZIP of DBFs (or a single .dbf)", type=["zip", "dbf"])
with colR:
    layout = st.selectbox(
        "JSON layout",
        ["Nested by date (default)", "Row records (long)", "Wide table with meta"],
        index=0
    )
    indent = st.number_input("JSON indent (pretty print)", min_value=0, max_value=8, value=2, step=1)
    drop_empty_cols = st.checkbox("Drop columns that are entirely empty", value=True)

st.divider()

if uploaded is None:
    st.info("Choose a **.zip** containing files like `CP20250925.dbf` (or a single `.dbf`).")
    st.stop()

# ----------------- Processing -----------------
try:
    pairs: List[Tuple[pd.Timestamp, pd.DataFrame]] = []

    if uploaded.name.lower().endswith(".zip"):
        # multiple days
        contents = uploaded.read()
        triplets = read_zip_of_dbfs(contents)  # (date, df, member_name)
        # show quick summary
        st.success(f"Found {len(triplets)} DBF file(s) in the ZIP.")
        with st.expander("Files detected"):
            st.write(pd.DataFrame(
                [{"member": m, "date": str(d.date()), "rows": len(df)} for (d, df, m) in triplets]
            ))
        pairs = [(d, df) for (d, df, _) in triplets]

    else:
        # single day
        # Parse date from the single DBF filename
        date = parse_cp_date_from_name(uploaded.name)
        df = read_dbf_bytes(uploaded.read())
        pairs = [(date, df)]
        st.success(f"Read {uploaded.name} with {len(df)} rows for date {str(date.date())}.")

    # Build price matrix
    prices = build_price_matrix(pairs)
    if prices.empty:
        st.error("No data parsed from the provided file(s).")
        st.stop()

    # Optionally drop entirely empty columns (all-NaN)
    if drop_empty_cols:
        prices = prices.dropna(axis=1, how="all")

    # Summary block
    n_days = prices.shape[0]
    n_codes = prices.shape[1]
    date_min = str(prices.index.min().date())
    date_max = str(prices.index.max().date())

    met1, met2, met3 = st.columns(3)
    met1.metric("Trading days", f"{n_days}")
    met2.metric("Unique codes", f"{n_codes}")
    met3.metric("Date range", f"{date_min} → {date_max}")

    with st.expander("Preview (tail) of wide price table"):
        st.dataframe(prices.tail())

    # Build JSON
    if layout.startswith("Nested by date"):
        json_obj = to_json_nested_by_date(prices)
        default_name = f"close_prices_nested_{date_min}_{date_max}.json"
    elif layout.startswith("Row records"):
        json_obj = to_json_records(prices)
        default_name = f"close_prices_records_{date_min}_{date_max}.json"
    else:
        json_obj = to_json_wide(prices)
        default_name = f"close_prices_wide_{date_min}_{date_max}.json"

    json_bytes = json.dumps(json_obj, indent=indent, ensure_ascii=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download JSON",
        data=json_bytes,
        file_name=default_name,
        mime="application/json",
        use_container_width=True
    )

    st.success("JSON is ready. Use the preview above to validate a few rows before exporting.")
except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
