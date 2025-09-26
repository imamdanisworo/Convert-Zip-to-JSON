# app.py — DBF → JSON (daily close), multi-file, robust parsing, code-length filter
# Accepts any file types; ignores non-DBF/ZIP but lists them; exports combined JSON and per-day JSON ZIP.

import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

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
Upload **one or more** files — the app will accept everything (DBF, ZIP, CSV, TXT, etc.).
It will **process only DBFs** (including DBFs inside ZIPs) and **ignore the rest** (logged below).

**Assumptions for DBF files:**
- Filename carries the date: `CPyyyymmdd.dbf` **or** `CPyymmdd.dbf`
- Each DBF contains the **closing price per stock** (at least one code column and one price column)
- Codes are normalized (trim + uppercase)
- Only codes with **length 4–7** are exported to JSON
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
      - CP20250925.dbf  (YYYYMMDD)
      - CP250925.dbf    (YYMMDD)
    """
    base = Path(name).name.upper()
    m8 = re.search(r'CP(\d{8})', base)
    if m8:
        return pd.Timestamp(datetime.strptime(m8.group(1), "%Y%m%d").date())
    m6 = re.search(r'CP(\d{6})', base)
    if m6:
        return pd.Timestamp(datetime.strptime(m6.group(1), "%y%m%d").date())
    raise ValueError("No CPyymmdd/CPyyyymmdd date pattern found")

def _clean_code(v) -> str:
    if isinstance(v, (bytes, bytearray)):
        v = bytes(v).decode("latin-1", errors="ignore")
    return str(v).replace("\x00", "").strip().upper()

def _clean_close(v):
    """
    Convert DBF field to float, tolerating:
    - real bytes (b'...')
    - byte-literal strings ("b'...\\x00'")
    - thousand separators, NULs, parentheses-negatives
    - stray non-numeric characters
    """
    import ast
    from decimal import Decimal

    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    if isinstance(v, Decimal):
        return float(v)

    # Decode bytes
    if isinstance(v, (bytes, bytearray)):
        try:
            v = bytes(v).decode("latin-1", errors="ignore")
        except Exception:
            v = str(bytes(v))  # repr-like fallback

    # Handle "byte-literal strings" e.g., "b'6875.00\\x00...'"
    if isinstance(v, str):
        s0 = v.strip()
        if re.match(r"""^b(['"]).*\1$""", s0):
            try:
                lit = ast.literal_eval(s0)  # -> bytes
                if isinstance(lit, (bytes, bytearray)):
                    v = lit.decode("latin-1", errors="ignore")
                else:
                    v = str(lit)
            except Exception:
                v = s0
        else:
            v = s0

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

def _choose_columns_for_code_close(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Choose the most likely price (numeric) and code (string) columns from >=2 columns.
    Heuristic:
      - price_col = column with highest count of successfully parsed numbers (_clean_close)
      - code_col  = column with lowest numeric success (or name hint: CODE/KODE/SYMBOL)
    """
    cols = df.columns.tolist()
    # Score numeric success
    scores = {c: df[c].map(_clean_close).notna().sum() for c in cols}
    # Try name hints for code column
    name_hints = [c for c in cols if str(c).strip().lower() in ("code", "kode", "symbol", "ticker", "stock", "saham")]
    price_col = max(scores, key=scores.get)
    # Candidate code columns: prefer hinted names not equal to price_col
    code_candidates = [c for c in name_hints if c != price_col] or [c for c in cols if c != price_col]
    # Among candidates, pick the one with lowest numeric score (more string-like)
    code_col = min(code_candidates, key=lambda c: scores[c]) if code_candidates else (cols[0] if cols[0] != price_col else cols[1])
    return code_col, price_col

def read_dbf_bytes(dbf_bytes: bytes) -> pd.DataFrame:
    """
    Read a DBF (raw bytes) into DataFrame with columns ['code','close'].
    Handles DBFs with 2+ columns by auto-detecting code vs price columns.
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
    if df.shape[1] < 2:
        # Not enough columns to extract code/close
        return pd.DataFrame(columns=["code", "close"])

    code_col, price_col = _choose_columns_for_code_close(df)

    out = pd.DataFrame({
        "code": df[code_col].map(_clean_code),
        "close": df[price_col].map(_clean_close)
    })
    out = out.dropna(subset=["close"])
    # Deduplicate codes per file (keep last)
    out = out.drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)
    return out

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

def ingest_uploaded_files(uploaded_files) -> Tuple[List[Tuple[pd.Timestamp, pd.DataFrame]], List[Dict[str, str]], List[str]]:
    """
    Accept many UploadedFile objects of any type.
    Returns:
      pairs:   [(date, df(code,close)), ...]  # only for DBFs (and inside ZIP)
      skipped: [{"member": str, "reason": str}, ...]  # bad DBFs or zip members
      others:  [filename, ...]  # non-DBF/ZIP files that we simply ignore
    """
    pairs, skipped, others = [], [], []
    for uf in uploaded_files:
        name = uf.name
        try:
            lower = name.lower()
            if lower.endswith(".zip"):
                triplets, sk = read_zip_of_dbfs(uf.read())
                pairs.extend([(d, df) for (d, df, _m) in triplets])
                skipped.extend(sk)
            elif lower.endswith(".dbf"):
                try:
                    date = parse_cp_date_from_name(name)
                except Exception as e:
                    skipped.append({"member": name, "reason": f"Bad filename/date: {e}"})
                    continue
                df = read_dbf_bytes(uf.read())
                pairs.append((date, df))
            else:
                # Accept upload but ignore for processing
                others.append(name)
        except Exception as e:
            skipped.append({"member": name, "reason": f"Processing error: {e}"})
    return pairs, skipped, others

def build_price_matrix(pairs: List[Tuple[pd.Timestamp, pd.DataFrame]]) -> pd.DataFrame:
    """
    Pivot (date, df[code,close]) pairs to wide matrix: index=date, columns=code, values=close.
    Ensures a 1-row DataFrame per date so setting index works.
    """
    if not pairs:
        return pd.DataFrame()

    frames = []
    for date, df in pairs:
        if df.empty:
            continue
        dfc = df.drop_duplicates(subset=["code"], keep="last").set_index("code")
        w = dfc[["close"]].T
        w.index = [pd.to_datetime(date)]
        frames.append(w)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=0).sort_index()
    return prices.apply(pd.to_numeric, errors="coerce")

def filter_stock_codes(prices: pd.DataFrame, min_len: int = 4, max_len: int = 7) -> pd.DataFrame:
    valid_cols = [c for c in prices.columns if min_len <= len(str(c)) <= max_len]
    return prices[valid_cols]

def to_json_nested_by_date(prices: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    nested = {}
    for d, row in prices.iterrows():
        nested[str(pd.to_datetime(d).date())] = {c: float(v) for c, v in row.dropna().to_dict().items()}
    return nested

def to_json_records(prices: pd.DataFrame) -> List[Dict[str, object]]:
    records = []
    for d, row in prices.iterrows():
        ds = str(pd.to_datetime(d).date())
        for code, val in row.dropna().items():
            records.append({"date": ds, "code": code, "close": float(val)})
    return records

def to_json_wide(prices: pd.DataFrame) -> Dict[str, object]:
    data_rows = prices.copy()
    data_rows.insert(0, "date", data_rows.index.date.astype(str))
    data_rows = data_rows.reset_index(drop=True)
    return {
        "meta": {"index": "date", "columns": list(prices.columns),
                 "notes": "Rows are dates; columns are stock codes; values are closing prices."},
        "data": data_rows.to_dict(orient="records")
    }

def filter_codes_in_df(df: pd.DataFrame, min_len=4, max_len=7) -> pd.DataFrame:
    """Filter a 2+ col df(code, close) to code length range."""
    df = df.copy()
    df["code"] = df["code"].astype(str)
    mask = df["code"].map(lambda c: min_len <= len(c) <= max_len)
    return df[mask]

def make_per_day_json_zip(pairs: List[Tuple[pd.Timestamp, pd.DataFrame]], indent: int = 2) -> bytes:
    """
    Build a ZIP in memory with one JSON per date:
      close_YYYY-MM-DD.json → { "CODE": close, ... }   (after code-length filtering)
    """
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for date, df in pairs:
            if df.empty:
                continue
            df_day = filter_codes_in_df(df, 4, 7)
            if df_day.empty:
                continue
            payload = {row["code"]: float(row["close"]) for _, row in df_day.dropna(subset=["close"]).iterrows()}
            if not payload:
                continue
            date_str = str(pd.to_datetime(date).date())
            z.writestr(f"close_{date_str}.json", json.dumps(payload, indent=indent, ensure_ascii=False))
    mem.seek(0)
    return mem.read()

# ------------------------ UI Controls ------------------------
colL, colR = st.columns([2, 1])
with colL:
    # Accept ANY file type so your error-log CSV can be uploaded together
    uploaded_files = st.file_uploader(
        "Upload your files (DBF / ZIP preferred; others are accepted but ignored)",
        type=None,  # accept everything
        accept_multiple_files=True
    )
with colR:
    layout = st.selectbox("Combined JSON layout", ["Nested by date (default)", "Row records (long)", "Wide table with meta"], index=0)
    indent = st.number_input("JSON indent", min_value=0, max_value=8, value=2, step=1)
    drop_empty_cols = st.checkbox("Drop empty columns (combined view)", value=True)

st.divider()

if not uploaded_files:
    st.info("Upload DBF/ZIP files (others allowed but ignored). Examples: `CP250925.dbf`, `CP20250925.dbf`, `CP_September.zip`.")
    st.stop()

with st.expander("All uploaded filenames"):
    st.write([f.name for f in uploaded_files])

# ------------------------ Processing ------------------------
try:
    pairs, skipped, others = ingest_uploaded_files(uploaded_files)

    if others:
        with st.expander("Other uploaded (ignored) files"):
            st.write(others)

    if not pairs and skipped:
        st.error("No valid DBFs found in the uploaded files.")
        with st.expander("Skipped files / reasons"):
            st.write(pd.DataFrame(skipped))
        st.stop()

    # --- Build combined wide price table ---
    prices = build_price_matrix(pairs)
    if prices.empty:
        st.error("No data parsed from the provided file(s).")
        if skipped:
            with st.expander("Skipped files / reasons"):
                st.write(pd.DataFrame(skipped))
        st.stop()

    if drop_empty_cols:
        prices = prices.dropna(axis=1, how="all")

    # filter codes 4–7 for combined JSON/table
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
        st.write(
            pd.DataFrame([{"date": str(d.date()), "codes": n_codes} for d in prices.index]).drop_duplicates()
        )

    if skipped:
        with st.expander("Skipped files / reasons"):
            st.write(pd.DataFrame(skipped))

    with st.expander("Preview (tail) of wide price table (combined)"):
        st.dataframe(prices.tail())

    # --- Combined JSON (single file) ---
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
        "⬇️ Download COMBINED JSON",
        data=json_bytes,
        file_name=fname,
        mime="application/json",
        use_container_width=True
    )

    # --- Per-day JSON (one file per DBF/day) packed into a ZIP ---
    zip_bytes = make_per_day_json_zip(pairs, indent=indent)
    st.download_button(
        "⬇️ Download PER-DAY JSONs (ZIP)",
        data=zip_bytes,
        file_name=f"close_prices_per_day_{date_min}_{date_max}.zip",
        mime="application/zip",
        use_container_width=True
    )

    st.success("JSONs are ready.")
except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
