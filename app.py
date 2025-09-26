import io
import json
import zipfile
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---- Optional dependency for DBF parsing ----
try:
    from dbfread import DBF
    DBF_AVAILABLE = True
except Exception:
    DBF_AVAILABLE = False

st.set_page_config(page_title="DBF → JSON (Daily Close Prices)", layout="wide")
st.title("📦➡️🧾 DBF (CPyyyymmdd) → JSON Converter — Multi-file")

st.markdown(
    """
Upload **one or more** files (`.dbf` and/or `.zip` containing `.dbf`).
The app will parse daily close prices and export **JSON**.

**Assumptions:**
- Each daily file is named `CPyyyymmdd.dbf` (date from filename)
- Each DBF contains exactly **two columns**: stock **code**, **close**
- Codes are normalized (trim + uppercase), prices to numeric
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
    Robustly parse date from filenames.
    Preferred: CPyyyyMMdd.*  (e.g., CP20250925.dbf)
    Also tolerates separators/noise; fallback to any YYYYMMDD.
    """
    base = Path(name).name
    base_clean = base.replace("-", "").replace("_", "")
    m = re.search(r'(?i)\bCP(\d{8})\b', base_clean)
    if m:
        return pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%d").date())
    m2 = re.search(r'(\d{4})(\d{2})(\d{2})', base_clean)
    if m2:
        return pd.Timestamp(datetime.strptime("".join(m2.groups()), "%Y%m%d").date())
    raise ValueError(f"Filename '{base}' does not contain a parsable date (expected CPyyyyMMdd).")

def read_dbf_bytes(dbf_bytes: bytes) -> pd.DataFrame:
    """
    Read a DBF (raw bytes) into DataFrame with columns: ['code','close'].
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
        raise ValueError(f"DBF expected 2 columns but found {len(cols)}: {cols}")

    c0_is_num = pd.api.types.is_numeric_dtype(df[cols[0]])
    if c0_is_num:
        price_col, code_col = cols[0], cols[1]
    else:
        price_col, code_col = cols[1], cols[0]

    out = df[[code_col, price_col]].copy()
    out.columns = ["code", "close"]
    out["code"] = out["code"].astype(str).str.strip().str.upper()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["close"])
    out = out.drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)
    return out

def read_zip_of_dbfs(zip_bytes: bytes):
    """
    Reads a ZIP (bytes) of DBFs.
    Returns:
      triplets: List[(date: pd.Timestamp, df: DataFrame(code,close), member_name: str)]
      skipped:  List[{member: str, reason: str}]
    """
    results = []
    skipped = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        members = [
            m for m in z.namelist()
            if m.lower().endswith(".dbf")
            and "__MACOSX/" not in m
            and not Path(m).name.startswith("._")
        ]
        if not members:
            skipped.append({"member": "<zip>", "reason": "No .dbf files found in ZIP."})
            return results, skipped

        for m in sorted(members):
            try:
                date = parse_cp_date_from_name(m)
            except Exception as e:
                skipped.append({"member": m, "reason": str(e)})
                continue
            try:
                with z.open(m) as f:
                    dbf_bytes = f.read()
                df = read_dbf_bytes(dbf_bytes)
                results.append((date, df, m))
            except Exception as e:
                skipped.append({"member": m, "reason": f"Failed to read DBF: {e}"})
    return results, skipped

def build_price_matrix(pairs: List[Tuple[pd.Timestamp, pd.DataFrame]]) -> pd.DataFrame:
    """
    Given list of (date, df[code, close]), pivot to wide:
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
    prices = prices.apply(pd.to_numeric, errors="coerce")
    return prices

def to_json_nested_by_date(prices: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    nested = {}
    for d, row in prices.iterrows():
        nested[str(pd.to_datetime(d).date())] = {
            c: float(v) for c, v in row.dropna().to_dict().items()
        }
    return nested

def to_json_records(prices: pd.DataFrame) -> List[Dict[str, object]]:
    records = []
    for d, row in prices.iterrows():
        date_str = str(pd.to_datetime(d).date())
        for code, val in row.dropna().items():
            records.append({"date": date_str, "code": code, "close": float(val)})
    return records

def to_json_wide(prices: pd.DataFrame) -> Dict[str, object]:
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

def ingest_uploaded_files(uploaded_files) -> Tuple[List[Tuple[pd.Timestamp, pd.DataFrame]], List[Dict[str, str]]]:
    """
    Accepts a list of UploadedFile objects (.dbf or .zip).
    Returns:
      pairs:   List[(date, df(code,close))]
      skipped: List[{member: str, reason: str}]
    """
    pairs = []
    skipped = []
    for uf in uploaded_files:
        name = uf.name
        try:
            if name.lower().endswith(".zip"):
                triplets, sk = read_zip_of_dbfs(uf.read())
                pairs.extend([(d, df) for (d, df, _m) in triplets])
                skipped.extend(sk)
            elif name.lower().endswith(".dbf"):
                # single DBF (one day)
                try:
                    date = parse_cp_date_from_name(name)
                except Exception as e:
                    skipped.append({"member": name, "reason": str(e)})
                    continue
                df = read_dbf_bytes(uf.read())
                pairs.append((date, df))
            else:
                skipped.append({"member": name, "reason": "Unsupported file type (use .dbf or .zip)."})
        except Exception as e:
            skipped.append({"member": name, "reason": f"Failed to process: {e}"})
    return pairs, skipped

# ----------------- UI -----------------
colL, colR = st.columns([2, 1])
with colL:
    uploaded_files = st.file_uploader(
        "Upload one or more files (.zip and/or .dbf)",
        type=["zip", "dbf"],
        accept_multiple_files=True
    )
with colR:
    layout = st.selectbox(
        "JSON layout",
        ["Nested by date (default)", "Row records (long)", "Wide table with meta"],
        index=0
    )
    indent = st.number_input("JSON indent (pretty print)", min_value=0, max_value=8, value=2, step=1)
    drop_empty_cols = st.checkbox("Drop columns that are entirely empty", value=True)

st.divider()

if not uploaded_files:
    st.info("Upload **one or more** `.dbf`/`.zip` files (e.g., `CP20250925.dbf` or `CP_September.zip`).")
    st.stop()

# ----------------- Processing -----------------
try:
    pairs, skipped = ingest_uploaded_files(uploaded_files)
    if not pairs and skipped:
        st.error("No valid DBFs found in the uploaded files.")
        with st.expander("Skipped files / reasons"):
            st.write(pd.DataFrame(skipped))
        st.stop()

    # Build price matrix
    prices = build_price_matrix(pairs)
    if prices.empty:
        st.error("No data parsed from the provided file(s).")
        if skipped:
            with st.expander("Skipped files / reasons"):
                st.write(pd.DataFrame(skipped))
        st.stop()

    # Optionally drop empty columns
    if drop_empty_cols:
        prices = prices.dropna(axis=1, how="all")

    # Summary
    n_days = prices.shape[0]
    n_codes = prices.shape[1]
    date_min = str(prices.index.min().date())
    date_max = str(prices.index.max().date())

    m1, m2, m3 = st.columns(3)
    m1.metric("Trading days", f"{n_days}")
    m2.metric("Unique codes", f"{n_codes}")
    m3.metric("Date range", f"{date_min} → {date_max}")

    with st.expander("Files processed (by date)"):
        st.write(pd.DataFrame(
            [{"date": str(d.date()), "rows": len(df)} for (d, df) in pairs]
        ).sort_values("date"))

    if skipped:
        with st.expander("Skipped files / reasons"):
            st.write(pd.DataFrame(skipped))

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

    st.success("JSON is ready.")
except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
