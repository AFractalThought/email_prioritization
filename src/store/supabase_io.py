import os
from pathlib import Path
from typing import Optional

import pandas as pd
from supabase import create_client


def make_supabase_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_ANON_KEY"]  
    return create_client(url, key)

def fetch_df(table_name: str, limit: int = 10000) -> pd.DataFrame:
    supabase = make_supabase_client()
    resp = supabase.table(table_name).select("*").limit(limit).execute()
    return pd.DataFrame(resp.data)

def upload_artifact(local_path: str | Path, bucket: str, object_path: str, upsert: bool = True) -> None:
    supabase = make_supabase_client()
    p = Path(local_path)

    with p.open("rb") as f:
        supabase.storage.from_(bucket).upload(
            path=object_path,
            file=f,
            file_options={"upsert": "true" if upsert else "false", "cache-control": "3600"},
        )

def download_artifact(
    bucket: str,
    object_path: str,
    local_path: str | Path,
    force: bool = False,
) -> Path:
    """
    Download an object from Supabase Storage to a local file.

    - bucket: storage bucket name
    - object_path: remote key inside bucket (e.g. "resend/models/model.joblib")
    - local_path: where to save locally
    - force: if True, re-download even if local file exists
    """
    supabase = make_supabase_client()
    lp = Path(local_path)
    lp.parent.mkdir(parents=True, exist_ok=True)

    if lp.exists() and not force:
        return lp

    data = supabase.storage.from_(bucket).download(object_path)  # bytes
    lp.write_bytes(data)
    return lp