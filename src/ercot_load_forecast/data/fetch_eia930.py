from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential


# EIA API v2 route for hourly demand / forecast / interchange data (EIA-930)
# Docs: requests are made to https://api.eia.gov/v2/<route>/data with facets + frequency
# See EIA API v2 documentation for request structure.  :contentReference[oaicite:5]{index=5}
EIA_V2_BASE = "https://api.eia.gov/v2"
ROUTE = "electricity/rto/region-data"


@dataclass(frozen=True)
class EIAQuery:
    respondent: str = "ERCO"  # ERCOT balancing authority code in EIA-930
    frequency: str = "hourly"
    # 'type' values are dataset-specific; we will fetch all types for ERCO and filter later.
    start: Optional[str] = "2015-01-01"
    end: Optional[str] = None  # None = up to latest
    page_size: int = 5000


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=20))
def _get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_region_data(q: EIAQuery, api_key: str) -> pd.DataFrame:
    """
    Fetch all hourly rows for a balancing authority respondent (ERCO) from EIA API v2.
    We page through results using offset/length.
    """
    url = f"{EIA_V2_BASE}/{ROUTE}/data/"

    offset = 0
    out: List[pd.DataFrame] = []

    while True:
        params: Dict[str, Any] = {
            "api_key": api_key,
            "frequency": q.frequency,
            "data[0]": "value",
            "facets[respondent][0]": q.respondent,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": q.page_size,
        }
        if q.start:
            params["start"] = q.start
        if q.end:
            params["end"] = q.end

        payload = _get_json(url, params=params)
        data = payload.get("response", {}).get("data", [])
        if not data:
            break

        df = pd.DataFrame(data)
        out.append(df)

        # If returned fewer than page_size rows, we're done.
        if len(df) < q.page_size:
            break

        offset += q.page_size

    if not out:
        raise RuntimeError(
            "No data returned from EIA API. Check API key and query facets."
        )

    return pd.concat(out, ignore_index=True)


def clean_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names and parse timestamps.
    EIA API returns many fields including 'period', 'respondent', 'type', etc.
    """
    df = df.copy()

    # Parse period; EIA often uses ISO-like strings.
    df["timestamp"] = pd.to_datetime(df["period"], errors="coerce")

    # Values may arrive as strings (API standardized as strings in some routes). :contentReference[oaicite:6]{index=6}
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Basic sanity
    df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp")

    # Keep a tidy subset, but preserve type/type-name if present
    keep = [
        c
        for c in [
            "timestamp",
            "respondent",
            "respondent-name",
            "type",
            "type-name",
            "timezone",
            "value",
        ]
        if c in df.columns
    ]
    return df[keep]


def main() -> None:
    load_dotenv()
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing EIA_API_KEY. Put it in a .env file or your shell env."
        )

    q = EIAQuery(respondent="ERCO", start="2015-01-01")
    raw = fetch_region_data(q, api_key=api_key)
    tidy = clean_and_standardize(raw)

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    raw_path = "data/raw/eia930_erco_raw.parquet"
    tidy_path = "data/processed/eia930_erco_tidy.parquet"

    raw.to_parquet(raw_path, index=False)
    tidy.to_parquet(tidy_path, index=False)

    print(f"Wrote:\n  {raw_path}\n  {tidy_path}")
    print("\nSample:")
    print(tidy.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
