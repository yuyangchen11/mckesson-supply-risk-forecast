#!/usr/bin/env python3
"""
UC2 Drug Shortage - Full 3-Year Training Script (No EDA)

This script extracts the essential train/validate/test workflow from
CapstoneUC2_Source_Code.ipynb and runs it as a standalone Python pipeline.

Default behavior:
- Uses all available years in train_P2.
- Builds item-week features and 4-week-ahead label (label_4w).
- Runs temporal split (train/val/test).
- Trains available tree models, tunes threshold on validation (F1), and evaluates on test.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import time
import warnings
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except Exception:
    optuna = None
    TPESampler = None
    OPTUNA_AVAILABLE = False

try:
    from pytrends.request import TrendReq
    from pytrends.exceptions import TooManyRequestsError

    PYTRENDS_AVAILABLE = True
except Exception:
    TrendReq = None
    TooManyRequestsError = Exception
    PYTRENDS_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except Exception:
    xgb = None
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except Exception:
    lgb = None
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except Exception:
    CatBoostClassifier = None
    CATBOOST_AVAILABLE = False


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("uc2_train")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - [%(levelname)s] - %(message)s",
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def safe_usecols(path: Path, desired_cols: List[str]) -> List[str]:
    try:
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        return [c for c in desired_cols if c in cols]
    except Exception:
        return desired_cols


def normalize_item_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"^I", "", regex=True).str.strip(),
        errors="coerce",
    ).astype("Int64")


def normalize_din(series: pd.Series, width: int = 8) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype("Int64")
    return x.astype("string").str.zfill(width)


def normalize_molecule(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.upper()
    return s.replace({"": pd.NA, "NAN": pd.NA, "<NA>": pd.NA, "NONE": pd.NA})


def yearweek_to_monday(year_week: pd.Series) -> pd.Series:
    yw = pd.to_numeric(year_week, errors="coerce").astype("Int64")
    year = (yw // 100).astype("Int64")
    week = (yw % 100).astype("Int64")

    def _iso_monday(y, w):
        if pd.isna(y) or pd.isna(w):
            return pd.NaT
        try:
            return datetime.fromisocalendar(int(y), int(w), 1).date()
        except Exception:
            return pd.NaT

    return pd.to_datetime([_iso_monday(y, w) for y, w in zip(year, week)])


def to_monday(dt: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt, errors="coerce")
    return dt - pd.to_timedelta(dt.dt.weekday, unit="D")


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = safe_num(num) / safe_num(den)
    return out.replace([np.inf, -np.inf], np.nan)


def mode_or_nan(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    m = s.mode()
    return m.iloc[0] if len(m) > 0 else s.iloc[0]


def collapse_molecule_string(mol_str: str):
    if not isinstance(mol_str, str) or mol_str.strip() == "":
        return pd.NA
    tokens = [t.strip() for t in mol_str.split("|") if t.strip()]
    if len(tokens) <= 1:
        return tokens[0] if tokens else pd.NA
    joined = " ".join(tokens).upper()
    if any(x in joined for x in ["VITAMIN", "BIOTIN", "FOLIC", "FOLATE"]):
        if any(x in joined for x in ["IRON", "CALCIUM", "ZINC", "MAGNESIUM", "MINERAL"]):
            return "MULTIVIT_MINERALS"
        return "MULTIVITAMIN"
    if any(x in joined for x in ["SODIUM", "POTASSIUM", "CHLORIDE"]):
        return "ELECTROLYTE_COMBINATION"
    return "COMBINATION_PRODUCT"


class TextNormalizer:
    NOISE_WORDS = {
        "BAG",
        "VIAL",
        "KIT",
        "BOX",
        "BTL",
        "UNT",
        "IU",
        "PCT",
        "MEQ",
        "VOL",
        "ML",
        "MG",
        "G",
        "L",
        "MCG",
        "KG",
        "TAB",
        "CAP",
        "SOL",
        "SOLUTION",
        "INJ",
        "INJECTION",
        "CREAM",
        "OINTMENT",
        "SUSP",
        "SYRUP",
        "ELIXIR",
        "SUPP",
        "PATCH",
        "AEROSOL",
        "DROPS",
        "USP",
        "STERILE",
        "PRESERVATIVE-FREE",
        "CONCENTRATE",
        "IRRIGATION",
    }
    ABBREVIATIONS = {
        "SOD": "SODIUM",
        "CHLOR": "CHLORIDE",
        "CALC": "CALCIUM",
        "POT": "POTASSIUM",
        "MAG": "MAGNESIUM",
        "DEXT": "DEXTROSE",
        "BICARB": "BICARBONATE",
        "ACET": "ACETAMINOPHEN",
        "IBUP": "IBUPROFEN",
        "SALB": "SALBUTAMOL",
        "VIT": "VITAMIN",
        "MULTI": "MULTIVITAMIN",
    }
    BRAND_MAP = {
        "TYLENOL": "ACETAMINOPHEN",
        "ADVIL": "IBUPROFEN",
        "MOTRIN": "IBUPROFEN",
        "VOLTAREN": "DICLOFENAC",
        "ASPIRIN": "ACETYLSALICYLIC ACID",
        "ANACIN": "ACETYLSALICYLIC ACID|CAFFEINE",
        "PAXLOVID": "NIRMATRELVIR|RITONAVIR",
    }
    SUPPLY_KEYWORDS = {
        "SYR",
        "SYRINGE",
        "NEEDLE",
        "LANCET",
        "STRIP",
        "TEST",
        "METER",
        "MONITOR",
        "WIPE",
        "SWAB",
        "GAUZE",
        "BANDAGE",
        "DEVICE",
        "SPACER",
        "AEROCHAMBER",
    }

    @staticmethod
    def parse(description: str) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(description, str) or not description:
            return np.nan, None

        desc = description.upper()
        for kw in TextNormalizer.SUPPLY_KEYWORDS:
            if kw in desc:
                return "SUPPLY", "V"
        for brand, generic in TextNormalizer.BRAND_MAP.items():
            if brand in desc:
                return generic, None

        desc = re.sub(r"\s(W/|WITH)\s", "|", desc)
        desc = re.sub(r"[+/&]", "|", desc)
        raw_tokens = desc.split("|")
        clean_tokens = []

        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue
            words = []
            for part in token.split():
                w = TextNormalizer.ABBREVIATIONS.get(part.strip(), part.strip())
                if w in TextNormalizer.NOISE_WORDS:
                    continue
                if w.replace(".", "", 1).isdigit():
                    continue
                words.append(w)
            if words:
                clean_tokens.append(" ".join(words))

        if not clean_tokens:
            return np.nan, None
        return "|".join(sorted(set(clean_tokens))), None


class SmartMatcher:
    def __init__(self, known_molecules: List[str], similarity_threshold: float = 0.55):
        self.threshold = similarity_threshold
        self.known_molecules = [m for m in known_molecules if isinstance(m, str) and len(m) > 0]
        if self.known_molecules:
            self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            self.molecule_vectors = self.vectorizer.fit_transform(self.known_molecules)
        else:
            self.vectorizer = None
            self.molecule_vectors = None

    def match_batch(self, series: pd.Series, batch_size: int = 5000) -> pd.Series:
        if self.vectorizer is None or series is None or len(series) == 0:
            return pd.Series([np.nan] * (0 if series is None else len(series)), index=None if series is None else series.index)

        results = []
        n_batches = (len(series) + batch_size - 1) // batch_size
        for i in range(n_batches):
            s = i * batch_size
            e = min((i + 1) * batch_size, len(series))
            batch = series.iloc[s:e]
            X = self.vectorizer.transform(batch.fillna("").astype(str))
            sim = cosine_similarity(X, self.molecule_vectors)
            out = []
            for row in sim:
                idx = int(row.argmax())
                out.append(self.known_molecules[idx] if row[idx] >= self.threshold else np.nan)
            results.extend(out)
        return pd.Series(results, index=series.index)


# -----------------------------------------------------------------------------
# Specs
# -----------------------------------------------------------------------------
ACTIVE_ITEMS_COLS = [
    "DIN_NUM",
    "ITEM_NUM",
    "VENDOR_NUM",
    "MOLECULE_NM",
    "ITEM_EN_DESC",
    "SHORT_EN_DESC",
    "EN_FORM",
    "THRP_CL_CD",
    "ITEM_CL_GRP_EN_SHORT_DESC",
]

PURCHASE_ORDERS_COLS = [
    "PO_DT",
    "ITEM_NUM",
    "PO_NUM",
    "VENDOR_NUM",
    "DC_CD",
    "PO_QTY_ORD",
    "PO_QTY_RCV",
]

RECEPTIONS_COLS = [
    "RCV_DT",
    "DC_CD",
    "RCV_NUM",
    "ITEM_NUM",
    "VENDOR_NUM",
    "PO_NUM",
    "RCV_QTY",
]

LEAD_TIMES_COLS = [
    "DC",
    "U_Supplier_Number",
    "LoadTime",
    "TransLeadTime",
    "UnloadTime",
]

TRAIN_P2_BASE_COLS = [
    "ITEM_NUM",
    "MOLECULE_NM_LOWER",
    "DIN_NUM",
    "YEAR_WEEK",
    "QTY_ORD",
    "SALES_DC_COUNT",
    "SALES_DATES_COUNT",
    "QTY_DELV",
    "QTY_MCS",
    "QTY_MCK",
    "QTY_NOT_DELV",
    "QTY_ISSUE_OTHER",
    "PO_QTY_ORD_sum",
    "PO_DT_nunique",
    "DC_CD_nunique",
    "PO_NUM_nunique",
    "RCV_NUM_nunique",
    "RCV_DT_ADJUSTED_nunique",
    "RCV_QTY_ADJUSTED_sum",
    "VENDOR_SHIP_DUE_DT_nunique",
    "VENDOR_SHIP_DUE_DT_minus_EXP_DUE_DT_FROM_LEAD_TIME",
    "DIF_RCV_DATE_ORD_DATE_MINUS_LEADTIME",
    "DIF_PO_ORD_DATE_VENDOR_SHIP_DUE_DATE",
    "DIF_RCV_DATE_VENDOR_SHIP_DUE_DATE",
    "DIF_RCV_DATE_ORD_DATE",
    "WEEKS_SINCE_LAST_RECEIVE_EXP",
    "SHORT_QTY_WEEKLY",
    "SHORT_2WEEKBUFFER_PERC",
    "SHORT_2WEEKBUFFER_BINARY",
    "QTY_DELV_DIVIDEDBY_QTY_ORD",
    "QTY_MCS_DIVIDEDBY_QTY_ORD",
    "QTY_MCK_DIVIDEDBY_QTY_ORD",
    "QTY_NOT_DELV_DIVIDEDBY_QTY_ORD",
    "QTY_ISSUE_OTHER_DIVIDEDBY_QTY_ORD",
    "RCV_QTY_ADJUSTED_sum_DIVIDEDBY_PO_QTY_ORD_sum",
    "SHORT_QTY_WEEKLY_DIVIDEDBY_PO_QTY_ORD_sum",
    "DIF_PO_ORD_DATE_VENDOR_SHIP_DUE_DATE_DIVIDEDBY_DIF_RCV_DATE_ORD_DATE",
    "DIF_RCV_DATE_VENDOR_SHIP_DUE_DATE_DIVIDEDBY_DIF_RCV_DATE_ORD_DATE",
    "VENDOR_SHIP_DUE_DT_minus_EXP_DUE_DT_FROM_LEAD_TIME_DIVIDEDBY_DIF_RCV_DATE_ORD_DATE_MINUS_LEADTIME",
    "RCV_QTY_ADJUSTED_sum_DIVIDEDBY_QTY_ORD",
    "PO_QTY_ORD_sum_DIVIDEDBY_QTY_ORD",
    "QTY_MCS_DIVIDEDBY_QTY_NOT_DELV",
    "SHORT_2WEEKBUFFER_PERC_CLASS",
    "SHORT_2WEEKBUFFER_PERC_CLASS_in6_weeks",
    "SHORT_2WEEKBUFFER_BINARY_in6_weeks",
    "OMIT_BINARY_in6_weeks",
]

WEEKLY_SALES_COLS = [
    "CAL_INVC_DT",
    "WEEK_ENDING_DT",
    "ITEM_NUM",
    "QTY_ORD",
    "QTY_DELV",
    "QTY_MCS",
    "QTY_MCK",
]


# -----------------------------------------------------------------------------
# Data Builder (Sections 2-3 core)
# -----------------------------------------------------------------------------
class UC2DataBuilder:
    def __init__(
        self,
        data_dir: Path,
        p2_chunksize: int = 300_000,
        ws_chunksize: int = 300_000,
        year_filter: Optional[List[int]] = None,
        start_yearweek: Optional[int] = None,
        end_yearweek: Optional[int] = None,
        refresh_dpd: bool = False,
        refresh_google_trends: bool = False,
        refresh_cpi: bool = False,
        refresh_flu: bool = False,
    ):
        self.data_dir = data_dir
        self.p2_chunksize = min(300_000, int(p2_chunksize))
        self.ws_chunksize = min(300_000, int(ws_chunksize))
        self.year_filter = year_filter
        self.start_yearweek = start_yearweek
        self.end_yearweek = end_yearweek
        self.refresh_dpd = refresh_dpd
        self.refresh_google_trends = refresh_google_trends
        self.refresh_cpi = refresh_cpi
        self.refresh_flu = refresh_flu

    def _path(self, filename: str) -> Path:
        return self.data_dir / filename

    def fetch_canada_dpd(self) -> None:
        dpd_page_url = (
            "https://www.canada.ca/en/health-canada/services/drugs-health-products/"
            "drug-products/drug-product-database/what-data-extract-drug-product-database.html"
        )
        zip_targets = {
            "allfiles.zip": ["drug.txt", "ingred.txt", "ther.txt", "status.txt", "comp.txt"],
            "allfiles_ia.zip": ["drug_ia.txt", "ingred_ia.txt", "ther_ia.txt", "comp_ia.txt", "status_ia.txt"],
        }

        logger.info("Fetching DPD page to resolve ZIP links")
        r = requests.get(dpd_page_url, timeout=120)
        r.raise_for_status()
        hrefs = re.findall(r'href=[\"\\\']([^\"\\\']+)[\"\\\']', r.text, flags=re.IGNORECASE)

        zip_links = {}
        for zip_name in zip_targets:
            link = None
            for h in hrefs:
                if h.lower().endswith(zip_name.lower()):
                    link = h
                    break
            if link is None:
                raise RuntimeError(f"Could not find {zip_name} link on DPD page.")
            zip_links[zip_name] = urljoin(dpd_page_url, link)

        for zip_name, zip_url in zip_links.items():
            logger.info("Downloading DPD zip: %s", zip_name)
            resp = requests.get(zip_url, timeout=300)
            resp.raise_for_status()
            zip_path = self._path(zip_name)
            zip_path.write_bytes(resp.content)

            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                names_lower = [n.lower() for n in names]
                for target_file in zip_targets[zip_name]:
                    idx = next((i for i, n in enumerate(names_lower) if n.endswith(target_file.lower())), None)
                    if idx is None:
                        logger.warning("DPD ZIP %s missing expected member %s", zip_name, target_file)
                        continue
                    out = self._path(target_file)
                    with zf.open(names[idx]) as src, open(out, "wb") as dst:
                        dst.write(src.read())
                    logger.info("Extracted DPD file: %s", out)

    def ensure_dpd_files(self) -> None:
        required = [
            "drug.txt",
            "ingred.txt",
            "ther.txt",
            "status.txt",
            "comp.txt",
            "drug_ia.txt",
            "ingred_ia.txt",
            "ther_ia.txt",
            "comp_ia.txt",
            "status_ia.txt",
        ]
        missing = [f for f in required if not self._path(f).exists()]
        if self.refresh_dpd or missing:
            logger.info("Refreshing DPD files (missing=%s, forced=%s)", len(missing), self.refresh_dpd)
            self.fetch_canada_dpd()

    def fetch_cpi_rx_otc_pharma(self) -> None:
        cpi_url = "https://www150.statcan.gc.ca/n1/tbl/csv/18100004-eng.zip"
        out_weekly = self._path("CPI_DRUG_WEEKLY.csv")
        zip_path = self._path("18100004-eng.zip")

        if self.refresh_cpi or (not zip_path.exists()):
            logger.info("Downloading CPI ZIP from %s", cpi_url)
            r = requests.get(cpi_url, timeout=120)
            r.raise_for_status()
            zip_path.write_bytes(r.content)
        else:
            logger.info("Using cached CPI ZIP: %s", zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            df = pd.read_csv(zf.open(csv_name), low_memory=False)

        df = df[["REF_DATE", "GEO", "Products and product groups", "UOM", "VALUE"]].copy()
        df = df[df["GEO"].str.contains("Canada", case=False, na=False)]
        df = df[df["UOM"].astype(str).str.contains("2002=100", na=False)]

        rx_name = "Prescribed medicines (excluding medicinal cannabis)"
        otc_name = "Non-prescribed medicines"
        pharma_name = "Medicinal and pharmaceutical products"

        rx = df[df["Products and product groups"] == rx_name][["REF_DATE", "VALUE"]].rename(columns={"VALUE": "CPI_RX"})
        otc = df[df["Products and product groups"] == otc_name][["REF_DATE", "VALUE"]].rename(columns={"VALUE": "CPI_OTC"})
        ph = df[df["Products and product groups"] == pharma_name][["REF_DATE", "VALUE"]].rename(columns={"VALUE": "CPI_PHARMA"})

        rx["date_month"] = pd.to_datetime(rx["REF_DATE"], errors="coerce")
        otc["date_month"] = pd.to_datetime(otc["REF_DATE"], errors="coerce")
        ph["date_month"] = pd.to_datetime(ph["REF_DATE"], errors="coerce")

        m = rx.merge(otc, on="date_month", how="outer").merge(ph, on="date_month", how="outer").sort_values("date_month")
        m["CPI_RX_mom"] = m["CPI_RX"].pct_change()
        m["CPI_OTC_mom"] = m["CPI_OTC"].pct_change()
        m["CPI_PHARMA_mom"] = m["CPI_PHARMA"].pct_change()
        m["CPI_RX_lag1"] = m["CPI_RX"].shift(1)
        m["CPI_OTC_lag1"] = m["CPI_OTC"].shift(1)
        m["CPI_PHARMA_lag1"] = m["CPI_PHARMA"].shift(1)
        m["CPI_RX_mom_lag1"] = m["CPI_RX_mom"].shift(1)
        m["CPI_OTC_mom_lag1"] = m["CPI_OTC_mom"].shift(1)
        m["CPI_PHARMA_mom_lag1"] = m["CPI_PHARMA_mom"].shift(1)

        rows = []
        for _, r in m.iterrows():
            if pd.isna(r["date_month"]):
                continue
            month_start = pd.Timestamp(r["date_month"]).normalize()
            month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
            mondays = pd.date_range(month_start, month_end, freq="W-MON")
            if len(mondays) == 0:
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "date_monday": mondays,
                        "CPI_RX_lag1": r["CPI_RX_lag1"],
                        "CPI_OTC_lag1": r["CPI_OTC_lag1"],
                        "CPI_PHARMA_lag1": r["CPI_PHARMA_lag1"],
                        "CPI_RX_mom_lag1": r["CPI_RX_mom_lag1"],
                        "CPI_OTC_mom_lag1": r["CPI_OTC_mom_lag1"],
                        "CPI_PHARMA_mom_lag1": r["CPI_PHARMA_mom_lag1"],
                    }
                )
            )
        weekly = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        weekly.to_csv(out_weekly, index=False)
        logger.info("Saved CPI weekly to %s", out_weekly)

    def fetch_flunet_canada_weekly(self) -> None:
        output_file = self._path("FLUNET_CANADA_WEEKLY.csv")
        url = "https://xmart-api-public.who.int/FLUMART/VIW_FNT"
        params = {"$format": "csv", "$filter": "ISO2 eq 'CA'"}
        logger.info("Downloading FluNet weekly data")
        r = requests.get(url, params=params, timeout=180)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), low_memory=False)

        num_cols = [
            "SPEC_PROCESSED_NB",
            "SPEC_RECEIVED_NB",
            "INF_ALL",
            "INF_A",
            "INF_B",
            "RSV",
            "RSV_PROCESSED",
            "HUMAN_CORONA",
            "RHINO",
            "PARAINFLUENZA",
            "METAPNEUMO",
            "ADENO",
            "BOCA",
        ]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        df["date_monday"] = pd.to_datetime(df["ISO_WEEKSTARTDATE"], errors="coerce")
        df = df.dropna(subset=["date_monday"]).sort_values("date_monday")

        def safe_ratio(num, den):
            return np.where((den is not None) & (den > 0), num / den, 0.0)

        df["FLU_INF_ALL_POS"] = safe_ratio(df.get("INF_ALL", 0), df.get("SPEC_PROCESSED_NB", 0))
        df["FLU_INF_A_SHARE"] = safe_ratio(df.get("INF_A", 0), df.get("INF_ALL", 0))
        df["FLU_INF_B_SHARE"] = safe_ratio(df.get("INF_B", 0), df.get("INF_ALL", 0))
        df["FLU_RSV_POS"] = safe_ratio(df.get("RSV", 0), df.get("RSV_PROCESSED", 0))

        other_cols = [c for c in ["HUMAN_CORONA", "RHINO", "PARAINFLUENZA", "METAPNEUMO", "ADENO", "BOCA"] if c in df.columns]
        if other_cols:
            df["FLU_OTHER_RESP_SUM"] = df[other_cols].sum(axis=1, skipna=True)
            df["FLU_OTHER_RESP_POS"] = safe_ratio(df["FLU_OTHER_RESP_SUM"], df.get("SPEC_PROCESSED_NB", 0))
        else:
            df["FLU_OTHER_RESP_POS"] = 0.0

        for col in ["FLU_INF_ALL_POS", "FLU_INF_A_SHARE", "FLU_INF_B_SHARE", "FLU_RSV_POS", "FLU_OTHER_RESP_POS"]:
            df[f"{col}_lag1"] = df[col].shift(1)

        out = df[
            [
                "date_monday",
                "FLU_INF_ALL_POS_lag1",
                "FLU_INF_A_SHARE_lag1",
                "FLU_INF_B_SHARE_lag1",
                "FLU_RSV_POS_lag1",
                "FLU_OTHER_RESP_POS_lag1",
            ]
        ].copy()
        out = out.fillna(0.0)
        out.to_csv(output_file, index=False)
        logger.info("Saved FluNet weekly to %s", output_file)

    def fetch_google_trends(self) -> None:
        if not PYTRENDS_AVAILABLE:
            raise RuntimeError(
                "ATC_COMPOSITE_TRENDS.csv is missing and pytrends is not installed. "
                "Install with: pip install pytrends"
            )
        output_file = self._path("ATC_COMPOSITE_TRENDS.csv")
        now = pd.Timestamp.utcnow().tz_localize(None).normalize()

        existing = None
        full_refresh = True
        last_date = None
        if output_file.exists() and not self.refresh_google_trends:
            try:
                existing = pd.read_csv(output_file)
                existing["date"] = pd.to_datetime(existing["date"], utc=True).dt.tz_localize(None)
                last_date = existing["date"].max()
                if pd.notna(last_date):
                    full_refresh = False
            except Exception:
                existing = None
                full_refresh = True

        if full_refresh:
            timeframe = "today 5-y"
        else:
            start_date = (last_date + pd.Timedelta(days=1)).date()
            end_date = now.date()
            if start_date > end_date:
                logger.info("Google Trends already up to date")
                return
            timeframe = f"{start_date} {end_date}"

        atc_keywords = {
            "A": ["Stomach pain", "Ozempic", "Antacid", "Digestive health"],
            "B": ["Blood thinner", "Eliquis", "Xarelto", "Warfarin"],
            "C": ["Blood pressure", "Hypertension", "Beta blocker", "Amlodipine"],
            "D": ["Eczema", "Skin rash", "Cortisone", "Hydrocortisone"],
            "G": ["Birth control", "Plan B", "Menopause", "Hormone replacement"],
            "H": ["Levothyroxine", "Synthroid", "Prednisone", "Steroid"],
            "J": ["Antibiotics", "Amoxicillin", "Azithromycin", "Penicillin"],
            "L": ["Chemotherapy", "Cancer treatment", "Immunotherapy", "Oncology"],
            "M": ["Arthritis", "Back pain", "Muscle pain", "Anti inflammatory"],
            "N": ["Tylenol", "Advil", "Ibuprofen", "Painkiller"],
            "P": ["Lice treatment", "Head lice", "Permethrin", "Worm medicine"],
            "R": ["Flu", "Cold medicine", "Cough", "Inhaler"],
            "S": ["Eye drops", "Pink eye", "Dry eyes", "Ear drops"],
            "V": ["First aid", "Bandage", "Medical supplies", "Syringe"],
        }

        pytrends = TrendReq(hl="en-CA", tz=-300, timeout=(10, 25), retries=0)
        rows = []
        for atc, kws in atc_keywords.items():
            try:
                pytrends.build_payload(kws, timeframe=timeframe, geo="CA")
                df = pytrends.interest_over_time()
                if df.empty:
                    continue
                df = df.drop(columns=["isPartial"], errors="ignore")
                valid_cols = [c for c in df.columns if df[c].std() > 1e-6 and df[c].sum() > 0]
                if not valid_cols:
                    continue
                x = df[valid_cols].astype(float)
                z = (x - x.mean()) / x.std()
                comp = z.mean(axis=1)
                tmp = comp.reset_index()
                tmp.columns = ["date", "COMPOSITE_INDEX"]
                tmp["ATC_LEVEL1"] = atc
                tmp["YEAR_WEEK"] = tmp["date"].dt.strftime("%G%V").astype(int)
                rows.append(tmp)
            except TooManyRequestsError:
                logger.warning("Google Trends rate limited on ATC %s", atc)
            except Exception as e:
                logger.warning("Google Trends failed on ATC %s (%s)", atc, str(e)[:120])

        if not rows and (existing is None or existing.empty):
            raise RuntimeError("Google Trends fetch returned no rows and no existing ATC_COMPOSITE_TRENDS.csv.")

        final = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "COMPOSITE_INDEX", "ATC_LEVEL1", "YEAR_WEEK"])
        if not final.empty:
            final["date"] = pd.to_datetime(final["date"], utc=True).dt.tz_localize(None)

        if existing is not None and not existing.empty and not full_refresh:
            existing["ATC_LEVEL1"] = existing["ATC_LEVEL1"].astype(str).str.upper().str.strip()
            final["ATC_LEVEL1"] = final["ATC_LEVEL1"].astype(str).str.upper().str.strip()
            out = pd.concat([existing, final], ignore_index=True)
            out = out.drop_duplicates(subset=["ATC_LEVEL1", "YEAR_WEEK"], keep="last").sort_values(["ATC_LEVEL1", "YEAR_WEEK"])
        else:
            out = final.sort_values(["ATC_LEVEL1", "YEAR_WEEK"])

        out.to_csv(output_file, index=False)
        logger.info("Saved Google Trends to %s", output_file)

    def load_active_items(self) -> pd.DataFrame:
        path = self._path("activeitems.csv")
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        usecols = safe_usecols(path, ACTIVE_ITEMS_COLS)
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        df["ITEM_NUM_CLEAN"] = normalize_item_num(df["ITEM_NUM"])
        df["DIN_KEY"] = normalize_din(df["DIN_NUM"])
        df["MOLECULE_NM"] = normalize_molecule(df.get("MOLECULE_NM", pd.Series([pd.NA] * len(df))))

        if "THRP_CL_CD" in df.columns:
            df["ATC_LEVEL1"] = df["THRP_CL_CD"].astype("string").str.strip().str.upper().str[:1]
        else:
            df["ATC_LEVEL1"] = "X"
        df["ATC_LEVEL1"] = df["ATC_LEVEL1"].fillna("X")

        cls = (
            df.get("ITEM_CL_GRP_EN_SHORT_DESC", pd.Series(["OTHER"] * len(df)))
            .astype("string")
            .str.upper()
            .fillna("OTHER")
        )
        df["item_class_group"] = np.where(cls.eq("RX"), "RX", np.where(cls.eq("OTC"), "OTC", "OTHER"))
        df["is_rx"] = (df["item_class_group"] == "RX").astype(int)
        df["is_otc"] = (df["item_class_group"] == "OTC").astype(int)
        df["is_other"] = (df["item_class_group"] == "OTHER").astype(int)
        return df

    def load_purchase_orders(self) -> pd.DataFrame:
        path = self._path("promitto_purchase_orders (2).csv")
        if not path.exists():
            logger.warning("Missing purchase orders file: %s", path)
            return pd.DataFrame()
        usecols = safe_usecols(path, PURCHASE_ORDERS_COLS)
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        df["PO_DT"] = pd.to_datetime(df["PO_DT"], errors="coerce")
        df["date_monday"] = to_monday(df["PO_DT"])
        df["ITEM_NUM_CLEAN"] = normalize_item_num(df["ITEM_NUM"])
        df["VENDOR_NUM"] = pd.to_numeric(df["VENDOR_NUM"], errors="coerce").astype("Int64")
        df["PO_QTY_ORD"] = pd.to_numeric(df["PO_QTY_ORD"], errors="coerce").fillna(0)
        df["PO_QTY_RCV"] = pd.to_numeric(df.get("PO_QTY_RCV", 0), errors="coerce").fillna(0)
        return df

    def load_receptions(self) -> pd.DataFrame:
        path = self._path("promitto_receptions (2).csv")
        if not path.exists():
            logger.warning("Missing receptions file: %s", path)
            return pd.DataFrame()
        usecols = safe_usecols(path, RECEPTIONS_COLS)
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        if "RCV_DT" in df.columns and "RCV_DT_ADJUSTED" not in df.columns:
            df = df.rename(columns={"RCV_DT": "RCV_DT_ADJUSTED"})
        if "RCV_QTY" in df.columns and "RCV_QTY_ADJUSTED" not in df.columns:
            df = df.rename(columns={"RCV_QTY": "RCV_QTY_ADJUSTED"})
        if "RCV_DT_ADJUSTED" not in df.columns:
            return pd.DataFrame()
        df["RCV_DT_ADJUSTED"] = pd.to_datetime(df["RCV_DT_ADJUSTED"], errors="coerce")
        df["date_monday"] = to_monday(df["RCV_DT_ADJUSTED"])
        df["ITEM_NUM_CLEAN"] = normalize_item_num(df["ITEM_NUM"])
        df["VENDOR_NUM"] = pd.to_numeric(df["VENDOR_NUM"], errors="coerce").astype("Int64")
        df["RCV_QTY_ADJUSTED"] = pd.to_numeric(df.get("RCV_QTY_ADJUSTED", 0), errors="coerce").fillna(0)
        return df

    def load_lead_times(self) -> pd.DataFrame:
        path = self._path("network lead times.csv")
        if not path.exists():
            logger.warning("Missing lead times file: %s", path)
            return pd.DataFrame()
        usecols = safe_usecols(path, LEAD_TIMES_COLS)
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        df["VENDOR_NUM"] = pd.to_numeric(df["U_Supplier_Number"], errors="coerce").astype("Int64")
        df["LoadTime"] = pd.to_numeric(df.get("LoadTime", 0), errors="coerce").fillna(0)
        df["TransLeadTime"] = pd.to_numeric(df.get("TransLeadTime", 0), errors="coerce").fillna(0)
        df["UnloadTime"] = pd.to_numeric(df.get("UnloadTime", 0), errors="coerce").fillna(0)
        df["TotalLeadTime"] = df["LoadTime"] + df["TransLeadTime"] + df["UnloadTime"]
        return df

    def load_dc_info(self) -> pd.DataFrame:
        path = self._path("promitto active dcs.csv")
        if not path.exists():
            logger.warning("Missing DC info file: %s", path)
            return pd.DataFrame()
        return pd.read_csv(path, low_memory=False)

    def load_trends(self) -> pd.DataFrame:
        path = self._path("ATC_COMPOSITE_TRENDS.csv")
        if self.refresh_google_trends or (not path.exists()):
            logger.info("Refreshing Google Trends (missing or forced refresh).")
            self.fetch_google_trends()
        if not path.exists():
            raise FileNotFoundError(f"ATC_COMPOSITE_TRENDS.csv still missing after fetch: {path}")
        df = pd.read_csv(path, low_memory=False)
        if "ATC_LEVEL1" not in df.columns:
            return pd.DataFrame()
        if "YEAR_WEEK" in df.columns:
            df["date_monday"] = yearweek_to_monday(df["YEAR_WEEK"])
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["date_monday"] = to_monday(df["date"])
        else:
            return pd.DataFrame()
        df["ATC_LEVEL1"] = df["ATC_LEVEL1"].astype(str).str.strip().str.upper()
        df["COMPOSITE_INDEX"] = pd.to_numeric(df.get("COMPOSITE_INDEX", np.nan), errors="coerce")
        return df

    def load_cpi_weekly(self) -> pd.DataFrame:
        path = self._path("CPI_DRUG_WEEKLY.csv")
        if self.refresh_cpi or (not path.exists()):
            logger.info("Refreshing CPI weekly (missing or forced refresh).")
            self.fetch_cpi_rx_otc_pharma()
        if not path.exists():
            raise FileNotFoundError(f"CPI_DRUG_WEEKLY.csv still missing after fetch: {path}")
        df = pd.read_csv(path, low_memory=False)
        if "date_monday" in df.columns:
            df["date_monday"] = pd.to_datetime(df["date_monday"], errors="coerce")
        return df

    def load_flunet_weekly(self) -> pd.DataFrame:
        path = self._path("FLUNET_CANADA_WEEKLY.csv")
        if self.refresh_flu or (not path.exists()):
            logger.info("Refreshing FluNet weekly (missing or forced refresh).")
            self.fetch_flunet_canada_weekly()
        if not path.exists():
            raise FileNotFoundError(f"FLUNET_CANADA_WEEKLY.csv still missing after fetch: {path}")
        df = pd.read_csv(path, low_memory=False)
        if "date_monday" in df.columns:
            df["date_monday"] = pd.to_datetime(df["date_monday"], errors="coerce")
        return df

    def build_train_p2_item_maps(self, chunksize: int = 1_000_000) -> Tuple[Dict[int, int], pd.DataFrame]:
        path = self._path("train_P2_inbound_allitems_3years.csv")
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        usecols = safe_usecols(path, ["ITEM_NUM", "DIN_NUM", "MOLECULE_NM_LOWER"])
        item_din_map: Dict[int, int] = {}
        conflicts = set()
        main_name_counter_by_item: Dict[int, Counter] = {}

        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            item = normalize_item_num(chunk["ITEM_NUM"])
            din = pd.to_numeric(chunk["DIN_NUM"], errors="coerce")
            main = normalize_molecule(chunk.get("MOLECULE_NM_LOWER", pd.Series([pd.NA] * len(chunk))))

            for it, di, mn in zip(item, din, main):
                if pd.isna(it):
                    continue
                it = int(it)

                if not pd.isna(di):
                    di = int(di)
                    if it not in item_din_map:
                        item_din_map[it] = di
                    elif item_din_map[it] != di:
                        conflicts.add(it)

                if isinstance(mn, str) and mn.strip():
                    if it not in main_name_counter_by_item:
                        main_name_counter_by_item[it] = Counter()
                    main_name_counter_by_item[it][mn] += 1

        for it in conflicts:
            item_din_map.pop(it, None)

        rows = []
        for it, cnt in main_name_counter_by_item.items():
            best_name, best_freq = cnt.most_common(1)[0]
            rows.append({"ITEM_NUM_CLEAN": it, "MOLECULE_MAIN_NAME": best_name, "MAIN_NAME_FREQ": best_freq})

        df_item_main = pd.DataFrame(rows)
        if not df_item_main.empty:
            main_item_count = (
                df_item_main.groupby("MOLECULE_MAIN_NAME")["ITEM_NUM_CLEAN"]
                .nunique()
                .reset_index(name="MAIN_NAME_ITEM_COUNT")
            )
            df_item_main = df_item_main.merge(main_item_count, on="MOLECULE_MAIN_NAME", how="left")
        else:
            df_item_main = pd.DataFrame(columns=["ITEM_NUM_CLEAN", "MOLECULE_MAIN_NAME", "MAIN_NAME_ITEM_COUNT"])

        return item_din_map, df_item_main

    def build_item_din_map_from_activeitems(self, df_raw: pd.DataFrame) -> Dict[int, int]:
        item_map, conflicts = {}, set()
        tmp = df_raw[["ITEM_NUM_CLEAN", "DIN_NUM"]].copy()
        tmp["ITEM_NUM_CLEAN"] = pd.to_numeric(tmp["ITEM_NUM_CLEAN"], errors="coerce")
        tmp["DIN_NUM"] = pd.to_numeric(tmp["DIN_NUM"], errors="coerce")
        tmp = tmp.dropna(subset=["ITEM_NUM_CLEAN", "DIN_NUM"])
        for it, di in zip(tmp["ITEM_NUM_CLEAN"], tmp["DIN_NUM"]):
            it, di = int(it), int(di)
            if it not in item_map:
                item_map[it] = di
            elif item_map[it] != di:
                conflicts.add(it)
        for it in conflicts:
            item_map.pop(it, None)
        return item_map

    def process_dpd_data(self) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
        cols_drug = [
            "DRUG_CODE",
            "PROD_CATEG",
            "CLASS",
            "DIN",
            "BRAND_NAME",
            "DESCRIPTOR",
            "PEDIATRIC_FLAG",
            "ACCESSION_NUMBER",
            "NUMBER_OF_AIS",
            "LAST_UPDATE_DATE",
            "AI_GROUP_NO",
            "CLASS_F",
            "BRAND_NAME_F",
            "DESCRIPTOR_F",
        ]
        cols_ther = [
            "DRUG_CODE",
            "TC_ATC_NUMBER",
            "TC_ATC",
            "TC_ATC_F",
            "TC_AHFS_NUMBER",
            "TC_AHFS",
            "TC_AHFS_F",
        ]
        cols_ingred = [
            "DRUG_CODE",
            "ACTIVE_INGREDIENT_CODE",
            "INGREDIENT",
            "INGREDIENT_SUPPLIED_IND",
            "STRENGTH",
            "STRENGTH_UNIT",
            "STRENGTH_TYPE",
            "DOSAGE_VALUE",
            "BASE",
            "YESNO",
            "NOTES",
            "INGREDIENT_F",
            "STRENGTH_UNIT_F",
            "STRENGTH_TYPE_F",
            "DOSAGE_VALUE_F",
        ]

        def ingest(f1, f2, cols):
            dfs = []
            for f in [f1, f2]:
                p = self._path(f)
                if p.exists():
                    dfs.append(pd.read_csv(p, names=cols, header=None, quotechar='"', encoding="latin1"))
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        df_drug = ingest("drug.txt", "drug_ia.txt", cols_drug)
        df_ther = ingest("ther.txt", "ther_ia.txt", cols_ther)
        df_ingred = ingest("ingred.txt", "ingred_ia.txt", cols_ingred)
        if df_drug.empty:
            return pd.DataFrame(), {}, []

        for d in [df_drug, df_ther, df_ingred]:
            if not d.empty and "DRUG_CODE" in d.columns:
                d["DRUG_CODE"] = pd.to_numeric(d["DRUG_CODE"], errors="coerce").fillna(0).astype(int)

        df_drug["DIN_KEY"] = normalize_din(df_drug["DIN"])
        base = df_drug[["DRUG_CODE", "DIN_KEY"]].copy()

        if not df_ingred.empty and "INGREDIENT" in df_ingred.columns:
            df_ingred["INGREDIENT"] = normalize_molecule(df_ingred["INGREDIENT"])
            mol_map = (
                df_ingred.dropna(subset=["INGREDIENT"])
                .groupby("DRUG_CODE")["INGREDIENT"]
                .apply(lambda x: "|".join(sorted(set(x.astype(str)))))
                .reset_index(name="DPD_MOLECULE")
            )
            base = base.merge(mol_map, on="DRUG_CODE", how="left")
        else:
            base["DPD_MOLECULE"] = pd.NA

        if not df_ther.empty and "TC_ATC_NUMBER" in df_ther.columns:
            df_ther["TC_ATC_NUMBER"] = df_ther["TC_ATC_NUMBER"].astype(str).str.strip().str.upper()
            tmap = df_ther.groupby("DRUG_CODE", as_index=False)[["TC_ATC_NUMBER"]].first()
            tmap["DPD_ATC_L1"] = tmap["TC_ATC_NUMBER"].str[:1]
            base = base.merge(tmap[["DRUG_CODE", "DPD_ATC_L1"]], on="DRUG_CODE", how="left")
        else:
            base["DPD_ATC_L1"] = pd.NA

        df_dpd_din = (
            base.groupby("DIN_KEY", as_index=False).agg({"DPD_MOLECULE": mode_or_nan, "DPD_ATC_L1": mode_or_nan})
        )
        tmp = df_dpd_din.copy()
        tmp["DPD_MOLECULE_COLLAPSED"] = tmp["DPD_MOLECULE"].apply(collapse_molecule_string)
        kb_df = tmp.dropna(subset=["DPD_MOLECULE_COLLAPSED", "DPD_ATC_L1"]).copy()
        knowledge_base = (
            kb_df.groupby("DPD_MOLECULE_COLLAPSED")["DPD_ATC_L1"]
            .agg(lambda x: x.mode().iat[0] if len(x.mode()) > 0 else x.dropna().iat[0])
            .to_dict()
        )
        known_molecules = sorted(
            [
                m
                for m in kb_df["DPD_MOLECULE_COLLAPSED"].dropna().unique().tolist()
                if isinstance(m, str) and len(m) > 0
            ]
        )
        return df_dpd_din, knowledge_base, known_molecules

    def build_item_scope(
        self,
        df_active: pd.DataFrame,
        item_map_train: Dict[int, int],
        item_map_active: Dict[int, int],
        df_item_main: pd.DataFrame,
        df_dpd_din: pd.DataFrame,
        knowledge_base: Dict[str, str],
        known_molecules: List[str],
    ) -> pd.DataFrame:
        base_cols = [
            "ITEM_NUM_CLEAN",
            "DIN_KEY",
            "MOLECULE_NM",
            "ITEM_EN_DESC",
            "SHORT_EN_DESC",
            "ATC_LEVEL1",
            "item_class_group",
            "is_rx",
            "is_otc",
            "is_other",
            "EN_FORM",
            "THRP_CL_CD",
        ]
        base_cols = [c for c in base_cols if c in df_active.columns]

        df_item = df_active[base_cols].copy()
        df_item = df_item.dropna(subset=["ITEM_NUM_CLEAN"])
        df_item = df_item.sort_values("ITEM_NUM_CLEAN").drop_duplicates("ITEM_NUM_CLEAN", keep="first")

        train_din = df_item["ITEM_NUM_CLEAN"].map(item_map_train)
        active_din = df_item["ITEM_NUM_CLEAN"].map(item_map_active)
        din_pref = normalize_din(train_din.fillna(active_din))
        df_item["DIN_KEY"] = din_pref.fillna(df_item["DIN_KEY"])

        if not df_item_main.empty:
            df_item = df_item.merge(
                df_item_main[["ITEM_NUM_CLEAN", "MOLECULE_MAIN_NAME", "MAIN_NAME_ITEM_COUNT"]],
                on="ITEM_NUM_CLEAN",
                how="left",
            )
        else:
            df_item["MOLECULE_MAIN_NAME"] = pd.NA
            df_item["MAIN_NAME_ITEM_COUNT"] = pd.NA

        df_item["MOLECULE_MAIN_NAME"] = normalize_molecule(df_item["MOLECULE_MAIN_NAME"])
        unique_main_mask = pd.to_numeric(df_item["MAIN_NAME_ITEM_COUNT"], errors="coerce").fillna(0).eq(1)
        df_item["TRAIN_MAIN_UNIQUE"] = np.where(unique_main_mask, df_item["MOLECULE_MAIN_NAME"], pd.NA)

        if not df_dpd_din.empty:
            df_item = df_item.merge(df_dpd_din[["DIN_KEY", "DPD_MOLECULE", "DPD_ATC_L1"]], on="DIN_KEY", how="left")
        else:
            df_item["DPD_MOLECULE"] = pd.NA
            df_item["DPD_ATC_L1"] = pd.NA

        desc_col = "ITEM_EN_DESC" if "ITEM_EN_DESC" in df_item.columns else ("SHORT_EN_DESC" if "SHORT_EN_DESC" in df_item.columns else None)
        if desc_col is not None:
            parsed = df_item[desc_col].apply(TextNormalizer.parse)
            df_item["PARSED_MOLECULE"] = parsed.apply(lambda x: x[0])
        else:
            df_item["PARSED_MOLECULE"] = pd.NA

        df_item["PARSED_MOLECULE"] = normalize_molecule(df_item["PARSED_MOLECULE"])
        df_item["DPD_MOLECULE_COLLAPSED"] = normalize_molecule(df_item["DPD_MOLECULE"].apply(collapse_molecule_string))

        df_item["FINAL_MOLECULE"] = (
            df_item["TRAIN_MAIN_UNIQUE"]
            .fillna(df_item["MOLECULE_NM"])
            .fillna(df_item["DPD_MOLECULE_COLLAPSED"])
            .fillna(df_item["PARSED_MOLECULE"])
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
        )

        df_item["ATC_LEVEL1"] = df_item["DPD_ATC_L1"].fillna(df_item["ATC_LEVEL1"])
        mask_kb = df_item["ATC_LEVEL1"].isna() & (df_item["FINAL_MOLECULE"] != "UNKNOWN")
        if mask_kb.any():
            inferred = df_item.loc[mask_kb, "FINAL_MOLECULE"].map(knowledge_base)
            upd = mask_kb & inferred.notna()
            df_item.loc[upd, "ATC_LEVEL1"] = inferred
        if known_molecules:
            mask_nlp = df_item["ATC_LEVEL1"].isna() & (df_item["FINAL_MOLECULE"] != "UNKNOWN")
            if mask_nlp.any():
                matcher = SmartMatcher(known_molecules, similarity_threshold=0.55)
                best = matcher.match_batch(df_item.loc[mask_nlp, "FINAL_MOLECULE"])
                inferred2 = best.map(knowledge_base)
                upd2 = mask_nlp & inferred2.notna()
                df_item.loc[upd2, "ATC_LEVEL1"] = inferred2

        bad_din = ~df_item["DIN_KEY"].astype("string").str.match(r"^\d{8}$", na=False)
        bad_mol = df_item["FINAL_MOLECULE"].isin(["UNKNOWN", "", "<NA>", "NAN"])
        df_item = df_item[~bad_din & ~bad_mol].copy()
        df_item["ATC_LEVEL1"] = df_item["ATC_LEVEL1"].fillna("X").astype(str).str.upper().str[:1]
        return df_item

    @staticmethod
    def _reduce_item_panel(parts: List[pd.DataFrame], label_cols: List[str], sum_cols: List[str], mean_cols: List[str]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)

        agg = {}
        for c in label_cols:
            if c in df.columns:
                agg[c] = "max"
        for c in sum_cols:
            if c in df.columns:
                agg[c] = "sum"
        for c in mean_cols:
            s = f"{c}__sum"
            n = f"{c}__count"
            if s in df.columns:
                agg[s] = "sum"
            if n in df.columns:
                agg[n] = "sum"

        df = df.groupby(["ITEM_NUM_CLEAN", "date_monday"], as_index=False).agg(agg)
        return df

    def load_train_p2_panel(self, valid_items: set) -> pd.DataFrame:
        path = self._path("train_P2_inbound_allitems_3years.csv")
        usecols = safe_usecols(path, TRAIN_P2_BASE_COLS)

        label_cols = [c for c in ["SHORT_2WEEKBUFFER_BINARY", "SHORT_2WEEKBUFFER_BINARY_in6_weeks"] if c in usecols]
        id_cols = [c for c in ["ITEM_NUM", "DIN_NUM", "MOLECULE_NM_LOWER", "YEAR_WEEK"] if c in usecols]
        numeric_cols = [c for c in usecols if c not in id_cols and c not in label_cols]

        sum_markers = ["QTY", "COUNT", "_sum", "_nunique"]
        sum_cols = [c for c in numeric_cols if any(m in c for m in sum_markers)]
        mean_cols = [c for c in numeric_cols if c not in sum_cols]

        agg_spec = {}
        for c in label_cols:
            agg_spec[c] = (c, "max")
        for c in sum_cols:
            agg_spec[c] = (c, "sum")
        for c in mean_cols:
            agg_spec[f"{c}__sum"] = (c, "sum")
            agg_spec[f"{c}__count"] = (c, "count")

        years = set(self.year_filter) if self.year_filter else None
        parts: List[pd.DataFrame] = []
        rows_read = 0

        t0 = time.time()
        for idx, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=self.p2_chunksize, low_memory=False), start=1):
            rows_read += len(chunk)

            yw = pd.to_numeric(chunk["YEAR_WEEK"], errors="coerce")
            if years is not None:
                chunk = chunk[(yw // 100).isin(years)]
                yw = pd.to_numeric(chunk["YEAR_WEEK"], errors="coerce")
            if self.start_yearweek is not None:
                chunk = chunk[yw >= self.start_yearweek]
                yw = pd.to_numeric(chunk["YEAR_WEEK"], errors="coerce")
            if self.end_yearweek is not None:
                chunk = chunk[yw <= self.end_yearweek]

            if chunk.empty:
                continue

            chunk["ITEM_NUM_CLEAN"] = normalize_item_num(chunk["ITEM_NUM"])
            chunk = chunk.dropna(subset=["ITEM_NUM_CLEAN"])
            chunk = chunk[chunk["ITEM_NUM_CLEAN"].isin(valid_items)]

            chunk["date_monday"] = yearweek_to_monday(chunk["YEAR_WEEK"])
            chunk = chunk.dropna(subset=["date_monday"])

            for c in numeric_cols + label_cols:
                if c in chunk.columns:
                    chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

            g = chunk.groupby(["ITEM_NUM_CLEAN", "date_monday"]).agg(**agg_spec).reset_index()
            parts.append(g)

            if len(parts) >= 10:
                parts = [self._reduce_item_panel(parts, label_cols, sum_cols, mean_cols)]

            if idx % 5 == 0:
                elapsed = time.time() - t0
                logger.info("P2 chunks=%d rows_read=%s elapsed=%.1fs", idx, f"{rows_read:,}", elapsed)

        if not parts:
            raise RuntimeError("No rows available in train_P2 after filters.")

        df = self._reduce_item_panel(parts, label_cols, sum_cols, mean_cols)
        for c in mean_cols:
            s = f"{c}__sum"
            n = f"{c}__count"
            if s in df.columns and n in df.columns:
                df[c] = df[s] / df[n].replace(0, np.nan)
                df = df.drop(columns=[s, n])

        return df.sort_values(["ITEM_NUM_CLEAN", "date_monday"]).reset_index(drop=True)

    def load_weekly_sales_panel(self, valid_items: set) -> pd.DataFrame:
        path = self._path("promitto_weekly_sales.csv")
        if not path.exists():
            logger.warning("promitto_weekly_sales.csv not found; continuing without weekly sales.")
            return pd.DataFrame(columns=["ITEM_NUM_CLEAN", "date_monday"])

        usecols = safe_usecols(path, WEEKLY_SALES_COLS)
        qty_cols = [c for c in ["QTY_ORD", "QTY_DELV", "QTY_MCS", "QTY_MCK"] if c in usecols]
        if not qty_cols:
            return pd.DataFrame(columns=["ITEM_NUM_CLEAN", "date_monday"])

        parts: List[pd.DataFrame] = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=self.ws_chunksize, low_memory=False):
            chunk["ITEM_NUM_CLEAN"] = normalize_item_num(chunk["ITEM_NUM"])
            chunk = chunk.dropna(subset=["ITEM_NUM_CLEAN"])
            chunk = chunk[chunk["ITEM_NUM_CLEAN"].isin(valid_items)]

            if "WEEK_ENDING_DT" in chunk.columns:
                dt = pd.to_datetime(chunk["WEEK_ENDING_DT"], errors="coerce")
            else:
                dt = pd.to_datetime(chunk["CAL_INVC_DT"], errors="coerce")
            chunk["date_monday"] = to_monday(dt)
            chunk = chunk.dropna(subset=["date_monday"])

            for c in qty_cols:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce").fillna(0)

            g = chunk.groupby(["ITEM_NUM_CLEAN", "date_monday"], as_index=False)[qty_cols].sum()
            parts.append(g)

            if len(parts) >= 10:
                tmp = pd.concat(parts, ignore_index=True)
                parts = [tmp.groupby(["ITEM_NUM_CLEAN", "date_monday"], as_index=False)[qty_cols].sum()]

        if not parts:
            return pd.DataFrame(columns=["ITEM_NUM_CLEAN", "date_monday"])

        df = pd.concat(parts, ignore_index=True)
        return df.groupby(["ITEM_NUM_CLEAN", "date_monday"], as_index=False)[qty_cols].sum()

    def build_vendor_maps(self, df_purchase_orders: pd.DataFrame, df_item_scope: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_vendor_item_map = pd.DataFrame()
        df_vendor_din_map = pd.DataFrame()
        if df_purchase_orders.empty or df_item_scope.empty:
            return df_vendor_item_map, df_vendor_din_map

        po = df_purchase_orders.dropna(subset=["ITEM_NUM_CLEAN", "VENDOR_NUM"]).copy()
        po["PO_QTY_ORD"] = pd.to_numeric(po["PO_QTY_ORD"], errors="coerce").fillna(0)
        v_item = po.groupby(["ITEM_NUM_CLEAN", "VENDOR_NUM"], as_index=False)["PO_QTY_ORD"].sum().rename(
            columns={"PO_QTY_ORD": "po_volume"}
        )
        t_item = v_item.groupby("ITEM_NUM_CLEAN", as_index=False)["po_volume"].sum().rename(columns={"po_volume": "item_total_volume"})
        df_vendor_item_map = v_item.merge(t_item, on="ITEM_NUM_CLEAN", how="left")
        df_vendor_item_map["vendor_weight"] = (
            df_vendor_item_map["po_volume"] / df_vendor_item_map["item_total_volume"].replace(0, np.nan)
        ).fillna(0.0)

        tmp_d = df_vendor_item_map.merge(
            df_item_scope[["ITEM_NUM_CLEAN", "DIN_KEY"]].drop_duplicates("ITEM_NUM_CLEAN"),
            on="ITEM_NUM_CLEAN",
            how="inner",
        )
        v_din = tmp_d.groupby(["DIN_KEY", "VENDOR_NUM"], as_index=False)["po_volume"].sum()
        t_din = v_din.groupby("DIN_KEY", as_index=False)["po_volume"].sum().rename(columns={"po_volume": "din_total_volume"})
        df_vendor_din_map = v_din.merge(t_din, on="DIN_KEY", how="left")
        df_vendor_din_map["vendor_weight"] = (
            df_vendor_din_map["po_volume"] / df_vendor_din_map["din_total_volume"].replace(0, np.nan)
        ).fillna(0.0)
        return df_vendor_item_map, df_vendor_din_map

    def calculate_common_date_range(
        self, df_train_p2: pd.DataFrame, df_weekly_sales: pd.DataFrame, df_trends: pd.DataFrame, df_cpi: pd.DataFrame, df_flu: pd.DataFrame
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        date_ranges = []
        for name, frame in [
            ("Train_P2", df_train_p2),
            ("WeeklySales", df_weekly_sales),
            ("Trends", df_trends),
            ("CPI", df_cpi),
            ("FluNet", df_flu),
        ]:
            if frame is not None and not frame.empty and "date_monday" in frame.columns:
                d = pd.to_datetime(frame["date_monday"], errors="coerce").dropna()
                if not d.empty:
                    date_ranges.append((name, d.min(), d.max()))
        if not date_ranges:
            return pd.NaT, pd.NaT
        return max(x[1] for x in date_ranges), min(x[2] for x in date_ranges)

    def run(self) -> Dict[str, pd.DataFrame]:
        logger.info("Loading active items")
        df_active = self.load_active_items()
        df_purchase_orders = self.load_purchase_orders()
        df_receptions = self.load_receptions()
        df_lead_times = self.load_lead_times()
        df_dc_info = self.load_dc_info()
        df_trends = self.load_trends()
        df_cpi = self.load_cpi_weekly()
        df_flu = self.load_flunet_weekly()

        logger.info("Building ITEM->DIN and molecule maps from train_P2")
        item_map_train, df_item_main = self.build_train_p2_item_maps()
        item_map_active = self.build_item_din_map_from_activeitems(df_active)
        self.ensure_dpd_files()
        df_dpd_din, knowledge_base, known_molecules = self.process_dpd_data()

        logger.info("Building item scope")
        df_item_scope = self.build_item_scope(
            df_active=df_active,
            item_map_train=item_map_train,
            item_map_active=item_map_active,
            df_item_main=df_item_main,
            df_dpd_din=df_dpd_din,
            knowledge_base=knowledge_base,
            known_molecules=known_molecules,
        )
        valid_items = set(df_item_scope["ITEM_NUM_CLEAN"].dropna().astype(int).tolist())
        logger.info("Valid items after scope gate: %s", f"{len(valid_items):,}")

        logger.info("Building train_P2 item-week panel")
        df_train_p2_panel = self.load_train_p2_panel(valid_items)
        logger.info("train_P2 panel rows: %s", f"{len(df_train_p2_panel):,}")

        logger.info("Building weekly_sales item-week panel")
        df_weekly_sales = self.load_weekly_sales_panel(valid_items)
        logger.info("weekly_sales panel rows: %s", f"{len(df_weekly_sales):,}")

        df_vendor_item_map, df_vendor_din_map = self.build_vendor_maps(df_purchase_orders, df_item_scope)
        common_start, common_end = self.calculate_common_date_range(
            df_train_p2=df_train_p2_panel,
            df_weekly_sales=df_weekly_sales,
            df_trends=df_trends,
            df_cpi=df_cpi,
            df_flu=df_flu,
        )

        return {
            "df_item_scope": df_item_scope,
            "df_train_p2_panel": df_train_p2_panel,
            "df_weekly_sales": df_weekly_sales,
            "df_vendor_item_map": df_vendor_item_map,
            "df_vendor_din_map": df_vendor_din_map,
            "df_purchase_orders": df_purchase_orders,
            "df_receptions": df_receptions,
            "df_lead_times": df_lead_times,
            "df_dc_info": df_dc_info,
            "df_trends": df_trends,
            "df_cpi": df_cpi,
            "df_flu": df_flu,
            "common_start": common_start,
            "common_end": common_end,
        }


class FeatureEngineer:
    def __init__(self, sales_window: int = 12, gap_window: int = 4):
        self.sales_window = sales_window
        self.gap_window = gap_window
        self.tracked_forms = ["TABLET", "CAPSULE", "SOLUTION", "LIQUID", "POWDER", "CREAM"]

    @staticmethod
    def build_vendor_stats_item(df_vendor_item_map: pd.DataFrame) -> pd.DataFrame:
        if df_vendor_item_map is None or df_vendor_item_map.empty:
            return pd.DataFrame()
        g = df_vendor_item_map.groupby("ITEM_NUM_CLEAN")
        stats = g["vendor_weight"].agg([("vendor_count_item", "count"), ("vendor_top_share_item", "max")]).reset_index()
        hhi = g["vendor_weight"].apply(lambda w: float((w**2).sum())).reset_index(name="vendor_hhi_item")
        return stats.merge(hhi, on="ITEM_NUM_CLEAN", how="left")

    @staticmethod
    def build_vendor_stats_din(df_vendor_din_map: pd.DataFrame) -> pd.DataFrame:
        if df_vendor_din_map is None or df_vendor_din_map.empty:
            return pd.DataFrame()
        g = df_vendor_din_map.groupby("DIN_KEY")
        stats = g["vendor_weight"].agg([("vendor_count_din", "count"), ("vendor_top_share_din", "max")]).reset_index()
        hhi = g["vendor_weight"].apply(lambda w: float((w**2).sum())).reset_index(name="vendor_hhi_din")
        return stats.merge(hhi, on="DIN_KEY", how="left")

    @staticmethod
    def build_weighted_leadtime_item(df_vendor_item_map: pd.DataFrame, df_lead_times: pd.DataFrame) -> pd.DataFrame:
        if df_vendor_item_map is None or df_vendor_item_map.empty or df_lead_times is None or df_lead_times.empty:
            return pd.DataFrame()
        tmp = df_vendor_item_map.merge(df_lead_times[["VENDOR_NUM", "TotalLeadTime"]], on="VENDOR_NUM", how="left")
        tmp = tmp.dropna(subset=["TotalLeadTime"])
        if tmp.empty:
            return pd.DataFrame()
        return (
            tmp.groupby("ITEM_NUM_CLEAN")
            .apply(lambda g: np.average(g["TotalLeadTime"], weights=g["vendor_weight"]))
            .reset_index(name="leadtime_wavg_item")
        )

    @staticmethod
    def build_weighted_leadtime_din(df_vendor_din_map: pd.DataFrame, df_lead_times: pd.DataFrame) -> pd.DataFrame:
        if df_vendor_din_map is None or df_vendor_din_map.empty or df_lead_times is None or df_lead_times.empty:
            return pd.DataFrame()
        tmp = df_vendor_din_map.merge(df_lead_times[["VENDOR_NUM", "TotalLeadTime"]], on="VENDOR_NUM", how="left")
        tmp = tmp.dropna(subset=["TotalLeadTime"])
        if tmp.empty:
            return pd.DataFrame()
        return (
            tmp.groupby("DIN_KEY")
            .apply(lambda g: np.average(g["TotalLeadTime"], weights=g["vendor_weight"]))
            .reset_index(name="leadtime_wavg_din")
        )

    def build_form_features(self, df_item_scope: pd.DataFrame, df_train_p2_panel: pd.DataFrame) -> pd.DataFrame:
        if df_item_scope is None or df_item_scope.empty:
            return pd.DataFrame()
        if "EN_FORM" in df_item_scope.columns:
            base = df_item_scope[["ITEM_NUM_CLEAN", "EN_FORM"]].copy()
        else:
            base = df_item_scope[["ITEM_NUM_CLEAN"]].copy()
            base["EN_FORM"] = pd.NA
        base["FORM_NORM"] = (
            base["EN_FORM"]
            .astype("string")
            .str.upper()
            .str.strip()
            .replace({"": pd.NA, "<NA>": pd.NA, "NONE": pd.NA, "NAN": pd.NA})
        )
        df_form = base[["ITEM_NUM_CLEAN", "FORM_NORM"]].copy()
        for f in self.tracked_forms:
            df_form[f"form_{f.lower()}"] = df_form["FORM_NORM"].eq(f).fillna(False).astype(int)
        df_form["form_other"] = (df_form["FORM_NORM"].notna() & ~df_form["FORM_NORM"].isin(self.tracked_forms)).fillna(False).astype(int)
        df_form["form_missing"] = df_form["FORM_NORM"].isna().astype(int)
        return df_form.drop(columns=["FORM_NORM"])

    def align_current_week_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["ITEM_NUM_CLEAN", "date_monday"]).reset_index(drop=True)
        g = df.groupby("ITEM_NUM_CLEAN", group_keys=False)
        lag_cols = [
            "QTY_ORD",
            "SALES_DC_COUNT",
            "SALES_DATES_COUNT",
            "QTY_DELV",
            "QTY_MCS",
            "QTY_MCK",
            "QTY_NOT_DELV",
            "QTY_ISSUE_OTHER",
            "PO_QTY_ORD_sum",
            "PO_DT_nunique",
            "DC_CD_nunique",
            "PO_NUM_nunique",
            "RCV_NUM_nunique",
            "RCV_DT_ADJUSTED_nunique",
            "RCV_QTY_ADJUSTED_sum",
            "VENDOR_SHIP_DUE_DT_nunique",
            "VENDOR_SHIP_DUE_DT_minus_EXP_DUE_DT_FROM_LEAD_TIME",
            "DIF_RCV_DATE_ORD_DATE_MINUS_LEADTIME",
            "DIF_PO_ORD_DATE_VENDOR_SHIP_DUE_DATE",
            "DIF_RCV_DATE_VENDOR_SHIP_DUE_DATE",
            "DIF_RCV_DATE_ORD_DATE",
            "WEEKS_SINCE_LAST_RECEIVE_EXP",
            "QTY_DELV_DIVIDEDBY_QTY_ORD",
            "QTY_MCS_DIVIDEDBY_QTY_ORD",
            "QTY_MCK_DIVIDEDBY_QTY_ORD",
            "QTY_NOT_DELV_DIVIDEDBY_QTY_ORD",
            "QTY_ISSUE_OTHER_DIVIDEDBY_QTY_ORD",
            "RCV_QTY_ADJUSTED_sum_DIVIDEDBY_PO_QTY_ORD_sum",
            "DIF_PO_ORD_DATE_VENDOR_SHIP_DUE_DATE_DIVIDEDBY_DIF_RCV_DATE_ORD_DATE",
            "DIF_RCV_DATE_VENDOR_SHIP_DUE_DATE_DIVIDEDBY_DIF_RCV_DATE_ORD_DATE",
            "VENDOR_SHIP_DUE_DT_minus_EXP_DUE_DT_FROM_LEAD_TIME_DIVIDEDBY_DIF_RCV_DATE_ORD_DATE_MINUS_LEADTIME",
            "RCV_QTY_ADJUSTED_sum_DIVIDEDBY_QTY_ORD",
            "PO_QTY_ORD_sum_DIVIDEDBY_QTY_ORD",
            "QTY_MCS_DIVIDEDBY_QTY_NOT_DELV",
            "WS_QTY_ORD",
            "WS_QTY_DELV",
            "WS_QTY_MCS",
            "WS_QTY_MCK",
        ]
        lag_cols = [c for c in lag_cols if c in df.columns]
        for col in lag_cols:
            df[col] = g[col].shift(1)
        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["ITEM_NUM_CLEAN", "date_monday"]).reset_index(drop=True)
        g = df.groupby("ITEM_NUM_CLEAN", group_keys=False)
        for col in [c for c in ["QTY_ORD", "QTY_DELV", "WS_QTY_ORD", "WS_QTY_DELV"] if c in df.columns]:
            df[f"{col}_mean_{self.sales_window}w"] = g[col].transform(lambda x: x.rolling(self.sales_window, min_periods=1).mean())
            df[f"{col}_std_{self.sales_window}w"] = g[col].transform(lambda x: x.rolling(self.sales_window, min_periods=2).std())
            df[f"{col}_sum_{self.gap_window}w"] = g[col].transform(lambda x: x.rolling(self.gap_window, min_periods=1).sum())
        if "QTY_ORD" in df.columns and "QTY_DELV" in df.columns:
            df["fill_rate"] = safe_div(df["QTY_DELV"], df["QTY_ORD"])
        return df

    def add_shortage_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["ITEM_NUM_CLEAN", "date_monday"]).reset_index(drop=True)
        g = df.groupby("ITEM_NUM_CLEAN", group_keys=False)
        base_short_cols = [c for c in ["SHORT_QTY_WEEKLY", "SHORT_2WEEKBUFFER_PERC", "SHORT_2WEEKBUFFER_BINARY", "SHORT_QTY_WEEKLY_DIVIDEDBY_PO_QTY_ORD_sum"] if c in df.columns]
        for col in base_short_cols:
            for k in [1, 2, 4]:
                df[f"{col}_lag{k}"] = g[col].shift(k)
            df[f"{col}_roll4_lag1"] = g[col].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
            df[f"{col}_roll12_lag1"] = g[col].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
        return df

    @staticmethod
    def build_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        if "SHORT_2WEEKBUFFER_BINARY" in df.columns:
            tmp = df.copy()
            tmp["SHORT_2WEEKBUFFER_BINARY"] = safe_num(tmp["SHORT_2WEEKBUFFER_BINARY"]).fillna(0).astype(int)
            tmp = tmp.sort_values(["ITEM_NUM_CLEAN", "date_monday"])
            g = tmp.groupby("ITEM_NUM_CLEAN")
            future_cols = []
            for k in range(1, 5):
                col = f"_s{k}"
                tmp[col] = g["SHORT_2WEEKBUFFER_BINARY"].shift(-k)
                future_cols.append(col)
            tmp["label_4w"] = tmp[future_cols].max(axis=1)
            tmp.loc[tmp[future_cols].isna().any(axis=1), "label_4w"] = pd.NA
            return tmp.drop(columns=future_cols), ["label_4w"]
        if "SHORT_2WEEKBUFFER_PERC" in df.columns:
            tmp = df.copy()
            tmp["SHORT_2WEEKBUFFER_PERC"] = safe_num(tmp["SHORT_2WEEKBUFFER_PERC"]).fillna(0)
            tmp = tmp.sort_values(["ITEM_NUM_CLEAN", "date_monday"])
            g = tmp.groupby("ITEM_NUM_CLEAN")
            future_cols = []
            for k in range(1, 5):
                col = f"_s{k}"
                tmp[col] = g["SHORT_2WEEKBUFFER_PERC"].shift(-k)
                future_cols.append(col)
            tmp["label_4w"] = (tmp[future_cols].max(axis=1) > 0).astype(int)
            tmp.loc[tmp[future_cols].isna().any(axis=1), "label_4w"] = pd.NA
            return tmp.drop(columns=future_cols), ["label_4w"]
        raise RuntimeError("Could not build label_4w: shortage source columns are missing.")

    def run(
        self,
        df_item_scope: pd.DataFrame,
        df_train_p2_panel: pd.DataFrame,
        df_weekly_sales: pd.DataFrame,
        df_vendor_item_map: pd.DataFrame,
        df_vendor_din_map: pd.DataFrame,
        df_lead_times: pd.DataFrame,
        df_trends: pd.DataFrame,
        df_cpi: pd.DataFrame,
        df_flu: pd.DataFrame,
    ) -> Dict[str, Any]:
        df = df_train_p2_panel.copy()

        if df_weekly_sales is not None and not df_weekly_sales.empty:
            ws = df_weekly_sales.copy()
            ws_cols = [c for c in ws.columns if c not in ["ITEM_NUM_CLEAN", "date_monday"]]
            ws = ws.rename(columns={c: f"WS_{c}" for c in ws_cols})
            df = df.merge(ws, on=["ITEM_NUM_CLEAN", "date_monday"], how="left")

        scope_cols = [
            "ITEM_NUM_CLEAN",
            "DIN_KEY",
            "FINAL_MOLECULE",
            "ATC_LEVEL1",
            "item_class_group",
            "is_rx",
            "is_otc",
            "is_other",
            "EN_FORM",
        ]
        scope_cols = [c for c in scope_cols if c in df_item_scope.columns]
        df = df.merge(df_item_scope[scope_cols], on="ITEM_NUM_CLEAN", how="left")

        v_item = self.build_vendor_stats_item(df_vendor_item_map)
        v_din = self.build_vendor_stats_din(df_vendor_din_map)
        if not v_item.empty:
            df = df.merge(v_item, on="ITEM_NUM_CLEAN", how="left")
        if not v_din.empty and "DIN_KEY" in df.columns:
            df = df.merge(v_din, on="DIN_KEY", how="left")

        lt_item = self.build_weighted_leadtime_item(df_vendor_item_map, df_lead_times)
        lt_din = self.build_weighted_leadtime_din(df_vendor_din_map, df_lead_times)
        if not lt_item.empty:
            df = df.merge(lt_item, on="ITEM_NUM_CLEAN", how="left")
        if not lt_din.empty and "DIN_KEY" in df.columns:
            df = df.merge(lt_din, on="DIN_KEY", how="left")

        df_form = self.build_form_features(df_item_scope, df_train_p2_panel)
        if not df_form.empty:
            df = df.merge(df_form, on="ITEM_NUM_CLEAN", how="left")

        if df_trends is not None and not df_trends.empty and "ATC_LEVEL1" in df.columns:
            t = df_trends.copy()
            t["ATC_LEVEL1"] = t["ATC_LEVEL1"].astype(str).str.upper().str.strip()
            t = t.sort_values(["ATC_LEVEL1", "date_monday"]).reset_index(drop=True)
            if "COMPOSITE_INDEX" in t.columns:
                t["COMPOSITE_INDEX"] = t.groupby("ATC_LEVEL1")["COMPOSITE_INDEX"].shift(1)
            df = df.merge(t[["ATC_LEVEL1", "date_monday", "COMPOSITE_INDEX"]], on=["ATC_LEVEL1", "date_monday"], how="left")

        if df_cpi is not None and not df_cpi.empty:
            df = df.merge(df_cpi, on="date_monday", how="left")
        if df_flu is not None and not df_flu.empty:
            df = df.merge(df_flu, on="date_monday", how="left")

        df = self.align_current_week_inputs(df)
        df = self.add_rolling_features(df)
        df = self.add_shortage_lag_features(df)
        df, label_cols = self.build_labels(df)

        id_cols = {"ITEM_NUM_CLEAN", "date_monday", "DIN_KEY", "FINAL_MOLECULE"}
        feature_cols = [c for c in df.columns if c not in id_cols and c not in set(label_cols)]
        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(0)
        return {"df_panel": df, "feature_cols": feature_cols, "label_cols": label_cols}


# -----------------------------------------------------------------------------
# Section 4 equivalent
# -----------------------------------------------------------------------------
class DataPreparator:
    def __init__(self, target_horizon: int = 4, drop_raw_sales: bool = True):
        self.target_col = f"label_{target_horizon}w"
        self.drop_raw_sales = drop_raw_sales

        self.feature_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.numerical_cols: List[str] = []
        self.constant_cols: List[str] = []

    def temporal_split(self, df_panel: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
        df = df_panel.copy()
        df["date_monday"] = pd.to_datetime(df["date_monday"], errors="coerce")
        if self.target_col not in df.columns:
            raise ValueError(f"Target column {self.target_col} not found.")

        before = len(df)
        df = df.dropna(subset=[self.target_col]).copy()
        logger.info("Dropped rows with missing %s: %s", self.target_col, f"{before - len(df):,}")

        dates = np.sort(df["date_monday"].unique())
        if len(dates) < 10:
            raise ValueError(f"Not enough unique weeks to split: {len(dates)}")

        n = len(dates)
        train_end_idx = max(int(np.floor(train_frac * n)) - 1, 0)
        val_end_idx = max(int(np.floor((train_frac + val_frac) * n)) - 1, train_end_idx + 1)

        train_end_date = dates[train_end_idx]
        val_end_date = dates[val_end_idx]

        train_df = df[df["date_monday"] <= train_end_date].copy()
        val_df = df[(df["date_monday"] > train_end_date) & (df["date_monday"] <= val_end_date)].copy()
        test_df = df[df["date_monday"] > val_end_date].copy()

        logger.info(
            "Split date ranges | train: %s -> %s | val: %s -> %s | test: %s -> %s",
            train_df["date_monday"].min().date(),
            train_df["date_monday"].max().date(),
            val_df["date_monday"].min().date(),
            val_df["date_monday"].max().date(),
            test_df["date_monday"].min().date(),
            test_df["date_monday"].max().date(),
        )
        return train_df, val_df, test_df

    def identify_features(self, df_panel: pd.DataFrame) -> None:
        exclude_cols = ["ITEM_NUM_CLEAN", "DIN_KEY", "date_monday", "FINAL_MOLECULE"]
        exclude_cols += [c for c in df_panel.columns if c.startswith("label_")]

        def is_shortage_leaky(col: str) -> bool:
            if "SHORT_" in col or "OMIT_BINARY" in col or "in6_weeks" in col:
                if "lag" in col:
                    return False
                return True
            return False

        exclude_cols += [c for c in df_panel.columns if is_shortage_leaky(c)]

        if self.drop_raw_sales:
            raw_sales = [
                "QTY_ORD",
                "QTY_DELV",
                "QTY_MCS",
                "QTY_MCK",
                "QTY_NOT_DELV",
                "QTY_ISSUE_OTHER",
                "PO_QTY_ORD_sum",
                "RCV_QTY_ADJUSTED_sum",
                "WS_QTY_ORD",
                "WS_QTY_DELV",
                "WS_QTY_MCS",
                "WS_QTY_MCK",
            ]
            exclude_cols += [c for c in raw_sales if c in df_panel.columns]

        feature_cols = [c for c in df_panel.columns if c not in exclude_cols]
        categorical_cols = []
        numerical_cols = []
        for c in feature_cols:
            if df_panel[c].dtype == "object" or df_panel[c].dtype.name in ("string", "category"):
                categorical_cols.append(c)
            else:
                numerical_cols.append(c)

        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols

    def prepare_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        X_train = train_df[self.feature_cols].copy()
        y_train = train_df[self.target_col].copy().astype(int)
        X_val = val_df[self.feature_cols].copy()
        y_val = val_df[self.target_col].copy().astype(int)
        X_test = test_df[self.feature_cols].copy()
        y_test = test_df[self.target_col].copy().astype(int)

        for c in self.categorical_cols:
            X_train[c] = X_train[c].fillna("UNKNOWN")
            X_val[c] = X_val[c].fillna("UNKNOWN")
            X_test[c] = X_test[c].fillna("UNKNOWN")

        for c in self.numerical_cols:
            X_train[c] = X_train[c].fillna(0)
            X_val[c] = X_val[c].fillna(0)
            X_test[c] = X_test[c].fillna(0)

        nunique = X_train.nunique(dropna=False)
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            X_train = X_train.drop(columns=constant_cols)
            X_val = X_val.drop(columns=constant_cols, errors="ignore")
            X_test = X_test.drop(columns=constant_cols, errors="ignore")
            self.feature_cols = [c for c in self.feature_cols if c not in constant_cols]
            self.categorical_cols = [c for c in self.categorical_cols if c not in constant_cols]
            self.numerical_cols = [c for c in self.numerical_cols if c not in constant_cols]
        self.constant_cols = constant_cols

        return X_train, y_train, X_val, y_val, X_test, y_test

    def run(self, df_panel: pd.DataFrame) -> Dict[str, Any]:
        train_df, val_df, test_df = self.temporal_split(df_panel)
        self.identify_features(df_panel)
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_features(train_df, val_df, test_df)

        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        return {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "feature_cols": self.feature_cols,
            "categorical_cols": self.categorical_cols,
            "numerical_cols": self.numerical_cols,
            "constant_cols": self.constant_cols,
            "scale_pos_weight": float(scale_pos_weight),
        }


# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------
class FeatureEncoder:
    def __init__(self):
        self.maps: Dict[str, Dict[str, int]] = {}

    def fit(self, X: pd.DataFrame, categorical_cols: List[str]) -> None:
        for col in categorical_cols:
            values = pd.Series(X[col].astype(str).fillna("UNKNOWN")).unique()
            self.maps[col] = {v: i for i, v in enumerate(values)}

    def transform(self, X: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        X_enc = X.copy()
        for col in categorical_cols:
            mapping = self.maps[col]
            X_enc[col] = (
                X_enc[col].astype(str).fillna("UNKNOWN").map(lambda v: mapping.get(v, -1)).astype(int)
            )
        return X_enc


def best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds is None or len(thresholds) == 0:
        return 0.5
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, model_name: str) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model_name": model_name,
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def run_section5_baselines(prepared: Dict[str, Any]) -> pd.DataFrame:
    X_train = prepared["X_train"].copy()
    y_train = prepared["y_train"].copy()
    X_val = prepared["X_val"].copy()
    y_val = prepared["y_val"].copy()
    categorical_cols = prepared["categorical_cols"]

    enc = FeatureEncoder()
    enc.fit(X_train, categorical_cols)
    X_train_enc = enc.transform(X_train, categorical_cols)
    X_val_enc = enc.transform(X_val, categorical_cols)

    rows = []

    # Majority baseline
    majority_class = int(y_train.mode().iloc[0]) if len(y_train) > 0 else 0
    y_prob = np.full(len(y_val), 1.0 if majority_class == 1 else 0.0)
    y_pred = np.full(len(y_val), majority_class)
    rows.append(
        {
            "model_name": "MAJORITY_CLASS",
            "pr_auc": float(average_precision_score(y_val.values, y_prob)) if y_val.nunique() > 1 else np.nan,
            "roc_auc": float(roc_auc_score(y_val.values, y_prob)) if y_val.nunique() > 1 else np.nan,
            "precision": float(precision_score(y_val.values, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val.values, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val.values, y_pred, zero_division=0)),
        }
    )

    # Random stratified baseline
    np.random.seed(42)
    pos_rate = float(y_train.mean()) if len(y_train) > 0 else 0.5
    y_prob = np.full(len(y_val), pos_rate)
    y_pred = np.random.binomial(1, pos_rate, size=len(y_val))
    rows.append(
        {
            "model_name": "RANDOM_STRATIFIED",
            "pr_auc": float(average_precision_score(y_val.values, y_prob)) if y_val.nunique() > 1 else np.nan,
            "roc_auc": float(roc_auc_score(y_val.values, y_prob)) if y_val.nunique() > 1 else np.nan,
            "precision": float(precision_score(y_val.values, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val.values, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val.values, y_pred, zero_division=0)),
        }
    )

    # Logistic regression baseline (scaled + robust solver to avoid lbfgs convergence warnings)
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            solver="saga",
            max_iter=5000,
            tol=1e-3,
            random_state=42,
            n_jobs=-1,
        ),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr.fit(X_train_enc, y_train)
    y_prob = lr.predict_proba(X_val_enc)[:, 1]
    y_pred = lr.predict(X_val_enc)
    rows.append(
        {
            "model_name": "LOGISTIC_REGRESSION",
            "pr_auc": float(average_precision_score(y_val.values, y_prob)) if y_val.nunique() > 1 else np.nan,
            "roc_auc": float(roc_auc_score(y_val.values, y_prob)) if y_val.nunique() > 1 else np.nan,
            "precision": float(precision_score(y_val.values, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val.values, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val.values, y_pred, zero_division=0)),
        }
    )

    return pd.DataFrame(rows)


def build_model(model_type: str, scale_pos_weight: float, try_gpu: bool, params: Optional[Dict[str, Any]] = None):
    params = params or {}
    if model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            return None
        base = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "aucpr",
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
        }
        if try_gpu:
            base["device"] = "cuda"
        base.update(params)
        return xgb.XGBClassifier(**base)

    if model_type == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            return None
        base = {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": {0: 1.0, 1: scale_pos_weight},
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        if try_gpu:
            base["device_type"] = "gpu"
        base.update(params)
        return lgb.LGBMClassifier(**base)

    if model_type == "randomforest":
        base = {
            "n_estimators": 300,
            "max_depth": 20,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        base.update(params)
        return RandomForestClassifier(**base)

    if model_type == "catboost":
        if not CATBOOST_AVAILABLE:
            return None
        base = {
            "depth": 8,
            "learning_rate": 0.05,
            "iterations": 500,
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "random_seed": 42,
            "class_weights": [1.0, scale_pos_weight],
            "verbose": False,
        }
        if try_gpu:
            base["task_type"] = "GPU"
        base.update(params)
        return CatBoostClassifier(**base)

    return None


def make_rolling_folds(train_df: pd.DataFrame, target_col: str, min_train_weeks: int = 8, val_weeks: int = 2, max_folds: int = 3):
    tmp = train_df.copy()
    tmp["date_monday"] = pd.to_datetime(tmp["date_monday"], errors="coerce")
    tmp = tmp.dropna(subset=["date_monday", target_col]).copy()
    weeks = np.sort(tmp["date_monday"].unique())
    if len(weeks) < (min_train_weeks + val_weeks):
        fallback_min_train = max(4, len(weeks) - val_weeks)
        if len(weeks) < (fallback_min_train + val_weeks):
            raise ValueError("Not enough training weeks to build rolling tuning folds.")
        min_train_weeks = fallback_min_train

    folds = []
    for split_idx in range(min_train_weeks, len(weeks) - val_weeks + 1, val_weeks):
        train_weeks = weeks[:split_idx]
        val_weeks_arr = weeks[split_idx : split_idx + val_weeks]
        if len(val_weeks_arr) < val_weeks:
            continue
        fold_train = tmp[tmp["date_monday"].isin(train_weeks)].copy()
        fold_val = tmp[tmp["date_monday"].isin(val_weeks_arr)].copy()
        if fold_train.empty or fold_val.empty:
            continue
        if fold_train[target_col].nunique() < 2 or fold_val[target_col].nunique() < 2:
            continue
        folds.append({"train_df": fold_train, "val_df": fold_val})
    if not folds:
        raise ValueError("No valid rolling folds could be created.")
    return folds[-max_folds:]


def prepare_fold_data(
    train_fold_df: pd.DataFrame,
    val_fold_df: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str],
    target_col: str,
) -> Dict[str, Any]:
    numerical_cols = [c for c in feature_cols if c not in categorical_cols]
    X_tr = train_fold_df[feature_cols].copy()
    X_va = val_fold_df[feature_cols].copy()
    y_tr = pd.Series(train_fold_df[target_col]).astype(int).reset_index(drop=True)
    y_va = pd.Series(val_fold_df[target_col]).astype(int).reset_index(drop=True)

    for c in categorical_cols:
        if c in X_tr.columns:
            X_tr[c] = X_tr[c].fillna("UNKNOWN").astype(str)
        if c in X_va.columns:
            X_va[c] = X_va[c].fillna("UNKNOWN").astype(str)
    for c in numerical_cols:
        if c in X_tr.columns:
            X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce").fillna(0)
        if c in X_va.columns:
            X_va[c] = pd.to_numeric(X_va[c], errors="coerce").fillna(0)

    fold_encoder = FeatureEncoder()
    fold_encoder.fit(X_tr, categorical_cols)
    X_tr_encoded = fold_encoder.transform(X_tr, categorical_cols)
    X_va_encoded = fold_encoder.transform(X_va, categorical_cols)

    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    fold_spw = (n_neg / n_pos) if n_pos > 0 else 1.0
    return {
        "X_tr_encoded": X_tr_encoded,
        "X_va_encoded": X_va_encoded,
        "X_tr_cb": X_tr,
        "X_va_cb": X_va,
        "y_tr": y_tr,
        "y_va": y_va,
        "scale_pos_weight": float(fold_spw),
    }


def fit_predict_model(
    model_type: str,
    model: Any,
    X_train_enc: pd.DataFrame,
    y_train: pd.Series,
    X_val_enc: pd.DataFrame,
    y_val: pd.Series,
    X_test_enc: pd.DataFrame,
    X_train_cb: pd.DataFrame,
    X_val_cb: pd.DataFrame,
    X_test_cb: pd.DataFrame,
    cat_features_idx: List[int],
    try_gpu: bool,
    scale_pos_weight: float,
    model_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, np.ndarray, np.ndarray, str]:
    device_used = "gpu" if try_gpu else "cpu"
    try:
        if model_type == "catboost":
            model.fit(X_train_cb, y_train, eval_set=(X_val_cb, y_val), cat_features=cat_features_idx, verbose=False)
            y_val_prob = model.predict_proba(X_val_cb)[:, 1]
            y_test_prob = model.predict_proba(X_test_cb)[:, 1]
        elif model_type == "lightgbm":
            model.fit(
                X_train_enc,
                y_train,
                eval_set=[(X_val_enc, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            y_val_prob = model.predict_proba(X_val_enc)[:, 1]
            y_test_prob = model.predict_proba(X_test_enc)[:, 1]
        elif model_type == "xgboost":
            model.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], verbose=False)
            y_val_prob = model.predict_proba(X_val_enc)[:, 1]
            y_test_prob = model.predict_proba(X_test_enc)[:, 1]
        else:
            model.fit(X_train_enc, y_train)
            y_val_prob = model.predict_proba(X_val_enc)[:, 1]
            y_test_prob = model.predict_proba(X_test_enc)[:, 1]
    except Exception as e:
        if not try_gpu:
            raise
        logger.warning("%s gpu mode failed (%s). Falling back to CPU.", model_type, str(e)[:180])
        device_used = "cpu"
        cpu_model = build_model(model_type, scale_pos_weight, try_gpu=False, params=model_params)
        return fit_predict_model(
            model_type=model_type,
            model=cpu_model,
            X_train_enc=X_train_enc,
            y_train=y_train,
            X_val_enc=X_val_enc,
            y_val=y_val,
            X_test_enc=X_test_enc,
            X_train_cb=X_train_cb,
            X_val_cb=X_val_cb,
            X_test_cb=X_test_cb,
            cat_features_idx=cat_features_idx,
            try_gpu=False,
            scale_pos_weight=scale_pos_weight,
            model_params=model_params,
        )
    return model, y_val_prob, y_test_prob, device_used


def run_section6_models(prepared: Dict[str, Any], try_gpu: bool = False):
    X_train = prepared["X_train"]
    y_train = prepared["y_train"]
    X_val = prepared["X_val"]
    y_val = prepared["y_val"]
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]
    categorical_cols = prepared["categorical_cols"]
    spw = prepared["scale_pos_weight"]

    encoder = FeatureEncoder()
    encoder.fit(X_train, categorical_cols)
    X_train_enc = encoder.transform(X_train, categorical_cols)
    X_val_enc = encoder.transform(X_val, categorical_cols)
    X_test_enc = encoder.transform(X_test, categorical_cols)

    X_train_cb = X_train.copy()
    X_val_cb = X_val.copy()
    X_test_cb = X_test.copy()
    for c in categorical_cols:
        X_train_cb[c] = X_train_cb[c].astype(str).fillna("UNKNOWN")
        X_val_cb[c] = X_val_cb[c].astype(str).fillna("UNKNOWN")
        X_test_cb[c] = X_test_cb[c].astype(str).fillna("UNKNOWN")
    cat_features_idx = [X_train_cb.columns.get_loc(c) for c in categorical_cols if c in X_train_cb.columns]

    model_order = ["xgboost", "randomforest", "lightgbm", "catboost"]
    val_metrics: List[Dict[str, Any]] = []
    test_metrics: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}

    for model_type in model_order:
        model = build_model(model_type, spw, try_gpu=try_gpu)
        if model is None:
            continue
        logger.info("Section 6 training model: %s", model_type)
        t0 = time.time()
        model, y_val_prob, y_test_prob, device_used = fit_predict_model(
            model_type=model_type,
            model=model,
            X_train_enc=X_train_enc,
            y_train=y_train,
            X_val_enc=X_val_enc,
            y_val=y_val,
            X_test_enc=X_test_enc,
            X_train_cb=X_train_cb,
            X_val_cb=X_val_cb,
            X_test_cb=X_test_cb,
            cat_features_idx=cat_features_idx,
            try_gpu=try_gpu,
            scale_pos_weight=spw,
            model_params=None,
        )
        best_th = best_threshold_f1(y_val.values, y_val_prob)
        m_val = evaluate(y_val.values, y_val_prob, best_th, model_type.upper())
        m_test = evaluate(y_test.values, y_test_prob, best_th, model_type.upper())
        m_val["train_time_sec"] = float(time.time() - t0)
        m_val["device_used"] = device_used
        m_test["device_used"] = device_used

        val_metrics.append(m_val)
        test_metrics.append(m_test)
        trained_models[model_type] = model

    if not trained_models:
        raise RuntimeError("No trainable models found.")
    return val_metrics, test_metrics, trained_models, encoder


def run_section7_tuning(
    prepared: Dict[str, Any],
    try_gpu: bool,
    output_dir: Path,
    trials_xgb: int = 20,
    trials_lgb: int = 20,
    trials_rf: int = 5,
    trials_cb: int = 15,
):
    train_df = prepared["train_df"].copy()
    X_train = prepared["X_train"].copy()
    y_train = prepared["y_train"].copy()
    categorical_cols = prepared["categorical_cols"]
    feature_cols = list(X_train.columns)
    target_col = y_train.name if getattr(y_train, "name", None) else "label_4w"

    encoder = FeatureEncoder()
    encoder.fit(X_train, categorical_cols)
    X_train_encoded = encoder.transform(X_train, categorical_cols)
    X_train_cb = X_train.copy()
    for c in categorical_cols:
        X_train_cb[c] = X_train_cb[c].astype(str).fillna("UNKNOWN")
    cat_features_idx = [X_train_cb.columns.get_loc(c) for c in categorical_cols if c in X_train_cb.columns]

    folds = make_rolling_folds(train_df, target_col=target_col, min_train_weeks=8, val_weeks=2, max_folds=3)
    configs = [("xgboost", trials_xgb), ("lightgbm", trials_lgb), ("randomforest", trials_rf), ("catboost", trials_cb)]
    tuned_models: Dict[str, Any] = {}
    tuning_results: Dict[str, Any] = {}
    rows = []
    rng = np.random.default_rng(42)

    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available; using built-in random search tuner for Section 7.")

    def suggest_params_optuna(model_type: str, trial):
        if model_type == "xgboost":
            return {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 80, 320),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
        if model_type == "lightgbm":
            return {
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 80, 320),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
        if model_type == "randomforest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 80, 240),
                "max_depth": trial.suggest_int("max_depth", 10, 25),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            }
        return {
            "depth": trial.suggest_int("depth", 6, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "iterations": trial.suggest_int("iterations", 250, 650),
        }

    def suggest_params_random(model_type: str) -> Dict[str, Any]:
        if model_type == "xgboost":
            return {
                "max_depth": int(rng.integers(3, 9)),
                "learning_rate": float(10 ** rng.uniform(np.log10(0.01), np.log10(0.2))),
                "n_estimators": int(rng.integers(80, 321)),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample_bytree": float(rng.uniform(0.6, 1.0)),
            }
        if model_type == "lightgbm":
            return {
                "num_leaves": int(rng.integers(20, 101)),
                "learning_rate": float(10 ** rng.uniform(np.log10(0.01), np.log10(0.2))),
                "n_estimators": int(rng.integers(80, 321)),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample_bytree": float(rng.uniform(0.6, 1.0)),
            }
        if model_type == "randomforest":
            return {
                "n_estimators": int(rng.integers(80, 241)),
                "max_depth": int(rng.integers(10, 26)),
                "min_samples_split": int(rng.integers(2, 16)),
                "min_samples_leaf": int(rng.integers(1, 9)),
            }
        return {
            "depth": int(rng.integers(6, 11)),
            "learning_rate": float(10 ** rng.uniform(np.log10(0.01), np.log10(0.2))),
            "iterations": int(rng.integers(250, 651)),
        }

    def score_params(model_type: str, params: Dict[str, Any]) -> float:
        scores = []
        for fold in folds:
            fd = prepare_fold_data(
                train_fold_df=fold["train_df"],
                val_fold_df=fold["val_df"],
                feature_cols=feature_cols,
                categorical_cols=categorical_cols,
                target_col=target_col,
            )
            model = build_model(model_type, fd["scale_pos_weight"], try_gpu=try_gpu, params=params)
            if model is None:
                continue
            try:
                if model_type == "catboost":
                    model.fit(
                        fd["X_tr_cb"],
                        fd["y_tr"],
                        eval_set=(fd["X_va_cb"], fd["y_va"]),
                        cat_features=cat_features_idx,
                        verbose=False,
                    )
                    y_prob = model.predict_proba(fd["X_va_cb"])[:, 1]
                elif model_type == "lightgbm":
                    model.fit(
                        fd["X_tr_encoded"],
                        fd["y_tr"],
                        eval_set=[(fd["X_va_encoded"], fd["y_va"])],
                        callbacks=[lgb.early_stopping(15, verbose=False)],
                    )
                    y_prob = model.predict_proba(fd["X_va_encoded"])[:, 1]
                elif model_type == "xgboost":
                    model.fit(fd["X_tr_encoded"], fd["y_tr"], eval_set=[(fd["X_va_encoded"], fd["y_va"])], verbose=False)
                    y_prob = model.predict_proba(fd["X_va_encoded"])[:, 1]
                else:
                    model.fit(fd["X_tr_encoded"], fd["y_tr"])
                    y_prob = model.predict_proba(fd["X_va_encoded"])[:, 1]
            except Exception:
                model = build_model(model_type, fd["scale_pos_weight"], try_gpu=False, params=params)
                if model_type == "catboost":
                    model.fit(
                        fd["X_tr_cb"],
                        fd["y_tr"],
                        eval_set=(fd["X_va_cb"], fd["y_va"]),
                        cat_features=cat_features_idx,
                        verbose=False,
                    )
                    y_prob = model.predict_proba(fd["X_va_cb"])[:, 1]
                elif model_type == "lightgbm":
                    model.fit(
                        fd["X_tr_encoded"],
                        fd["y_tr"],
                        eval_set=[(fd["X_va_encoded"], fd["y_va"])],
                        callbacks=[lgb.early_stopping(15, verbose=False)],
                    )
                    y_prob = model.predict_proba(fd["X_va_encoded"])[:, 1]
                elif model_type == "xgboost":
                    model.fit(fd["X_tr_encoded"], fd["y_tr"], eval_set=[(fd["X_va_encoded"], fd["y_va"])], verbose=False)
                    y_prob = model.predict_proba(fd["X_va_encoded"])[:, 1]
                else:
                    model.fit(fd["X_tr_encoded"], fd["y_tr"])
                    y_prob = model.predict_proba(fd["X_va_encoded"])[:, 1]
            scores.append(float(average_precision_score(fd["y_va"], y_prob)))
        return float(np.mean(scores)) if scores else 0.0

    for model_type, n_trials in configs:
        if model_type == "xgboost" and not XGBOOST_AVAILABLE:
            continue
        if model_type == "lightgbm" and not LIGHTGBM_AVAILABLE:
            continue
        if model_type == "catboost" and not CATBOOST_AVAILABLE:
            continue

        if OPTUNA_AVAILABLE:
            def objective(trial):
                params = suggest_params_optuna(model_type, trial)
                return score_params(model_type, params)

            sampler = TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            best_params = study.best_params
            best_score = float(study.best_value)
        else:
            best_params = None
            best_score = -1.0
            for _ in range(n_trials):
                params = suggest_params_random(model_type)
                score = score_params(model_type, params)
                if score > best_score:
                    best_score = score
                    best_params = params
            if best_params is None:
                continue

        scale_pos_weight = prepared["scale_pos_weight"]
        model = build_model(model_type, scale_pos_weight, try_gpu=try_gpu, params=best_params)
        try:
            if model_type == "catboost":
                model.fit(X_train_cb, y_train, cat_features=cat_features_idx, verbose=False)
            else:
                model.fit(X_train_encoded, y_train)
        except Exception:
            model = build_model(model_type, scale_pos_weight, try_gpu=False, params=best_params)
            if model_type == "catboost":
                model.fit(X_train_cb, y_train, cat_features=cat_features_idx, verbose=False)
            else:
                model.fit(X_train_encoded, y_train)

        tuned_models[model_type] = model
        tuning_results[model_type] = {"best_params": best_params, "best_score": best_score, "n_folds": len(folds)}
        rows.append({"model_type": model_type, "best_score": best_score, "best_params": json.dumps(best_params)})

        with open(output_dir / f"{model_type}_tuned_params.json", "w") as f:
            json.dump(tuning_results[model_type], f, indent=2)

    if not tuned_models:
        return None, None, None
    pd.DataFrame(rows).sort_values("best_score", ascending=False).to_csv(output_dir / "tuning_summary.csv", index=False)
    return tuned_models, tuning_results, encoder


def run_section8_thresholds(prepared: Dict[str, Any], tuned_models: Dict[str, Any], encoder: FeatureEncoder, output_dir: Path):
    X_train = prepared["X_train"]
    X_val = prepared["X_val"]
    y_val = prepared["y_val"]
    categorical_cols = prepared["categorical_cols"]
    if encoder is None:
        encoder = FeatureEncoder()
        encoder.fit(X_train, categorical_cols)
    X_val_encoded = encoder.transform(X_val, categorical_cols)
    X_val_cb = X_val.copy()
    for c in categorical_cols:
        X_val_cb[c] = X_val_cb[c].astype(str).fillna("UNKNOWN")

    def find_optimal(y_true, y_prob, strategy="f1", min_precision=0.20, min_recall=0.60):
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        if thresholds is None or len(thresholds) == 0:
            thr = 0.5
            y_pred = (y_prob >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            return {"threshold": float(thr), "precision": float(precision_score(y_true, y_pred, zero_division=0)), "recall": float(recall_score(y_true, y_pred, zero_division=0)), "f1": float(f1_score(y_true, y_pred, zero_division=0)), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        if strategy == "f1":
            idx = int(np.argmax(f1_scores))
        elif strategy == "recall_constrained":
            valid = precision >= min_precision
            idx = np.where(valid)[0][int(np.argmax(recall[valid]))] if valid.sum() else int(np.argmax(f1_scores))
        else:
            valid = recall >= min_recall
            idx = np.where(valid)[0][int(np.argmax(precision[valid]))] if valid.sum() else int(np.argmax(f1_scores))
        thr = thresholds[min(idx, len(thresholds) - 1)]
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {"threshold": float(thr), "precision": float(precision_score(y_true, y_pred, zero_division=0)), "recall": float(recall_score(y_true, y_pred, zero_division=0)), "f1": float(f1_score(y_true, y_pred, zero_division=0)), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

    strategies = [("f1", {}), ("recall_constrained", {"min_precision": 0.20}), ("precision_constrained", {"min_recall": 0.60})]
    threshold_results = {}
    rows = []

    for model_type, model in tuned_models.items():
        if model_type == "catboost":
            y_prob = model.predict_proba(X_val_cb)[:, 1]
        else:
            y_prob = model.predict_proba(X_val_encoded)[:, 1]
        model_results = {}
        for st, kwargs in strategies:
            r = find_optimal(y_val.values, y_prob, strategy=st, **kwargs)
            model_results[st] = r
            rows.append({"model_type": model_type, "strategy": st, "threshold": r["threshold"], "precision": r["precision"], "recall": r["recall"], "f1": r["f1"]})
        threshold_results[model_type] = model_results
        with open(output_dir / f"{model_type}_threshold_results.json", "w") as f:
            json.dump(model_results, f, indent=2)

    pd.DataFrame(rows).to_csv(output_dir / "threshold_optimization_summary.csv", index=False)
    return threshold_results


def run_section9_selection(prepared: Dict[str, Any], tuned_models: Dict[str, Any], tuning_results: Dict[str, Any], threshold_results: Dict[str, Any], encoder: FeatureEncoder):
    best_model_type = max(tuning_results, key=lambda k: tuning_results[k]["best_score"])

    def get_f1_recall_thr(model_type):
        r = threshold_results.get(model_type, {})
        if "f1" not in r:
            return None, None, None
        return r["f1"]["f1"], r["f1"]["recall"], r["f1"]["threshold"]

    best_f1, best_recall, best_thr = get_f1_recall_thr(best_model_type)
    if best_f1 is not None:
        for mt in tuned_models:
            if mt == best_model_type:
                continue
            f1, rec, thr = get_f1_recall_thr(mt)
            if f1 is None:
                continue
            if abs(f1 - best_f1) <= 0.05 and rec > best_recall:
                best_model_type = mt
                best_f1, best_recall, best_thr = f1, rec, thr

    X_train = prepared["X_train"]
    X_val = prepared["X_val"]
    y_val = prepared["y_val"]
    categorical_cols = prepared["categorical_cols"]
    if encoder is None:
        encoder = FeatureEncoder()
        encoder.fit(X_train, categorical_cols)
    X_val_encoded = encoder.transform(X_val, categorical_cols)
    X_val_cb = X_val.copy()
    for c in categorical_cols:
        X_val_cb[c] = X_val_cb[c].astype(str).fillna("UNKNOWN")

    model = tuned_models[best_model_type]
    if best_model_type == "catboost":
        y_prob = model.predict_proba(X_val_cb)[:, 1]
    else:
        y_prob = model.predict_proba(X_val_encoded)[:, 1]
    thr = float(best_thr if best_thr is not None else best_threshold_f1(y_val.values, y_prob))
    y_pred = (y_prob >= thr).astype(int)
    val_metrics = {
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
    }
    return best_model_type, thr, val_metrics


def run_section10_test(
    prepared: Dict[str, Any],
    tuned_models: Dict[str, Any],
    encoder: FeatureEncoder,
    final_model_name: str,
    final_threshold: float,
    output_dir: Path,
):
    test_df = prepared["test_df"]
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]
    categorical_cols = prepared["categorical_cols"]
    model = tuned_models[final_model_name]

    if final_model_name == "catboost":
        X_eval = X_test.copy()
        for c in categorical_cols:
            X_eval[c] = X_eval[c].astype(str).fillna("UNKNOWN")
    else:
        X_eval = encoder.transform(X_test.copy(), categorical_cols)
    y_true = pd.Series(y_test).astype(int).reset_index(drop=True)
    y_prob = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob >= final_threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    summary = {
        "model": final_model_name.upper(),
        "threshold": float(final_threshold),
        "test_start": str(pd.to_datetime(test_df["date_monday"]).min().date()),
        "test_end": str(pd.to_datetime(test_df["date_monday"]).max().date()),
        "rows": int(len(test_df)),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "test_summary.csv", index=False)
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_year_filter(raw: str) -> Optional[List[int]]:
    raw = raw.strip()
    if raw.lower() in {"none", "", "all"}:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="UC2 train/val/test pipeline aligned to notebook Sections 2-10 (3-year default, no EDA)."
    )
    p.add_argument("--data-dir", type=str, default=".", help="Directory containing CSV files.")
    p.add_argument("--output-dir", type=str, default="Output_py", help="Output directory.")

    p.add_argument(
        "--year-filter",
        type=str,
        default="none",
        help="Year filter for YEAR_WEEK (e.g., '2023' or '2021,2022,2023' or 'none'). Default: none (all years).",
    )
    p.add_argument("--start-yearweek", type=int, default=None, help="Optional start YEAR_WEEK (e.g., 202101).")
    p.add_argument("--end-yearweek", type=int, default=None, help="Optional end YEAR_WEEK (e.g., 202352).")
    p.add_argument("--p2-chunksize", type=int, default=300_000)
    p.add_argument("--ws-chunksize", type=int, default=300_000)
    p.add_argument("--refresh-dpd", action="store_true", help="Force refresh Health Canada DPD files.")
    p.add_argument("--refresh-google-trends", action="store_true", help="Force refresh Google Trends from pytrends.")
    p.add_argument("--refresh-cpi", action="store_true", help="Force refresh CPI weekly from StatCan URL.")
    p.add_argument("--refresh-flu", action="store_true", help="Force refresh FluNet weekly from WHO URL.")

    p.add_argument("--drop-raw-sales", action="store_true", default=True, help="Drop raw sales columns from features.")
    p.add_argument("--keep-raw-sales", action="store_true", help="Keep raw sales columns in features.")
    p.add_argument("--try-gpu", action="store_true", help="Try model GPU mode first, fallback to CPU if unavailable.")
    p.add_argument("--skip-tuning", action="store_true", help="Skip Section 7 Optuna tuning and use Section 6 models.")
    p.add_argument("--trials-xgb", type=int, default=20)
    p.add_argument("--trials-lgb", type=int, default=20)
    p.add_argument("--trials-rf", type=int, default=5)
    p.add_argument("--trials-cb", type=int, default=15)
    p.add_argument("--log-level", type=str, default="INFO")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    year_filter = parse_year_filter(args.year_filter)
    drop_raw_sales = args.drop_raw_sales and (not args.keep_raw_sales)

    logger.info("UC2 standalone training started")
    logger.info("Data dir: %s", data_dir)
    logger.info("Output dir: %s", output_dir)
    logger.info("Year filter: %s", year_filter if year_filter is not None else "ALL")
    logger.info("Yearweek window: %s -> %s", args.start_yearweek, args.end_yearweek)
    logger.info("Try GPU: %s", args.try_gpu)
    logger.info(
        "Model availability | xgboost=%s lightgbm=%s catboost=%s",
        XGBOOST_AVAILABLE,
        LIGHTGBM_AVAILABLE,
        CATBOOST_AVAILABLE,
    )

    # Section 2
    builder = UC2DataBuilder(
        data_dir=data_dir,
        p2_chunksize=args.p2_chunksize,
        ws_chunksize=args.ws_chunksize,
        year_filter=year_filter,
        start_yearweek=args.start_yearweek,
        end_yearweek=args.end_yearweek,
        refresh_dpd=args.refresh_dpd,
        refresh_google_trends=args.refresh_google_trends,
        refresh_cpi=args.refresh_cpi,
        refresh_flu=args.refresh_flu,
    )
    section2 = builder.run()

    # Section 3
    fe = FeatureEngineer(sales_window=12, gap_window=4)
    section3 = fe.run(
        df_item_scope=section2["df_item_scope"],
        df_train_p2_panel=section2["df_train_p2_panel"],
        df_weekly_sales=section2["df_weekly_sales"],
        df_vendor_item_map=section2["df_vendor_item_map"],
        df_vendor_din_map=section2["df_vendor_din_map"],
        df_lead_times=section2["df_lead_times"],
        df_trends=section2["df_trends"],
        df_cpi=section2["df_cpi"],
        df_flu=section2["df_flu"],
    )
    df_panel = section3["df_panel"]
    logger.info("Feature panel rows: %s", f"{len(df_panel):,}")
    logger.info("Feature columns (pre-section4 filtering): %d", len(section3["feature_cols"]))

    # Section 4
    preparator = DataPreparator(target_horizon=4, drop_raw_sales=drop_raw_sales)
    prepared = preparator.run(df_panel)

    train_df = prepared["train_df"]
    val_df = prepared["val_df"]
    test_df = prepared["test_df"]
    logger.info(
        "Split summary | train=%s val=%s test=%s",
        f"{len(train_df):,}",
        f"{len(val_df):,}",
        f"{len(test_df):,}",
    )

    # Section 5
    df_baselines = run_section5_baselines(prepared)
    df_baselines.to_csv(output_dir / "section5_baseline_metrics.csv", index=False)

    # Section 6
    val_metrics_s6, test_metrics_s6, section6_models, section6_encoder = run_section6_models(prepared, try_gpu=args.try_gpu)
    df_val_s6 = pd.DataFrame(val_metrics_s6).sort_values("pr_auc", ascending=False)
    df_test_s6 = pd.DataFrame(test_metrics_s6)
    df_val_s6.to_csv(output_dir / "section6_metrics_validation.csv", index=False)
    df_test_s6.to_csv(output_dir / "section6_metrics_test.csv", index=False)

    # Section 7
    tuned_models = None
    tuning_results = None
    tuning_encoder = None
    if not args.skip_tuning:
        tuned_models, tuning_results, tuning_encoder = run_section7_tuning(
            prepared=prepared,
            try_gpu=args.try_gpu,
            output_dir=output_dir,
            trials_xgb=args.trials_xgb,
            trials_lgb=args.trials_lgb,
            trials_rf=args.trials_rf,
            trials_cb=args.trials_cb,
        )

    if tuned_models is None or tuning_results is None:
        logger.warning("Using Section 6 models as fallback for downstream Sections 8-10.")
        tuned_models = section6_models
        tuning_encoder = section6_encoder
        tuning_results = {
            k: {"best_score": float(df_val_s6[df_val_s6["model_name"] == k.upper()]["pr_auc"].iloc[0]), "best_params": {}, "n_folds": 0}
            for k in tuned_models.keys()
        }

    # Section 8
    threshold_results = run_section8_thresholds(
        prepared=prepared,
        tuned_models=tuned_models,
        encoder=tuning_encoder,
        output_dir=output_dir,
    )

    # Section 9
    final_model_name, final_threshold, validation_metrics = run_section9_selection(
        prepared=prepared,
        tuned_models=tuned_models,
        tuning_results=tuning_results,
        threshold_results=threshold_results,
        encoder=tuning_encoder,
    )

    # Section 10
    test_summary = run_section10_test(
        prepared=prepared,
        tuned_models=tuned_models,
        encoder=tuning_encoder,
        final_model_name=final_model_name,
        final_threshold=final_threshold,
        output_dir=output_dir,
    )

    run_summary = {
        "data_dir": str(data_dir),
        "year_filter": year_filter,
        "start_yearweek": args.start_yearweek,
        "end_yearweek": args.end_yearweek,
        "try_gpu": bool(args.try_gpu),
        "panel_date_min": str(pd.to_datetime(df_panel["date_monday"]).min().date()),
        "panel_date_max": str(pd.to_datetime(df_panel["date_monday"]).max().date()),
        "train_date_min": str(pd.to_datetime(train_df["date_monday"]).min().date()),
        "train_date_max": str(pd.to_datetime(train_df["date_monday"]).max().date()),
        "val_date_min": str(pd.to_datetime(val_df["date_monday"]).min().date()),
        "val_date_max": str(pd.to_datetime(val_df["date_monday"]).max().date()),
        "test_date_min": str(pd.to_datetime(test_df["date_monday"]).min().date()),
        "test_date_max": str(pd.to_datetime(test_df["date_monday"]).max().date()),
        "n_rows_panel": int(len(df_panel)),
        "n_rows_train": int(len(train_df)),
        "n_rows_val": int(len(val_df)),
        "n_rows_test": int(len(test_df)),
        "n_features": int(len(prepared["feature_cols"])),
        "n_categorical": int(len(prepared["categorical_cols"])),
        "n_numerical": int(len(prepared["numerical_cols"])),
        "common_start": str(section2["common_start"].date()) if pd.notna(section2["common_start"]) else None,
        "common_end": str(section2["common_end"].date()) if pd.notna(section2["common_end"]) else None,
        "section6_models": [m for m in section6_models.keys()],
        "tuned_models": [m for m in tuned_models.keys()],
        "section5_baselines": df_baselines.to_dict(orient="records"),
        "best_model": final_model_name,
        "best_threshold": float(final_threshold),
        "validation_metrics": validation_metrics,
        "best_test_metrics": test_summary,
    }
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    logger.info("Section 6 validation metrics saved: %s", output_dir / "section6_metrics_validation.csv")
    logger.info("Section 6 test metrics saved: %s", output_dir / "section6_metrics_test.csv")
    logger.info("Section 10 test summary saved: %s", output_dir / "test_summary.csv")
    logger.info("Run summary saved: %s", output_dir / "run_summary.json")

    print("\n" + "=" * 90)
    print("UC2 TRAIN/VAL/TEST COMPLETE")
    print("=" * 90)
    print(f"Panel range: {run_summary['panel_date_min']} -> {run_summary['panel_date_max']}")
    print(f"Train range: {run_summary['train_date_min']} -> {run_summary['train_date_max']}")
    print(f"Val range:   {run_summary['val_date_min']} -> {run_summary['val_date_max']}")
    print(f"Test range:  {run_summary['test_date_min']} -> {run_summary['test_date_max']}")
    print(f"Best model:  {final_model_name}")
    print(f"Best thr:    {float(final_threshold):.4f}")
    print(f"Best test F1:{float(test_summary['f1']):.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
