"""One-time preparation of the bundled section databases.

Reads a flat CSV export of the AISC Shapes Database (US customary units,
e.g. https://github.com/ambaker1/aisc-csv, v15.0) and writes the two trimmed
catalogs the package ships with:

    aisc_w_shapes.csv   rolled W-shapes (Type == "W")
    aisc_hss_square.csv square HSS (Type == "HSS", Ht == B; the round HSS
                        rows carry OD instead and are dropped by the Ht/B
                        filter)

W-shapes: rts and ho are not present in the source export; both are computed
from their exact definitions for doubly symmetric I-shapes:
    rts^2 = sqrt(Iy * Cw) / Sx        (AISC 360 Eq. F2-7)
    ho    = d - tf                    (distance between flange centroids)
Spot-checked against AISC Manual values (e.g. W18X35: rts = 1.51 in,
ho = 17.3 in).

Square HSS: every property is taken straight from the source. A and the
b/tdes, h/tdes slendernesses are the DESIGN-wall values (tdes = 0.93*tnom
for ERW product), which is what AISC 360 Chapters E, F and G require. The
flat widths are recovered as (b/tdes)*tdes, so the corner radii stay
consistent with the published ratios rather than being re-assumed here.

Usage:  python tools/prepare_sections_csv.py <path-to-Shapes-US.csv>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "src" / "frame_optimizer" / "sections" / "data"
OUT_W = DATA / "aisc_w_shapes.csv"
OUT_HSS = DATA / "aisc_hss_square.csv"

# source column -> bundled column
W_COLUMNS = {
    "AISC_Manual_Label": "name",
    "W": "weight_plf",
    "A": "A",
    "d": "d",
    "bf": "bf",
    "tf": "tf",
    "tw": "tw",
    "Ix": "Ix",
    "Zx": "Zx",
    "Sx": "Sx",
    "rx": "rx",
    "Iy": "Iy",
    "Zy": "Zy",
    "Sy": "Sy",
    "ry": "ry",
    "J": "J",
    "Cw": "Cw",
    "bf/2tf": "bf_2tf",
    "h/tw": "h_tw",
}

HSS_COLUMNS = {
    "AISC_Manual_Label": "name",
    "W": "weight_plf",
    "A": "A",
    "Ht": "Ht",
    "B": "B",
    "tnom": "tnom",
    "tdes": "tdes",
    "Ix": "Ix",
    "Zx": "Zx",
    "Sx": "Sx",
    "rx": "rx",
    "Iy": "Iy",
    "Zy": "Zy",
    "Sy": "Sy",
    "ry": "ry",
    "J": "J",
    "C": "C",
    "b/tdes": "b_tdes",
    "h/tdes": "h_tdes",
}


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col != "name":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["name"] = df["name"].str.upper().str.replace(" ", "", regex=False)
    return df.dropna().sort_values("weight_plf").reset_index(drop=True)


def prepare_w_shapes(df: pd.DataFrame) -> pd.DataFrame:
    w = df[df["Type"] == "W"].copy()
    w = _numeric(w[list(W_COLUMNS)].rename(columns=W_COLUMNS))
    w["rts"] = np.sqrt(np.sqrt(w["Iy"] * w["Cw"]) / w["Sx"])
    w["ho"] = w["d"] - w["tf"]
    return w


def prepare_hss_square(df: pd.DataFrame) -> pd.DataFrame:
    h = df[df["Type"] == "HSS"].copy()
    h = _numeric(h[list(HSS_COLUMNS)].rename(columns=HSS_COLUMNS))
    # square only: the round HSS rows have no Ht/B and were already dropped
    # by dropna(); this keeps the rectangular ones out too.
    return h[(h["Ht"] - h["B"]).abs() < 1e-9].reset_index(drop=True)


def main(src: str) -> None:
    df = pd.read_csv(src, low_memory=False)
    DATA.mkdir(parents=True, exist_ok=True)

    w = prepare_w_shapes(df)
    w.to_csv(OUT_W, index=False)
    print(f"Wrote {len(w)} W-shapes to {OUT_W}")

    hss = prepare_hss_square(df)
    hss.to_csv(OUT_HSS, index=False)
    print(f"Wrote {len(hss)} square HSS to {OUT_HSS}")
    print(f"  max wall slenderness b/tdes = {hss['b_tdes'].max():.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
