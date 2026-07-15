"""One-time preparation of the bundled W-shape section database.

Reads a flat CSV export of the AISC Shapes Database (US customary units,
e.g. https://github.com/ambaker1/aisc-csv, v15.0), filters to W-shapes, and
writes the trimmed catalog the package ships with.

rts and ho are not present in the source export; both are computed from
their exact definitions for doubly symmetric I-shapes:
    rts^2 = sqrt(Iy * Cw) / Sx        (AISC 360 Eq. F2-7)
    ho    = d - tf                    (distance between flange centroids)
Spot-checked against AISC Manual values (e.g. W18X35: rts = 1.51 in,
ho = 17.3 in).

Usage:  python tools/prepare_sections_csv.py <path-to-Shapes-US.csv>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "src" / "frame_optimizer" / "sections" / "data" / "aisc_w_shapes.csv"

# source column -> bundled column
COLUMNS = {
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


def main(src: str) -> None:
    df = pd.read_csv(src)
    w = df[df["Type"] == "W"].copy()
    w = w[list(COLUMNS)].rename(columns=COLUMNS)
    for col in w.columns:
        if col != "name":
            w[col] = pd.to_numeric(w[col], errors="coerce")
    w["name"] = w["name"].str.upper().str.replace(" ", "", regex=False)
    w["rts"] = np.sqrt(np.sqrt(w["Iy"] * w["Cw"]) / w["Sx"])
    w["ho"] = w["d"] - w["tf"]
    w = w.dropna().sort_values("weight_plf").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(OUT, index=False)
    print(f"Wrote {len(w)} W-shapes to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
