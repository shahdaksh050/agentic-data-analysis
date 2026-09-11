"""
Edge-case dataset corpus — programmatic frame/file factories.

Each factory writes a file into the given directory (normally pytest's
`tmp_path`) and returns its Path. Nothing here is committed as a binary
fixture; everything is generated on demand.

See tests/test_data_shapes.py for the matrix test these back.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_RNG = np.random.default_rng(42)


def tsv_file(tmp_path: Path) -> Path:
    """3x3 tab-separated file — the classic Excel European export trap."""
    p = tmp_path / "data.tsv"
    p.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n", encoding="utf-8")
    return p


def semicolon_csv(tmp_path: Path) -> Path:
    """3x3 semicolon-delimited .csv — EU-locale Excel's default CSV export."""
    p = tmp_path / "data.csv"
    p.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")
    return p


def cp1252_csv(tmp_path: Path) -> Path:
    """Latin-1/cp1252 accented French text — not valid UTF-8. Real prose
    (not just a couple of accented names) so a statistical encoding
    detector has enough signal to place it confidently."""
    sentences = [
        "Le système français a été conçu pour améliorer la qualité générale des données",
        "Notre équipe a déjà vérifié plusieurs résultats importants cette année académique",
        "La sécurité reste une priorité absolue pour chaque projet réalisé ici même",
        "François et José ont présenté leurs conclusions à Zürich et à Montréal",
        "Cette étude a été menée près de la côte avec une équipe très expérimentée",
    ]
    content = "note\n" + "\n".join(sentences * 4) + "\n"
    p = tmp_path / "data.csv"
    p.write_bytes(content.encode("cp1252"))
    return p


def json_records(tmp_path: Path) -> Path:
    """`[{...}, {...}]` records-oriented JSON — unsupported today (item 8)."""
    p = tmp_path / "data.json"
    p.write_text('[{"a": 1, "b": 2}, {"a": 3, "b": 4}]', encoding="utf-8")
    return p


def empty_file(tmp_path: Path) -> Path:
    """0-byte file."""
    p = tmp_path / "empty.csv"
    p.write_bytes(b"")
    return p


def headers_only(tmp_path: Path) -> Path:
    """Header row, zero data rows."""
    p = tmp_path / "headers_only.csv"
    p.write_text("a,b,c\n", encoding="utf-8")
    return p


def single_row(tmp_path: Path) -> Path:
    """Exactly one data row."""
    p = tmp_path / "single_row.csv"
    p.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    return p


def single_column(tmp_path: Path) -> Path:
    """One column x 50 rows."""
    df = pd.DataFrame({"value": _RNG.normal(0, 1, 50)})
    p = tmp_path / "single_column.csv"
    df.to_csv(p, index=False)
    return p


def all_categorical(tmp_path: Path) -> Path:
    """No numeric columns at all."""
    df = pd.DataFrame({
        "color": _RNG.choice(["red", "green", "blue"], 40),
        "size": _RNG.choice(["S", "M", "L"], 40),
    })
    p = tmp_path / "all_categorical.csv"
    df.to_csv(p, index=False)
    return p


def all_text(tmp_path: Path) -> Path:
    """Varied free-text prose across two columns."""
    reviews = [
        "This product exceeded my expectations in almost every way imaginable",
        "Terrible experience, the packaging was damaged and support was unhelpful",
        "Solid value for the price, would recommend to a friend without hesitation",
        "Not what I expected at all, the description was misleading and vague",
        "Absolutely fantastic quality, fast shipping, and great customer service",
        "Mediocre at best, works but nothing special about the overall design",
        "I have used this daily for months and it still performs beautifully",
        "Broke within a week of light use, very disappointed with the durability",
    ]
    df = pd.DataFrame({
        "review": (reviews * 5)[:40],
        "comment": (reviews[::-1] * 5)[:40],
    })
    p = tmp_path / "all_text.csv"
    df.to_csv(p, index=False)
    return p


def mixed_type_column(tmp_path: Path) -> Path:
    """A column with a stray non-numeric literal: 1,2,NOT_A_NUMBER,4."""
    p = tmp_path / "mixed_type.csv"
    p.write_text("id,value\n1,1\n2,2\n3,NOT_A_NUMBER\n4,4\n", encoding="utf-8")
    return p


def duplicate_headers(tmp_path: Path) -> Path:
    """Header row with a repeated column name: a,a,b."""
    p = tmp_path / "duplicate_headers.csv"
    p.write_text("a,a,b\n1,2,3\n4,5,6\n", encoding="utf-8")
    return p


def high_missing(tmp_path: Path) -> Path:
    """~60% missing cells."""
    n = 50
    vals = _RNG.normal(0, 1, n)
    mask = _RNG.random(n) < 0.6
    vals = vals.astype(object)
    vals[mask] = None
    df = pd.DataFrame({"a": vals, "b": range(n)})
    p = tmp_path / "high_missing.csv"
    df.to_csv(p, index=False)
    return p


def wide_frame(tmp_path: Path) -> Path:
    """200 numeric columns."""
    df = pd.DataFrame(_RNG.normal(0, 1, (30, 200)), columns=[f"f{i}" for i in range(200)])
    p = tmp_path / "wide_frame.csv"
    df.to_csv(p, index=False)
    return p


def million_rows(tmp_path: Path) -> Path:
    """1,000,000 rows x 4 columns. Slow — see @pytest.mark.slow usage."""
    n = 1_000_000
    df = pd.DataFrame({
        "id": range(n),
        "value": _RNG.normal(0, 1, n),
        "category": _RNG.choice(["a", "b", "c"], n),
        "flag": _RNG.choice([True, False], n),
    })
    p = tmp_path / "million_rows.csv"
    df.to_csv(p, index=False)
    return p


def time_series(tmp_path: Path) -> Path:
    """Daily datetime column + numeric measurement."""
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    df = pd.DataFrame({"date": dates, "value": _RNG.normal(0, 1, 60)})
    p = tmp_path / "time_series.csv"
    df.to_csv(p, index=False)
    return p


def panel(tmp_path: Path) -> Path:
    """Datetime + a low-cardinality group column (panel/grouped structure)."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "date": list(dates) * 3,
        "group": ["a"] * 30 + ["b"] * 30 + ["c"] * 30,
        "value": _RNG.normal(0, 1, 90),
    })
    p = tmp_path / "panel.csv"
    df.to_csv(p, index=False)
    return p


def geo(tmp_path: Path) -> Path:
    """Latitude/longitude columns."""
    df = pd.DataFrame({
        "lat": _RNG.uniform(-90, 90, 40),
        "lon": _RNG.uniform(-180, 180, 40),
        "value": _RNG.normal(0, 1, 40),
    })
    p = tmp_path / "geo.csv"
    df.to_csv(p, index=False)
    return p


def high_cardinality(tmp_path: Path) -> Path:
    """A categorical column with far more than HIGH_CARDINALITY_THRESHOLD
    levels but repeats (so it's genuinely categorical, not identifier-like)."""
    df = pd.DataFrame({
        "sku": _RNG.choice([f"SKU-{i}" for i in range(80)], 200),
        "value": _RNG.normal(0, 1, 200),
    })
    p = tmp_path / "high_cardinality.csv"
    df.to_csv(p, index=False)
    return p
