"""
Unified dataset reader — bytes to DataFrame, one code path.

Before this module, five near-identical `_read_df` functions were
duplicated across src/tools/*.py and src/core/controller.py, one of them
(controller's) not even knowing about .tsv. That divergence is exactly how
a semicolon- or tab-delimited export silently parses as a single column:
a confident, complete analysis of data that doesn't actually exist.

Detection chain: extension -> format, encoding, delimiter, header
validation. Every guess is recorded on the returned ReadReport instead of
being applied silently — the report is what item 6 (report restructure)
surfaces as "what was detected vs assumed at read time".

Pure I/O + parsing. No LLM calls, no profiling, no coercion (src.core.coercion).
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

#: Extensions read_any knows how to ingest. The single source of truth —
#: src.core.security.ALLOWED_EXTENSIONS and the Streamlit uploader's type
#: list both derive from this instead of maintaining their own copies.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".tsv", ".xlsx", ".xls"})

#: Delimiters considered when sniffing a .csv file's separator.
_CSV_DELIMITER_CANDIDATES = ",;\t|"

#: Bytes read from the front of a text file to sniff its delimiter/headers.
_SNIFF_SAMPLE_BYTES = 64 * 1024

_FORMAT_BY_SUFFIX = {".csv": "csv", ".tsv": "tsv", ".xlsx": "xlsx", ".xls": "xls"}


@dataclass
class ReadReport:
    """What read_any actually did — surfaced, never assumed."""

    path: str
    format: str                    # csv | tsv | xlsx | xls
    encoding: str
    encoding_confident: bool       # False when guessed via the cp1252 fallback
    delimiter: str | None
    delimiter_sniffed: bool        # False when it came from the extension
    duplicate_headers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class DatasetReadError(Exception):
    """Raised when a dataset cannot be read at all: empty, corrupt, or an
    unsupported format. Never raised for data that merely looks unusual —
    that's the profiler's and coercion's job to flag, not the reader's."""


#: Within this much chaos of the top candidate, prefer cp1252 over an
#: equally-clean but far rarer codepage — charset_normalizer's tiebreak is a
#: generic multi-language coherence score that has no special preference for
#: cp1252 even though it's overwhelmingly the real-world encoding behind
#: "Excel European export" files (the exact case this detector exists for).
_CP1252_TIEBREAK_MARGIN = 0.05


def _detect_encoding(raw: bytes) -> tuple[str, bool, list[str]]:
    notes: list[str] = []
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", True, notes
    try:
        raw.decode("utf-8")
        return "utf-8", True, notes
    except UnicodeDecodeError:
        pass
    try:
        from charset_normalizer import from_bytes
        matches = from_bytes(raw)
        best = matches.best()
    except Exception:
        matches, best = None, None
    if best is not None and best.encoding:
        encoding = best.encoding
        if encoding != "cp1252" and matches is not None:
            for candidate in matches:
                if candidate.encoding == "cp1252" and candidate.chaos <= best.chaos + _CP1252_TIEBREAK_MARGIN:
                    encoding = "cp1252"
                    break
        notes.append(f"Encoding auto-detected as {encoding} (file is not valid UTF-8).")
        return encoding, True, notes
    notes.append(
        "Encoding could not be confidently detected; assumed cp1252. "
        "Re-export as UTF-8 if any characters look wrong."
    )
    return "cp1252", False, notes


def _sniff_delimiter(text_sample: str) -> tuple[str, bool]:
    try:
        dialect = csv.Sniffer().sniff(text_sample, delimiters=_CSV_DELIMITER_CANDIDATES)
        return dialect.delimiter, True
    except csv.Error:
        # Genuinely single-column data (or too small a sample) — csv.Sniffer
        # raises rather than returning a best guess. Comma is a safe default:
        # a single-column file has no delimiter to get wrong either way.
        return ",", False


def _find_duplicate_headers(text_sample: str, delimiter: str) -> list[str]:
    first_line = text_sample.splitlines()[0] if text_sample else ""
    try:
        names = next(csv.reader([first_line], delimiter=delimiter))
    except (csv.Error, StopIteration):
        return []
    seen: set[str] = set()
    dupes: list[str] = []
    for name in names:
        if name in seen and name not in dupes:
            dupes.append(name)
        seen.add(name)
    return dupes


def _read_delimited(path: Path, format_: str) -> tuple[pd.DataFrame, ReadReport]:
    raw = path.read_bytes()
    if not raw:
        raise DatasetReadError(f"'{path.name}' is empty — no data to read.")

    encoding, encoding_confident, notes = _detect_encoding(raw)
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise DatasetReadError(
            f"'{path.name}' could not be decoded as {encoding}: {exc}"
        ) from exc
    sample = text[:_SNIFF_SAMPLE_BYTES]

    if format_ == "tsv":
        # The extension is unambiguous — sniffing a small sample of quoted
        # or numeric text is what mis-detects, so trust the extension.
        delimiter, delimiter_sniffed = "\t", False
    else:
        delimiter, delimiter_sniffed = _sniff_delimiter(sample)

    duplicate_headers = _find_duplicate_headers(sample, delimiter)
    if duplicate_headers:
        notes.append(
            f"Duplicate column name(s) {duplicate_headers} were disambiguated "
            "(e.g. 'a' -> 'a.1')."
        )

    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter)
    except pd.errors.EmptyDataError as exc:
        raise DatasetReadError(f"'{path.name}' has no columns to parse.") from exc

    report = ReadReport(
        path=str(path),
        format=format_,
        encoding=encoding,
        encoding_confident=encoding_confident,
        delimiter=delimiter,
        delimiter_sniffed=delimiter_sniffed,
        duplicate_headers=duplicate_headers,
        notes=notes,
    )
    return df, report


def _read_excel(path: Path, format_: str) -> tuple[pd.DataFrame, ReadReport]:
    engine = "openpyxl" if format_ == "xlsx" else "xlrd"
    try:
        df = pd.read_excel(path, engine=engine)
    except Exception as exc:
        raise DatasetReadError(f"'{path.name}' could not be read as {format_}: {exc}") from exc
    report = ReadReport(
        path=str(path),
        format=format_,
        encoding="n/a",
        encoding_confident=True,
        delimiter=None,
        delimiter_sniffed=False,
    )
    return df, report


def read_any(file_path: str) -> tuple[pd.DataFrame, ReadReport]:
    """
    Read a dataset from disk, detecting format/encoding/delimiter.

    Raises:
        DatasetReadError: file is unsupported, empty, or unreadable as
            claimed (corrupt Excel container, undecodable text, ...).
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    format_ = _FORMAT_BY_SUFFIX.get(suffix)
    if format_ is None:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DatasetReadError(
            f"Unsupported file extension '{path.suffix or '(none)'}'. Supported: {allowed}."
        )
    if format_ in ("csv", "tsv"):
        return _read_delimited(path, format_)
    return _read_excel(path, format_)
