"""Fetch historical flag SVGs from Wikimedia Commons.

Uses Special:FilePath redirects (most stable URL form) and saves to
data/raw_svg/hist-<code>.svg. Sleeps 1s between requests with a User-Agent
to avoid rate-limit issues.
"""
from __future__ import annotations

import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SVG_DIR = ROOT / "data" / "raw_svg"

UA = "flag2vec-research-bot/0.1 (https://github.com/Rome-1/flag2vec)"

# (hist code, Wikimedia filename). The Special:FilePath endpoint resolves to
# the canonical upload.wikimedia.org URL with hash prefix.
FLAGS = [
    ("ussr",                       "Flag of the Soviet Union.svg"),
    ("russian-empire",             "Romanov Flag.svg"),
    ("yugoslavia-sfry",            "Flag of Yugoslavia (1946–1992).svg"),
    ("yugoslavia-kingdom",         "Flag of the Kingdom of Yugoslavia.svg"),
    ("czechoslovakia",             "Flag of Czechoslovakia.svg"),
    ("ddr",                        "Flag of East Germany.svg"),
    ("rhodesia",                   "Flag of Rhodesia (1968–1979).svg"),
    ("south-africa-apartheid",     "Flag of South Africa (1928–1994).svg"),
    ("south-vietnam",              "Flag of South Vietnam.svg"),
    ("south-yemen",                "Flag of South Yemen.svg"),
    ("north-yemen",                "Flag of North Yemen.svg"),
    ("csa-stars-bars",             "Flag of the Confederate States (1861–1863).svg"),
    ("csa-battle",                 "Battle flag of the Confederate States of America.svg"),
    ("british-raj",                "British Raj Red Ensign.svg"),
    ("roc-mainland",               "Flag of the Republic of China.svg"),
    ("manchukuo",                  "Flag of Manchukuo.svg"),
    ("imperial-japan",             "Naval ensign of Japan (1889–1945).svg"),
    ("iran-pahlavi",               "State flag of Iran (1933–1964).svg"),
    ("khmer-republic",             "Flag of the Khmer Republic.svg"),
    ("democratic-kampuchea",       "Flag of Democratic Kampuchea.svg"),
    ("republic-of-texas",          "Flag of Texas (1839–1879).svg"),
    ("hawaii-kingdom",             "Flag of Hawaii.svg"),
    ("tibet-independent",          "Flag of Tibet.svg"),
    ("biafra",                     "Flag of Biafra.svg"),
    ("katanga",                    "Flag of Katanga.svg"),
    ("ottoman",                    "Flag of the Ottoman Empire (1844–1922).svg"),
    ("austria-hungary",            "Flag of Austria-Hungary (1869-1918).svg"),
    ("holy-roman-empire",          "Banner of the Holy Roman Emperor (after 1400).svg"),
    ("mongolia-bogd",              "Flag of Bogd Khaanate Mongolia.svg"),
    ("uar",                        "Flag of the United Arab Republic.svg"),
]


def fetch(filename: str) -> bytes:
    quoted = urllib.parse.quote(filename.replace(" ", "_"))
    url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{quoted}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    return data


def main() -> int:
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for code, filename in FLAGS:
        out = SVG_DIR / f"hist-{code}.svg"
        if out.exists() and out.stat().st_size > 200:
            print(f"  skip {code} (already present, {out.stat().st_size} bytes)")
            continue
        try:
            data = fetch(filename)
            if not data.startswith(b"<?xml") and b"<svg" not in data[:2000]:
                raise RuntimeError(f"not an SVG (got {len(data)} bytes, head={data[:80]!r})")
            out.write_bytes(data)
            print(f"  ok   {code:30s} -> {out.name} ({len(data)} bytes)")
        except Exception as e:
            failed.append((code, filename, str(e)))
            print(f"  FAIL {code:30s}: {e}", file=sys.stderr)
        time.sleep(1.0)

    if failed:
        print(f"\n{len(failed)} failures:", file=sys.stderr)
        for c, f, e in failed:
            print(f"  {c} ({f}): {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
