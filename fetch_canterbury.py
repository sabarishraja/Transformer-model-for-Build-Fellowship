# fetch_canterbury.py
import re, requests
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/22120/pg22120.txt"

def strip_gutenberg(txt: str) -> str:
    s = re.search(r"\*\*\* START OF(.*)\*\*\*", txt)
    e = re.search(r"\*\*\* END OF(.*)\*\*\*", txt)
    if s and e and s.end() < e.start():
        txt = txt[s.end():e.start()]
    return txt.replace("\r\n","\n").strip()

def strip_editorial_apparatus(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # remove italicized/underscore apparatus, e.g., _om._, _rest_
    text = re.sub(r"_(?:[^_]{0,40})_", "", text)
    # bare line-numbers
    text = re.sub(r"^\s*\d{1,5}\.?\s*$", "", text, flags=re.MULTILINE)
    # lines like "B. 1270. ..." or "2278. E. seen ..."
    text = re.sub(r"^\s*[A-Z]\.\s*\d{1,5}\.?.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d{1,5}\.\s*[A-Z]\..*$", "", text, flags=re.MULTILINE)
    # collapse
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

print("[data] downloading Canterbury (Middle English)…")
raw = requests.get(URL, timeout=60).text
core = strip_gutenberg(raw)
core = strip_editorial_apparatus(core)
Path("input.txt").write_text(core, encoding="utf-8")
print("[data] wrote input.txt with", len(core), "chars")
