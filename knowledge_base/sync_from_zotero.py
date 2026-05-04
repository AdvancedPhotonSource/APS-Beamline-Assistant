#!/usr/bin/env python3
"""
Sync papers + citations from a Zotero collection into knowledge_base/papers/.

Walks the local Zotero SQLite DB (~/Zotero/zotero.sqlite), finds the named
collection, and for every item:
  - copies the attached PDF (if any) to papers/<Author><Year>_<slug>.pdf
  - writes a sibling <stem>.bib with author/year/journal/DOI for the indexer

Usage:
    uv run python knowledge_base/sync_from_zotero.py
    uv run python knowledge_base/sync_from_zotero.py --collection "Pawan/APEXA/KnowledgeBase"
    uv run python knowledge_base/sync_from_zotero.py --reindex   # also rebuild ChromaDB

Notes:
    - Zotero often holds an exclusive lock on zotero.sqlite while running.
      We work on a temp snapshot copy to avoid the lock.
    - Filenames are deterministic from first-author + year; safe to re-run.
    - Existing PDFs are overwritten only if the source is newer.
"""

from __future__ import annotations
import argparse
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

ZOTERO_DIR = Path.home() / "Zotero"
DEFAULT_COLLECTION = "GroupMembers/Pawan/APEXA/KnowledgeBase"
PAPERS_DIR = Path(__file__).parent / "papers"


def snapshot_db() -> Path:
    """Copy the live Zotero DB to a temp file so we don't fight the lock."""
    src = ZOTERO_DIR / "zotero.sqlite"
    if not src.exists():
        sys.exit(f"❌ No Zotero DB at {src}")
    fd, tmp = tempfile.mkstemp(prefix="zotero_snap_", suffix=".sqlite")
    Path(tmp).unlink()  # mkstemp creates the file; shutil.copy will replace
    shutil.copy2(src, tmp)
    return Path(tmp)


def find_collection_id(con: sqlite3.Connection, path: str) -> int:
    """Resolve 'Parent/Child/Grandchild' into the leaf collectionID."""
    parts = [p.strip() for p in path.split("/") if p.strip()]
    if not parts:
        sys.exit("❌ Empty collection path")
    parent_id: Optional[int] = None
    for part in parts:
        if parent_id is None:
            row = con.execute(
                "SELECT collectionID FROM collections WHERE collectionName=? AND parentCollectionID IS NULL",
                (part,),
            ).fetchone()
        else:
            row = con.execute(
                "SELECT collectionID FROM collections WHERE collectionName=? AND parentCollectionID=?",
                (part, parent_id),
            ).fetchone()
        if not row:
            # Show siblings to help debug
            sibs = con.execute(
                "SELECT collectionName FROM collections WHERE parentCollectionID IS ?",
                (parent_id,),
            ).fetchall()
            sib_names = ", ".join(s[0] for s in sibs) or "(none)"
            sys.exit(f"❌ Collection '{part}' not found under {parts[:parts.index(part)]}.\n   Siblings: {sib_names}")
        parent_id = row[0]
    return parent_id  # type: ignore


def get_field(con: sqlite3.Connection, item_id: int, name: str) -> str:
    row = con.execute(
        """SELECT idv.value FROM itemData id
           JOIN itemDataValues idv ON id.valueID=idv.valueID
           JOIN fields f ON id.fieldID=f.fieldID
           WHERE id.itemID=? AND f.fieldName=?""",
        (item_id, name),
    ).fetchone()
    return row[0] if row else ""


def get_authors(con: sqlite3.Connection, item_id: int) -> list[tuple[str, str]]:
    """Return ordered list of (lastName, firstName) tuples."""
    rows = con.execute(
        """SELECT c.firstName, c.lastName FROM itemCreators ic
           JOIN creators c ON c.creatorID=ic.creatorID
           WHERE ic.itemID=? ORDER BY ic.orderIndex""",
        (item_id,),
    ).fetchall()
    return [(r[1] or "", r[0] or "") for r in rows]


def get_pdf_attachment(con: sqlite3.Connection, item_id: int) -> Optional[Path]:
    row = con.execute(
        """SELECT attach.key, ia.path FROM itemAttachments ia
           JOIN items attach ON ia.itemID=attach.itemID
           WHERE ia.parentItemID=? AND ia.contentType='application/pdf'
           LIMIT 1""",
        (item_id,),
    ).fetchone()
    if not row:
        return None
    key, path = row[0], row[1] or ""
    if not path.startswith("storage:"):
        return None
    return ZOTERO_DIR / "storage" / key / path[len("storage:"):]


def clean_title(t: str) -> str:
    return re.sub(r"<[^>]+>", "", t).strip()


def year_from_date(d: str) -> str:
    m = re.search(r"\b(19|20)\d{2}\b", d or "")
    return m.group(0) if m else ""


_SLUG_BAD = re.compile(r"[^A-Za-z0-9]+")


def make_stem(authors: list[tuple[str, str]], year: str, title: str) -> str:
    """e.g. ('Sharma',...), '2012', 'A fast methodology...' -> 'Sharma2012_FastMethodology'"""
    if authors:
        last = _SLUG_BAD.sub("", authors[0][0]) or "Unknown"
    else:
        last = "Unknown"
    yr = year or "ND"
    title_words = [w for w in re.split(r"\s+", clean_title(title)) if w and w.lower() not in
                   {"a", "an", "the", "of", "for", "and", "in", "on", "to", "with", "using"}]
    slug = "".join(w.capitalize() for w in title_words[:3])
    slug = _SLUG_BAD.sub("", slug)
    return f"{last}{yr}" + (f"_{slug}" if slug else "")


def make_bibkey(authors: list[tuple[str, str]], year: str) -> str:
    if authors:
        last = _SLUG_BAD.sub("", authors[0][0]) or "Unknown"
    else:
        last = "Unknown"
    return f"{last}{year or 'ND'}"


def write_bib(stem_path: Path, item_type: str, bibkey: str,
              authors: list[tuple[str, str]], title: str, journal: str,
              volume: str, pages: str, year: str, doi: str) -> None:
    auth_str = " and ".join(f"{ln}, {fn}" for ln, fn in authors)
    entry_type = "book" if item_type == "book" else "article"
    lines = [f"@{entry_type}{{{bibkey},"]
    lines.append(f"  author  = {{{auth_str}}},")
    lines.append(f"  title   = {{{title}}},")
    if journal: lines.append(f"  journal = {{{journal}}},")
    if volume:  lines.append(f"  volume  = {{{volume}}},")
    if pages:   lines.append(f"  pages   = {{{pages}}},")
    if year:    lines.append(f"  year    = {{{year}}},")
    if doi:     lines.append(f"  doi     = {{{doi}}},")
    # Remove trailing comma on last field
    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    lines.append("}\n")
    stem_path.write_text("\n".join(lines), encoding="utf-8")


def sync(collection_path: str, dry_run: bool = False) -> tuple[int, int]:
    """Returns (papers_imported, pdfs_copied)."""
    snap = snapshot_db()
    try:
        con = sqlite3.connect(snap)
        cid = find_collection_id(con, collection_path)
        items = con.execute(
            "SELECT i.itemID, it.typeName FROM collectionItems ci "
            "JOIN items i ON ci.itemID=i.itemID "
            "JOIN itemTypes it ON i.itemTypeID=it.itemTypeID "
            "WHERE ci.collectionID=?",
            (cid,),
        ).fetchall()
        if not items:
            sys.exit(f"❌ Collection '{collection_path}' (id {cid}) has no items")
        print(f"Found {len(items)} items in '{collection_path}' (id {cid})\n")
        PAPERS_DIR.mkdir(exist_ok=True)

        papers, pdfs = 0, 0
        for iid, itype in items:
            authors = get_authors(con, iid)
            title = clean_title(get_field(con, iid, "title"))
            year = year_from_date(get_field(con, iid, "date"))
            doi = get_field(con, iid, "DOI")
            journal = get_field(con, iid, "publicationTitle")
            volume = get_field(con, iid, "volume")
            pages = get_field(con, iid, "pages")

            stem = make_stem(authors, year, title)
            bibkey = make_bibkey(authors, year)

            bib_path = PAPERS_DIR / f"{stem}.bib"
            if dry_run:
                print(f"  · would write {bib_path.name}")
            else:
                write_bib(bib_path, itype, bibkey, authors, title, journal,
                          volume, pages, year, doi)
                print(f"  ✓ {bib_path.name}")

            src_pdf = get_pdf_attachment(con, iid)
            if src_pdf and src_pdf.exists():
                dst_pdf = PAPERS_DIR / f"{stem}.pdf"
                size_kb = src_pdf.stat().st_size // 1024
                if dry_run:
                    print(f"  · would copy {dst_pdf.name}  ({size_kb} KB)")
                elif not dst_pdf.exists() or src_pdf.stat().st_mtime > dst_pdf.stat().st_mtime:
                    shutil.copy2(src_pdf, dst_pdf)
                    print(f"  ✓ {dst_pdf.name}  ({size_kb} KB)")
                else:
                    print(f"    {dst_pdf.name}  (up-to-date, skipped)")
                pdfs += 1
            else:
                print(f"    (no PDF in Zotero — citation only)")
            papers += 1
            print()
        con.close()
        return papers, pdfs
    finally:
        snap.unlink(missing_ok=True)


def reindex() -> None:
    print("Re-indexing ChromaDB...")
    indexer = Path(__file__).parent / "index_knowledge.py"
    subprocess.run([sys.executable, str(indexer)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help=f"Slash-separated Zotero collection path (default: {DEFAULT_COLLECTION!r})")
    ap.add_argument("--reindex", action="store_true",
                    help="Re-run index_knowledge.py after sync")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be written without touching the filesystem")
    args = ap.parse_args()

    papers, pdfs = sync(args.collection, dry_run=args.dry_run)
    print("=" * 60)
    print(f"✅ Sync complete: {papers} citations written, {pdfs} PDFs copied")
    print(f"   Output: {PAPERS_DIR}")
    print("=" * 60)

    if args.reindex:
        print()
        reindex()
    else:
        print("\nRun the indexer to update the vector DB:")
        print("  uv run python knowledge_base/index_knowledge.py")


if __name__ == "__main__":
    main()
