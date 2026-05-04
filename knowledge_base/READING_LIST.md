# APEXA Knowledge Base — Curated Reading List

Authoritative sources for HEDM, WAXS, GSAS-II, MIDAS, and pyFAI domains.
Drop the corresponding PDF into `knowledge_base/papers/` (filename suggested in
the **File** column). Each PDF should have a sibling `.bib` file with the same
stem (the indexer reads it for citation metadata). When no `.bib` exists, the
indexer falls back to PDF metadata + first-page DOI regex — it works, but the
sidecar is more reliable.

Status legend: ✅ indexed (PDF) · 📇 metadata-only (no PDF, citation-searchable) · ⬜ not present

---

## MIDAS / 3DXRD core methodology

| Status | File                                | Citation                                                                                       | DOI                                  |
|--------|-------------------------------------|------------------------------------------------------------------------------------------------|--------------------------------------|
| ✅     | `HEDM-I.pdf`                        | Sharma, Huizenga & Offerman (2012a). *J. Appl. Cryst.* **45**:693–704                          | 10.1107/S0021889812025563            |
| ✅     | `HEDM-II.pdf`                       | Sharma, Huizenga & Offerman (2012b). *J. Appl. Cryst.* **45**:705–718                          | 10.1107/S0021889812025599            |
| ⬜     | `Sharma2024_repeatability.pdf`      | Sharma et al. (2024). MIDAS repeatability/sensitivity. IUCr `fv5137`                           | (look up on IUCr)                    |
| ✅     | `Lienert2011_JOM.pdf`               | Lienert et al. (2011). HEDM at the Advanced Photon Source. *JOM* **63**:70–77                  | 10.1007/s11837-011-0116-0            |

## HEDM reviews & forward modeling

| Status | File                                | Citation                                                                                            | DOI                                  |
|--------|-------------------------------------|-----------------------------------------------------------------------------------------------------|--------------------------------------|
| ✅     | `Bernier2020_AnnRev.pdf`            | Bernier, Suter, Rollett & Almer (2020). *Annu. Rev. Mater. Res.* **50**:395–436                     | 10.1146/annurev-matsci-070616-124125 |
| 📇     | `Bernier2011_FFHEDM.bib`            | Bernier, Barton, Lienert & Miller (2011). *J. Strain Anal. Eng. Des.* **46**:527–547                | 10.1177/0309324711405761             |
| 📇     | `Suter2006_NF.bib`                  | Suter, Hennessy, Xiao & Lienert (2006). *Rev. Sci. Instrum.* **77**:123905 (NF-HEDM forward model)  | 10.1063/1.2400017                    |
| 📇     | `Poulsen2004_3DXRD.bib`             | Poulsen (2004). *Three-Dimensional X-Ray Diffraction Microscopy* (Springer). Foundational text.     | 10.1007/b97884                       |

## Coherent / multiscale (BCDI bridge)

| Status | File                                | Citation                                                                                | DOI                              |
|--------|-------------------------------------|-----------------------------------------------------------------------------------------|----------------------------------|
| ✅     | `PhysRevApplied.14.024085.pdf`      | Maddali et al. (2020). *Phys. Rev. Applied* **14**:024085                               | 10.1103/PhysRevApplied.14.024085 |

## In-situ / application case studies

| Status | File                                | Citation                                                                                | DOI                          |
|--------|-------------------------------------|-----------------------------------------------------------------------------------------|------------------------------|
| ✅     | `vb5013.pdf`                        | Moslehy et al. (2021). *J. Appl. Cryst.* **54**:1379–1393 (rock salt)                   | 10.1107/S1600576721007809    |

## GSAS-II (Rietveld + powder)

| Status | File                                | Citation                                                                                                                      | DOI                          |
|--------|-------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| ✅     | `Toby2013_GSASII.pdf`               | Toby & Von Dreele (2013). GSAS-II genesis. *J. Appl. Cryst.* **46**:544–549                                                   | 10.1107/S0021889813003531    |
| ✅     | `Park2024_SharkVertebra.pdf`        | Park et al. (2024). Energy-dispersive diffraction tomography of shark vertebral centra. *Powder Diffr.* **39**:69–75          | 10.1017/S0885715624000307    |

## WAXS / azimuthal integration (pyFAI)

| Status | File                                | Citation                                                                                          | DOI                                 |
|--------|-------------------------------------|---------------------------------------------------------------------------------------------------|-------------------------------------|
| ✅     | `Kieffer2013_pyFAI.pdf`             | Kieffer & Karkoulis (2013). pyFAI azimuthal regrouping. *J. Phys. Conf. Ser.* **425**:202012      | 10.1088/1742-6596/425/20/202012     |
| ✅     | `Ashiotis2015_pyFAI.pdf`            | Ashiotis et al. (2015). The fast azimuthal integration Python library: pyFAI. *J. Appl. Cryst.* **48**:510–519 | 10.1107/S1600576715004306 |

---

## Code repositories (referenced in citations, not indexed as PDFs)

- MIDAS — https://github.com/marinerhemant/MIDAS
- GSAS-II — https://github.com/AdvancedPhotonSource/GSAS-II
- pyFAI — https://github.com/silx-kit/pyFAI
- HEXRD — https://github.com/HEXRD/hexrd

---

## How to add a paper

1. Drop `Foo2024.pdf` into `knowledge_base/papers/`.
2. Create `knowledge_base/papers/Foo2024.bib` with one BibTeX entry. Required keys:
   `author`, `title`, `journal`, `year`, `doi`. Optional: `volume`, `pages`, `topics`.
3. Run `uv run python knowledge_base/index_knowledge.py` to re-index.
4. The MCP `query_hedm_knowledge` tool will return chunks with formatted citations,
   and `get_bibtex` returns the `.bib` entry for the source.
