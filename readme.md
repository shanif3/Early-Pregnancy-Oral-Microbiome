# Early Pregnancy Marks Significant Shifts in the Oral Microbiome

Companion code and processed data for the manuscript.

## Overview
Longitudinal analysis of the oral microbiome across pregnancy:
- **Israel**: 346 women (T1/T2/T3)  
- **Russia**: 154 women (T2/T3, validation)

**Key results (brief):**
- Significant community shifts, strongest from **T1 → T2** (PERMANOVA; Bray–Curtis, Jaccard).
- **Alpha diversity** decreases across pregnancy (Shannon: T1 > T2 > T3).
- Taxonomic trends: ↓ **Verrucomicrobiota** (*A. muciniphila*), ↑ **Synergistota**; ↑ Gammaproteobacteria, ↓ Erysipelotrichia.
- Within-woman profiles are more stable than between-woman profiles.
- Maternal factors: strongest associations with **gluten-free diet**, plus **smoking history** and **conception method**.

## Repo structure
```
Analysis/                      # scripts (correlations, diversity, figures)
Paper_plots                    # figure PDFs (Figure 1–3)
Data/
  Israel/                      # processed taxa tables + metadata
  Russia/                      # processed taxa tables + metadata
Appendix_plots                 # figure PDFs ( supplements)

```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


## Run (examples)
```bash
python Analysis/SCRIPT_NAME.py
# set cohort = 'Israel' or 'Russia' in the script
```


## Data availability
- **Processed tables + metadata:** in `Data/` (this repo).  
- **Raw sequencing:** ENA/EBI accession **ERP143097**.

## Cite
```
Finkelstein S, Frishman S, Turjeman S, Shtossel O, Tikhonov E, Nuriel-Ohayon M, Pinto Y,
Popova P, Tkachuk AS, Vasukova EA, Anopova AD, Pustozerov EA, Pervunina TM, Grineva EN,
Hod M, Schwartz B, Hadar E, Koren O*, Louzoun Y*. Early Pregnancy Marks Significant Shifts in the Oral Microbiome. 2025.
```
## License
Add a `LICENSE` (e.g., MIT for code; CC BY 4.0 for figures).
