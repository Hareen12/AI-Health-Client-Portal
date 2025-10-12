
# Mental Health Forms – Clinician Dashboard (Streamlit)

**Now supports CSV *and* PDF upload.** On upload, the app immediately shows a consolidated table of **all detected form scores** for analysis, plus per-patient cards.

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- PDF parsing is best-effort. Include text patterns like `phq9_q1: 2` in exported/fillable PDFs to enable automatic parsing. CSV is fully supported.
- Extend `scoring.py` to add SDQ, Brief COPE, Buss-Perry, FAM; they will automatically appear in the consolidated table once implemented.


### Demographic fields
- Optional CSV columns supported: `name`, `gender`, `dob`.
- PDF parser will also try to extract `Name:`, `Gender:`, and `DOB:` if present in text.


## Supported forms now
- PHQ-9 (severity bands + item 9 flag)
- GAD-7 (severity bands)
- PCL-5 (probable PTSD cutoff at 33)
- SDQ total difficulties + prosocial (bands: normal/borderline/abnormal)
- Brief COPE (14 subscales + composite coping styles)
- Buss-Perry Aggression (4 subscales)
- FAM (total)

## Triggers
- PHQ-9 item 9 > 0 → **Immediate risk screen**
- NLP risk cues present → **Immediate risk screen**
- PCL-5 ≥ 33 → probable PTSD (priority)
- GAD-7 ≥ 15 → severe anxiety (priority)
