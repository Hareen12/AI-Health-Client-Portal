# app.py (v2) — adds CSV template + charts
# pip install: streamlit PyPDF2 reportlab pandas numpy python-dateutil matplotlib

import io
import re
import json
import math
import base64
from datetime import datetime, date
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import streamlit as st

# NEW: charts
import matplotlib.pyplot as plt

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
except Exception:
    canvas = None

st.set_page_config(page_title="Mental Health Forms Analyzer", layout="wide")

# ---------------------------
# Constants & Mappings
# ---------------------------

PHQ9_ITEMS = [f"phq{i}" for i in range(1, 10)]
PHQ9_SEVERITY = [(0,4,"Minimal"),(5,9,"Mild"),(10,14,"Moderate"),(15,19,"Moderately severe"),(20,27,"Severe")]

GAD7_ITEMS = [f"gad{i}" for i in range(1, 8)]
GAD7_SEVERITY = [(0,4,"Minimal"),(5,9,"Mild"),(10,14,"Moderate"),(15,21,"Severe")]

PCL5_ITEMS = [f"pcl{i}" for i in range(1, 21)]
PCL5_SUGGESTED_CUTOFF = 33

BRIEF_COPE_ITEMS = [f"brief_{i}" for i in range(1, 29)]
BRIEF_COPE_SCALES = {
    "Self-Distraction":[1,19],"Active Coping":[2,7],"Denial":[3,8],"Substance Use":[4,11],
    "Emotional Support":[5,15],"Behavioral Disengagement":[6,16],"Positive Reframing":[12,17],
    "Planning":[14,25],"Humor":[18,28],"Acceptance":[20,24],"Religion":[22,27],
    "Self-Blame":[13,26],"Vent/Emotional Expression":[9,21],"Instrumental Support":[10,23]
}

BP_ITEMS = [f"bp_{i}" for i in range(1, 30)]
BP_REVERSE = {7,18}
BP_SCALES = {
    "Physical Aggression":[1,2,3,4,5,6,7,8,9],
    "Verbal Aggression":[10,11,12,13,14],
    "Anger":[15,16,17,18,19,20,21],
    "Hostility":[22,23,24,25,26,27,28,29]
}

FAM_ITEMS = [f"fam_{i}" for i in range(1, 31)]

SDQ_FLAGS = [
    "sdq_kindness","sdq_restless","sdq_somatic","sdq_share","sdq_tantrum",
    "sdq_alone","sdq_obedient","sdq_worry","sdq_helpful","sdq_fidget",
    "sdq_friend","sdq_fight","sdq_unhappy","sdq_liked","sdq_distracted",
    "sdq_nervous","sdq_kind_young","sdq_lying","sdq_bullied","sdq_volunteer",
    "sdq_think_before","sdq_take_things","sdq_get_along_adults","sdq_fears",
    "sdq_finish_work"
]

PERSONAL_FIELDS = ["name","gender","dob"]
FREE_TEXT_FIELDS = ["free_text","notes","comments","chief_complaint","story","summary"]

SUICIDE_CUES = {
    "ideation":[
        "suicide","kill myself","end my life","better off dead","hurt myself",
        "self-harm","self harm","cut myself","jump off","overdose","wish i were dead",
        "no reason to live","die","ending it"
    ],
    "immediate_action_phrases":[ "right now","tonight","this week","i will do it","i have a plan" ]
}

POS_WORDS = set("calm hopeful safe supported relieved encouraged optimistic good okay fine better happy joy satisfied".split())
NEG_WORDS = set("sad hopeless worthless tired anxious panicked afraid scared angry ashamed guilty irritable restless depressed miserable terrible horrible bad worse crying alone empty helpless".split())
STOPWORDS = set("a an the and or but if on in at to for from by of with without about this that those these is are was were be been am i you he she they we it as into over under again same just very more most less least own so too not only than then once doing being having".split())

# ---------------------------
# Utils
# ---------------------------

def safe_int(x):
    try:
        if x is None or (isinstance(x,float) and math.isnan(x)): return None
        return int(str(x).strip())
    except: return None

def classify_band(score, bands):
    for lo,hi,label in bands:
        if lo<=score<=hi: return label
    return "Unclassified"

def quick_sentiment(text):
    words = [w for w in re.findall(r"[a-zA-Z']+", text.lower()) if w not in STOPWORDS]
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)
    s = pos - neg
    label = "Neutral"
    if s>=2: label="Positive"
    elif s<=-2: label="Negative"
    return {"pos_hits":pos,"neg_hits":neg,"score":s,"label":label}

def keyword_tags(text, top_k=8):
    words = [w for w in re.findall(r"[a-zA-Z']+", text.lower()) if w not in STOPWORDS and len(w)>2]
    return [w for w,_ in Counter(words).most_common(top_k)]

def short_summary(text, max_sent=2):
    sents = [s.strip() for s in re.split(r"[\.!?]\s+", text) if s.strip()]
    return " ".join(sents[:max_sent])

def suicide_screen(text):
    t = text.lower()
    ideation = any(p in t for p in SUICIDE_CUES["ideation"])
    imminent = ideation and any(p in t for p in SUICIDE_CUES["immediate_action_phrases"])
    return {"ideation":ideation,"imminent_language":imminent}

def read_pdf_form_fields(file_bytes):
    if PdfReader is None: return {}
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        fields = {}
        if "/AcroForm" in reader.trailer["/Root"]:
            form = reader.trailer["/Root"]["/AcroForm"]
            for f in form.get("/Fields", []):
                field = f.get_object()
                name = field.get("/T")
                value = field.get("/V")
                if name:
                    if isinstance(value, str): fields[name.strip()] = value.strip()
                    else: fields[name.strip()] = str(value) if value is not None else ""
        return fields
    except: return {}

def normalize_keys(d):
    return {re.sub(r"[^a-z0-9_]+","", k.lower().replace(" ","_")): v for k,v in d.items()}

def try_parse_number(v):
    if v is None: return None
    s = str(v).strip()
    if s=="": return None
    if re.match(r"^\d+$", s): return int(s)
    if s.lower()=="off": return None
    try: return int(float(s))
    except: return None

def map_form_fields(raw):
    d = normalize_keys(raw)
    out = {k:None for k in PERSONAL_FIELDS + PHQ9_ITEMS + GAD7_ITEMS + PCL5_ITEMS +
           BRIEF_COPE_ITEMS + BP_ITEMS + FAM_ITEMS + SDQ_FLAGS + FREE_TEXT_FIELDS}

    for k in list(d.keys()):
        val = d[k]
        if any(x in k for x in ["name","your_name","student_name","patient_name"]): out["name"] = str(val).strip()
        if any(x in k for x in ["gender","sex"]): out["gender"] = str(val).strip()
        if "date_of_birth" in k or k.endswith("dob") or "birth" in k: out["dob"] = str(val).strip()

    for i in range(1,10):
        for key in [f"phq{i}", f"phq_{i}", f"phq{i}_score"]:
            if key in d: out[f"phq{i}"] = try_parse_number(d[key]); break

    for i in range(1,8):
        for key in [f"gad{i}", f"gad_{i}", f"gad{i}_score"]:
            if key in d: out[f"gad{i}"] = try_parse_number(d[key]); break

    for i in range(1,21):
        for key in [f"pcl{i}", f"pcl_{i}", f"pcl5_{i}"]:
            if key in d: out[f"pcl{i}"] = try_parse_number(d[key]); break

    for i in range(1,29):
        for key in [f"brief_{i}", f"cope_{i}", f"briefcope_{i}"]:
            if key in d: out[f"brief_{i}"] = try_parse_number(d[key]); break

    for i in range(1,30):
        for key in [f"bp_{i}", f"bussperry_{i}", f"aggression_{i}"]:
            if key in d: out[f"bp_{i}"] = try_parse_number(d[key]); break

    for i in range(1,31):
        for key in [f"fam_{i}", f"functional_antisocial_{i}"]:
            if key in d: out[f"fam_{i}"] = try_parse_number(d[key]); break

    for key in SDQ_FLAGS:
        if key in d: out[key] = str(d[key]).strip()

    for key in FREE_TEXT_FIELDS:
        if key in d and str(d[key]).strip(): out[key] = str(d[key]).strip()

    return out

# ---------------------------
# Scoring
# ---------------------------

def score_phq9(rec):
    vals = [safe_int(rec.get(k)) for k in PHQ9_ITEMS if rec.get(k) is not None]
    if not vals: return None
    score = sum(v for v in vals if v is not None)
    band = classify_band(score, PHQ9_SEVERITY)
    item9 = safe_int(rec.get("phq9")) or 0
    return {"total":score,"severity":band,"cutoffs":PHQ9_SEVERITY,"flags":{"suicide_item_positive": item9>0}}

def score_gad7(rec):
    vals = [safe_int(rec.get(k)) for k in GAD7_ITEMS if rec.get(k) is not None]
    if not vals: return None
    score = sum(v for v in vals if v is not None)
    band = classify_band(score, GAD7_SEVERITY)
    return {"total":score,"severity":band,"cutoffs":GAD7_SEVERITY}

def score_pcl5(rec):
    vals = [safe_int(rec.get(k)) for k in PCL5_ITEMS if rec.get(k) is not None]
    if not vals: return None
    total = sum(v for v in vals if v is not None)
    return {"total":total,"suggested_cutoff":PCL5_SUGGESTED_CUTOFF,"meets_screen": total>=PCL5_SUGGESTED_CUTOFF}

def summarize_brief_cope(rec):
    vals = {k: safe_int(rec.get(k)) for k in BRIEF_COPE_ITEMS}
    if all(v is None for v in vals.values()): return None
    out = {}
    for name, idxs in BRIEF_COPE_SCALES.items():
        items = [vals.get(f"brief_{i}") for i in idxs]
        items = [v for v in items if v is not None]
        out[name] = round(float(np.mean(items)),2) if items else None
    return out

def score_buss_perry(rec):
    vals = {}
    any_present = False
    for i in range(1,30):
        v = safe_int(rec.get(f"bp_{i}"))
        if v is not None: any_present = True
        vals[i] = (8 - v) if (i in BP_REVERSE and v is not None) else v
    if not any_present: return None
    out = {}
    for name, idxs in BP_SCALES.items():
        items = [vals.get(i) for i in idxs if vals.get(i) is not None]
        out[name] = round(sum(items),2) if items else None
    out["Total (observed)"] = round(sum(v for v in vals.values() if v is not None),2)
    return out

def score_fam(rec):
    vals = [safe_int(rec.get(k)) for k in FAM_ITEMS]
    if all(v is None for v in vals): return None
    vals = [v for v in vals if v is not None]
    return {"mean": round(float(np.mean(vals)),2), "sum": int(sum(vals)), "n_items": len(vals)}

def sdq_snapshot(rec):
    present = [k for k in SDQ_FLAGS if rec.get(k)]
    if not present: return None
    return {"present_items": len(present), "noted_fields": present[:10] + (["..."] if len(present) > 10 else [])}

def analyze_free_text(rec):
    texts = []
    for k in FREE_TEXT_FIELDS:
        v = rec.get(k)
        if v and isinstance(v,str) and v.strip(): texts.append(v.strip())
    if not texts: return None
    full = "\n\n".join(texts)
    senti = quick_sentiment(full)
    tags = keyword_tags(full, top_k=8)
    summ = short_summary(full, max_sent=2)
    sui = suicide_screen(full)
    flags = []
    if sui["ideation"]: flags.append("Self-harm/suicide language detected")
    if sui["imminent_language"]: flags.append("Imminent language detected (time-bound intent)")
    return {"sentiment":senti,"tags":tags,"summary":summ,"flags":flags,"raw_text":full}

def trigger_rules(scored, nlp):
    banners = []
    phq = scored.get("phq9")
    if phq and phq.get("flags",{}).get("suicide_item_positive"):
        banners.append(("danger","Immediate risk screen",
                        "PHQ-9 item 9 is positive (self-harm thoughts).",
                        ["Engage safety protocol","Direct risk assessment","Consider emergency contact/911 if imminent"]))
    if nlp and ("Self-harm/suicide language detected" in nlp.get("flags",[])):
        msg = "Free-text includes self-harm/suicide cues."
        if "Imminent language detected (time-bound intent)" in nlp.get("flags",[]): msg += " Imminent language detected."
        banners.append(("danger","Immediate risk screen", msg,
                        ["Engage safety protocol","Do not leave patient alone","Escalate per clinic policy"]))
    gad = scored.get("gad7")
    if gad and gad["severity"]=="Severe":
        banners.append(("warning","High anxiety severity","GAD-7 indicates severe anxiety.",
                        ["Consider CBT/medication evaluation","Schedule follow-up"]))
    pcl = scored.get("pcl5")
    if pcl and pcl.get("meets_screen"):
        banners.append(("warning","PTSD screen positive",
                        f"PCL-5 ≥ {pcl['suggested_cutoff']} (observed {pcl['total']}).",
                        ["PTSD diagnostic interview","Trauma-focused therapy referral"]))
    return banners

# ---------------------------
# PDF exporter
# ---------------------------

def render_pdf_summary(respondent_id, personal, scored, nlp, out_bytes: io.BytesIO):
    if canvas is None: return False
    c = canvas.Canvas(out_bytes, pagesize=letter)
    width, height = letter

    def line(y, txt, size=11, bold=False):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(0.75*inch, y, txt)

    y = height - 0.75*inch
    line(y, f"Mental Health Forms Summary — {respondent_id}", 14, True); y -= 0.3*inch

    line(y, "Personal:", 12, True); y -= 0.2*inch
    for k in ["name","gender","dob"]:
        v = personal.get(k) or "-"
        line(y, f"• {k.capitalize()}: {v}"); y -= 0.18*inch

    y -= 0.1*inch
    line(y, "Scores:", 12, True); y -= 0.2*inch
    def score_line(label, value):
        nonlocal y
        line(y, f"• {label}: {value}"); y -= 0.18*inch

    phq = scored.get("phq9")
    if phq:
        score_line("PHQ-9 total", f"{phq['total']} ({phq['severity']})")
        if phq["flags"].get("suicide_item_positive"): score_line("PHQ-9 item 9","Positive")

    gad = scored.get("gad7")
    if gad: score_line("GAD-7 total", f"{gad['total']} ({gad['severity']})")

    pcl = scored.get("pcl5")
    if pcl:
        meet = "meets screen" if pcl["meets_screen"] else "below screen"
        score_line("PCL-5 total", f"{pcl['total']} ({meet} vs {pcl['suggested_cutoff']})")

    bc = scored.get("brief_cope")
    if bc:
        line(y, "Brief COPE (means):", 11, True); y -= 0.18*inch
        for k, v in bc.items():
            if v is not None: score_line(f"  {k}", v)

    bp = scored.get("buss_perry")
    if bp:
        line(y, "Buss-Perry:", 11, True); y -= 0.18*inch
        for k, v in bp.items():
            score_line(f"  {k}", v)

    fam = scored.get("fam")
    if fam:
        score_line("FAM mean", fam["mean"]); score_line("FAM sum", fam["sum"])

    sdq = scored.get("sdq")
    if sdq: score_line("SDQ snapshot", f"{sdq['present_items']} items present")

    y -= 0.1*inch
    if nlp:
        line(y, "NLP:", 12, True); y -= 0.2*inch
        s = nlp.get("sentiment", {})
        score_line("Sentiment", f"{s.get('label','-')} (score {s.get('score',0)})")
        tags = ", ".join(nlp.get("tags", []))
        score_line("Tags", tags or "-")
        summ = nlp.get("summary") or "-"
        line(y, f"Summary: {summ}"); y -= 0.18*inch

    y -= 0.1*inch
    line(y, "Notes:", 10, True); y -= 0.16*inch
    line(y, "This summary is informational and not a diagnosis.", 9); y -= 0.12*inch
    line(y, "If self-harm risk is suspected, follow local emergency protocols.", 9); y -= 0.12*inch

    c.showPage(); c.save(); return True

# ---------------------------
# CSV Template (NEW)
# ---------------------------

def csv_template_df():
    cols = []
    cols += PERSONAL_FIELDS
    cols += PHQ9_ITEMS + GAD7_ITEMS + PCL5_ITEMS
    cols += BRIEF_COPE_ITEMS + BP_ITEMS + FAM_ITEMS
    cols += SDQ_FLAGS
    cols += FREE_TEXT_FIELDS
    # one empty example row
    df = pd.DataFrame(columns=cols)
    return df

# ---------------------------
# App UI
# ---------------------------

st.title("🧠 Mental Health Forms Analyzer")
st.caption("Upload filled PDFs and/or CSV to score, analyze, visualize, and export.")

with st.expander("Input format help", expanded=False):
    st.markdown("""
- **PDF**: Upload filled, form-fillable AcroForm PDFs (your form works).
- **CSV**: Use the downloadable **CSV template** below, or ensure columns like:
  `phq1..phq9`, `gad1..gad7`, `pcl1..pcl20`, `brief_1..brief_28`, `bp_1..bp_29`, `fam_1..fam_30`,
  personal (`name`,`gender`,`dob`), SDQ flags, and optional free-text fields.
""")

# NEW: CSV template download
st.subheader("CSV template")
templ = csv_template_df()
csv_bytes = templ.to_csv(index=False).encode("utf-8")
st.download_button("Download blank CSV template", data=csv_bytes, file_name="mental_health_forms_template.csv", mime="text/csv")
st.caption("Fill one row per respondent. Leave any unused columns blank.")

uploads = st.file_uploader("Upload PDF(s) and/or CSV", type=["pdf","csv"], accept_multiple_files=True)

respondents = []
if uploads:
    for f in uploads:
        if f.name.lower().endswith(".pdf"):
            if PdfReader is None:
                st.error("PyPDF2 not installed. `pip install PyPDF2` to parse PDF forms."); continue
            data = f.read()
            fields = read_pdf_form_fields(data)
            if not fields:
                st.warning(f"No form fields detected in {f.name}; skipping."); continue
            mapped = map_form_fields(fields); mapped["_source_file"] = f.name
            respondents.append(mapped)
        else:
            try:
                df = pd.read_csv(f)
                for _, row in df.iterrows():
                    mapped = map_form_fields(row.to_dict()); mapped["_source_file"] = f.name
                    respondents.append(mapped)
            except Exception as e:
                st.error(f"Failed reading CSV {f.name}: {e}")

if not respondents:
    st.info("Upload files to begin."); st.stop()

# Score all
scored_list = []
for idx, rec in enumerate(respondents, start=1):
    phq = score_phq9(rec)
    gad = score_gad7(rec)
    pcl = score_pcl5(rec)
    bc  = summarize_brief_cope(rec)
    bp  = score_buss_perry(rec)
    fam = score_fam(rec)
    sdq = sdq_snapshot(rec)
    nlp = analyze_free_text(rec)
    trig = trigger_rules({"phq9":phq,"gad7":gad,"pcl5":pcl,"brief_cope":bc,"buss_perry":bp,"fam":fam,"sdq":sdq}, nlp)

    scored_list.append({
        "id": f"Respondent {idx}",
        "personal": {k: rec.get(k) for k in PERSONAL_FIELDS},
        "scores": {"phq9":phq,"gad7":gad,"pcl5":pcl,"brief_cope":bc,"buss_perry":bp,"fam":fam,"sdq":sdq},
        "nlp": nlp,
        "triggers": trig,
        "_source_file": rec.get("_source_file","")
    })

# ---------------------------
# Overview table
# ---------------------------

st.subheader("Overview")
ov = []
for r in scored_list:
    ov.append({
        "Respondent": r["id"],
        "Source": r["_source_file"],
        "Name": r["personal"].get("name"),
        "PHQ-9": (r["scores"]["phq9"] or {}).get("total"),
        "PHQ-9 Sev": (r["scores"]["phq9"] or {}).get("severity"),
        "GAD-7": (r["scores"]["gad7"] or {}).get("total"),
        "GAD-7 Sev": (r["scores"]["gad7"] or {}).get("severity"),
        "PCL-5": (r["scores"]["pcl5"] or {}).get("total"),
        "PTSD Screen": "Yes" if ((r["scores"]["pcl5"] or {}).get("meets_screen")) else "No"
    })
ov_df = pd.DataFrame(ov)
# If there's exactly one respondent, show the row vertically (key: value) for easier reading
if len(ov_df) == 1:
    single_row = ov_df.iloc[0].to_frame(name="Value")
    # Show a compact vertical table
    st.table(single_row)
else:
    st.dataframe(ov_df, use_container_width=True)

# ---------------------------
# NEW: Charts (Overview)
# ---------------------------

# st.markdown("---")
# st.subheader("Charts")

# def _hist(column_name, title):
#     vals = ov_df[column_name].dropna().astype(float)
#     if len(vals)==0: st.info(f"No data for {title}."); return
#     fig, ax = plt.subplots()
#     ax.hist(vals, bins=min(10, max(3, len(vals)//2)))
#     ax.set_title(title)
#     ax.set_xlabel(column_name)
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Distributions
# colA, colB, colC = st.columns(3)
# with colA: _hist("PHQ-9", "PHQ-9 Distribution")
# with colB: _hist("GAD-7", "GAD-7 Distribution")
# with colC: _hist("PCL-5", "PCL-5 Distribution")

# # Brief COPE mean across respondents (bar)
# brief_all = defaultdict(list)
# for r in scored_list:
#     bc = r["scores"]["brief_cope"]
#     if bc:
#         for k,v in bc.items():
#             if v is not None: brief_all[k].append(v)

# if brief_all:
#     brief_mean = {k: float(np.mean(vs)) for k,vs in brief_all.items() if vs}
#     if brief_mean:
#         st.write("**Brief COPE — mean of subscale means (across respondents)**")
#         fig, ax = plt.subplots(figsize=(8,4))
#         labels = list(brief_mean.keys())
#         vals   = [brief_mean[k] for k in labels]
#         ax.barh(labels, vals)
#         ax.set_xlabel("Mean score")
#         st.pyplot(fig)

# # Buss-Perry subscale averages (bar)
# bp_all = defaultdict(list)
# for r in scored_list:
#     bp = r["scores"]["buss_perry"]
#     if bp:
#         for k,v in bp.items():
#             if k!="Total (observed)" and v is not None:
#                 bp_all[k].append(v)

# if bp_all:
#     bp_mean = {k: float(np.mean(vs)) for k,vs in bp_all.items() if vs}
#     if bp_mean:
#         st.write("**Buss-Perry — average subscale scores**")
#         fig, ax = plt.subplots(figsize=(6,3.5))
#         labels = list(bp_mean.keys())
#         vals   = [bp_mean[k] for k in labels]
#         ax.bar(labels, vals)
#         ax.set_ylabel("Score")
#         ax.set_xticklabels(labels, rotation=30, ha="right")
#         st.pyplot(fig)

# # FAM mean distribution
# fam_vals = []
# for r in scored_list:
#     fam = r["scores"]["fam"]
#     if fam: fam_vals.append(fam["mean"])
# if fam_vals:
#     fig, ax = plt.subplots()
#     ax.hist(fam_vals, bins=min(10, max(3, len(fam_vals)//2)))
#     ax.set_title("FAM Mean Distribution")
#     ax.set_xlabel("FAM mean")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# st.markdown("---")

# ---------------------------
# Per-respondent sections + export
# ---------------------------

for r in scored_list:
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        st.markdown(f"### {r['id']}  \n*Source:* `{r['_source_file']}`")

        for level, title, msg, steps in r["triggers"]:
            if level == "danger":
                st.error(f"**{title}** — {msg}\n\n**Next steps:** " + "; ".join(steps))
            elif level == "warning":
                st.warning(f"**{title}** — {msg}\n\n**Next steps:** " + "; ".join(steps))
            elif level == "info":
                st.info(f"**{title}** — {msg}\n\n**Next steps:** " + "; ".join(steps))
            else:
                st.success(f"**{title}** — {msg}\n\n**Next steps:** " + "; ".join(steps))

        with st.expander("Personal details"):
            st.write(r["personal"])

        with st.expander("Scores & Summaries", expanded=True):
            phq = r["scores"]["phq9"]; gad = r["scores"]["gad7"]; pcl = r["scores"]["pcl5"]
            bc  = r["scores"]["brief_cope"]; bp = r["scores"]["buss_perry"]
            fam = r["scores"]["fam"]; sdq = r["scores"]["sdq"]

            if phq:
                st.markdown(f"**PHQ-9:** {phq['total']} ({phq['severity']})")
                st.caption("Bands: 0-4 Minimal, 5-9 Mild, 10-14 Moderate, 15-19 Moderately severe, 20-27 Severe.")
                if phq["flags"].get("suicide_item_positive"):
                    st.warning("PHQ-9 item 9 positive for self-harm thoughts.")

            if gad:
                st.markdown(f"**GAD-7:** {gad['total']} ({gad['severity']})")
                st.caption("Bands: 0-4 Minimal, 5-9 Mild, 10-14 Moderate, 15-21 Severe.")

            if pcl:
                st.markdown(f"**PCL-5:** {pcl['total']} (Screen {'positive' if pcl['meets_screen'] else 'negative'} vs cutoff {pcl['suggested_cutoff']})")

            if bc:
                st.markdown("**Brief COPE (mean per subscale)**"); st.json(bc)

            if bp:
                st.markdown("**Buss-Perry Aggression**"); st.json(bp)

            if fam:
                st.markdown("**FAM:**")
                st.write(f"Mean = {fam['mean']}, Sum = {fam['sum']} (n={fam['n_items']})")

            if sdq:
                st.markdown("**SDQ snapshot (presence check):**"); st.json(sdq)

        with st.expander("Free-text analysis"):
            nlp = r["nlp"]
            if not nlp:
                st.write("No free text found.")
            else:
                st.write(f"**Sentiment:** {nlp['sentiment']['label']} (score {nlp['sentiment']['score']})")
                st.write("**Keyword tags:**", ", ".join(nlp["tags"]))
                st.write("**Short summary:**", nlp["summary"])
                if nlp["flags"]:
                    st.warning("Flags: " + "; ".join(nlp["flags"]))
                with st.popover("View raw text"):
                    st.code(nlp["raw_text"])

    with col2:
        st.markdown("#### Export")
        if canvas is None:
            st.info("Install `reportlab` to enable PDF export: `pip install reportlab`")
        else:
            b = io.BytesIO()
            ok = render_pdf_summary(
                respondent_id=r["id"],
                personal=r["personal"],
                scored={
                    "phq9": r["scores"]["phq9"], "gad7": r["scores"]["gad7"], "pcl5": r["scores"]["pcl5"],
                    "brief_cope": r["scores"]["brief_cope"], "buss_perry": r["scores"]["buss_perry"],
                    "fam": r["scores"]["fam"], "sdq": r["scores"]["sdq"]
                },
                nlp=r["nlp"],
                out_bytes=b
            )
            if ok:
                st.download_button(
                    label=f"Download PDF: {r['id']}",
                    data=b.getvalue(),
                    file_name=f"{r['id'].replace(' ', '_')}_summary.pdf",
                    mime="application/pdf"
                )

st.markdown("---")
st.subheader("Combined export")
if canvas is None:
    st.info("Install `reportlab` for PDF export.")
else:
    import zipfile
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, r in enumerate(scored_list, start=1):
            b = io.BytesIO()
            render_pdf_summary(
                respondent_id=r["id"],
                personal=r["personal"],
                scored={
                    "phq9": r["scores"]["phq9"], "gad7": r["scores"]["gad7"], "pcl5": r["scores"]["pcl5"],
                    "brief_cope": r["scores"]["brief_cope"], "buss_perry": r["scores"]["buss_perry"],
                    "fam": r["scores"]["fam"], "sdq": r["scores"]["sdq"]
                },
                nlp=r["nlp"],
                out_bytes=b
            )
            zf.writestr(f"{r['id'].replace(' ','_')}_summary.pdf", b.getvalue())
    st.download_button("Download ZIP of all PDFs", data=zbuf.getvalue(),
                       file_name="all_summaries.zip", mime="application/zip")

st.caption("This tool summarizes standard self-report instruments; it is **not** a diagnosis. Follow local protocols for assessment and safety.")
