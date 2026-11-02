import io
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import app_utils as U

BASE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Student check-in helper", page_icon="🎓", layout="wide")


@st.cache_resource
def load_assets():
    cfg = json.load(open(os.path.join(BASE, "app_config.json")))
    meta = json.load(open(os.path.join(BASE, "model_meta.json")))
    models = {t: {"reg": joblib.load(os.path.join(BASE, f"reg_{t}.joblib")),
                  "clf": joblib.load(os.path.join(BASE, f"clf_{t}.joblib"))}
              for t in ("early", "all")}
    return cfg, meta, models


try:
    cfg, meta, models = load_assets()
except Exception as e:                      # most often a scikit-learn version mismatch
    st.error("The model files could not be loaded. This is usually a scikit-learn version "
             "mismatch: install the version listed in requirements.txt.")
    st.code(str(e))
    st.stop()

# ---------------- sidebar ----------------
st.sidebar.header("Settings")
stage = st.sidebar.radio("When is the prediction made?",
                         ["Start of term (no grades yet)", "After period 2 (G1 and G2 known)"])
tag = "early" if stage.startswith("Start") else "all"
info = meta["models"][tag]
features = meta["features_early"] if tag == "early" else meta["features_all"]
reg, clf, thr = models[tag]["reg"], models[tag]["clf"], info["threshold"]

st.sidebar.markdown("---")
st.sidebar.caption("Nothing you enter is saved by this app. Do not enter names or ID numbers.")
st.sidebar.caption(f"Grade model: {info['regression']} | flag model: {info['classification']}")
st.sidebar.caption(f"Test RMSE {info['test_rmse']:.2f} grade points | test AUC {info['test_auc']:.2f}")

# ---------------- header and honest warnings ----------------
st.title("Student check-in helper")
st.warning("Decision support only. A flag means a conversation might help, never a decision, label or "
           "penalty. Built as a course project on public data from two schools. Do not use it for "
           "grading, ranking, admissions or discipline.")
if not info["regression_useful"]:
    st.warning("At this stage the grade prediction was not clearly better than simply predicting the "
               "class average when tested on unseen students. Treat the predicted grade as a rough guide only.")
if not info["classification_usable"]:
    st.warning("At this stage the check-in flag did not reach the 0.70 AUC level set in advance. "
               "Do not rely on it.")
st.caption(f"The flag threshold ({thr:.2f}) was set to catch about 80% of at-risk students in the training "
           "data, so some flagged students will not turn out to be at risk.")


# ---------------- helpers ----------------
def pretty(name):
    return name.replace("_", " = ", 1) if "_" in name else name


def render(field, container):
    key, label, kind, spec, hint = field
    default = cfg["defaults"][key]
    wkey = f"in_{key}"
    if kind in ("slider", "number"):
        lo, hi = spec
        val = int(min(max(default, lo), hi))
        if kind == "slider":
            return container.slider(label, lo, hi, val, help=hint, key=wkey)
        return container.number_input(label, min_value=lo, max_value=hi, value=val, step=1, help=hint, key=wkey)
    options = list(spec)
    idx = options.index(default) if default in options else 0
    return container.selectbox(label, options, index=idx, format_func=lambda o, s=spec: s[o],
                               help=hint, key=wkey)


def input_notes(raw_row):
    notes = []
    for c, (lo, hi) in cfg["numeric_ranges"].items():
        if c in raw_row and not (lo <= raw_row[c] <= hi):
            notes.append(f"{c} = {raw_row[c]} is outside the range seen in training ({lo} to {hi}), "
                         "so the prediction is less reliable.")
    if raw_row.get("absences", 0) > cfg["absences_cap"]:
        notes.append(f"Absences above {cfg['absences_cap']:.0f} are treated as {cfg['absences_cap']:.0f}, as in training.")
    if raw_row.get("G1", 1) == 0 or raw_row.get("G2", 1) == 0:
        notes.append("A period grade of 0 may mean the student did not sit an exam. Students whose final "
                     "grade was 0 were excluded from training, so this tool does not apply to them.")
    return notes


def explanation_chart(contrib, risk_mode):
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)[::-1]
    colors = ["indianred" if (v > 0) == risk_mode else "seagreen" for v in top.values]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh([pretty(c) for c in top.index], top.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Change in risk score vs a typical student" if risk_mode
                  else "Change in predicted grade vs a typical student")
    fig.tight_layout()
    return fig


tab_single, tab_batch, tab_about = st.tabs(["One student", "Upload a CSV", "About this tool"])

# ---------------- one student ----------------
with tab_single:
    with st.form("student_form"):
        st.subheader("Main information")
        vals = {}
        cols = st.columns(3)
        for i, f in enumerate(U.MAIN):
            vals[f[0]] = render(f, cols[i % 3])
        if tag == "all":
            st.subheader("Period grades")
            gcols = st.columns(2)
            for i, f in enumerate(U.GRADE_FIELDS):
                vals[f[0]] = render(f, gcols[i])
        with st.expander("Background (optional): typical values are pre-filled"):
            bcols = st.columns(3)
            for i, f in enumerate(U.BACKGROUND):
                vals[f[0]] = render(f, bcols[i % 3])
        submitted = st.form_submit_button("Check")

    if submitted:
        try:
            enc = U.encode_raw(pd.DataFrame([vals]), cfg["absences_cap"], need_grades=(tag == "all"))
            X = enc.reindex(columns=features)
            if X.isna().any().any():
                raise ValueError("Some model columns could not be built from the inputs.")
            st.session_state["last"] = {"tag": tag, "X": X, "raw": dict(vals)}
        except ValueError as e:
            st.error(str(e))

    last = st.session_state.get("last")
    if last and last["tag"] == tag:
        X = last["X"]
        res = U.score_students(X, reg, clf, thr).iloc[0]
        st.divider()
        a, b, c = st.columns(3)
        a.metric("Predicted final grade (0-20)", f"{res['predicted_grade']:.1f}")
        b.metric("Risk score (0-1)", f"{res['risk_score']:.2f}")
        c.metric("Check-in suggested?", "Yes" if res["flagged"] else "No")
        st.caption(f"On students the model had never seen, the predicted grade was typically off by about "
                   f"{info['test_rmse']:.1f} grade points (test RMSE). The risk score is a ranking score, not a "
                   "guaranteed probability.")
        for n in input_notes(last["raw"]):
            st.warning(n)

        st.subheader("What moved this result?")
        what = st.radio("Explain", ["Risk score", "Predicted grade"], horizontal=True)
        ref = pd.Series(cfg["reference_median"]).reindex(features)
        if what == "Risk score":
            fn = lambda D: clf.predict_proba(D)[:, 1]
        else:
            fn = lambda D: reg.predict(D)
        p0, contrib = U.occlusion(fn, X, ref)
        p_ref = float(fn(pd.DataFrame([ref]))[0])
        st.pyplot(explanation_chart(contrib, risk_mode=(what == "Risk score")))
        st.caption(f"This student: {p0:.2f}. A typical student (median values): {p_ref:.2f}. Bars show how much "
                   "each value moves the result compared with a typical student. These are patterns the model "
                   "uses to predict, not causes. Please do not tell a student that any one factor is the reason "
                   "for a result.")
    elif submitted is False:
        st.info("Fill in the form and press Check.")

# ---------------- batch ----------------
with tab_batch:
    st.write("Upload a CSV in the same layout as the public dataset (semicolon or comma separated). "
             "The results show only a row number, predicted grade, risk score and flag.")
    st.warning("This app may be hosted publicly. Never upload real student data to a public app. "
               "Use the sample file or public data.")
    sure = st.checkbox("I confirm this file has no names, ID numbers or other real student data.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None and not sure:
        st.info("Tick the box above to continue.")
    elif up is not None:
        try:
            if up.size > 2_000_000:
                raise ValueError("The file is larger than 2 MB.")
            text = up.getvalue().decode("utf-8-sig", errors="replace")
            first = text.splitlines()[0] if text.strip() else ""
            sep = ";" if first.count(";") > first.count(",") else ","
            df = pd.read_csv(io.StringIO(text), sep=sep)
            df.columns = [str(c).strip().strip('"') for c in df.columns]
            if len(df) == 0:
                raise ValueError("The file has no rows.")
            if len(df) > 2000:
                raise ValueError("More than 2000 rows; please upload fewer.")
            enc = U.encode_raw(df, cfg["absences_cap"], need_grades=(tag == "all"))
            X = enc.reindex(columns=features)
            out = U.score_students(X, reg, clf, thr)
            out.insert(0, "row", np.arange(1, len(out) + 1))
            n_flag = int(out["flagged"].sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Students", len(out))
            m2.metric("Check-in suggested", n_flag)
            m3.metric("Share flagged", f"{100 * n_flag / len(out):.0f}%")
            st.dataframe(out, hide_index=True, width="stretch")
            st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                               "check_in_results.csv", "text/csv")
            st.caption("Nothing is stored by the app. Remember that a flag is a prompt for a conversation, "
                       "not a decision.")
        except ValueError as e:
            st.error(str(e))

# ---------------- about ----------------
with tab_about:
    st.subheader("What this tool is for")
    st.markdown(
        "- Helps a teacher decide which students to check in with. It does not make decisions about students.\n"
        "- Two stages: **start of term** (no grades yet) and **after period 2** (period grades known).\n"
        "- Trained on public data from maths classes in two schools. Students whose final grade was 0 were "
        "excluded, so it says nothing about students who do not sit the final exam.\n"
        "- It offers no what-if sliders on purpose: the model finds patterns, and cannot say what would "
        "change a student's grade.")
    st.subheader("Performance on students the models never saw (one-time test)")
    rows = []
    for t, label in [("early", "Start of term"), ("all", "After period 2")]:
        m = meta["models"][t]
        rows.append({"stage": label, "grade model": m["regression"],
                     "test RMSE (grade points)": round(m["test_rmse"], 2),
                     "grade model clearly beats the average?": "Yes" if m["regression_useful"] else "No",
                     "flag model": m["classification"], "test AUC": round(m["test_auc"], 2),
                     "flag usable (AUC 0.70 or more)?": "Yes" if m["classification_usable"] else "No"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("The figures come from the models trained on the training students only. The models running in "
               "this app were refit on all students with identical settings.")
    card = os.path.join(BASE, "MODEL_CARD.md")
    if os.path.exists(card):
        with st.expander("Model card"):
            st.markdown(open(card, encoding="utf-8").read())

st.divider()
st.caption("Data: Cortez and Silva (2008), UCI Student Performance dataset. Course project, not a "
           "validated or certified tool. Version 1.0.")
