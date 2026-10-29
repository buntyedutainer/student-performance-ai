# Cell 2: one reusable function. Your Streamlit app will import this on Day 25.
import pandas as pd

def add_features(df):
    out = df.copy()
    out["avg_parent_edu"] = (out["Medu"] + out["Fedu"]) / 2
    out["support_score"] = out["famsup"] + out["schoolsup"] + out["paid"]
    out["alcohol_avg"] = (out["Dalc"] + out["Walc"]) / 2
    out["social_load"] = out["goout"] + out["freetime"]
    out["has_failures"] = (out["failures"] > 0).astype(int)
    out["no_absences"] = (out["absences"] == 0).astype(int)
    if {"G1", "G2"}.issubset(out.columns):
        out["grade_trend"] = out["G2"] - out["G1"]
    return out
