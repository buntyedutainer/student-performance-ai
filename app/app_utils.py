"""Shared code for the Student check-in helper (used by the app and by the tests)."""
import numpy as np
import pandas as pd

YES_NO = ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]
BINARY_MAPS = {
    "school": {"GP": 0, "MS": 1},
    "sex": {"F": 0, "M": 1},
    "address": {"R": 0, "U": 1},
    "famsize": {"LE3": 0, "GT3": 1},
    "Pstatus": {"A": 0, "T": 1},
}
for _c in YES_NO:
    BINARY_MAPS[_c] = {"no": 0, "yes": 1}

# first category in each list is the dropped baseline (drop_first=True in training)
NOMINAL = {
    "Mjob": ["at_home", "health", "other", "services", "teacher"],
    "Fjob": ["at_home", "health", "other", "services", "teacher"],
    "reason": ["course", "home", "other", "reputation"],
    "guardian": ["father", "mother", "other"],
}
NUMERIC = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
           "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]
GRADES = ["G1", "G2"]


def _clean_text(series):
    return series.astype(str).str.strip().str.strip('"')


def encode_raw(df, absences_cap, need_grades=False):
    """Turn rows in the public dataset's layout into the encoded columns the models were trained on."""
    needed = list(BINARY_MAPS) + list(NOMINAL) + NUMERIC + (GRADES if need_grades else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    out = pd.DataFrame(index=df.index)
    for col, mapping in BINARY_MAPS.items():
        s = _clean_text(df[col])
        bad = sorted(set(s) - set(mapping))
        if bad:
            raise ValueError(f"Column '{col}' has unexpected values {bad}. Allowed: {list(mapping)}")
        out[col] = s.map(mapping).astype(int)
    for col, cats in NOMINAL.items():
        s = _clean_text(df[col])
        bad = sorted(set(s) - set(cats))
        if bad:
            raise ValueError(f"Column '{col}' has unexpected values {bad}. Allowed: {cats}")
        for cat in cats[1:]:
            out[f"{col}_{cat}"] = (s == cat).astype(int)
    for col in NUMERIC + (GRADES if need_grades else []):
        v = pd.to_numeric(df[col], errors="coerce")
        if v.isna().any():
            raise ValueError(f"Column '{col}' has missing or non-numeric values")
        out[col] = v
    out["absences"] = out["absences"].clip(lower=0, upper=absences_cap)
    return out


def score_students(X, reg, clf, threshold):
    grade = np.clip(reg.predict(X), 0, 20)
    risk = clf.predict_proba(X)[:, 1]
    return pd.DataFrame({"predicted_grade": np.round(grade, 1),
                         "risk_score": np.round(risk, 3),
                         "flagged": risk >= threshold}, index=X.index)


def occlusion(predict_fn, X, ref):
    """How much does each value move the result compared with a typical student?"""
    base = float(predict_fn(X)[0])
    contrib = {}
    for c in X.columns:
        Xr = X.copy()
        Xr[c] = ref[c]
        contrib[c] = base - float(predict_fn(Xr)[0])
    return base, pd.Series(contrib)


# ---- form layout: (key, label, kind, spec, hint) ----
LIKERT5 = {1: "1 (very low)", 2: "2", 3: "3", 4: "4", 5: "5 (very high)"}
QUALITY5 = {1: "1 (very bad)", 2: "2", 3: "3", 4: "4", 5: "5 (excellent)"}
EDU = {0: "none", 1: "primary (up to 4th grade)", 2: "5th to 9th grade", 3: "secondary", 4: "higher education"}
JOBS = {"at_home": "at home", "health": "health care", "other": "other",
        "services": "civil services", "teacher": "teacher"}
YN = {"yes": "Yes", "no": "No"}

MAIN = [
    ("failures", "Past class failures", "slider", (0, 3), "0 to 3 (3 means three or more)"),
    ("studytime", "Weekly study time", "choice",
     {1: "under 2 hours", 2: "2 to 5 hours", 3: "5 to 10 hours", 4: "over 10 hours"}, None),
    ("absences", "School absences so far", "number", (0, 93), "Values above the training cap are treated as the cap"),
    ("higher", "Wants to continue to higher education", "choice", YN, None),
    ("age", "Age", "slider", (15, 22), None),
]
GRADE_FIELDS = [
    ("G1", "Grade in period 1 (0-20)", "slider", (0, 20), None),
    ("G2", "Grade in period 2 (0-20)", "slider", (0, 20), None),
]
BACKGROUND = [
    ("school", "School", "choice", {"GP": "GP", "MS": "MS"}, None),
    ("sex", "Sex (as recorded in the data)", "choice", {"F": "Female", "M": "Male"}, None),
    ("address", "Home area", "choice", {"U": "Urban", "R": "Rural"}, None),
    ("famsize", "Family size", "choice", {"LE3": "3 or fewer", "GT3": "more than 3"}, None),
    ("Pstatus", "Parents live", "choice", {"T": "together", "A": "apart"}, None),
    ("Medu", "Mother's education", "choice", EDU, None),
    ("Fedu", "Father's education", "choice", EDU, None),
    ("Mjob", "Mother's job", "choice", JOBS, None),
    ("Fjob", "Father's job", "choice", JOBS, None),
    ("reason", "Reason for choosing this school", "choice",
     {"course": "course", "home": "close to home", "reputation": "reputation", "other": "other"}, None),
    ("guardian", "Guardian", "choice", {"mother": "mother", "father": "father", "other": "other"}, None),
    ("traveltime", "Travel time to school", "choice",
     {1: "under 15 min", 2: "15 to 30 min", 3: "30 to 60 min", 4: "over 1 hour"}, None),
    ("schoolsup", "Extra educational support from school", "choice", YN, None),
    ("famsup", "Family educational support", "choice", YN, None),
    ("paid", "Paid extra classes in this subject", "choice", YN, None),
    ("activities", "Extracurricular activities", "choice", YN, None),
    ("nursery", "Attended nursery school", "choice", YN, None),
    ("internet", "Internet at home", "choice", YN, None),
    ("romantic", "In a romantic relationship", "choice", YN, None),
    ("famrel", "Quality of family relationships", "choice", QUALITY5, None),
    ("freetime", "Free time after school", "choice", LIKERT5, None),
    ("goout", "Going out with friends", "choice", LIKERT5, None),
    ("Dalc", "Weekday alcohol use", "choice", LIKERT5, None),
    ("Walc", "Weekend alcohol use", "choice", LIKERT5, None),
    ("health", "Current health", "choice", QUALITY5, None),
]
