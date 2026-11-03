# Student Performance Predictor

Title: Student Performance Predictor
Problem: Schools want to identify students at risk of low grades early so they can support them.
Data: UCI Student Performance dataset, 395 students, 33 features.
Goal: Predict final grade G3 and understand which factors matter most.
Methods: Regression, ensemble learning, explainability checks.

Predicting final student grades and flagging students who might benefit from a check-in. 
This is an AI internship project built over 30 days with the public UCI Student Performance dataset.

**Live app:** (https://student-checkin.streamlit.app/)

> **Decision support only.** A flag means a conversation might help. It is never a decision, label or penalty. Do not use this tool for grading, ranking, admissions or discipline.

App screenshot - images/Screenshot 1.png
App screenshot - images/Screenshot 2.png

## What it does

- **Grade model:** predicts a student's final grade (0 to 20).
- **Check-in flag:** flags students predicted to be at risk (final grade below 10).
- **Two stages:** *start of term* (no grades yet) and *after period 2* (G1 and G2 known).
- **Explanations:** shows which values moved a prediction compared with a typical student.
- **Two ways to use it:** a single-student form, or a CSV upload for a group.
- Nothing entered in the app is stored.

## Key results

All results come from students the final models never saw, measured once. Intervals are 95% bootstrap intervals, and the test set is small, so they are wide.

| Stage | Grade model | Test RMSE (95% CI) | Beats predicting the class average? | Flag model | Test AUC (95% CI) | Usable (AUC 0.70 or more)? |
|---|---|---|---|---|---|---|
| Start of term | Ridge | 2.602 (2.25 to 2.98) | No by the pre-set rule; test gain 16.2% (95% CI 6.9 to 25.1) | Logistic | 0.748 (0.61 to 0.88) | No by the pre-set rule | (inside expected range? grade: Yes | flag: Yes)
| After period 2 | Ridge | 0.898 (0.73 to 1.07) | Yes by the pre-set rule; test gain 71.1% (95% CI 63.5 to 77.5) | Logistic | 0.991 (0.97 to 1.00) | Yes by the pre-set rule | (inside expected range? grade: Yes | flag: NO)

Fill the table from `data/test_results_regression.csv` and `data/test_results_classification.csv`.

**Main finding:** accuracy comes from information that arrives late. At the start of term, neither the grade model nor the flag met the usefulness rules fixed in advance. After period 2, test RMSE 0.898 (95% CI 0.73 to 1.07), a 71.1% change versus predicting the class average (95% CI 63.5 to 77.5). The flag reached a test AUC of 0.991 (95% CI 0.97 to 1.00), with recall 0.89 and precision 0.94, on only 19 at-risk test students.

Inside expected range? grade: Yes | flag: NO

## How it was built

| Phase | Days | What was done |
|---|---|---|
| 1. Foundations | 1-5 | Python, pandas, NumPy, visualisation, data audit |
| 2. Exploration and preparation | 6-10 | Hypothesis tests, encoding, outliers, train/test split, feature engineering |
| 3. Models and evaluation | 11-16 | Linear regression from scratch, Ridge and Lasso, metrics, learning curves, classification |
| 4. Ensembles | 17-21 | Bagging, Random Forest, gradient boosting, voting, stacking, nested-CV tuning, one-time test evaluation |
| 5. Explainability and fairness | 22-24 | Permutation importance, fairness audit, mitigation, data minimisation, model card |
| 6. App, report and submission | 25-30 | Streamlit app, deployment, report|

Key design choices:

- Students with a final grade of 0 were removed, because they differ systematically from the rest.
- All models were compared on identical repeated cross-validation folds against a predict-the-average baseline.
- Hyperparameters were tuned inside nested cross-validation, and the simplest model within one standard deviation of the best was chosen.
- The test set was opened once, and the expected result ranges were saved beforehand (`data/preregistration.json`).

## Repository layout

```
app/          Streamlit app: app.py, app_utils.py, saved models, config, sample CSV
Notebooks/    Analysis notebooks, by day and phase
data/         Dataset splits, result tables, model metadata
images/       Figures used in the report
models/       Saved model files
reports/      Final report
src/          Reusable code (linear regression from scratch)
MODEL_CARD.md Purpose, data, performance, limits and recommended use
```

## Run the app locally

You need **Python 3.12**. The models were saved with scikit-learn 1.6.1, so newer Python versions won't install the pinned version.

```bash
cd app
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Mac / Linux:  source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, use `py -3.12 -m venv .venv` if you have several Python versions installed.

## Reproduce the analysis

The notebooks in `Notebooks/` were written for Google Colab. Upload the files from `data/` as each notebook asks, and run the cells in order.

## Responsible use and limitations

- Small data: a few hundred students from two schools, one subject, collected years ago.
- About a quarter of students are at risk, so every estimate has wide intervals.
- The start-of-term stage adds little over predicting the class average.
- Many student groups are too small to audit, and the data records sex as two values.
- The explanations show what the model uses to predict, **not** what causes grades.
- Removing a sensitive column does not remove its influence, because other columns act as proxies.
- Students who did not sit the final exam were excluded, so the tool says nothing about them.

See `MODEL_CARD.md` for the full discussion and recommended use policy.

## Data

Student Performance dataset - (https://archive.ics.uci.edu/dataset/320/student+performance) (UCI Machine Learning Repository), mathematics class.
Cortez, P., and Silva, A. (2008). *Using data mining to predict secondary school student performance.* Proceedings of the 5th Future Business Technology Conference (FUBUTEC 2008), pp. 5-12.

## Internship context

This project completes an online internship in Artificial Intelligence and applies material from three courses:

- **Artificial Intelligence Course Online:** regression from scratch, ensemble learning, visualisation, evaluation, explainability and bias.
- **ChatGPT for Professionals:** AI-assisted drafting and documentation, with every result checked against the source files, and a log of what was verified and corrected.

## Use of AI tools

ChatGPT was used for explanations, drafting and review. Every number and claim was checked against the project files, and corrections are recorded in the AI assistance log in the report. 

## Author

Bunty Deb · Program Manager
