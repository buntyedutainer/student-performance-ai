# Model card: Student Performance Predictor

*DRAFT generated from saved results. Replace every [[double-bracket]] item with your own words.*

## 1. Purpose and intended use
- Helps a teacher decide which students to check in with. It does not make decisions about students.
- Two models: an EARLY one (before any term grades exist) and a LATER one (after periods 1 and 2).
- Not for: grading, ranking students, admissions, discipline, or any automated decision.

## 2. Data
- UCI Student Performance dataset (maths class), two schools. 285 training students, after removing students with a final grade of 0.
- Target: final grade G3 (0-20). At risk = G3 below 10.
- Students with G3 = 0 were removed because they differ systematically (Day 8). The model says nothing about students who do not sit the final exam.

## 3. Models
- early (no G1/G2): grade model = Ridge (useful vs baseline: False); at-risk flag = Logistic (usable, AUC >= 0.70: False), threshold 0.12.
- all features: grade model = Ridge (useful vs baseline: True); at-risk flag = Logistic (usable, AUC >= 0.70: True), threshold 0.36.

## 4. Performance (one-time test evaluation, Day 21)
- early (no G1/G2), grade: test RMSE 2.602 (95% CI 2.25 to 2.98); 16.2% better than predicting the average (95% CI 6.9 to 25.1).
- all features, grade: test RMSE 0.898 (95% CI 0.73 to 1.07); 71.1% better than predicting the average (95% CI 63.5 to 77.5).
- early (no G1/G2), flag: test AUC 0.748 (95% CI 0.61 to 0.88); recall 0.84, precision 0.33.
- all features, flag: test AUC 0.991 (95% CI 0.97 to 1.00); recall 0.89, precision 0.94.

## 5. Fairness audit (training students, out-of-fold predictions)
- 40 group comparisons across 10 attributes; 10 showed a gap that was both detectable and practically large (thresholds fixed in advance).
- 3 of 20 groups were small, so most comparisons cannot rule out a gap.
- Mitigation tested (Day 24): dropping sensitive columns and reweighting. [[State what changed beyond noise, if anything.]]
- [[Describe any gap worth discussing, with its group sizes and interval.]]

## 6. Data minimisation
- early (no G1/G2): 5 column(s) perform within one standard deviation of using all 39. [[State what this means for what a school should collect.]]
- all features: 1 column(s) perform within one standard deviation of using all 41. [[State what this means for what a school should collect.]]

## 7. Limitations
- Small data: a few hundred students, and about a quarter at risk, so every estimate has wide intervals.
- Two schools in one country, collected years ago, one subject. Patterns may not transfer.
- The early model adds little over predicting the class average [[confirm with your own results]].
- Students with a final grade of 0 are excluded.
- Groups are recorded coarsely (for example sex as two values), and many are too small to audit.
- Explanations describe what the model uses to predict, not what causes grades.

## 8. Ethical considerations and use policy (recommended, edit to your own judgement)
- A flag starts a conversation. It is never a decision, a label, or a reason to restrict a student.
- Human review of every flag; tell students and guardians that the tool exists and how it is used.
- Do not use for grading, ranking, admissions or discipline.
- Collect only the columns the task needs (Section 6), and keep them only as long as they are needed.
- Restrict access; do not paste real student data into public AI tools.
- Re-run the fairness audit and the calibration check every term.
- [[State when you would stop using the tool, for example if recall on new students falls below a level you set.]]
- Check the data-protection law that applies where the tool would be used. This card is not legal advice.
