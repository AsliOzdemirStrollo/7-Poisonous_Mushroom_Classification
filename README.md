# 🍄 Poisonous Mushroom Classification  

## Data Science Project — WBS Coding School  
This project applies **cost-based machine learning** to predict whether a mushroom is **poisonous or edible**, prioritizing **safety (recall)** over raw accuracy.  

---

## 🎯 Objective  
The goal of this project is to build a model that identifies poisonous mushrooms based on physical traits such as **cap color**, **odor**, **shape**, and **bruising**, while minimizing the risk of false negatives (predicting a poisonous mushroom as safe).  

> 🧠 *Eat as many mushrooms as we can — but nobody dies.*

---

## 📂 Repository Contents  
- 📓 **Mushroom_Model_Training.ipynb** — main notebook with cost-based training and SAFE threshold tuning  
- 📓 **Mushroom_Predict_Submission.ipynb** — inference + submission creation  
- 📂 **images/** — project visuals  
  - **safe_confusion_matrix_final.png** — final SAFE confusion matrix (**FN = 0**)  
- 📑 **mush_train.csv**, **mush_test.csv** — datasets  
- 📄 **submission_labels_HistGB_COST_OPT.csv** — predictions at OPT threshold (`0.035661`, leaderboard result)  
- 📄 **submission_labels_HistGB_COST_SAFE.csv** — predictions at SAFE threshold (`0.025661`, recall = 1.0)  

---

## 🧑‍💻 Approach  

### ✅ Final Workflow Summary  
1. **Load + prepare data**  
   - Loaded `mush_train.csv` and `mush_test.csv`.  
   - Separated target **`poisonous`** and defined categorical and boolean features.  

2. **Encode + impute**  
   - Applied **Target Encoding** (not one-hot) to capture category–label relationships.  
   - Converted boolean columns to integers.  
   - Imputed missing values using the most frequent strategy.  

3. **Model**  
   - Used **HistGradientBoostingClassifier** inside a scikit-learn **Pipeline**.  
   - Tuned via **GridSearchCV** over `learning_rate`, `max_depth`, `l2_regularization`, and `min_samples_leaf`.  

4. **Cost-based optimization**  
   - Custom scoring function penalized false negatives 100× more than false positives:  
     ```python
     cost = 100 * FN + 1 * FP
     ```  
   - This aligned the model’s optimization with real-world safety priorities.  

5. **Threshold tuning**  
   - Instead of the default 0.5 cutoff, thresholds were tuned to minimize total cost:  
     - **OPT threshold:** 0.035661 → best recall–precision balance (used in leaderboard submission)  
     - **SAFE threshold:** 0.025661 → stricter cutoff guaranteeing **FN = 0**  

6. **Submissions**  
   - Created two output files:  
     ```python
     save_submission(thr_opt,  "HistGB_COST_OPT")
     save_submission(thr_safe, "HistGB_COST_SAFE")
     ```  

---

## 🎧 Results  

### Confusion Matrix — SAFE Threshold (FN = 0)

![Confusion Matrix — SAFE (FN=0)](./images/safe_confusion_matrix_final.png)  

| Metric | Value |
|:--|--:|
| Accuracy | 0.8971 |
| Precision | 0.8251 |
| Recall | **1.0000** |
| Threshold | 0.008752 |

**Interpretation:**  
- **FN = 0:** no poisonous mushrooms were predicted safe — nobody dies.  
- **669 false positives:** some edible mushrooms marked as poisonous, a safe and acceptable trade-off.  
- **Perfect recall (1.0)** ensures full safety in deployment.  

> 🍄 The final SAFE model guarantees absolute safety — all poisonous mushrooms are correctly identified.

---

## 🛠 Tools Used  
1. **Python** — pandas, numpy, scikit-learn, category_encoders  
2. **JupyterLab** — exploration and experimentation  
3. **Matplotlib / Seaborn** — visualization  
4. **Custom cost function + make_scorer()** — recall-driven optimization  
5. **StratifiedKFold cross-validation** — robust model validation  

---

## 🎓 Key Learnings  
1. **Cost-sensitive modeling** is essential when false negatives have severe real-world consequences.  
2. **Threshold tuning** can change safety outcomes more than algorithm choice.  
3. **Target Encoding** efficiently handled high-cardinality categorical data.  
4. **Recall-first optimization** aligns model design with ethical and safety requirements.  
5. Final leaderboard confirmed real-world performance — **0 hospitalized** on hidden test set.  

---

## 💡 Final Summary  
✅ **OPT model (HistGB_COST_OPT):** best recall–precision balance; 0 hospitalized on test data.  
✅ **SAFE model (HistGB_COST_SAFE):** conservative fallback; **FN = 0**, recall = 1.0.  

> ⚖️ *Better to misclassify an edible mushroom as poisonous than a poisonous one as edible.*  

✅ **Final Goal Achieved:**  
> *Nobody dies — and we still eat plenty of mushrooms.* 🍽️  
