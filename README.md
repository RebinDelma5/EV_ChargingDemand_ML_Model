# ⚡ EV Charging Demand Prediction

A beginner-friendly machine-learning project on **72,856 EV charging sessions** (Sep 2021 – Sep 2022).

* **Regression** – predict how much energy (kWh) a session will use.
* **Classification** – label a session as **High** or **Normal** demand
  (High = demand ≥ the mean of the training data, 17.37 kWh).
* **Dashboard** – a Streamlit app with EDA, model results and live predictions.

## Files

| File | Purpose |
|---|---|
| `ChargingRecords.csv` | The dataset (you provide this) |
| `EV_Charging_ML.ipynb` | Full workflow: cleaning → EDA → features → models → SHAP/LIME → save with pickle |
| `app.py` | Streamlit dashboard (EDA, Regression, Classification) |
| `requirements.txt` | Python libraries to install |
| `EV_Charging_Presentation.pptx` | 7-slide summary presentation |
| `ev_models.pkl` | **Created by the notebook** – models, results and data used by the app |

Keep all of these files **in the same folder**.

## Quick start

**1. Install** (Python 3.10+ recommended; a virtual environment is a good idea)

```bash
pip install -r requirements.txt
```

**2. Run the notebook** (about 5–7 minutes)

```bash
jupyter notebook EV_Charging_ML.ipynb
```

Choose **Kernel → Restart & Run All**. When it finishes, `ev_models.pkl` appears in the folder.

**3. Start the dashboard**

```bash
streamlit run app.py
```

Your browser opens at <http://localhost:8501>.

## What the notebook does

| Step | Description |
|---|---|
| 1 | Load the data |
| 2 | Clean: remove duplicates, recalculate `Duration` from start/end times, drop impossible rows |
| 3 | EDA: demand by location, charger type, company, hour, weekday and month |
| 4 | New columns: `Day`, `IsWeekend`, `TimeOfDay`, and the average demand of a user's / charger's *earlier* sessions |
| 5 | Chronological split: train on the oldest 80%, test on the newest 20% |
| 6 | Regression: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM |
| 7 | Create the `DemandLevel` column (High / Normal) and compare Random Oversampling, SMOTE and ADASYN |
| 8 | Classification: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost, LightGBM |
| 9 | Explain the models with feature importance, SHAP and LIME |
| 10 | Save everything to `ev_models.pkl` with `pickle` |

### Avoiding data leakage
* `Duration` and the end time are **not** model inputs – they are unknown when a car plugs in.
  (Adding `Duration` would raise R² from 0.35 to 0.80, but the model could not be used in real life.)
* History columns only use sessions that happened **before** the current one.
* The test set is the **newest** 20% of the data, and all scaling/filling values come from the training data only.
* `UserID 0` (43% of rows) looks like a guest account, so it gets no user history and an `IsAnonymous` flag.

## Results (newest 20% of sessions)

**Regression** – best model: **LightGBM**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| LightGBM | 8.18 | 11.22 | 0.349 |
| Random Forest | 8.19 | 11.27 | 0.344 |
| XGBoost | 8.29 | 11.29 | 0.341 |
| Gradient Boosting | 8.55 | 11.42 | 0.326 |
| Linear Regression | 8.55 | 11.50 | 0.317 |
| Decision Tree | 8.51 | 11.70 | 0.293 |

**Classification** – best model: **XGBoost** (class balance 42% High / 58% Normal, balanced with ADASYN)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.711 | 0.612 | 0.826 | 0.703 | 0.811 |
| Random Forest | 0.724 | 0.634 | 0.787 | 0.702 | 0.813 |
| LightGBM | 0.716 | 0.622 | 0.805 | 0.702 | 0.810 |
| SVM | 0.700 | 0.603 | 0.809 | 0.691 | 0.780 |
| Decision Tree | 0.701 | 0.606 | 0.796 | 0.688 | 0.789 |
| KNN | 0.693 | 0.598 | 0.783 | 0.678 | 0.777 |
| Logistic Regression | 0.704 | 0.619 | 0.742 | 0.675 | 0.790 |

**Main findings**
* The strongest predictors are the **average demand of the user's and the charger's earlier sessions**, then the **time of day**.
* Demand peaks in the evening (busiest start hour: 18:00; about 42% of energy starts between 17:00 and 21:59).
* Slow (type 0) chargers deliver about 78% of the energy. Apartments and public areas are the top locations.

## Dashboard pages
* **📊 EDA** – KPIs, filters (location, charger type, company) and demand charts, including a weekday × hour heatmap.
* **📈 Regression** – model comparison, actual vs predicted, important columns, and a form to predict kWh for a new session.
* **🏷️ Classification** – class balance, imbalance techniques, model comparison, and a form that gives the chance of *High* demand.

## Troubleshooting
* **`ev_models.pkl not found`** – run the notebook to the end first, and keep `app.py` and the `.pkl` in the same folder.
* **Pickle errors after upgrading libraries** – the `.pkl` only loads with the library versions that created it. Re-run the notebook.
* **`ChargingRecords.csv` not found** – put the CSV next to the notebook.
* **Notebook is slow** – Random Forest, SVM and SHAP take the longest; this is normal.

## Limitations
* Predicting the exact kWh of one session is modest (R² ≈ 0.35, average error ≈ 8 kWh). Use predictions for planning, not billing.
* Brand-new users and chargers have no history, so the model relies on time, type and location only.
* The data covers one year, so yearly seasonality cannot be separated from overall growth.
