"""
EV Charging dashboard (EDA + Regression + Classification).
Run the notebook first (it creates ev_models.pkl), then:   streamlit run app.py
"""
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EV Charging Analysis", page_icon="⚡", layout="wide")
sns.set_theme(style="whitegrid")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------------------------------------------------------------- load the pickle file
@st.cache_resource
def load_pickle():
    with open("ev_models.pkl", "rb") as f:
        return pickle.load(f)


try:
    saved = load_pickle()
except FileNotFoundError:
    st.error("ev_models.pkl not found. Run the notebook first and keep this file in the same folder.")
    st.stop()

df = saved["eda_data"]


# ---------------------------------------------------------------- helper: turn the form inputs into model input
def session_form(key):
    """Shows the input widgets and returns a one-row table the models understand."""
    c1, c2, c3 = st.columns(3)
    location = c1.selectbox("Location", sorted(df["Location"].unique()), key=key + "loc")
    ctype = c2.selectbox("Charger type", [0, 1], key=key + "type")
    company = c3.selectbox("Charger company", [0, 1], index=1, key=key + "comp")
    d1, d2 = st.columns(2)
    day = d1.date_input("Start date", df["Start"].max().date(), key=key + "date")
    time = d2.time_input("Start time", pd.Timestamp("18:00").time(), key=key + "time")
    h1, h2 = st.columns(2)
    ch_mean = h1.number_input("Average kWh of this charger's earlier sessions", 0.0, 100.0, 17.0, key=key + "chm")
    ch_count = h2.number_input("Number of earlier sessions on this charger", 0, 2000, 50, key=key + "chc")
    guest = st.checkbox("Unknown / guest user", value=True, key=key + "guest")
    u_mean, u_dur, u_count = np.nan, np.nan, 0
    if not guest:
        u1, u2, u3 = st.columns(3)
        u_mean = u1.number_input("User's average kWh (earlier sessions)", 0.0, 100.0, 17.0, key=key + "um")
        u_dur = u2.number_input("User's average duration in minutes", 0.0, 2000.0, 140.0, key=key + "ud")
        u_count = u3.number_input("User's earlier sessions", 0, 2000, 10, key=key + "uc")

    when = pd.Timestamp.combine(day, time)
    hour = when.hour
    time_of_day = "Morning" if 5 <= hour < 12 else "Afternoon" if 12 <= hour < 17 else "Evening" if 17 <= hour < 22 else "Night"
    row = {"ChargerCompany": company, "ChargerType": ctype, "Hour": hour, "Day": when.day, "Month": when.month,
           "DayOfWeek": when.dayofweek, "IsWeekend": int(when.dayofweek >= 5),
           "Charger_prev_mean": ch_mean, "Charger_prev_count": ch_count, "User_prev_mean": u_mean,
           "User_prev_duration": u_dur, "User_prev_count": u_count, "IsAnonymous": int(guest),
           "Location": location, "TimeOfDay": time_of_day}
    X = pd.get_dummies(pd.DataFrame([row]), dtype=int)                    # text -> 0/1 columns
    X = X.reindex(columns=saved["columns"], fill_value=0)                 # same columns as training
    return X.fillna(saved["fill_values"])                                 # same missing-value rule as training


def bar(series, title):
    st.caption(title)
    st.bar_chart(series)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚡ EV Charging")
page = st.sidebar.radio("Go to", ["📊 EDA", "📈 Regression", "🏷️ Classification"])


# ================================================================ EDA
if page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    st.sidebar.subheader("Filters")
    locs = st.sidebar.multiselect("Location", sorted(df["Location"].unique()))
    types = st.sidebar.multiselect("Charger type", [0, 1])
    comps = st.sidebar.multiselect("Charger company", [0, 1])
    f = df
    if locs:
        f = f[f["Location"].isin(locs)]
    if types:
        f = f[f["ChargerType"].isin(types)]
    if comps:
        f = f[f["ChargerCompany"].isin(comps)]
    if f.empty:
        st.warning("No sessions match these filters."); st.stop()

    k = st.columns(5)
    k[0].metric("Sessions", f"{len(f):,}")
    k[1].metric("Total energy (kWh)", f"{f['Demand'].sum():,.0f}")
    k[2].metric("Average kWh / session", f"{f['Demand'].mean():.1f}")
    k[3].metric("Average duration (min)", f"{f['Duration'].mean():.0f}")
    k[4].metric("Chargers", f"{f['ChargerID'].nunique():,}")

    a, b = st.columns(2)
    with a:
        bar(f.groupby("Location")["Demand"].sum().sort_values(), "Total demand (kWh) by location")
        bar(f.groupby("ChargerType")["Demand"].sum(), "Total demand (kWh) by charger type")
        bar(f.groupby("ChargerCompany")["Demand"].sum(), "Total demand (kWh) by charger company")
    with b:
        bar(f.groupby("Hour")["Demand"].sum(), "Total demand (kWh) by start hour")
        week = f.groupby("DayOfWeek")["Demand"].sum(); week.index = [DAYS[i] for i in week.index]
        bar(week, "Total demand (kWh) by day of week")
        bar(f.groupby("Month")["Demand"].sum(), "Total demand (kWh) by month")

    st.caption("Peak periods (darker = more demand)")
    heat = f.pivot_table(index="DayOfWeek", columns="Hour", values="Demand", aggfunc="sum")
    heat.index = [DAYS[i] for i in heat.index]
    fig, ax = plt.subplots(figsize=(12, 3.2)); sns.heatmap(heat, cmap="YlOrRd", ax=ax); st.pyplot(fig)

    c1, c2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 3)); f["Demand"].plot.hist(bins=50, ax=ax, title="Demand per session (kWh)"); c1.pyplot(fig)
    fig, ax = plt.subplots(figsize=(6, 3)); f["Duration"].clip(upper=1000).plot.hist(bins=50, ax=ax, title="Duration (minutes)"); c2.pyplot(fig)


# ================================================================ REGRESSION
elif page == "📈 Regression":
    st.title("📈 Regression – predict energy demand (kWh)")
    st.write(f"Best model: **{saved['reg_name']}**. Trained on the oldest 80% of sessions and tested on the newest 20%.")
    st.subheader("Model comparison")
    st.dataframe(saved["reg_results"])
    st.caption("MAE / RMSE: smaller is better. R²: closer to 1 is better.")

    a, b = st.columns(2)
    with a:
        st.caption("Error by model (smaller is better)")
        st.bar_chart(saved["reg_results"][["MAE", "RMSE"]])
    with b:
        n = min(3000, len(saved["test_actual"]))
        idx = np.random.RandomState(0).choice(len(saved["test_actual"]), n, replace=False)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(saved["test_actual"][idx], saved["test_pred"][idx], s=4, alpha=0.3)
        m = saved["test_actual"].max(); ax.plot([0, m], [0, m], "r--")
        ax.set(xlabel="Actual kWh", ylabel="Predicted kWh", title="Actual vs predicted (test data)")
        st.pyplot(fig)

    st.subheader("Most important columns")
    st.bar_chart(saved["reg_importance"])

    st.subheader("🔮 Predict a new session")
    X_new = session_form("reg_")
    if st.button("Predict energy", key="reg_btn"):
        pred = saved["reg_model"].predict(X_new)[0]
        st.metric("Predicted energy demand", f"{pred:.1f} kWh")
        st.caption("The model only uses information known when the car plugs in (no duration).")


# ================================================================ CLASSIFICATION
else:
    st.title("🏷️ Classification – High or Normal demand")
    st.write(f"A session is **High** if its energy is **≥ {saved['mean_demand']:.2f} kWh** (the average of the training data), otherwise **Normal**.")
    st.write(f"Best model: **{saved['clf_name']}**. Imbalance technique used: **{saved['sampler_used']}**.")

    a, b = st.columns(2)
    with a:
        st.caption("Number of High vs Normal sessions")
        st.bar_chart(df["DemandLevel"].value_counts())
    with b:
        st.caption("Imbalance techniques compared (LightGBM)")
        st.dataframe(saved["imbalance_results"])

    st.subheader("Model comparison")
    st.dataframe(saved["clf_results"])
    st.bar_chart(saved["clf_results"][["Accuracy", "F1", "ROC_AUC"]])
    st.subheader("Most important columns")
    st.bar_chart(saved["clf_importance"])

    st.subheader("🔮 Predict a new session")
    X_new = session_form("clf_")
    if st.button("Predict demand level", key="clf_btn"):
        prob = saved["clf_model"].predict_proba(X_new)[0, 1]
        c1, c2 = st.columns(2)
        c1.metric("Chance of HIGH demand", f"{prob:.0%}")
        c2.metric("Prediction", "🔴 High" if prob >= 0.5 else "🟢 Normal")
