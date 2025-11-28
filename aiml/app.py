# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    from prophet import Prophet
    PROPHET_OK = True
except Exception:
    PROPHET_OK = False

st.set_page_config(page_title="Mobile App KPI Dashboard + AI", layout="wide", initial_sidebar_state="expanded")
st.title("📱 Mobile App KPI Dashboard — with AI-lite")

# ---------- loading data ----------
@st.cache_data
def load_data(path="data/daily_kpis.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date","dau","sessions","avg_session_duration_sec"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"Missing columns in {path}: {required - set(df.columns)}")
    df = df.sort_values("date").reset_index(drop=True)
    return df

try:
    dashboard = load_data()
except Exception as e:
    st.error(f"Failed to load KPI CSV: {e}")
    st.stop()

# ---------- sidebar controls ----------
with st.sidebar:
    st.header("Controls")
    nonzero = dashboard[dashboard["dau"] > 0]
    default_min = nonzero["date"].min() if not nonzero.empty else dashboard["date"].min()
    default_max = nonzero["date"].max() if not nonzero.empty else dashboard["date"].max()
    start_date = st.date_input("Start date", value=default_min, min_value=dashboard["date"].min(), max_value=dashboard["date"].max())
    end_date = st.date_input("End date", value=default_max, min_value=dashboard["date"].min(), max_value=dashboard["date"].max())
    show_zero = st.checkbox("Show days with DAU = 0", value=False)
    min_dau = st.slider("Minimum DAU to display (filter)", int(dashboard['dau'].min()), int(dashboard['dau'].max()), 0)
    run_ai = st.checkbox("Enable AI analytics (anomalies + forecast)", value=True)
    forecast_horizon = st.number_input("Forecast horizon (days)", min_value=7, max_value=90, value=14, step=7)
    st.markdown("---")
    st.caption("AI features run locally (IsolationForest, Prophet fallback).")

# ---------- filtering ----------
mask = (dashboard["date"] >= pd.to_datetime(start_date)) & (dashboard["date"] <= pd.to_datetime(end_date))
df = dashboard.loc[mask].copy()
if not show_zero:
    df = df[df["dau"] > 0]
if min_dau > 0:
    df = df[df["dau"] >= min_dau]

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---------- KPIs ----------
c1, c2, c3 = st.columns(3)
c1.metric("Peak DAU", int(df["dau"].max()))
c2.metric("Peak Sessions", int(df["sessions"].max()))
c3.metric("Avg Session Duration (sec)", f"{df['avg_session_duration_sec'].mean():.2f}")
st.markdown("---")

# ---------- plots ----------
st.subheader("Daily Active Users")
fig_dau = px.line(df, x="date", y="dau", markers=True, title="Daily Active Users")
fig_dau.update_layout(hovermode="x unified", template="plotly_dark", yaxis_title="DAU")
st.plotly_chart(fig_dau, use_container_width=True)

st.subheader("Sessions per Day")
fig_sessions = px.line(df, x="date", y="sessions", markers=True, title="Sessions per Day")
fig_sessions.update_layout(hovermode="x unified", template="plotly_dark", yaxis_title="Sessions")
st.plotly_chart(fig_sessions, use_container_width=True)

st.subheader("Avg Session Duration (sec)")
fig_avg = px.line(df, x="date", y="avg_session_duration_sec", markers=True, title="Avg Session Duration (sec)")
fig_avg.update_layout(hovermode="x unified", template="plotly_dark", yaxis_title="Seconds")
st.plotly_chart(fig_avg, use_container_width=True)

# ---------- AI analytics functions ----------
def detect_anomalies_iforest(series_vals, n_estimators=100, contamination=0.05, random_state=42):
    # series_vals: 1D numeric numpy array
    X = np.array(series_vals).reshape(-1,1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    preds = clf.fit_predict(Xs)  # -1 anomaly, 1 normal
    scores = -clf.score_samples(Xs)  # higher => more anomalous
    return preds, scores

def simple_forecast_prophet(df_ts, horizon=14):
   
    if not PROPHET_OK:
        raise RuntimeError("Prophet not available")
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.fit(df_ts.rename(columns={"date":"ds","y":"y"}))
    future = m.make_future_dataframe(periods=horizon)
    fc = m.predict(future)
    return fc[['ds','yhat','yhat_lower','yhat_upper']]

def simple_rolling_forecast(series, horizon=14, window=7):
    # conservative fallback: last value or rolling mean
    # return dates/ad-hoc with predicted value = rolling mean of last 'window' days
    last = series.index.max()
    last_val = float(series.iloc[-window:].mean())
    preds = []
    for i in range(1,horizon+1):
        preds.append(last_val)  # naive constant forecast
    return preds


if run_ai:
    st.markdown("## AI Insights & Anomalies")
    # 1) Anomaly detection on DAU
    with st.expander("Anomaly detection (DAU)"):
        contamination = st.slider("Anomaly sensitivity (higher = more anomalies)", 1, 20, 5)
        cont_val = contamination / 100.0
        preds, scores = detect_anomalies_iforest(df['dau'].values, contamination=cont_val)
        df['anomaly_iforest'] = preds  # -1 anomaly
        df['anomaly_score'] = scores
        anomalies = df[df['anomaly_iforest'] == -1].sort_values('anomaly_score', ascending=False)
        st.write("Detected anomaly days (IsolationForest):")
        if anomalies.empty:
            st.success("No anomalies detected with current sensitivity.")
        else:
            st.dataframe(anomalies[['date','dau','sessions','avg_session_duration_sec','anomaly_score']])
            st.write(f"Total anomalies: {len(anomalies)}")
        # highlight anomalies on plot
        fig = px.line(df, x='date', y='dau', title='DAU with anomalies (red points)')
        if not anomalies.empty:
            fig.add_scatter(x=anomalies['date'], y=anomalies['dau'], mode='markers', marker=dict(color='red',size=10), name='anomaly')
        fig.update_layout(template='plotly_dark', yaxis_title='DAU')
        st.plotly_chart(fig, use_container_width=True)

    # 2) Forecasting
    with st.expander("Forecasting (short-term)"):
        st.write("We try Prophet if installed; otherwise we fallback to a simple rolling mean.")
        horizon = int(forecast_horizon)
        if PROPHET_OK:
            # prepare prophet input
            prophet_df = df[['date','dau']].rename(columns={'dau':'y'})
            try:
                fc = simple_forecast_prophet(prophet_df, horizon=horizon)
                fc_recent = fc.tail(horizon)
                st.write("Prophet forecast (next days):")
                st.dataframe(fc_recent.rename(columns={'ds':'date','yhat':'predicted_dau'})[['date','predicted_dau','yhat_lower','yhat_upper']].reset_index(drop=True))
                figf = px.line(df[['date','dau']], x='date', y='dau', title='DAU with forecast')
                figf.add_scatter(x=fc['ds'], y=fc['yhat'], mode='lines', name='forecast', line=dict(color='orange'))
                st.plotly_chart(figf, use_container_width=True)
            except Exception as ex:
                st.error(f"Prophet forecasting failed: {ex}. Using fallback.")
                preds = simple_rolling_forecast(df.set_index('date')['dau'], horizon=horizon)
                st.write("Fallback forecast (NAIVE constant):")
                st.dataframe(pd.DataFrame({'date': pd.date_range(df['date'].max() + pd.Timedelta(days=1), periods=horizon), 'predicted_dau': preds}))
        else:
            preds = simple_rolling_forecast(df.set_index('date')['dau'], horizon=horizon)
            st.write("Prophet not installed. Fallback forecast (naive constant):")
            st.dataframe(pd.DataFrame({'date': pd.date_range(df['date'].max() + pd.Timedelta(days=1), periods=horizon), 'predicted_dau': preds}))

    # 3) Correlation / explanatory insight
    with st.expander("Correlation & quick insight"):
        corr = df['dau'].corr(df['avg_session_duration_sec'])
        st.write(f"Pearson correlation (DAU vs Avg Session Duration): **{corr:.3f}**")
        # quick textual summary
        if abs(corr) < 0.1:
            txt = "Weak correlation between DAU and session duration — volume and engagement seem independent."
        elif corr > 0.1 and corr < 0.5:
            txt = "Moderate positive correlation: when DAU increases, session duration tends to increase slightly."
        elif corr >= 0.5:
            txt = "Strong positive correlation: more users come and they also stay longer — great engagement synergy."
        elif corr < -0.1:
            txt = "Negative correlation: more users are associated with shorter sessions, possibly due to new short-sessions users."
        else:
            txt = "No clear relationship."
        st.markdown(f"**Quick insight:** {txt}")

        # list top 5 days with highest avg_session_duration
        st.markdown("Top 5 days by avg session duration:")
        st.dataframe(df.sort_values('avg_session_duration_sec', ascending=False).head(5)[['date','dau','avg_session_duration_sec']])

    # 4) Auto-insights summary (rule-based)
    with st.expander("Autogenerated Insights (one-paragraph)"):
        mean_dau = df['dau'].mean()
        mean_dur = df['avg_session_duration_sec'].mean()
        num_anom = int((df['anomaly_iforest']==-1).sum()) if 'anomaly_iforest' in df.columns else 0
        insight_lines = []
        insight_lines.append(f"During the selected range, mean DAU = {mean_dau:.0f} and mean session duration = {mean_dur:.1f} sec (~{mean_dur/60:.1f} min).")
        if num_anom > 0:
            insight_lines.append(f"{num_anom} anomalous day(s) were detected (IsolationForest); inspect those dates for events or incidents.")
        if corr > 0.2:
            insight_lines.append("DAU and session duration show positive correlation, indicating traffic increases align with engagement.")
        elif corr < -0.2:
            insight_lines.append("Negative DAU-duration correlation detected; spikes in traffic may be driven by low-engagement cohorts.")
        else:
            insight_lines.append("No strong correlation between DAU and session duration; volume and engagement are largely independent.")
        st.write(" ".join(insight_lines))

st.caption("Data source: data/daily_kpis.csv — AI features run locally. IsolationForest for anomalies, Prophet optional for forecasting.")

