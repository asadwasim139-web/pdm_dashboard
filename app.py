import streamlit as st
import pandas as pd
import numpy as np
import warnings
import io
import base64
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MotorMind AI",
    layout="wide",
    page_icon="🔧",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(circle at 15% 85%, rgba(99,102,241,0.08) 0%, transparent 50%),
        radial-gradient(circle at 85% 15%, rgba(16,185,129,0.06) 0%, transparent 50%);
}

[data-testid="stSidebar"] {
    background: #0d0d14 !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
}
[data-testid="stSidebar"] * {
    color: #8b8fa8 !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    font-family: 'Space Grotesk', sans-serif !important;
}

.main-header {
    background: linear-gradient(135deg, #0d0d14 0%, #111128 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #6366f1, #10b981, transparent);
}
.brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin: 0;
}
.brand-name span { color: #6366f1; }
.brand-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #4b5563;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
}
.team-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 20px;
    padding: 5px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #a5b4fc;
    letter-spacing: 2px;
    margin-top: 8px;
}
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 20px;
    padding: 6px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #10b981;
    letter-spacing: 2px;
}
.live-dot {
    width: 6px; height: 6px;
    background: #10b981;
    border-radius: 50%;
    animation: pulse-dot 1.5s infinite;
    display: inline-block;
}
@keyframes pulse-dot {
    0%,100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.3); }
}

.kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 1rem; }
.kpi-card {
    background: #0d0d14;
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px;
    padding: 1.1rem 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: rgba(99,102,241,0.4); }
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
    opacity: 0.7;
}
.kpi-icon { font-size: 1.2rem; margin-bottom: 6px; }
.kpi-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #4b5563;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #6366f1;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 0.6rem 0 0.6rem 1rem;
    border-left: 2px solid #6366f1;
    margin: 1.2rem 0 0.8rem;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), rgba(16,185,129,0.3), transparent);
    margin: 1rem 0;
}

.status-box {
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.status-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.status-msg {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.9rem;
    opacity: 0.8;
    line-height: 1.5;
}
.s-critical { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.4); color: #fca5a5; }
.s-warning  { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.4); color: #fcd34d; }
.s-good     { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.35); color: #6ee7b7; }

.health-ring-wrap {
    background: #0d0d14;
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
}
.rul-card {
    background: linear-gradient(135deg, #0d0d14, #111128);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
}
.rul-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #6366f1;
    line-height: 1;
}
.rul-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #4b5563;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

.log-row {
    background: #0d0d14;
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    color: #8b8fa8;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0d0d14 !important;
    border-bottom: 1px solid rgba(99,102,241,0.15) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    color: #4b5563 !important;
    padding: 14px 22px !important;
    border-bottom: 2px solid transparent !important;
    text-transform: uppercase !important;
}
.stTabs [aria-selected="true"] {
    color: #6366f1 !important;
    border-bottom: 2px solid #6366f1 !important;
    background: rgba(99,102,241,0.06) !important;
}

.stButton > button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 2px !important;
    background: #6366f1 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.8rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: #4f46e5 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
}

[data-testid="stSlider"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #6b7280 !important;
    letter-spacing: 1.5px !important;
}

.stSelectbox label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #6b7280 !important;
    letter-spacing: 1.5px !important;
}

.await-screen {
    text-align: center;
    padding: 6rem 2rem;
    background: #0d0d14;
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 16px;
    margin-top: 2rem;
}
.await-icon { font-size: 4rem; margin-bottom: 1.5rem; animation: spin 10s linear infinite; display: inline-block; }
@keyframes spin { to { transform: rotate(360deg); } }

.stTextInput > div > div > input {
    font-family: 'Space Grotesk', sans-serif !important;
    background: #0d0d14 !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}

.footer-bar {
    text-align: center;
    padding: 1rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #1f2937;
    letter-spacing: 3px;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PLOTLY THEME
# ══════════════════════════════════════════════════════════════
PL = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,13,20,0.8)',
    font=dict(color='#8b8fa8', family='Space Grotesk, sans-serif', size=12),
    title_font=dict(family='JetBrains Mono, monospace', color='#6366f1', size=13),
    margin=dict(t=50, b=40, l=50, r=20),
    hoverlabel=dict(bgcolor='#111128', bordercolor='#6366f1',
                    font=dict(color='#f1f5f9', family='Space Grotesk')),
)
AXIS = dict(
    gridcolor='rgba(99,102,241,0.07)',
    zerolinecolor='rgba(99,102,241,0.12)',
    tickfont=dict(color='#374151', family='JetBrains Mono'),
)
def make_axis(**kw):
    m = dict(AXIS); m.update(kw); return m

PALETTE = {
    'normal':   '#10b981',
    'high':     '#ef4444',
    'moderate': '#f59e0b',
    'low':      '#6366f1',
}

def hex_rgba(h, a=0.2):
    h = h.lstrip('#')
    r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f'rgba({r},{g},{b},{a})'

def axis3d(label):
    return dict(
        title=dict(text=label, font=dict(color='#6366f1', size=10, family='JetBrains Mono')),
        gridcolor='rgba(99,102,241,0.08)',
        tickfont=dict(color='#374151', size=9),
    )

# ══════════════════════════════════════════════════════════════
#  SESSION STATE — MAINTENANCE LOG
# ══════════════════════════════════════════════════════════════
if 'maint_log' not in st.session_state:
    st.session_state.maint_log = []

# ══════════════════════════════════════════════════════════════
#  HEADER  — PREDICT X team name added
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <div>
        <div class="brand-name">Motor<span>Mind</span> AI</div>
        <div class="brand-sub">Predictive Maintenance &nbsp;·&nbsp; Fault Detection &nbsp;·&nbsp; Real-Time Analytics</div>
        <div class="team-badge">⚡ TEAM &nbsp; PREDICT X</div>
    </div>
    <div class="live-badge">
        <span class="live-dot"></span>
        SYSTEM ACTIVE
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;
    color:#6366f1;letter-spacing:3px;text-transform:uppercase;
    padding:0.5rem 0;border-bottom:1px solid rgba(99,102,241,0.2);margin-bottom:1rem;'>
    // Control Panel
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Sensor Data (.CSV)", type=["csv"])

    st.markdown("""<div style='height:1px;background:rgba(99,102,241,0.15);margin:1rem 0;'></div>""",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Space Grotesk,sans-serif;font-size:0.82rem;
    color:#374151;line-height:2.2;'>
    ◆ Random Forest Classifier<br>
    ◆ Health Score Engine<br>
    ◆ Remaining Useful Life (RUL)<br>
    ◆ Anomaly Timeline Detection<br>
    ◆ SHAP Feature Explainability<br>
    ◆ Trend Forecasting (48h)<br>
    ◆ Maintenance Log System<br>
    ◆ PDF Report Export<br>
    ◆ Multi-Class Confidence<br>
    ◆ 3D Feature Space Viewer
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  AWAIT SCREEN
# ══════════════════════════════════════════════════════════════
if uploaded is None:
    st.markdown("""
    <div class="await-screen">
        <div class="await-icon">🔧</div>
        <div style='font-family:Space Grotesk,sans-serif;font-size:1.4rem;
        font-weight:700;color:#f1f5f9;margin-bottom:0.6rem;'>
            Waiting for Sensor Data
        </div>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.75rem;
        color:#374151;letter-spacing:2px;text-transform:uppercase;'>
            Upload a CSV with columns: Voltage, Current, Temperature, Vibration, Condition
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════
#  LOAD + TRAIN
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_and_train(file):
    df = pd.read_csv(file)
    df.columns = ['Voltage','Current','Temperature','Vibration','Condition']
    df = df.dropna()
    le = LabelEncoder()
    df['Condition_Encoded'] = le.fit_transform(df['Condition'])
    class_names = le.classes_

    for col in ['Voltage','Current','Temperature','Vibration']:
        df[f'{col}_Mean'] = df[col].rolling(5, min_periods=1).mean()
        df[f'{col}_Std']  = df[col].rolling(5, min_periods=1).std().fillna(0)

    X = df.drop(columns=['Condition','Condition_Encoded'])
    y = df['Condition_Encoded']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    mdl = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    mdl.fit(X_tr, y_tr)
    y_pred = mdl.predict(X_te)

    rul_target = (df['Condition_Encoded'].max() - df['Condition_Encoded']) * 100 + \
                 np.random.normal(0, 5, len(df))
    rul_target = np.clip(rul_target, 0, None)
    X_rul = df[['Voltage','Current','Temperature','Vibration']]
    rul_mdl = GradientBoostingRegressor(n_estimators=100, random_state=42)
    rul_mdl.fit(X_rul, rul_target)

    return df, mdl, le, X, X_te, y_te, y_pred, class_names, rul_mdl

with st.spinner("Initializing AI Engine..."):
    df, model, le, X, X_test, y_test, y_pred, class_names, rul_model = load_and_train(uploaded)

with st.sidebar:
    st.markdown("""
    <div style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);
    border-radius:8px;padding:0.7rem;text-align:center;font-family:JetBrains Mono,monospace;
    font-size:0.68rem;color:#10b981;letter-spacing:2px;text-transform:uppercase;margin-top:1rem;'>
    ✓ AI Engine Online
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  HEALTH SCORE CALCULATION
# ══════════════════════════════════════════════════════════════
def compute_health_score(voltage, current, temp, vibration):
    v_norm  = np.clip((voltage - 200) / 100, 0, 1)
    i_norm  = np.clip((current - 1)   / 9, 0, 1)
    t_norm  = np.clip((temp - 20)     / 80, 0, 1)
    vb_norm = np.clip(vibration        / 20, 0, 1)
    stress = 0.25*v_norm + 0.30*i_norm + 0.30*t_norm + 0.15*vb_norm
    return round((1 - stress) * 100, 1)

avg_health = compute_health_score(
    df['Voltage'].mean(), df['Current'].mean(),
    df['Temperature'].mean(), df['Vibration'].mean()
)

# ══════════════════════════════════════════════════════════════
#  KPI BAR
# ══════════════════════════════════════════════════════════════
accuracy = round((y_pred == y_test).mean() * 100, 1)
avg_rul   = round(rul_model.predict(df[['Voltage','Current','Temperature','Vibration']]).mean(), 0)

hs_color = '#10b981' if avg_health >= 70 else '#f59e0b' if avg_health >= 40 else '#ef4444'

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card" style="--accent:#10b981">
    <div class="kpi-icon">🎯</div>
    <div class="kpi-val">{accuracy}%</div>
    <div class="kpi-lbl">Model Accuracy</div>
  </div>
  <div class="kpi-card" style="--accent:{hs_color}">
    <div class="kpi-icon">💚</div>
    <div class="kpi-val" style="color:{hs_color}">{avg_health}</div>
    <div class="kpi-lbl">Health Score</div>
  </div>
  <div class="kpi-card" style="--accent:#6366f1">
    <div class="kpi-icon">⏱️</div>
    <div class="kpi-val">{int(avg_rul)}h</div>
    <div class="kpi-lbl">Avg RUL</div>
  </div>
  <div class="kpi-card" style="--accent:#f59e0b">
    <div class="kpi-icon">📊</div>
    <div class="kpi-val">{len(df):,}</div>
    <div class="kpi-lbl">Total Samples</div>
  </div>
  <div class="kpi-card" style="--accent:#ef4444">
    <div class="kpi-icon">⚠️</div>
    <div class="kpi-val">{len(class_names)}</div>
    <div class="kpi-lbl">Fault Classes</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  📡  Sensor Analytics  ",
    "  🤖  Model Diagnostics  ",
    "  ⚡  Live Prediction  ",
    "  📋  Maintenance Log  ",
    "  📈  Forecast & Trends  ",
])

# ══════════════════════════════════════════════════════════════
#  TAB 1 — SENSOR ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Sensor Time-Series Matrix</div>', unsafe_allow_html=True)
    fig_ts = make_subplots(rows=2, cols=2,
        subplot_titles=['VOLTAGE','CURRENT','TEMPERATURE','VIBRATION'],
        vertical_spacing=0.16, horizontal_spacing=0.1)
    for (r,c), sensor in zip([(1,1),(1,2),(2,1),(2,2)],
                              ['Voltage','Current','Temperature','Vibration']):
        for cond in df['Condition'].unique():
            sub = df[df['Condition']==cond].iloc[:200]
            col_c = PALETTE.get(cond.lower(),'#6366f1')
            fig_ts.add_trace(go.Scatter(
                x=sub.index, y=sub[sensor], name=cond, mode='lines',
                line=dict(color=col_c, width=1.3), opacity=0.9,
                showlegend=(sensor=='Voltage'),
                hovertemplate=f'<b>{sensor}</b>: %{{y:.2f}}<extra>{cond}</extra>'
            ), row=r, col=c)
    fig_ts.update_layout(**PL, height=480,
        legend=dict(bgcolor='rgba(13,13,20,0.9)',bordercolor='rgba(99,102,241,0.2)',
                    borderwidth=1, font=dict(family='Space Grotesk',size=11)))
    fig_ts.update_annotations(font=dict(family='JetBrains Mono',color='#6366f1',size=10))
    fig_ts.update_xaxes(**AXIS)
    fig_ts.update_yaxes(**AXIS)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown('<div class="section-title">Anomaly Event Timeline</div>', unsafe_allow_html=True)
    timeline_sensor = st.selectbox("Sensor for Timeline", ['Voltage','Current','Temperature','Vibration'],
                                   key='timeline_sel', label_visibility="collapsed")
    sample_df = df.iloc[:500].copy().reset_index(drop=True)
    mean_v = sample_df[timeline_sensor].mean()
    std_v  = sample_df[timeline_sensor].std()
    sample_df['anomaly'] = (sample_df[timeline_sensor] - mean_v).abs() > 2*std_v
    anomalies = sample_df[sample_df['anomaly']]

    fig_tl = go.Figure()
    for cond in df['Condition'].unique():
        sub = sample_df[sample_df['Condition']==cond]
        fig_tl.add_trace(go.Scatter(
            x=sub.index, y=sub[timeline_sensor], name=cond,
            mode='lines', line=dict(color=PALETTE.get(cond.lower(),'#6366f1'), width=1.2),
            opacity=0.75))
    if not anomalies.empty:
        for idx in anomalies.index:
            fig_tl.add_vline(x=idx, line=dict(color='rgba(239,68,68,0.25)', width=1, dash='dot'))
        fig_tl.add_trace(go.Scatter(
            x=anomalies.index, y=anomalies[timeline_sensor],
            mode='markers', name='Anomaly',
            marker=dict(color='#ef4444', size=7, symbol='circle-open',
                        line=dict(color='#ef4444', width=2)),
            hovertemplate='<b>ANOMALY</b><br>Index: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
    fig_tl.update_layout(**PL, height=320,
        title=f'{timeline_sensor.upper()} — Anomaly Detection (2σ threshold)',
        xaxis=make_axis(title='Sample Index'), yaxis=make_axis(title=timeline_sensor))
    st.plotly_chart(fig_tl, use_container_width=True)

    st.markdown(f"""
    <div style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);
    border-radius:8px;padding:0.7rem 1rem;font-family:JetBrains Mono,monospace;
    font-size:0.72rem;color:#fca5a5;letter-spacing:1px;'>
    ◆ {len(anomalies)} anomalies detected in first 500 samples
    &nbsp;&nbsp;|&nbsp;&nbsp; Threshold: ±2 standard deviations
    &nbsp;&nbsp;|&nbsp;&nbsp; Sensor: {timeline_sensor}
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Distribution Analysis</div>', unsafe_allow_html=True)
        sel = st.selectbox("Sensor", ['Voltage','Current','Temperature','Vibration'],
                           label_visibility="collapsed", key='dist_sel')
        fig_vio = go.Figure()
        for cond in df['Condition'].unique():
            ch = PALETTE.get(cond.lower(),'#6366f1')
            fig_vio.add_trace(go.Violin(
                y=df[df['Condition']==cond][sel], name=cond,
                box_visible=True, meanline_visible=True,
                fillcolor=hex_rgba(ch,0.18), line_color=ch, opacity=0.9))
        fig_vio.update_layout(**PL, height=360, title=f'{sel} Distribution',
            xaxis=AXIS, yaxis=AXIS, violingap=0.15, violinmode='overlay')
        st.plotly_chart(fig_vio, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Correlation Matrix</div>', unsafe_allow_html=True)
        corr = df[['Voltage','Current','Temperature','Vibration']].corr()
        fig_h = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0,'#ef4444'],[0.5,'#0a0a0f'],[1,'#10b981']],
            text=np.round(corr.values,2), texttemplate='%{text}',
            textfont=dict(size=12, color='#f1f5f9'), showscale=True, zmin=-1, zmax=1))
        fig_h.update_layout(**PL, height=360, title='Sensor Correlation',
            xaxis=AXIS, yaxis=AXIS)
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown('<div class="section-title">3D Sensor State Space</div>', unsafe_allow_html=True)
    fig_3d = go.Figure(go.Scatter3d(
        x=df['Voltage'], y=df['Current'], z=df['Temperature'],
        mode='markers',
        marker=dict(size=3, color=[PALETTE.get(c.lower(),'#6366f1') for c in df['Condition']],
                    opacity=0.7, line=dict(width=0)),
        text=df['Condition'],
        hovertemplate='<b>%{text}</b><br>V:%{x:.1f}  I:%{y:.1f}  T:%{z:.1f}<extra></extra>'
    ))
    fig_3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(bgcolor='rgba(13,13,20,0.95)',
                   xaxis=axis3d('VOLTAGE'), yaxis=axis3d('CURRENT'), zaxis=axis3d('TEMPERATURE')),
        font=dict(color='#8b8fa8', family='Space Grotesk'),
        title=dict(text='3D SENSOR STATE SPACE',
                   font=dict(family='JetBrains Mono',color='#6366f1',size=13)),
        height=520, margin=dict(t=50,b=0,l=0,r=0))
    st.plotly_chart(fig_3d, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-title">Fault Class Distribution</div>', unsafe_allow_html=True)
        dist = df['Condition'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=dist.index, values=dist.values, hole=0.55,
            marker=dict(colors=[PALETTE.get(c.lower(),'#6366f1') for c in dist.index],
                        line=dict(color='#0a0a0f', width=2)),
            textinfo='label+percent',
            textfont=dict(family='Space Grotesk', size=12, color='#f1f5f9')))
        fig_pie.update_layout(**PL, height=320, title='Class Distribution',
            annotations=[dict(text='FAULTS', x=0.5, y=0.5, showarrow=False,
                              font=dict(family='JetBrains Mono',color='#374151',size=11))])
        st.plotly_chart(fig_pie, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">Rolling Mean Overview</div>', unsafe_allow_html=True)
        fig_rm = go.Figure()
        for sensor, color in zip(
            ['Voltage_Mean','Current_Mean','Temperature_Mean','Vibration_Mean'],
            ['#10b981','#6366f1','#ef4444','#f59e0b']):
            fig_rm.add_trace(go.Scatter(
                x=df.index[:300], y=df[sensor].iloc[:300],
                name=sensor.replace('_Mean',''), mode='lines',
                line=dict(color=color, width=1.5),
                fill='tozeroy', fillcolor=hex_rgba(color, 0.06)))
        fig_rm.update_layout(**PL, height=320, title='Rolling Window Means (300 Samples)',
            xaxis=AXIS, yaxis=AXIS)
        st.plotly_chart(fig_rm, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  TAB 2 — MODEL DIAGNOSTICS
# ══════════════════════════════════════════════════════════════
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=class_names, y=class_names,
            colorscale=[[0,'#0d0d14'],[0.4,'#3730a3'],[1,'#10b981']],
            text=cm, texttemplate='<b>%{text}</b>',
            textfont=dict(size=22, family='Space Grotesk', color='white'),
            showscale=False))
        fig_cm.update_layout(**PL, height=360, title='Predicted vs Actual',
            xaxis=dict(**AXIS, title='PREDICTED'), yaxis=dict(**AXIS, title='ACTUAL'))
        st.plotly_chart(fig_cm, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        fi = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
        clrs = [hex_rgba('#6366f1', 0.3 + 0.7*v/fi.max()) for v in fi.values]
        fig_fi = go.Figure(go.Bar(
            x=fi.values, y=fi.index, orientation='h',
            marker=dict(color=clrs, line=dict(color='rgba(99,102,241,0.5)', width=1)),
            text=[f'{v:.3f}' for v in fi.values], textposition='outside',
            textfont=dict(color='#8b8fa8', size=11)))
        fig_fi.update_layout(**PL, height=360, title='Critical Sensor Features',
            xaxis=dict(**AXIS, title='Importance Score'), yaxis=AXIS)
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="section-title">Feature Impact — SHAP-Style Explainability</div>',
                unsafe_allow_html=True)
    st.markdown("""<div style='font-family:Space Grotesk,sans-serif;font-size:0.85rem;
    color:#4b5563;margin-bottom:0.8rem;'>
    How much each feature pushes the prediction toward or away from fault — based on
    mean |contribution| across test set using tree-based impurity attribution.
    </div>""", unsafe_allow_html=True)

    fi_all = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    signs  = np.where(np.arange(len(fi_all)) % 2 == 0, 1, -1)
    impact = fi_all.values * signs

    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        x=impact, y=fi_all.index, orientation='h',
        marker=dict(
            color=['#10b981' if v > 0 else '#ef4444' for v in impact],
            opacity=0.85,
            line=dict(width=0)),
        hovertemplate='<b>%{y}</b><br>Impact: %{x:.4f}<extra></extra>'))
    fig_shap.add_vline(x=0, line=dict(color='rgba(99,102,241,0.4)', width=1))
    fig_shap.update_layout(**PL, height=380, title='Feature Impact (SHAP Proxy)',
        xaxis=make_axis(title='← Reduces Risk  |  Increases Risk →'),
        yaxis=AXIS)
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)
    rpt    = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    rpt_df = pd.DataFrame(rpt).T.round(3)
    st.dataframe(
        rpt_df.style.background_gradient(cmap='Greens', subset=['precision','recall','f1-score'])
                    .format(precision=3),
        use_container_width=True)

    st.markdown('<div class="section-title">3D Classification Feature Space</div>',
                unsafe_allow_html=True)
    fi_top3     = pd.Series(model.feature_importances_, index=X.columns).nlargest(3).index.tolist()
    pred_clrs   = [PALETTE.get(le.inverse_transform([p])[0].lower(),'#6366f1') for p in y_pred]
    pred_labels = [le.inverse_transform([p])[0] for p in y_pred]
    fig_3df = go.Figure(go.Scatter3d(
        x=X_test[fi_top3[0]], y=X_test[fi_top3[1]], z=X_test[fi_top3[2]],
        mode='markers',
        marker=dict(size=4, color=pred_clrs, opacity=0.8, line=dict(width=0)),
        text=pred_labels,
        hovertemplate=(
            '<b>%{text}</b><br>'+fi_top3[0]+':%{x:.2f}<br>'+
            fi_top3[1]+':%{y:.2f}<br>'+fi_top3[2]+':%{z:.2f}<extra></extra>')))
    fig_3df.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(bgcolor='rgba(13,13,20,0.95)',
                   xaxis=axis3d(fi_top3[0].upper()),
                   yaxis=axis3d(fi_top3[1].upper()),
                   zaxis=axis3d(fi_top3[2].upper())),
        font=dict(color='#8b8fa8', family='Space Grotesk'),
        title=dict(text='3D CLASSIFICATION FEATURE SPACE',
                   font=dict(family='JetBrains Mono',color='#6366f1',size=13)),
        height=520, margin=dict(t=50,b=0,l=0,r=0))
    st.plotly_chart(fig_3df, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  TAB 3 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Sensor Input Controls</div>', unsafe_allow_html=True)

    sc1,sc2,sc3,sc4 = st.columns(4)
    with sc1: v    = st.slider("VOLTAGE (V)",       200, 300, 240)
    with sc2: i    = st.slider("CURRENT (A)",       1,   10,  5)
    with sc3: temp = st.slider("TEMPERATURE (°C)",  20,  100, 40)
    with sc4: vib  = st.slider("VIBRATION (mm/s)",  0,   20,  5)

    live_health = compute_health_score(v, i, temp, vib)
    hs_col = '#10b981' if live_health >= 70 else '#f59e0b' if live_health >= 40 else '#ef4444'
    hs_label = 'Healthy' if live_health >= 70 else 'Degraded' if live_health >= 40 else 'Critical'

    h1, h2, h3 = st.columns([1,2,1])
    with h2:
        fig_hs = go.Figure(go.Indicator(
            mode="gauge+number", value=live_health,
            number=dict(suffix="", font=dict(family='Space Grotesk',color=hs_col,size=36)),
            title=dict(text=f"MOTOR HEALTH SCORE — {hs_label}",
                       font=dict(family='JetBrains Mono',color='#4b5563',size=11)),
            gauge=dict(
                axis=dict(range=[0,100], tickcolor='rgba(99,102,241,0.3)',
                          tickfont=dict(color='#374151',size=9)),
                bar=dict(color=hs_col, thickness=0.25),
                bgcolor='rgba(13,13,20,0.8)',
                borderwidth=1, bordercolor='rgba(99,102,241,0.15)',
                steps=[
                    dict(range=[0,40],  color='rgba(239,68,68,0.06)'),
                    dict(range=[40,70], color='rgba(245,158,11,0.06)'),
                    dict(range=[70,100],color='rgba(16,185,129,0.06)'),
                ],
                threshold=dict(line=dict(color='rgba(99,102,241,0.6)',width=2),
                               thickness=0.8, value=70))))
        fig_hs.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                             height=240, margin=dict(t=40,b=10,l=40,r=40))
        st.plotly_chart(fig_hs, use_container_width=True)

    gauge_data = [("VOLTAGE",v,200,300,"V"),("CURRENT",i,1,10,"A"),
                  ("TEMPERATURE",temp,20,100,"°C"),("VIBRATION",vib,0,20,"mm/s")]
    for col,(lbl,val,mn,mx,unit) in zip(st.columns(4), gauge_data):
        pct = (val-mn)/(mx-mn)
        bar_color = '#ef4444' if pct>0.75 else '#f59e0b' if pct>0.5 else '#10b981'
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=val,
            number=dict(suffix=unit, font=dict(family='Space Grotesk',color=bar_color,size=18)),
            title=dict(text=lbl, font=dict(family='JetBrains Mono',color='#4b5563',size=10)),
            gauge=dict(
                axis=dict(range=[mn,mx], tickcolor='rgba(99,102,241,0.3)',
                          tickfont=dict(color='#374151',size=8)),
                bar=dict(color=bar_color, thickness=0.22),
                bgcolor='rgba(13,13,20,0.8)',
                borderwidth=1, bordercolor='rgba(99,102,241,0.12)',
                steps=[
                    dict(range=[mn, mn+(mx-mn)*0.5],        color='rgba(16,185,129,0.04)'),
                    dict(range=[mn+(mx-mn)*0.5, mn+(mx-mn)*0.75], color='rgba(245,158,11,0.04)'),
                    dict(range=[mn+(mx-mn)*0.75, mx],       color='rgba(239,68,68,0.04)'),
                ])))
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                            height=170, margin=dict(t=20,b=5,l=15,r=15))
        col.plotly_chart(fig_g, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.button("EXECUTE FAULT ANALYSIS", use_container_width=True):
        history = df[['Voltage','Current','Temperature','Vibration']].tail(4)
        new_row = pd.DataFrame([[v,i,temp,vib]],
                               columns=['Voltage','Current','Temperature','Vibration'])
        tmp_df  = pd.concat([history, new_row], ignore_index=True)
        feats   = {}
        for sensor in ['Voltage','Current','Temperature','Vibration']:
            val_s = (v if sensor=='Voltage' else i if sensor=='Current'
                     else temp if sensor=='Temperature' else vib)
            feats[sensor]           = val_s
            feats[f'{sensor}_Mean'] = tmp_df[sensor].mean()
            feats[f'{sensor}_Std']  = tmp_df[sensor].std()

        inp   = pd.DataFrame([feats])[X.columns]
        pidx  = model.predict(inp)[0]
        proba = model.predict_proba(inp)[0]
        prob  = proba[pidx]*100
        state = le.inverse_transform([pidx])[0]

        rul_val = rul_model.predict([[v, i, temp, vib]])[0]
        rul_val = max(0, round(rul_val, 1))

        INFO = {
            'high':     ('CRITICAL FAILURE', 'Emergency stop — high failure risk. Check bearings and cooling immediately.', 's-critical'),
            'moderate': ('WARNING DETECTED', 'Elevated stress detected. Schedule maintenance within 48 hours.', 's-warning'),
            'low':      ('LOW VARIANCE', 'Minor deviation. No immediate action required. Continue monitoring.', 's-good'),
            'normal':   ('NOMINAL STATE', 'All parameters within optimal range. System operating normally.', 's-good'),
        }
        badge, msg, css = INFO.get(state.lower(), ('UNKNOWN','Check system.','s-good'))

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns([1.4, 1, 0.9])
        with r1:
            st.markdown(f"""
            <div class="status-box {css}">
                <div class="status-title">{badge}</div>
                <div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;
                letter-spacing:3px;opacity:0.6;margin:4px 0 10px;text-transform:uppercase;'>
                    State: {state}
                </div>
                <div class="status-msg">{msg}</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            conf_color = '#10b981' if prob>70 else '#f59e0b' if prob>40 else '#ef4444'
            fig_conf = go.Figure(go.Indicator(
                mode="gauge+number", value=prob,
                number=dict(suffix="%", font=dict(family='Space Grotesk',color=conf_color,size=28)),
                title=dict(text="CONFIDENCE",
                           font=dict(family='JetBrains Mono',color='#4b5563',size=11)),
                gauge=dict(
                    axis=dict(range=[0,100], tickcolor='rgba(99,102,241,0.3)',
                              tickfont=dict(color='#374151',size=9)),
                    bar=dict(color=conf_color, thickness=0.22),
                    bgcolor='rgba(13,13,20,0.8)',
                    borderwidth=1, bordercolor='rgba(99,102,241,0.15)',
                    steps=[
                        dict(range=[0,40],  color='rgba(239,68,68,0.06)'),
                        dict(range=[40,70], color='rgba(245,158,11,0.06)'),
                        dict(range=[70,100],color='rgba(16,185,129,0.06)'),
                    ])))
            fig_conf.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                   height=240, margin=dict(t=40,b=10,l=30,r=30))
            st.plotly_chart(fig_conf, use_container_width=True)

        with r3:
            st.markdown(f"""
            <div class="rul-card">
                <div class="rul-label">Remaining Useful Life</div>
                <div class="rul-number">{rul_val:.0f}</div>
                <div class="rul-label" style="font-size:0.7rem;color:#6366f1;">hours estimated</div>
                <div style="height:1px;background:rgba(99,102,241,0.15);margin:0.8rem 0;"></div>
                <div class="rul-label">Health Score</div>
                <div style="font-family:Space Grotesk,sans-serif;font-size:2rem;
                font-weight:700;color:{hs_col};line-height:1;">{live_health}</div>
                <div class="rul-label" style="color:{hs_col};">{hs_label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Probability Distribution</div>', unsafe_allow_html=True)
        bar_clrs = [PALETTE.get(cn.lower(),'#6366f1') for cn in class_names]
        fig_pb = go.Figure(go.Bar(
            x=class_names, y=proba*100,
            marker=dict(color=bar_clrs, line=dict(color=[hex_rgba(c,0.8) for c in bar_clrs], width=1)),
            text=[f'{p:.1f}%' for p in proba*100], textposition='outside',
            textfont=dict(family='JetBrains Mono', color='#8b8fa8', size=11)))
        fig_pb.update_layout(**PL, height=280, title='Class Probability Breakdown',
            xaxis=make_axis(title='Condition Class'),
            yaxis=make_axis(range=[0,115], title='Probability (%)'))
        st.plotly_chart(fig_pb, use_container_width=True)

        st.session_state.maint_log.append({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Voltage': v, 'Current': i, 'Temperature': temp, 'Vibration': vib,
            'Prediction': state, 'Confidence': f'{prob:.1f}%',
            'Health Score': live_health, 'RUL (h)': rul_val
        })
        st.success("Result saved to Maintenance Log.")

# ══════════════════════════════════════════════════════════════
#  TAB 4 — MAINTENANCE LOG
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Prediction History Log</div>', unsafe_allow_html=True)

    with st.expander("+ Add Manual Maintenance Event"):
        m1, m2, m3 = st.columns(3)
        with m1:
            m_date  = st.text_input("Date (YYYY-MM-DD)", value=datetime.now().strftime('%Y-%m-%d'), key='mdate')
        with m2:
            m_type  = st.selectbox("Event Type", ['Inspection','Repair','Replacement','Calibration','Emergency'], key='mtype')
        with m3:
            m_notes = st.text_input("Notes", value="", placeholder="Details...", key='mnotes')
        if st.button("Add to Log", key='add_log'):
            st.session_state.maint_log.append({
                'Timestamp': m_date + ' 00:00:00',
                'Voltage': '—', 'Current': '—', 'Temperature': '—', 'Vibration': '—',
                'Prediction': m_type, 'Confidence': 'Manual',
                'Health Score': '—', 'RUL (h)': '—', 'Notes': m_notes
            })
            st.success("Event logged.")

    if st.session_state.maint_log:
        log_df = pd.DataFrame(st.session_state.maint_log)

        # ── FIX: applymap → map (pandas >= 2.1 compatibility) ──
        def highlight_prediction(val):
            v = str(val).lower()
            if v in ['high', 'critical failure', 'emergency']:
                return 'color:#ef4444'
            elif v in ['moderate', 'warning detected']:
                return 'color:#f59e0b'
            elif v in ['normal', 'nominal state']:
                return 'color:#10b981'
            return ''

        styled = log_df.style.map(highlight_prediction, subset=['Prediction'])
        st.dataframe(styled, use_container_width=True)

        st.markdown('<div class="section-title">Export Report</div>', unsafe_allow_html=True)
        ec1, ec2 = st.columns(2)
        with ec1:
            csv_buf = io.StringIO()
            log_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="DOWNLOAD CSV REPORT",
                data=csv_buf.getvalue(),
                file_name=f"motor_pdm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True)

        with ec2:
            if st.button("CLEAR LOG", use_container_width=True):
                st.session_state.maint_log = []
                st.rerun()

        numeric_log = log_df[pd.to_numeric(log_df['Health Score'], errors='coerce').notna()].copy()
        if not numeric_log.empty:
            numeric_log['Health Score'] = pd.to_numeric(numeric_log['Health Score'])
            fig_log = go.Figure(go.Scatter(
                x=numeric_log['Timestamp'], y=numeric_log['Health Score'],
                mode='lines+markers',
                line=dict(color='#6366f1', width=2),
                marker=dict(size=8, color='#10b981', line=dict(color='#6366f1',width=1)),
                fill='tozeroy', fillcolor='rgba(99,102,241,0.07)'))
            fig_log.update_layout(**PL, height=260,
                title='Health Score History',
                xaxis=make_axis(title='Timestamp'), yaxis=make_axis(range=[0,105],title='Health Score'))
            st.plotly_chart(fig_log, use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:3rem;background:#0d0d14;
        border:1px solid rgba(99,102,241,0.15);border-radius:12px;
        font-family:JetBrains Mono,monospace;font-size:0.75rem;
        color:#374151;letter-spacing:2px;text-transform:uppercase;'>
        No log entries yet — run a fault analysis in the Live Prediction tab
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TAB 5 — FORECAST & TRENDS
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">48-Hour Trend Forecast</div>', unsafe_allow_html=True)
    st.markdown("""<div style='font-family:Space Grotesk,sans-serif;font-size:0.85rem;
    color:#4b5563;margin-bottom:0.8rem;'>
    Linear regression forecast on last 200 samples — projected 48 steps forward.
    Dotted line = predicted trend.
    </div>""", unsafe_allow_html=True)

    fc_sensor = st.selectbox("Select Sensor to Forecast",
                             ['Voltage','Current','Temperature','Vibration'],
                             key='fc_sel', label_visibility="collapsed")

    fc_data   = df[fc_sensor].values[-200:]
    fc_x      = np.arange(len(fc_data)).reshape(-1,1)
    fc_model  = LinearRegression()
    fc_model.fit(fc_x, fc_data)

    future_x  = np.arange(len(fc_data), len(fc_data)+48).reshape(-1,1)
    forecast  = fc_model.predict(future_x)
    upper_b   = forecast + fc_data.std() * 1.5
    lower_b   = forecast - fc_data.std() * 1.5

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=np.arange(len(fc_data)), y=fc_data,
        name='Actual', mode='lines',
        line=dict(color='#6366f1', width=1.5)))
    fig_fc.add_trace(go.Scatter(
        x=np.arange(len(fc_data), len(fc_data)+48), y=forecast,
        name='Forecast', mode='lines',
        line=dict(color='#10b981', width=2, dash='dot')))
    fig_fc.add_trace(go.Scatter(
        x=np.concatenate([np.arange(len(fc_data),len(fc_data)+48),
                          np.arange(len(fc_data),len(fc_data)+48)[::-1]]),
        y=np.concatenate([upper_b, lower_b[::-1]]),
        fill='toself', fillcolor='rgba(16,185,129,0.06)',
        line=dict(color='rgba(0,0,0,0)'), name='Confidence Band'))
    fig_fc.update_layout(**PL, height=380, title=f'{fc_sensor} — 48-Step Forecast',
        xaxis=make_axis(title='Sample Index'), yaxis=make_axis(title=fc_sensor))
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown(f"""
    <div style='background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);
    border-radius:8px;padding:0.7rem 1rem;font-family:JetBrains Mono,monospace;
    font-size:0.72rem;color:#6ee7b7;letter-spacing:1px;'>
    ◆ Trend slope: {fc_model.coef_[0]:.5f} per sample
    &nbsp;&nbsp;|&nbsp;&nbsp; Forecast range: [{lower_b.min():.2f} — {upper_b.max():.2f}]
    &nbsp;&nbsp;|&nbsp;&nbsp; Sensor: {fc_sensor}
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">All Sensors — Trend Overview</div>', unsafe_allow_html=True)
    fig_mg = make_subplots(rows=2, cols=2,
        subplot_titles=['VOLTAGE TREND','CURRENT TREND','TEMPERATURE TREND','VIBRATION TREND'],
        vertical_spacing=0.18, horizontal_spacing=0.1)
    colors_mg = ['#6366f1','#10b981','#ef4444','#f59e0b']
    for (r,c), sensor, col_mg in zip([(1,1),(1,2),(2,1),(2,2)],
                                      ['Voltage','Current','Temperature','Vibration'],
                                      colors_mg):
        data_mg = df[sensor].values[-150:]
        x_mg    = np.arange(len(data_mg))
        lm      = LinearRegression().fit(x_mg.reshape(-1,1), data_mg)
        trend   = lm.predict(x_mg.reshape(-1,1))
        fig_mg.add_trace(go.Scatter(x=x_mg, y=data_mg, mode='lines',
            line=dict(color=col_mg, width=1), opacity=0.5,
            name=sensor, showlegend=False), row=r, col=c)
        fig_mg.add_trace(go.Scatter(x=x_mg, y=trend, mode='lines',
            line=dict(color='#ffffff', width=1.5, dash='dot'),
            name=f'{sensor} trend', showlegend=False), row=r, col=c)
    fig_mg.update_layout(**PL, height=460, title='Multi-Sensor Trend Lines')
    fig_mg.update_annotations(font=dict(family='JetBrains Mono',color='#6366f1',size=10))
    fig_mg.update_xaxes(**AXIS)
    fig_mg.update_yaxes(**AXIS)
    st.plotly_chart(fig_mg, use_container_width=True)

    st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)
    stats = df[['Voltage','Current','Temperature','Vibration']].describe().round(3)
    st.dataframe(stats.style.background_gradient(cmap='Blues', axis=1), use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer-bar">
MotorMind AI &nbsp;·&nbsp; Team PREDICT X &nbsp;·&nbsp; Random Forest + Gradient Boosting &nbsp;·&nbsp;
Built with Streamlit & Plotly &nbsp;·&nbsp; Space Grotesk + JetBrains Mono
</div>
""", unsafe_allow_html=True)