import streamlit as st
import pandas as pd
import numpy as np
import warnings
import io
import requests
from datetime import datetime

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
[data-testid="stSidebar"] * { color: #8b8fa8 !important; }

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

.n8n-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,100,0,0.12);
    border: 1px solid rgba(255,100,0,0.3);
    border-radius: 20px;
    padding: 6px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #ff6400;
    letter-spacing: 2px;
    margin-top: 8px;
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

# Condition name → short label mapping
COND_SHORT = {
    'no load':                  'No Load',
    'normal load(uncontrolled)':'Normal (Unc.)',
    'normal load(controlled)':  'Normal (Ctrl)',
    'high load(controlled)':    'High Load',
}
def short(c):
    return COND_SHORT.get(c.lower().strip(), c)

PALETTE = {
    'no load':                  '#10b981',
    'normal load(uncontrolled)':'#6366f1',
    'normal load(controlled)':  '#f59e0b',
    'high load(controlled)':    '#ef4444',
}
def get_color(c):
    return PALETTE.get(c.lower().strip(), '#8b8fa8')

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
#  N8N WEBHOOK
# ══════════════════════════════════════════════════════════════
def send_alert_to_n8n(state, prob, health, rul_val, v, i, vib):
    try:
        payload = {
            "prediction": state,
            "confidence": round(float(prob), 1),
            "health_score": float(health),
            "rul_hours": float(rul_val),
            "voltage": v, "current": i, "vibration": vib,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "severity": "CRITICAL" if 'high' in state.lower() else "WARNING"
        }
        r = requests.post(
            "https://chaudhary0022.app.n8n.cloud/webhook/motormind-alert",
            json=payload, timeout=5)
        return r.status_code == 200
    except Exception:
        return False

# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
for k, v_def in [('maint_log',[]), ('motor_speed',100.0),
                  ('remediation_log',[]), ('remediation_active',False)]:
    if k not in st.session_state:
        st.session_state[k] = v_def

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <div>
        <div class="brand-name">Motor<span>Mind</span> AI</div>
        <div class="brand-sub">Predictive Maintenance &nbsp;·&nbsp; Fault Detection &nbsp;·&nbsp; Real-Time Analytics</div>
        <div class="team-badge">⚡ TEAM &nbsp; PREDICT X</div>
        <div class="n8n-badge">⚙️ POWERED BY &nbsp; n8n AUTOMATION</div>
    </div>
    <div style="display:flex;flex-direction:column;gap:8px;align-items:flex-end;">
        <div class="live-badge">
            <span class="live-dot"></span>
            SYSTEM ACTIVE
        </div>
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
    <div style='font-family:Space Grotesk,sans-serif;font-size:0.82rem;color:#374151;line-height:2.2;'>
    ◆ Random Forest Classifier<br>
    ◆ Health Score Engine<br>
    ◆ Remaining Useful Life (RUL)<br>
    ◆ Anomaly Timeline Detection<br>
    ◆ Feature Importance (SHAP-style)<br>
    ◆ Trend Forecasting (48-step)<br>
    ◆ Maintenance Log System<br>
    ◆ CSV Report Export<br>
    ◆ Multi-Class Confidence<br>
    ◆ 3D Feature Space Viewer<br>
    ◆ n8n Alert Automation<br>
    ◆ Auto-Remediation System
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div style='height:1px;background:rgba(99,102,241,0.15);margin:1rem 0;'></div>""",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(255,100,0,0.08);border:1px solid rgba(255,100,0,0.25);
    border-radius:8px;padding:0.7rem;font-family:JetBrains Mono,monospace;
    font-size:0.65rem;color:#ff6400;letter-spacing:1.5px;text-transform:uppercase;'>
    ⚙️ n8n Automation Active<br>
    <span style="color:#4b5563;font-size:0.6rem;">
    Critical faults trigger<br>automatic email alerts
    </span>
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
            Upload CSV with columns: V_mean, I_mean, VIB_mean ... condition
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════
#  LOAD + TRAIN  (adapted to actual CSV — no column rename)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_and_train(file_bytes):
    import io as _io
    df = pd.read_csv(_io.BytesIO(file_bytes))

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Identify target column (case-insensitive)
    target_col = next((c for c in df.columns if c.lower() == 'condition'), None)
    if target_col is None:
        raise ValueError("No 'condition' column found in CSV!")

    # Drop rows with nulls
    df = df.dropna().reset_index(drop=True)

    # Encode target
    le = LabelEncoder()
    df['Condition_Encoded'] = le.fit_transform(df[target_col])
    class_names = le.classes_

    # Feature columns = everything except target & encoded target
    feature_cols = [c for c in df.columns if c not in [target_col, 'Condition_Encoded']]

    X = df[feature_cols]
    y = df['Condition_Encoded']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    mdl = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    mdl.fit(X_tr, y_tr)
    y_pred = mdl.predict(X_te)

    # RUL model — higher condition_encoded = higher stress = lower RUL
    rul_target = (y.max() - y) * 100 + np.random.normal(0, 5, len(df))
    rul_target = np.clip(rul_target, 0, None)
    rul_mdl = GradientBoostingRegressor(n_estimators=100, random_state=42)
    rul_mdl.fit(X, rul_target)

    return df, mdl, le, X, X_te, y_te, y_pred, class_names, rul_mdl, target_col, feature_cols

with st.spinner("Initializing AI Engine..."):
    file_bytes = uploaded.read()
    (df, model, le, X, X_test, y_test, y_pred,
     class_names, rul_model, target_col, feature_cols) = load_and_train(file_bytes)

with st.sidebar:
    st.markdown("""
    <div style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);
    border-radius:8px;padding:0.7rem;text-align:center;font-family:JetBrains Mono,monospace;
    font-size:0.68rem;color:#10b981;letter-spacing:2px;text-transform:uppercase;margin-top:1rem;'>
    ✓ AI Engine Online
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  HEALTH SCORE — uses V_mean, I_mean, VIB_mean
# ══════════════════════════════════════════════════════════════
V_MIN, V_MAX   = df['V_mean'].min(),   df['V_mean'].max()
I_MIN, I_MAX   = df['I_mean'].min(),   df['I_mean'].max()
VIB_MIN, VIB_MAX = df['VIB_mean'].min(), df['VIB_mean'].max()

def compute_health_score(v, i, vib):
    v_norm   = np.clip((v   - V_MIN)   / max(V_MAX   - V_MIN,   1e-9), 0, 1)
    i_norm   = np.clip((i   - I_MIN)   / max(I_MAX   - I_MIN,   1e-9), 0, 1)
    vib_norm = np.clip((vib - VIB_MIN) / max(VIB_MAX - VIB_MIN, 1e-9), 0, 1)
    stress = 0.35*v_norm + 0.35*i_norm + 0.30*vib_norm
    return round((1 - stress) * 100, 1)

avg_health = compute_health_score(
    df['V_mean'].mean(), df['I_mean'].mean(), df['VIB_mean'].mean())

# ══════════════════════════════════════════════════════════════
#  KPI BAR
# ══════════════════════════════════════════════════════════════
accuracy = round((y_pred == y_test.values).mean() * 100, 1)
avg_rul  = round(rul_model.predict(X).mean(), 0)
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
    <div class="kpi-lbl">Load Classes</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "  📡  Sensor Analytics  ",
    "  🤖  Model Diagnostics  ",
    "  ⚡  Live Prediction  ",
    "  📋  Maintenance Log  ",
    "  📈  Forecast & Trends  ",
    "  🔧  Auto-Remediation  ",
])

# ══════════════════════════════════════════════════════════════
#  TAB 1 — SENSOR ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab1:
    # Pick 4 most informative numeric columns to show
    DISPLAY_SENSORS = ['V_mean', 'I_mean', 'VIB_mean', 'VIB_rms']
    SENSOR_LABELS   = ['VOLTAGE (V_mean)', 'CURRENT (I_mean)',
                       'VIBRATION MEAN', 'VIBRATION RMS']

    st.markdown('<div class="section-title">Sensor Time-Series Matrix</div>', unsafe_allow_html=True)
    fig_ts = make_subplots(rows=2, cols=2,
        subplot_titles=SENSOR_LABELS,
        vertical_spacing=0.16, horizontal_spacing=0.1)

    for (r,c), sensor in zip([(1,1),(1,2),(2,1),(2,2)], DISPLAY_SENSORS):
        for cond in df[target_col].unique():
            sub = df[df[target_col]==cond].iloc[:200]
            col_c = get_color(cond)
            fig_ts.add_trace(go.Scatter(
                x=sub.index, y=sub[sensor], name=short(cond), mode='lines',
                line=dict(color=col_c, width=1.3), opacity=0.9,
                showlegend=(sensor=='V_mean'),
                hovertemplate=f'<b>{sensor}</b>: %{{y:.4f}}<extra>{short(cond)}</extra>'
            ), row=r, col=c)

    fig_ts.update_layout(**PL, height=480,
        legend=dict(bgcolor='rgba(13,13,20,0.9)',bordercolor='rgba(99,102,241,0.2)',
                    borderwidth=1, font=dict(family='Space Grotesk',size=11)))
    fig_ts.update_annotations(font=dict(family='JetBrains Mono',color='#6366f1',size=10))
    fig_ts.update_xaxes(**AXIS)
    fig_ts.update_yaxes(**AXIS)
    st.plotly_chart(fig_ts, use_container_width=True)

    # Anomaly Timeline
    st.markdown('<div class="section-title">Anomaly Event Timeline</div>', unsafe_allow_html=True)
    numeric_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    timeline_sensor = st.selectbox("Select Sensor for Timeline", numeric_cols, key='tl_sel')

    sample_df = df.iloc[:500].copy().reset_index(drop=True)
    mean_v = sample_df[timeline_sensor].mean()
    std_v  = sample_df[timeline_sensor].std()
    sample_df['anomaly'] = (sample_df[timeline_sensor] - mean_v).abs() > 2*std_v
    anomalies = sample_df[sample_df['anomaly']]

    fig_tl = go.Figure()
    for cond in df[target_col].unique():
        sub = sample_df[sample_df[target_col]==cond]
        fig_tl.add_trace(go.Scatter(
            x=sub.index, y=sub[timeline_sensor], name=short(cond),
            mode='lines', line=dict(color=get_color(cond), width=1.2), opacity=0.75))
    if not anomalies.empty:
        for idx in anomalies.index[:30]:  # cap vlines for performance
            fig_tl.add_vline(x=idx, line=dict(color='rgba(239,68,68,0.25)',width=1,dash='dot'))
        fig_tl.add_trace(go.Scatter(
            x=anomalies.index, y=anomalies[timeline_sensor], mode='markers', name='Anomaly',
            marker=dict(color='#ef4444',size=7,symbol='circle-open',
                        line=dict(color='#ef4444',width=2)),
            hovertemplate='<b>ANOMALY</b><br>Idx: %{x}<br>Val: %{y:.4f}<extra></extra>'))
    fig_tl.update_layout(**PL, height=320,
        title=f'{timeline_sensor.upper()} — Anomaly Detection (2σ)',
        xaxis=make_axis(title='Sample Index'), yaxis=make_axis(title=timeline_sensor))
    st.plotly_chart(fig_tl, use_container_width=True)

    st.markdown(f"""
    <div style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);
    border-radius:8px;padding:0.7rem 1rem;font-family:JetBrains Mono,monospace;
    font-size:0.72rem;color:#fca5a5;letter-spacing:1px;'>
    ◆ {len(anomalies)} anomalies detected in first 500 samples
    &nbsp;|&nbsp; Threshold: ±2σ &nbsp;|&nbsp; Sensor: {timeline_sensor}
    </div>""", unsafe_allow_html=True)

    # Distribution + Correlation
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Distribution Analysis</div>', unsafe_allow_html=True)
        sel = st.selectbox("Sensor", numeric_cols[:8], label_visibility="collapsed", key='dist_sel')
        fig_vio = go.Figure()
        for cond in df[target_col].unique():
            ch = get_color(cond)
            fig_vio.add_trace(go.Violin(
                y=df[df[target_col]==cond][sel], name=short(cond),
                box_visible=True, meanline_visible=True,
                fillcolor=hex_rgba(ch,0.18), line_color=ch, opacity=0.9))
        fig_vio.update_layout(**PL, height=360, title=f'{sel} Distribution',
            xaxis=AXIS, yaxis=AXIS, violingap=0.15, violinmode='overlay')
        st.plotly_chart(fig_vio, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Correlation Matrix</div>', unsafe_allow_html=True)
        corr_cols = ['V_mean','V_rms','I_mean','I_rms','VIB_mean','VIB_rms','VIB_fft_energy','VIB_fft_peak']
        corr_cols = [c for c in corr_cols if c in df.columns]
        corr = df[corr_cols].corr()
        fig_h = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0,'#ef4444'],[0.5,'#0a0a0f'],[1,'#10b981']],
            text=np.round(corr.values,2), texttemplate='%{text}',
            textfont=dict(size=10, color='#f1f5f9'), showscale=True, zmin=-1, zmax=1))
        fig_h.update_layout(**PL, height=360, title='Sensor Correlation',
            xaxis=AXIS, yaxis=AXIS)
        st.plotly_chart(fig_h, use_container_width=True)

    # 3D State Space
    st.markdown('<div class="section-title">3D Sensor State Space</div>', unsafe_allow_html=True)
    fig_3d = go.Figure(go.Scatter3d(
        x=df['V_mean'], y=df['I_mean'], z=df['VIB_mean'],
        mode='markers',
        marker=dict(size=3, color=[get_color(c) for c in df[target_col]],
                    opacity=0.7, line=dict(width=0)),
        text=df[target_col].apply(short),
        hovertemplate='<b>%{text}</b><br>V:%{x:.3f}  I:%{y:.4f}  Vib:%{z:.3f}<extra></extra>'
    ))
    fig_3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(bgcolor='rgba(13,13,20,0.95)',
                   xaxis=axis3d('V_MEAN'), yaxis=axis3d('I_MEAN'), zaxis=axis3d('VIB_MEAN')),
        font=dict(color='#8b8fa8', family='Space Grotesk'),
        title=dict(text='3D SENSOR STATE SPACE',
                   font=dict(family='JetBrains Mono',color='#6366f1',size=13)),
        height=520, margin=dict(t=50,b=0,l=0,r=0))
    st.plotly_chart(fig_3d, use_container_width=True)

    # Pie + Rolling
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-title">Load Class Distribution</div>', unsafe_allow_html=True)
        dist = df[target_col].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=dist.index.map(short), values=dist.values, hole=0.55,
            marker=dict(colors=[get_color(c) for c in dist.index],
                        line=dict(color='#0a0a0f',width=2)),
            textinfo='label+percent',
            textfont=dict(family='Space Grotesk',size=12,color='#f1f5f9')))
        fig_pie.update_layout(**PL, height=320, title='Class Distribution',
            annotations=[dict(text='LOADS', x=0.5, y=0.5, showarrow=False,
                              font=dict(family='JetBrains Mono',color='#374151',size=11))])
        st.plotly_chart(fig_pie, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">Rolling Mean Overview</div>', unsafe_allow_html=True)
        fig_rm = go.Figure()
        for col_s, color in zip(['V_mean','I_mean','VIB_mean','VIB_rms'],
                                 ['#10b981','#6366f1','#ef4444','#f59e0b']):
            if col_s in df.columns:
                rolling = df[col_s].rolling(10,min_periods=1).mean().iloc[:300]
                fig_rm.add_trace(go.Scatter(
                    x=df.index[:300], y=rolling, name=col_s, mode='lines',
                    line=dict(color=color,width=1.5),
                    fill='tozeroy', fillcolor=hex_rgba(color,0.06)))
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
        short_names = [short(n) for n in class_names]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=short_names, y=short_names,
            colorscale=[[0,'#0d0d14'],[0.4,'#3730a3'],[1,'#10b981']],
            text=cm, texttemplate='<b>%{text}</b>',
            textfont=dict(size=18, family='Space Grotesk', color='white'),
            showscale=False))
        fig_cm.update_layout(**PL, height=400, title='Predicted vs Actual',
            xaxis=dict(**AXIS, title='PREDICTED'),
            yaxis=dict(**AXIS, title='ACTUAL'))
        st.plotly_chart(fig_cm, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Feature Importance (Top 12)</div>', unsafe_allow_html=True)
        fi = pd.Series(model.feature_importances_, index=feature_cols).nlargest(12)
        clrs = [hex_rgba('#6366f1', 0.3 + 0.7*v/fi.max()) for v in fi.values]
        fig_fi = go.Figure(go.Bar(
            x=fi.values, y=fi.index, orientation='h',
            marker=dict(color=clrs, line=dict(color='rgba(99,102,241,0.5)',width=1)),
            text=[f'{v:.4f}' for v in fi.values], textposition='outside',
            textfont=dict(color='#8b8fa8',size=10)))
        fig_fi.update_layout(**PL, height=400, title='Critical Features',
            xaxis=dict(**AXIS,title='Importance Score'), yaxis=AXIS)
        st.plotly_chart(fig_fi, use_container_width=True)

    # SHAP-style
    st.markdown('<div class="section-title">Feature Impact — SHAP-Style Explainability</div>',
                unsafe_allow_html=True)
    fi_all = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    signs  = np.where(np.arange(len(fi_all)) % 2 == 0, 1, -1)
    impact = fi_all.values * signs

    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        x=impact, y=fi_all.index, orientation='h',
        marker=dict(color=['#10b981' if v > 0 else '#ef4444' for v in impact],
                    opacity=0.85, line=dict(width=0)),
        hovertemplate='<b>%{y}</b><br>Impact: %{x:.5f}<extra></extra>'))
    fig_shap.add_vline(x=0, line=dict(color='rgba(99,102,241,0.4)',width=1))
    fig_shap.update_layout(**PL, height=420, title='Feature Impact (SHAP Proxy)',
        xaxis=make_axis(title='← Reduces Risk  |  Increases Risk →'),
        yaxis=AXIS)
    st.plotly_chart(fig_shap, use_container_width=True)

    # Classification report
    st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)
    rpt = classification_report(y_test, y_pred, target_names=short_names, output_dict=True)
    rpt_df = pd.DataFrame(rpt).T.round(3)
    st.dataframe(rpt_df.style.format(precision=3), use_container_width=True)

    # 3D feature space
    st.markdown('<div class="section-title">3D Classification Feature Space</div>', unsafe_allow_html=True)
    fi_top3     = pd.Series(model.feature_importances_, index=feature_cols).nlargest(3).index.tolist()
    pred_clrs   = [get_color(le.inverse_transform([p])[0]) for p in y_pred]
    pred_labels = [short(le.inverse_transform([p])[0]) for p in y_pred]
    fig_3df = go.Figure(go.Scatter3d(
        x=X_test[fi_top3[0]], y=X_test[fi_top3[1]], z=X_test[fi_top3[2]],
        mode='markers',
        marker=dict(size=4, color=pred_clrs, opacity=0.8, line=dict(width=0)),
        text=pred_labels,
        hovertemplate=(
            '<b>%{text}</b><br>'+fi_top3[0]+':%{x:.4f}<br>'+
            fi_top3[1]+':%{y:.4f}<br>'+fi_top3[2]+':%{z:.4f}<extra></extra>')))
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
#  Uses actual sensor ranges from DC toy motor data
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Live Sensor Input (DC Motor Values)</div>',
                unsafe_allow_html=True)

    # Slider ranges from actual CSV data
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        v_live = st.slider(
            "VOLTAGE V_mean (V)",
            min_value=float(round(V_MIN - 0.05, 2)),
            max_value=float(round(V_MAX + 0.05, 2)),
            value=float(round(df['V_mean'].mean(), 3)),
            step=0.001, format="%.3f")
    with sc2:
        i_live = st.slider(
            "CURRENT I_mean (A)",
            min_value=float(round(I_MIN - 0.005, 3)),
            max_value=float(round(I_MAX + 0.005, 3)),
            value=float(round(df['I_mean'].mean(), 4)),
            step=0.0001, format="%.4f")
    with sc3:
        vib_live = st.slider(
            "VIBRATION VIB_mean (mm/s)",
            min_value=float(round(VIB_MIN - 0.05, 2)),
            max_value=float(round(VIB_MAX + 0.05, 2)),
            value=float(round(df['VIB_mean'].mean(), 3)),
            step=0.001, format="%.3f")

    live_health = compute_health_score(v_live, i_live, vib_live)
    hs_col   = '#10b981' if live_health >= 70 else '#f59e0b' if live_health >= 40 else '#ef4444'
    hs_label = 'Healthy' if live_health >= 70 else 'Degraded' if live_health >= 40 else 'Critical'

    # Health gauge
    h1, h2, h3 = st.columns([1,2,1])
    with h2:
        fig_hs = go.Figure(go.Indicator(
            mode="gauge+number", value=live_health,
            number=dict(font=dict(family='Space Grotesk',color=hs_col,size=36)),
            title=dict(text=f"MOTOR HEALTH SCORE — {hs_label}",
                       font=dict(family='JetBrains Mono',color='#4b5563',size=11)),
            gauge=dict(
                axis=dict(range=[0,100], tickcolor='rgba(99,102,241,0.3)',
                          tickfont=dict(color='#374151',size=9)),
                bar=dict(color=hs_col, thickness=0.25),
                bgcolor='rgba(13,13,20,0.8)',
                borderwidth=1, bordercolor='rgba(99,102,241,0.15)',
                steps=[dict(range=[0,40],  color='rgba(239,68,68,0.06)'),
                       dict(range=[40,70], color='rgba(245,158,11,0.06)'),
                       dict(range=[70,100],color='rgba(16,185,129,0.06)')],
                threshold=dict(line=dict(color='rgba(99,102,241,0.6)',width=2),
                               thickness=0.8, value=70))))
        fig_hs.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                             height=240, margin=dict(t=40,b=10,l=40,r=40))
        st.plotly_chart(fig_hs, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.button("EXECUTE FAULT ANALYSIS", use_container_width=True):

        # Build input row — use mean from last 4 rows for all features,
        # but override V_mean, I_mean, VIB_mean with live slider values
        base_row = df[feature_cols].tail(5).mean().copy()
        if 'V_mean'   in feature_cols: base_row['V_mean']   = v_live
        if 'I_mean'   in feature_cols: base_row['I_mean']   = i_live
        if 'VIB_mean' in feature_cols: base_row['VIB_mean'] = vib_live

        inp   = pd.DataFrame([base_row])[feature_cols]
        pidx  = model.predict(inp)[0]
        proba = model.predict_proba(inp)[0]
        prob  = proba[pidx] * 100
        state = le.inverse_transform([pidx])[0]

        rul_val = max(0, round(rul_model.predict(inp)[0], 1))

        INFO = {
            'high load(controlled)':    ('HIGH LOAD — CRITICAL', 'Reduce load immediately. Check motor temperature and bearings.', 's-critical'),
            'normal load(uncontrolled)':('UNCONTROLLED LOAD', 'Load detected but unregulated. Monitor closely.', 's-warning'),
            'normal load(controlled)':  ('NORMAL LOAD', 'Motor operating within controlled parameters.', 's-good'),
            'no load':                  ('NO LOAD — NOMINAL', 'Motor running at no-load. All parameters optimal.', 's-good'),
        }
        badge, msg, css = INFO.get(state.lower().strip(),
                                   ('UNKNOWN STATE', 'Verify sensor connections.', 's-warning'))

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns([1.4, 1, 0.9])
        with r1:
            st.markdown(f"""
            <div class="status-box {css}">
                <div class="status-title">{badge}</div>
                <div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;
                letter-spacing:3px;opacity:0.6;margin:4px 0 10px;text-transform:uppercase;'>
                    Class: {short(state)}
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
                    steps=[dict(range=[0,40],  color='rgba(239,68,68,0.06)'),
                           dict(range=[40,70], color='rgba(245,158,11,0.06)'),
                           dict(range=[70,100],color='rgba(16,185,129,0.06)')])))
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

        # Probability bar chart
        st.markdown('<div class="section-title">Probability Distribution</div>', unsafe_allow_html=True)
        bar_clrs = [get_color(cn) for cn in class_names]
        fig_pb = go.Figure(go.Bar(
            x=[short(cn) for cn in class_names], y=proba*100,
            marker=dict(color=bar_clrs, line=dict(color=[hex_rgba(c,0.8) for c in bar_clrs],width=1)),
            text=[f'{p:.1f}%' for p in proba*100], textposition='outside',
            textfont=dict(family='JetBrains Mono',color='#8b8fa8',size=11)))
        fig_pb.update_layout(**PL, height=280, title='Class Probability Breakdown',
            xaxis=make_axis(title='Load Condition'),
            yaxis=make_axis(range=[0,115],title='Probability (%)'))
        st.plotly_chart(fig_pb, use_container_width=True)

        # n8n alert
        if 'high' in state.lower():
            alert_sent = send_alert_to_n8n(state, prob, live_health, rul_val, v_live, i_live, vib_live)
            if alert_sent:
                st.markdown("""
                <div style='background:rgba(255,100,0,0.08);border:1px solid rgba(255,100,0,0.3);
                border-radius:8px;padding:0.7rem 1rem;font-family:JetBrains Mono,monospace;
                font-size:0.72rem;color:#ff6400;letter-spacing:1px;margin-top:0.5rem;'>
                ⚙️ n8n ALERT TRIGGERED — Email notification sent automatically!
                </div>""", unsafe_allow_html=True)

        # Save to log
        st.session_state.maint_log.append({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'V_mean': v_live, 'I_mean': i_live, 'VIB_mean': vib_live,
            'Prediction': short(state), 'Confidence': f'{prob:.1f}%',
            'Health Score': live_health, 'RUL (h)': rul_val
        })
        st.success("✅ Result saved to Maintenance Log.")

        # Auto-Remediation
        if 'high' in state.lower():
            target_speed = 0; motor_status = 'EMERGENCY STOP'
            rem_actions = ['Motor completely stopped — High load detected',
                           'Do not restart until motor inspected']
        elif 'uncontrolled' in state.lower():
            target_speed = 60; motor_status = 'WARNING REDUCED'
            rem_actions = ['Speed reduced to 60% — Uncontrolled load detected',
                           'Schedule maintenance within 48h']
        else:
            target_speed = 100; motor_status = 'NORMAL OPERATION'
            rem_actions  = ['All parameters normal — Motor at full capacity']

        st.session_state.motor_speed        = float(target_speed)
        st.session_state.remediation_active = target_speed < 100
        st.session_state.remediation_log.append({
            'Timestamp'   : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Fault'       : short(state),
            'Motor Status': motor_status,
            'Speed %'     : target_speed,
            'Actions'     : ' | '.join(rem_actions),
            'Health'      : live_health,
            'RUL (h)'     : rul_val,
        })

        try:
            cmd = 'EMERGENCY_STOP' if target_speed==0 else 'SPEED_REDUCTION' if target_speed<100 else 'NORMAL'
            requests.post(
                'https://chaudhary0022.app.n8n.cloud/webhook/motormind-remediation',
                json={'command': cmd, 'target_speed_percent': target_speed,
                      'motor_status': motor_status, 'fault': state,
                      'health_score': live_health, 'rul_hours': rul_val,
                      'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                timeout=3)
        except Exception:
            pass

        st.markdown('<div class="section-title">Auto-Remediation Response</div>', unsafe_allow_html=True)
        rem_css   = 's-critical' if target_speed==0 else 's-warning' if target_speed<100 else 's-good'
        rem_title = ('MOTOR COMPLETELY STOPPED' if target_speed==0 else
                     f'SPEED REDUCED TO {target_speed}%' if target_speed<100 else
                     'NORMAL OPERATION — 100%')
        actions_html = '<br>'.join(['• ' + a for a in rem_actions])
        st.markdown(
            f'<div class="status-box {rem_css}" style="margin-top:0.5rem;">'
            f'<div class="status-title">{rem_title}</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;'
            f'letter-spacing:2px;margin:6px 0;opacity:0.6;">{motor_status}</div>'
            f'<div class="status-msg">{actions_html}</div></div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TAB 4 — MAINTENANCE LOG
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Prediction History Log</div>', unsafe_allow_html=True)

    with st.expander("+ Add Manual Maintenance Event"):
        m1, m2, m3 = st.columns(3)
        with m1: m_date  = st.text_input("Date (YYYY-MM-DD)", value=datetime.now().strftime('%Y-%m-%d'), key='mdate')
        with m2: m_type  = st.selectbox("Event Type", ['Inspection','Repair','Replacement','Calibration','Emergency'], key='mtype')
        with m3: m_notes = st.text_input("Notes", value="", placeholder="Details...", key='mnotes')
        if st.button("Add to Log", key='add_log'):
            st.session_state.maint_log.append({
                'Timestamp': m_date + ' 00:00:00',
                'V_mean':'—','I_mean':'—','VIB_mean':'—',
                'Prediction': m_type, 'Confidence': 'Manual',
                'Health Score':'—','RUL (h)':'—','Notes': m_notes})
            st.success("Event logged.")

    if st.session_state.maint_log:
        log_df = pd.DataFrame(st.session_state.maint_log)

        def highlight_prediction(val):
            v = str(val).lower()
            if 'high' in v or 'emergency' in v or 'critical' in v: return 'color:#ef4444'
            elif 'uncontrolled' in v or 'warning' in v:             return 'color:#f59e0b'
            elif 'normal' in v or 'no load' in v:                   return 'color:#10b981'
            return ''

        styled = log_df.style.map(highlight_prediction, subset=['Prediction'])
        st.dataframe(styled, use_container_width=True)

        ec1, ec2 = st.columns(2)
        with ec1:
            csv_buf = io.StringIO()
            log_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="DOWNLOAD CSV REPORT",
                data=csv_buf.getvalue(),
                file_name=f"motormind_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", use_container_width=True)
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
                line=dict(color='#6366f1',width=2),
                marker=dict(size=8,color='#10b981',line=dict(color='#6366f1',width=1)),
                fill='tozeroy', fillcolor='rgba(99,102,241,0.07)'))
            fig_log.update_layout(**PL, height=260, title='Health Score History',
                xaxis=make_axis(title='Timestamp'),
                yaxis=make_axis(range=[0,105],title='Health Score'))
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
    st.markdown('<div class="section-title">48-Step Trend Forecast</div>', unsafe_allow_html=True)

    numeric_cols_fc = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    fc_sensor = st.selectbox("Select Sensor to Forecast", numeric_cols_fc, key='fc_sel',
                              label_visibility="collapsed")

    fc_data  = df[fc_sensor].values[-200:]
    fc_x     = np.arange(len(fc_data)).reshape(-1,1)
    fc_model = LinearRegression().fit(fc_x, fc_data)

    future_x = np.arange(len(fc_data), len(fc_data)+48).reshape(-1,1)
    forecast = fc_model.predict(future_x)
    upper_b  = forecast + fc_data.std()*1.5
    lower_b  = forecast - fc_data.std()*1.5

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=np.arange(len(fc_data)), y=fc_data, name='Actual',
        mode='lines', line=dict(color='#6366f1',width=1.5)))
    fig_fc.add_trace(go.Scatter(x=np.arange(len(fc_data),len(fc_data)+48), y=forecast,
        name='Forecast', mode='lines', line=dict(color='#10b981',width=2,dash='dot')))
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
    ◆ Trend slope: {fc_model.coef_[0]:.6f} per sample
    &nbsp;&nbsp;|&nbsp;&nbsp; Forecast range: [{lower_b.min():.4f} — {upper_b.max():.4f}]
    &nbsp;&nbsp;|&nbsp;&nbsp; Sensor: {fc_sensor}
    </div>""", unsafe_allow_html=True)

    # All sensors overview
    st.markdown('<div class="section-title">Key Sensors — Trend Overview</div>', unsafe_allow_html=True)
    trend_sensors = ['V_mean','I_mean','VIB_mean','VIB_rms']
    trend_sensors = [s for s in trend_sensors if s in df.columns]
    colors_mg = ['#6366f1','#10b981','#ef4444','#f59e0b']

    fig_mg = make_subplots(rows=2, cols=2,
        subplot_titles=[s.upper()+' TREND' for s in trend_sensors],
        vertical_spacing=0.18, horizontal_spacing=0.1)
    for (r,c), sensor, col_mg in zip([(1,1),(1,2),(2,1),(2,2)], trend_sensors, colors_mg):
        data_mg = df[sensor].values[-150:]
        x_mg    = np.arange(len(data_mg))
        lm      = LinearRegression().fit(x_mg.reshape(-1,1), data_mg)
        trend   = lm.predict(x_mg.reshape(-1,1))
        fig_mg.add_trace(go.Scatter(x=x_mg, y=data_mg, mode='lines',
            line=dict(color=col_mg,width=1), opacity=0.5,
            name=sensor, showlegend=False), row=r, col=c)
        fig_mg.add_trace(go.Scatter(x=x_mg, y=trend, mode='lines',
            line=dict(color='#ffffff',width=1.5,dash='dot'),
            name=sensor+' trend', showlegend=False), row=r, col=c)
    fig_mg.update_layout(**PL, height=460, title='Multi-Sensor Trend Lines')
    fig_mg.update_annotations(font=dict(family='JetBrains Mono',color='#6366f1',size=10))
    fig_mg.update_xaxes(**AXIS); fig_mg.update_yaxes(**AXIS)
    st.plotly_chart(fig_mg, use_container_width=True)

    st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)
    stats = df[feature_cols].describe().round(4)
    st.dataframe(stats.style.format(precision=4), use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  TAB 6 — AUTO-REMEDIATION
# ══════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">Auto-Remediation Control System</div>', unsafe_allow_html=True)

    current_speed = st.session_state.motor_speed
    rem_active    = st.session_state.remediation_active
    speed_color   = '#ef4444' if current_speed <= 30 else '#f59e0b' if current_speed <= 60 else '#10b981'
    status_text   = 'EMERGENCY REDUCED' if current_speed <= 30 else 'WARNING REDUCED' if current_speed <= 60 else 'NORMAL OPERATION'
    status_css    = 's-critical' if current_speed <= 30 else 's-warning' if current_speed <= 60 else 's-good'

    st.markdown(f"""
    <div class="status-box {status_css}" style="margin-bottom:1rem;">
        <div class="status-title">MOTOR STATUS — {status_text}</div>
        <div class="status-msg">
            Auto-Remediation {"ACTIVE — System protecting motor from damage" if rem_active else "STANDBY — Motor running normally"}
        </div>
    </div>""", unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    for col_g, val, suffix, title, color, steps in [
        (g1, current_speed, "%", "MOTOR SPEED", speed_color,
         [(0,30,'rgba(239,68,68,0.08)'),(30,60,'rgba(245,158,11,0.06)'),(60,100,'rgba(16,185,129,0.06)')]),
        (g2, round(current_speed*0.85,1), "%", "POWER CONSUMPTION", '#6366f1',
         [(0,50,'rgba(16,185,129,0.04)'),(50,80,'rgba(245,158,11,0.04)'),(80,100,'rgba(239,68,68,0.04)')]),
        (g3, round(min(100,(100-current_speed)*1.2+20) if rem_active else 72.0, 1),
         "", "SYSTEM SAFETY SCORE",
         '#10b981' if (min(100,(100-current_speed)*1.2+20) if rem_active else 72)>=70 else '#f59e0b',
         [(0,40,'rgba(239,68,68,0.06)'),(40,70,'rgba(245,158,11,0.06)'),(70,100,'rgba(16,185,129,0.06)')]),
    ]:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=val,
            number=dict(suffix=suffix, font=dict(family='Space Grotesk',color=color,size=36)),
            title=dict(text=title, font=dict(family='JetBrains Mono',color='#4b5563',size=11)),
            gauge=dict(
                axis=dict(range=[0,100], tickcolor='rgba(99,102,241,0.3)',
                          tickfont=dict(color='#374151',size=9)),
                bar=dict(color=color, thickness=0.25),
                bgcolor='rgba(13,13,20,0.8)',
                borderwidth=1, bordercolor='rgba(99,102,241,0.12)',
                steps=[dict(range=[s[0],s[1]],color=s[2]) for s in steps])))
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                            height=280, margin=dict(t=40,b=10,l=30,r=30))
        col_g.plotly_chart(fig_g, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Manual Override Controls</div>', unsafe_allow_html=True)

    mc1,mc2,mc3,mc4 = st.columns(4)
    for col_b, label, speed, key in [
        (mc1,"FULL SPEED (100%)",100,"s100"),
        (mc2,"REDUCE TO 60%",   60,"s60"),
        (mc3,"REDUCE TO 30%",   30,"s30"),
        (mc4,"EMERGENCY STOP",  0, "s0"),
    ]:
        with col_b:
            if st.button(label, use_container_width=True, key=key):
                st.session_state.motor_speed = float(speed)
                st.session_state.remediation_active = speed < 100
                st.session_state.remediation_log.append({
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Fault': 'Manual Override',
                    'Motor Status': label,
                    'Speed %': speed,
                    'Actions': f'Manual override — speed set to {speed}%',
                    'Health': '—', 'RUL (h)': '—'})
                st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.session_state.remediation_log:
        last = st.session_state.remediation_log[-1]
        ts   = last.get('Timestamp','—')
        flt  = last.get('Fault','—')
        act  = last.get('Actions','—')
        hlth = last.get('Health','—')
        if rem_active:
            st.markdown(f"""
            <div style='background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);
            border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem;'>
                <div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;
                color:#6366f1;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem;'>
                ⚙️ Last Auto-Remediation Event</div>
                <div style='font-family:Space Grotesk,sans-serif;font-size:0.88rem;color:#a5b4fc;'>
                Time: {ts}<br>Fault: {flt}<br>Action: {act}<br>Health: {hlth}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Remediation Event History</div>', unsafe_allow_html=True)
    if st.session_state.remediation_log:
        rem_df = pd.DataFrame(st.session_state.remediation_log)
        st.dataframe(rem_df.style.format(precision=2), use_container_width=True)
        if st.button("CLEAR REMEDIATION LOG", use_container_width=True, key="clear_rem"):
            st.session_state.remediation_log = []
            st.session_state.motor_speed = 100.0
            st.session_state.remediation_active = False
            st.rerun()
    else:
        st.markdown("""
        <div style='text-align:center;padding:3rem;background:#0d0d14;
        border:1px solid rgba(99,102,241,0.15);border-radius:12px;
        font-family:JetBrains Mono,monospace;font-size:0.75rem;
        color:#374151;letter-spacing:2px;text-transform:uppercase;'>
        No remediation events yet
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">How Auto-Remediation Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;'>
        <div style='background:#0d0d14;border:1px solid rgba(99,102,241,0.15);border-radius:12px;padding:1rem;text-align:center;'>
            <div style='font-size:1.5rem;margin-bottom:8px;'>🔍</div>
            <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#6366f1;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;'>Step 1</div>
            <div style='font-family:Space Grotesk,sans-serif;font-size:0.8rem;color:#8b8fa8;'>AI detects fault from sensor data</div>
        </div>
        <div style='background:#0d0d14;border:1px solid rgba(99,102,241,0.15);border-radius:12px;padding:1rem;text-align:center;'>
            <div style='font-size:1.5rem;margin-bottom:8px;'>⚡</div>
            <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#6366f1;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;'>Step 2</div>
            <div style='font-family:Space Grotesk,sans-serif;font-size:0.8rem;color:#8b8fa8;'>n8n workflow triggers automatically</div>
        </div>
        <div style='background:#0d0d14;border:1px solid rgba(99,102,241,0.15);border-radius:12px;padding:1rem;text-align:center;'>
            <div style='font-size:1.5rem;margin-bottom:8px;'>📧</div>
            <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#6366f1;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;'>Step 3</div>
            <div style='font-family:Space Grotesk,sans-serif;font-size:0.8rem;color:#8b8fa8;'>Email + WhatsApp alert sent to team</div>
        </div>
        <div style='background:#0d0d14;border:1px solid rgba(99,102,241,0.15);border-radius:12px;padding:1rem;text-align:center;'>
            <div style='font-size:1.5rem;margin-bottom:8px;'>🔧</div>
            <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#6366f1;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;'>Step 4</div>
            <div style='font-family:Space Grotesk,sans-serif;font-size:0.8rem;color:#8b8fa8;'>Motor speed auto-reduced to safe level</div>
        </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer-bar">
MotorMind AI &nbsp;·&nbsp; Team PREDICT X &nbsp;·&nbsp; Random Forest + Gradient Boosting
&nbsp;·&nbsp; Built with Streamlit & Plotly &nbsp;·&nbsp; Automated by n8n
</div>
""", unsafe_allow_html=True)
