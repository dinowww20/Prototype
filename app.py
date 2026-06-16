# ============================================================
# app.py — Vector-Borne Disease CDSS
# Streamlit Cloud Deployment — Enhanced UI
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import shap
import plotly.graph_objects as go
import plotly.express as px
from groq import Groq

st.set_page_config(
    page_title="VBD Clinical Decision Support",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
/* Font & base */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header */
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    padding: 24px 32px; border-radius: 12px;
    color: white; margin-bottom: 24px;
}
.main-header h1 { font-size: 26px; font-weight: 700;
    margin: 0; color: white; }
.main-header p  { font-size: 14px; opacity: 0.85;
    margin: 4px 0 0 0; }

/* Cards */
.metric-card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.metric-card .label {
    font-size: 12px; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.05em;
    font-weight: 600;
}
.metric-card .value {
    font-size: 22px; font-weight: 700; color: #1e293b;
    margin-top: 4px;
}

/* Disease row */
.disease-card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 14px 18px;
    margin-bottom: 8px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    transition: all 0.2s;
}
.disease-card.above {
    border-left: 4px solid #ef4444;
    background: #fff8f8;
}
.disease-card .dis-name {
    font-size: 14px; font-weight: 600; color: #1e293b;
}
.disease-card .dis-prob {
    font-size: 20px; font-weight: 700;
}
.disease-card .dis-ci {
    font-size: 12px; color: #64748b;
}
.disease-card .action-badge {
    display: inline-block; padding: 4px 10px;
    border-radius: 99px; font-size: 12px;
    font-weight: 600;
}
.action-treat  { background: #fee2e2; color: #991b1b; }
.action-monitor{ background: #f1f5f9; color: #475569; }

/* SHAP bar */
.shap-feat { font-size: 12px; color: #475569;
    font-family: monospace; }
.shap-val-pos { color: #dc2626; font-weight: 600; }
.shap-val-neg { color: #2563eb; font-weight: 600; }

/* Section header */
.section-header {
    font-size: 15px; font-weight: 600; color: #1e293b;
    margin: 20px 0 12px 0; padding-bottom: 6px;
    border-bottom: 2px solid #e2e8f0;
}

/* Uncertainty banner */
.unc-high {
    background: #fef3c7; border: 1px solid #f59e0b;
    border-radius: 8px; padding: 10px 14px;
    font-size: 13px; color: #92400e;
    margin-bottom: 12px;
}
.unc-ok {
    background: #d1fae5; border: 1px solid #10b981;
    border-radius: 8px; padding: 10px 14px;
    font-size: 13px; color: #065f46;
    margin-bottom: 12px;
}

/* Disclaimer */
.disclaimer {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 10px 14px;
    font-size: 12px; color: #64748b;
    margin-top: 16px;
}

/* Chat */
.chat-container {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 16px;
}

/* Checklist */
.confirm-row {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 0; font-size: 13px;
    border-bottom: 1px solid #f1f5f9;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    with open('artifacts/metadata.json') as f:
        meta = json.load(f)
    models = []
    for i, label in enumerate(meta['target_labels']):
        label_clean = label.replace('/', '_').replace(' ', '_')
        m = joblib.load(f'artifacts/model_{i}_{label_clean}.pkl')
        models.append(m)
    with open('artifacts/feature_names.txt') as f:
        feat_names = [l.strip() for l in f.readlines()]
    boot_stats = joblib.load('artifacts/bootstrap_stats.pkl')
    return models, meta, feat_names, boot_stats

models, META, FEATURE_NAMES, BOOT_STATS = load_artifacts()

LABELS        = META['target_labels']
THRESHOLDS    = np.array(META['cv_thresholds'])
UNC_P75       = META['unc_p75']
POP_MEAN_STD  = BOOT_STATS['pop_mean_std']
POP_STD_LABEL = np.array(BOOT_STATS['pop_std_per_label'])

COLORS_HEX = ['#ef4444','#3b82f6','#f59e0b','#10b981','#8b5cf6']

# ============================================================
# GROQ AI CLIENT
# ============================================================
@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if api_key is None:
        return None
    return Groq(api_key=api_key)

SYSTEM_PROMPT = """You are a clinical decision support \
assistant helping healthcare workers interpret vector-borne \
disease probability assessments from an AI model trained on \
clinical data from Burkina Faso, West Africa.

The AI model is a Binary Relevance Random Forest trained on \
300 patients from two health facilities (CMA de DO and \
CMA de DAFRA). It predicts probabilities for 5 diseases: \
Malaria, Dengue, Yellow Fever, Typhoid, and Others.
Model CV F1 Macro: 0.6272 ± 0.0376 (5-fold CV, N=300).

You MUST follow these rules strictly:
1. Always frame output as probability-based decision support,
   never as definitive diagnosis.
2. Always recommend confirmation by a licensed medical
   professional.
3. Use precise clinical language for healthcare professionals.
4. Reference specific probability values from the context.
5. Acknowledge model uncertainty when flagged as HIGH.
6. Explain SHAP values in plain clinical terms.
7. Answer follow-up questions concisely.

You MUST NOT:
1. State a patient has or is diagnosed with a disease.
2. Provide specific drug dosages or treatment regimens.
3. Override or contradict the model probability outputs.
4. Make claims beyond what the model evidence supports."""

# ============================================================
# PREPROCESSING
# ============================================================
def build_fe_batch1(df):
    df = df.copy()
    syndrome_groups = {
        'neuro': [
            'Convulsions généralisées ou focales (Generalised or focal convulsion)',
            'Convulsions multiples (Multiple convulsions)',
            'Délire ',
            'Troubles de la conscience (Consciousness trouble)',
            'Prostration'
        ],
        'resp': [
            'Toux (Cough)',
            'Détresse respiratoire (Respiratory distress)',
            'Accumulation de liquide et détresse respiratoire (Accumulation of fluid and respiratory distress)'
        ],
        'gi': [
            'Vomissement (Vomiting)',
            'Diarrhée  (Diarrhea)',
            'Douleur abdominale (stomac pain)',
            'Nausée (Nausea)'
        ],
        'msk': [
            'Douleur articulaire (Joint pain)',
            'Douleur musculaire ( Muscle pain)',
            'Gonflement des articulations (Joint Swelling)',
            'Raideur (Stiffness)'
        ],
        'fever': [
            'Haute température.(temperature, Hyperpyrexia)',
            'Fièvre depuis 48 heures(Fever 48 hrs)',
            'Fièvre au cours des 7 derniers jours (Fever in the last 7 days)'
        ]
    }
    for name, cols in syndrome_groups.items():
        valid = [c for c in cols if c in df.columns]
        df[f'FE_{name}_score'] = df[valid].sum(axis=1)
        if name != 'fever':
            df[f'FE_has_{name}'] = (
                df[f'FE_{name}_score'] >= 2).astype(int)
    df['FE_persistent_fever'] = (
        (df.get('Fièvre depuis 48 heures(Fever 48 hrs)',
                pd.Series(0, index=df.index)) == 1) &
        (df.get('Fièvre au cours des 7 derniers jours (Fever in the last 7 days)',
                pd.Series(0, index=df.index)) == 1)
    ).astype(int)
    df['Âge (Age)'] = pd.to_numeric(
        df['Âge (Age)'], errors='coerce')
    df['FE_is_child']       = (df['Âge (Age)'] < 18).astype(int)
    df['FE_is_young_adult'] = (
        (df['Âge (Age)'] >= 18) &
        (df['Âge (Age)'] < 35)).astype(int)
    df['FE_is_elderly']     = (df['Âge (Age)'] >= 60).astype(int)
    df['FE_dengue_pattern']  = (
        df['FE_msk_score'] * df['FE_gi_score'])
    df['FE_typhoid_pattern'] = (
        df['FE_gi_score'] * (4 - df['FE_msk_score']))
    df['FE_others_pattern']  = (
        df['FE_resp_score'] * 2 - df['FE_msk_score'])
    return df

def clean_colname(col):
    col = re.sub(r'[^\w\s]', '', col)
    col = re.sub(r'\s+', '_', col.strip())
    return col

def fix_cols(X):
    X = X.copy()
    X.columns = [clean_colname(c) for c in X.columns]
    cols = pd.Series(X.columns)
    for dup in cols[cols.duplicated()].unique():
        dups = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dups[1:], 1):
            cols[idx] = f"{dup}_{i}"
    X.columns = cols
    return X

def apply_yf_rules(X_val, probas, feature_names):
    probas = probas.copy()
    fn = list(feature_names)
    muscle_col = [c for c in fn
                  if 'Muscle' in c and 'pain' in c.lower()]
    cough_col  = [c for c in fn
                  if 'Cough' in c or 'Toux' in c]
    conv_col   = [c for c in fn if 'Generalised' in c]
    mask = np.zeros(len(X_val), dtype=bool)
    if muscle_col:
        mask |= (X_val[:, fn.index(muscle_col[0])] == 0)
    if cough_col:
        mask |= (X_val[:, fn.index(cough_col[0])]  == 1)
    if conv_col:
        mask |= (X_val[:, fn.index(conv_col[0])]   == 1)
    probas[mask, 2] = 0.0
    return probas

def preprocess_df(df):
    drop_before = [c for c in
                   META['target_cols'] +
                   ['Dengue (Dengua)', 'n_diseases']
                   if c in df.columns]
    df = df.drop(columns=drop_before)
    df = build_fe_batch1(df)
    df = fix_cols(df)
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_NAMES].fillna(0).astype(np.float32)

# ============================================================
# PREDICTION ENGINE
# ============================================================
def predict_dataset_patient(patient_idx):
    return {
        'mean'  : BOOT_STATS['mean'][patient_idx],
        'std'   : float(BOOT_STATS['std'][patient_idx].mean()),
        'lower' : BOOT_STATS['lower'][patient_idx],
        'upper' : BOOT_STATS['upper'][patient_idx],
        'source': 'bootstrap_notebook',
    }

def predict_new_patient(X_arr):
    probas = np.column_stack([
        m.predict_proba(X_arr)[:, 1] for m in models
    ])
    probas = apply_yf_rules(X_arr, probas, FEATURE_NAMES)
    lower  = np.clip(probas - 1.96 * POP_STD_LABEL, 0, 1)
    upper  = np.clip(probas + 1.96 * POP_STD_LABEL, 0, 1)
    return {
        'mean'  : probas[0],
        'std'   : float(POP_MEAN_STD),
        'lower' : lower[0],
        'upper' : upper[0],
        'source': 'population_estimate',
    }

def get_shap_top3(X_arr, label_idx):
    exp = shap.TreeExplainer(models[label_idx])
    sv  = exp.shap_values(X_arr)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    top3 = np.argsort(np.abs(sv[0]))[::-1][:3]
    return [(FEATURE_NAMES[i][:45], float(sv[0, i]),
             float(X_arr[0, i])) for i in top3]

# ============================================================
# VISUALIZATIONS
# ============================================================
def plot_radar(mean_p):
    """Radar chart perbandingan 5 penyakit."""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[float(mean_p[i]) * 100 for i in range(5)] +
          [float(mean_p[0]) * 100],
        theta=LABELS + [LABELS[0]],
        fill='toself',
        fillcolor='rgba(99,102,241,0.15)',
        line=dict(color='#6366f1', width=2),
        name='Probability %'
    ))
    # Threshold line
    fig.add_trace(go.Scatterpolar(
        r=[float(THRESHOLDS[i]) * 100
           for i in range(5)] + [float(THRESHOLDS[0]) * 100],
        theta=LABELS + [LABELS[0]],
        fill='none',
        line=dict(color='#ef4444', width=1.5,
                  dash='dash'),
        name='Threshold'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100],
                ticksuffix='%', tickfont=dict(size=10),
                gridcolor='#e2e8f0',
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='#374151')
            ),
            bgcolor='white',
        ),
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='bottom',
            y=-0.15, xanchor='center', x=0.5,
            font=dict(size=11)
        ),
        margin=dict(l=40, r=40, t=20, b=40),
        height=320,
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    return fig

def plot_gauge(prob, threshold, label, color):
    """Gauge chart untuk satu label."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number=dict(suffix="%", font=dict(size=22,
                    color='#1e293b')),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor='#94a3b8',
                tickfont=dict(size=9),
            ),
            bar=dict(color=color, thickness=0.6),
            bgcolor='#f8fafc',
            borderwidth=1,
            bordercolor='#e2e8f0',
            steps=[
                dict(range=[0, threshold*100],
                     color='#f1f5f9'),
                dict(range=[threshold*100, 100],
                     color='#fee2e2'),
            ],
            threshold=dict(
                line=dict(color='#ef4444', width=2),
                thickness=0.8,
                value=threshold * 100
            )
        ),
        title=dict(text=label[:8],
                   font=dict(size=12, color='#374151'))
    ))
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='white',
    )
    return fig

def plot_shap_bar(top3, label, color):
    """Horizontal bar chart untuk SHAP values."""
    feats = [f[0][:30] for f in top3]
    vals  = [f[1] for f in top3]
    colors = ['#ef4444' if v > 0 else '#3b82f6' for v in vals]

    fig = go.Figure(go.Bar(
        x=vals[::-1],
        y=feats[::-1],
        orientation='h',
        marker_color=colors[::-1],
        marker_line_width=0,
        text=[f'{v:+.4f}' for v in vals[::-1]],
        textposition='outside',
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title=dict(text=f'SHAP — {label}',
                   font=dict(size=12, color='#374151')),
        xaxis=dict(
            title='SHAP value',
            tickfont=dict(size=9),
            gridcolor='#f1f5f9',
            zerolinecolor='#94a3b8',
        ),
        yaxis=dict(tickfont=dict(size=10)),
        height=180,
        margin=dict(l=10, r=60, t=36, b=30),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    return fig

# ============================================================
# REPORT RENDERER
# ============================================================
ACTIONS = {
    'Malaria'      : 'Confirm with RDT / blood smear',
    'Dengue'       : 'Observe, check platelet count',
    'Yellow Fever' : 'Isolate & notify health authority',
    'Typhoid'      : 'Blood culture & empirical antibiotics',
    'Others'       : 'Investigate further',
}

def render_report(pred, X_arr=None, show_shap=True,
                  true_dx=None, patient_info=None):
    mean_p   = pred['mean']
    lo       = pred['lower']
    hi       = pred['upper']
    std      = pred['std']
    source   = pred['source']
    unc_flag = std > UNC_P75

    # ── Patient summary card ──────────────────────────────
    if patient_info:
        cols = st.columns(4)
        info_items = [
            ("Patient", patient_info.get('id', '—')),
            ("Age",     patient_info.get('age', '—')),
            ("Gender",  patient_info.get('gender', '—')),
            ("Facility",patient_info.get('faskes', '—')),
        ]
        for col, (label, value) in zip(cols, info_items):
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='label'>{label}</div>"
                    f"<div class='value'>{value}</div>"
                    f"</div>",
                    unsafe_allow_html=True)
        st.markdown("")

    # ── Uncertainty banner ────────────────────────────────
    if source == 'population_estimate':
        st.info(
            "ℹ️ CI shown is a population-level estimate "
            "(±1.96 × population std). "
            "Individual bootstrap CI available in Tab 3.")

    if unc_flag:
        st.markdown(
            f"<div class='unc-high'>⚠️ <b>High uncertainty</b> "
            f"(std={std:.4f} > P75={UNC_P75:.4f}) "
            f"— second opinion recommended</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='unc-ok'>✓ <b>Confidence within "
            f"normal range</b> (std={std:.4f})</div>",
            unsafe_allow_html=True)

    # ── Radar chart ───────────────────────────────────────
    st.markdown(
        "<div class='section-header'>"
        "Disease Probability Overview</div>",
        unsafe_allow_html=True)

    col_radar, col_detail = st.columns([1, 2])
    with col_radar:
        st.plotly_chart(
            plot_radar(mean_p),
            use_container_width=True,
            config={'displayModeBar': False})

    with col_detail:
        sorted_idx = np.argsort(mean_p)[::-1]
        for i in sorted_idx:
            label  = LABELS[i]
            prob   = float(mean_p[i])
            above  = prob >= THRESHOLDS[i]
            color  = COLORS_HEX[i]
            action = ACTIONS[label] if above else "Monitor"
            badge_cls = "action-treat" if above else "action-monitor"
            marker    = "●" if above else "○"

            st.markdown(
                f"<div class='disease-card "
                f"{'above' if above else ''}'>"
                f"<div style='display:flex;justify-content:"
                f"space-between;align-items:center'>"
                f"<div>"
                f"<span style='color:{color};font-size:16px'>"
                f"{marker}</span> "
                f"<span class='dis-name'>{label}</span>"
                f"</div>"
                f"<div style='text-align:right'>"
                f"<span class='dis-prob' "
                f"style='color:{color}'>{prob*100:.1f}%</span>"
                f"</div>"
                f"</div>"
                f"<div style='margin-top:6px;display:flex;"
                f"justify-content:space-between;"
                f"align-items:center'>"
                f"<span class='dis-ci'>"
                f"95% CI: [{float(lo[i])*100:.1f}% – "
                f"{float(hi[i])*100:.1f}%] "
                f"| threshold={THRESHOLDS[i]:.2f}</span>"
                f"<span class='action-badge {badge_cls}'>"
                f"→ {action}</span>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True)

    # ── Gauge charts ──────────────────────────────────────
    st.markdown(
        "<div class='section-header'>Probability Gauges</div>",
        unsafe_allow_html=True)
    gauge_cols = st.columns(5)
    for i, col in enumerate(gauge_cols):
        with col:
            st.plotly_chart(
                plot_gauge(
                    float(mean_p[i]),
                    float(THRESHOLDS[i]),
                    LABELS[i],
                    COLORS_HEX[i]),
                use_container_width=True,
                config={'displayModeBar': False})

    # ── SHAP ──────────────────────────────────────────────
    if show_shap and X_arr is not None:
        above_labels = [i for i in range(5)
                        if float(mean_p[i]) >= THRESHOLDS[i]]
        if above_labels:
            st.markdown(
                "<div class='section-header'>"
                "Key Contributing Factors (SHAP)</div>",
                unsafe_allow_html=True)
            shap_cols = st.columns(min(len(above_labels), 3))
            for ci, i in enumerate(above_labels[:3]):
                top3 = get_shap_top3(X_arr, i)
                with shap_cols[ci]:
                    st.plotly_chart(
                        plot_shap_bar(top3, LABELS[i],
                                      COLORS_HEX[i]),
                        use_container_width=True,
                        config={'displayModeBar': False})
                    for feat, sv, fv in top3:
                        direction = ("↑ increases"
                                     if sv > 0 else
                                     "↓ decreases")
                        c = "#dc2626" if sv > 0 else "#2563eb"
                        st.markdown(
                            f"<div style='font-size:11px;"
                            f"padding:2px 0;color:#475569'>"
                            f"<code style='font-size:10px'>"
                            f"{feat[:28]}</code> "
                            f"val=<b>{fv:.2f}</b> "
                            f"<span style='color:{c}'>"
                            f"SHAP={sv:+.3f} {direction}"
                            f"</span></div>",
                            unsafe_allow_html=True)

    # ── Clinical confirmation checklist ───────────────────
    st.markdown(
        "<div class='section-header'>"
        "Clinical Confirmation Checklist</div>",
        unsafe_allow_html=True)
    st.caption(
        "Check findings confirmed by clinical examination")

    above_labels_check = [i for i in range(5)
                          if float(mean_p[i]) >= THRESHOLDS[i]]
    if above_labels_check:
        check_items = {
            'Malaria'      : ['RDT positive', 'Blood smear positive',
                              'Fever pattern consistent'],
            'Dengue'       : ['Platelet < 150k',
                              'Positive tourniquet test',
                              'NS1/IgM positive'],
            'Yellow Fever' : ['Jaundice present',
                              'Hemorrhagic signs',
                              'Epidemiological exposure'],
            'Typhoid'      : ['Blood culture sent',
                              'Relative bradycardia',
                              'Rose spots observed'],
            'Others'       : ['Alternative diagnosis considered',
                              'Specialist referral planned'],
        }
        check_cols = st.columns(min(len(above_labels_check), 3))
        for ci, i in enumerate(above_labels_check[:3]):
            label = LABELS[i]
            with check_cols[ci]:
                st.markdown(
                    f"**{label}** "
                    f"({float(mean_p[i])*100:.1f}%)")
                for item in check_items.get(label, []):
                    key = f"chk_{label}_{item}_{id(pred)}"
                    st.checkbox(item, key=key)
    else:
        st.caption(
            "No labels above threshold — "
            "all diseases below action threshold.")

    # ── Ground truth ──────────────────────────────────────
    if true_dx is not None:
        st.markdown("---")
        st.caption(
            f"[Evaluation only] True diagnosis: "
            f"{', '.join(true_dx) if true_dx else 'None'}")

    # ── Disclaimer ────────────────────────────────────────
    st.markdown(
        "<div class='disclaimer'>⚕️ <b>DISCLAIMER:</b> "
        "This output is for clinical decision support only. "
        "It does not constitute medical advice or diagnosis. "
        "All clinical decisions must be confirmed by a "
        "licensed medical professional based on full clinical "
        "assessment and laboratory findings.</div>",
        unsafe_allow_html=True)

# ============================================================
# AI CHAT
# ============================================================
def build_context_message(pred, shap_results, patient_info):
    mean_p = pred['mean']
    std    = pred['std']
    lo     = pred['lower']
    hi     = pred['upper']
    source = pred['source']
    unc    = std > UNC_P75

    prob_lines = []
    for i in np.argsort(mean_p)[::-1]:
        above  = mean_p[i] >= THRESHOLDS[i]
        marker = "ABOVE THRESHOLD" if above else "below threshold"
        prob_lines.append(
            f"- {LABELS[i]}: {mean_p[i]*100:.1f}% "
            f"[95% CI: {lo[i]*100:.1f}%-{hi[i]*100:.1f}%] "
            f"({marker}, threshold={THRESHOLDS[i]:.2f})"
        )

    shap_lines = []
    for label, top3 in shap_results.items():
        shap_lines.append(f"\n{label}:")
        for feat, sv, fv in top3:
            direction = "increases" if sv > 0 else "decreases"
            shap_lines.append(
                f"  - {feat}: value={fv:.2f}, "
                f"SHAP={sv:+.4f} ({direction} probability)"
            )

    ci_note = (
        "CI is population-level estimate (new patient)."
        if source == 'population_estimate'
        else "CI is exact individual bootstrap CI (N=50)."
    )

    return (
        f"Here is the full clinical assessment context.\n\n"
        f"PATIENT INFORMATION:\n"
        f"- Age: {patient_info.get('age', 'Unknown')}\n"
        f"- Gender: {patient_info.get('gender', 'Unknown')}\n"
        f"- Facility: {patient_info.get('faskes', 'Unknown')}\n\n"
        f"MODEL OUTPUT (sorted by probability):\n"
        f"{chr(10).join(prob_lines)}\n\n"
        f"UNCERTAINTY:\n"
        f"- Mean prediction std: {std:.4f}\n"
        f"- P75 threshold: {UNC_P75:.4f}\n"
        f"- Flag: "
        f"{'HIGH - second opinion recommended' if unc else 'Within normal range'}\n"
        f"- {ci_note}\n\n"
        f"SHAP CONTRIBUTING FACTORS:\n"
        f"{chr(10).join(shap_lines) if shap_lines else 'Not available.'}\n\n"
        f"Please provide:\n"
        f"### Prediction Interpretation\n"
        f"### Recommended Clinical Actions\n"
        f"### Key Contributing Factors (SHAP)\n\n"
        f"After your initial response, I will ask "
        f"follow-up questions."
    )

def render_ai_chat(pred, X_arr, patient_info, chat_key):
    client = get_groq_client()
    if client is None:
        st.warning(
            "AI chat unavailable — "
            "GROQ_API_KEY not found in Streamlit secrets.")
        return

    hist_key    = f"chat_history_{chat_key}"
    context_key = f"chat_context_set_{chat_key}"

    if hist_key not in st.session_state:
        st.session_state[hist_key] = []
    if context_key not in st.session_state:
        st.session_state[context_key] = False

    st.markdown(
        "<div class='section-header'>"
        "🤖 AI Clinical Assistant</div>",
        unsafe_allow_html=True)
    st.caption(
        "Ask follow-up questions about this assessment · "
        "For decision support only")

    col_init, col_reset = st.columns([3, 1])

    with col_init:
        if not st.session_state[context_key]:
            if st.button(
                    "Start AI consultation ↗",
                    type="primary",
                    key=f"start_{chat_key}"):

                mean_p       = pred['mean']
                above_labels = [
                    i for i in range(5)
                    if mean_p[i] >= THRESHOLDS[i]]
                shap_results = {}
                if X_arr is not None and above_labels:
                    with st.spinner("Computing SHAP..."):
                        for i in above_labels[:3]:
                            shap_results[LABELS[i]] = \
                                get_shap_top3(X_arr, i)

                context_msg = build_context_message(
                    pred, shap_results, patient_info)

                st.session_state[hist_key] = [
                    {"role": "user",
                     "content": context_msg}
                ]

                with st.spinner(
                        "Generating initial interpretation..."):
                    try:
                        response = \
                            client.chat.completions.create(
                                model="llama-3.1-8b-instant",
                                messages=[
                                    {"role"   : "system",
                                     "content": SYSTEM_PROMPT},
                                ] + st.session_state[hist_key],
                                max_tokens=800,
                                temperature=0.3,
                            )
                        assistant_msg = \
                            response.choices[0].message.content
                        st.session_state[hist_key].append({
                            "role"   : "assistant",
                            "content": assistant_msg,
                        })
                        st.session_state[context_key] = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"API error: {e}")
                        return

    with col_reset:
        if st.session_state[context_key]:
            if st.button("Reset chat",
                         key=f"reset_{chat_key}"):
                st.session_state[hist_key]    = []
                st.session_state[context_key] = False
                st.rerun()

    if st.session_state[context_key]:
        st.markdown(
            "<div class='chat-container'>",
            unsafe_allow_html=True)

        display_history = st.session_state[hist_key][1:]
        for msg in display_history:
            if msg["role"] == "assistant":
                with st.chat_message("assistant",
                                     avatar="⚕️"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("user",
                                     avatar="👤"):
                    st.markdown(msg["content"])

        st.markdown("</div>", unsafe_allow_html=True)

        user_input = st.chat_input(
            "Ask a follow-up question about this "
            "assessment...",
            key=f"chat_input_{chat_key}")

        if user_input:
            st.session_state[hist_key].append({
                "role"   : "user",
                "content": user_input,
            })
            with st.chat_message("assistant", avatar="⚕️"):
                response_placeholder = st.empty()
                full_response        = ""
                try:
                    stream = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role"   : "system",
                             "content": SYSTEM_PROMPT},
                        ] + st.session_state[hist_key],
                        max_tokens=600,
                        temperature=0.3,
                        stream=True,
                    )
                    for chunk in stream:
                        delta = \
                            chunk.choices[0].delta.content
                        if delta:
                            full_response += delta
                            response_placeholder.markdown(
                                full_response + "▌")
                    response_placeholder.markdown(
                        full_response)
                    st.session_state[hist_key].append({
                        "role"   : "assistant",
                        "content": full_response,
                    })
                except Exception as e:
                    st.error(f"API error: {e}")

        st.caption(
            "⚕️ AI responses are for clinical decision "
            "support only. Not medical advice.")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:8px 0'>"
        "<span style='font-size:32px'>⚕️</span>"
        "<div style='font-size:16px;font-weight:700;"
        "color:#1e293b;margin-top:4px'>VBD CDSS</div>"
        "<div style='font-size:11px;color:#64748b'>"
        "Clinical Decision Support</div>"
        "</div>",
        unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**🧠 Model**")
    st.caption("Binary Relevance Random Forest")
    st.caption("CV F1 Macro: **0.6272 ± 0.0376**")
    st.caption("N=300 · Burkina Faso · Seed=456")
    st.markdown("---")

    st.markdown("**📊 Thresholds** (median 5-fold CV)")
    for l, t, c in zip(LABELS, THRESHOLDS, COLORS_HEX):
        st.markdown(
            f"<div style='display:flex;justify-content:"
            f"space-between;font-size:12px;padding:2px 0'>"
            f"<span style='color:{c};font-weight:500'>"
            f"{l}</span>"
            f"<span style='color:#64748b'>{t:.2f}</span>"
            f"</div>",
            unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**📉 Uncertainty**")
    st.caption(f"P75 threshold : {UNC_P75:.4f}")
    st.caption(f"Pop mean std  : {POP_MEAN_STD:.4f}")
    st.markdown("---")

    st.markdown("**🤖 AI Assistant**")
    grok_ok = st.secrets.get(
        "GROQ_API_KEY", None) is not None
    if grok_ok:
        st.success("Groq AI: connected")
    else:
        st.warning("Groq AI: not configured")

# ============================================================
# HEADER
# ============================================================
st.markdown(
    "<div class='main-header'>"
    "<h1>⚕️ Vector-Borne Disease CDSS</h1>"
    "<p>Multi-label Clinical Decision Support · "
    "Malaria · Dengue · Yellow Fever · Typhoid · Others · "
    "Burkina Faso, West Africa</p>"
    "</div>",
    unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📋 Manual Input",
    "📁 Upload CSV",
    "🔍 Dataset Lookup",
])

# ── TAB 1: MANUAL INPUT ───────────────────────────────────
with tab1:
    st.markdown(
        "<div class='section-header'>"
        "Patient Information</div>",
        unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input(
            "Age (years)",
            min_value=0, max_value=120,
            value=25, step=1)
    with c2:
        gender = st.selectbox(
            "Gender", ["Male", "Female"], index=0)
    with c3:
        faskes = st.selectbox(
            "Facility",
            ["CMA de DO", "CMA de DAFRA"], index=0)

    st.markdown(
        "<div class='section-header'>"
        "Clinical Symptoms</div>",
        unsafe_allow_html=True)
    st.caption("Check all symptoms present in this patient")

    symptoms = {
        'Fièvre depuis 48 heures(Fever 48 hrs)':
            'Fever ≥ 48h',
        'Fièvre au cours des 7 derniers jours (Fever in the last 7 days)':
            'Fever last 7 days',
        'Haute température.(temperature, Hyperpyrexia)':
            'High temperature / hyperpyrexia',
        'Toux (Cough)': 'Cough',
        'Douleur musculaire ( Muscle pain)': 'Muscle pain',
        'Douleur articulaire (Joint pain)': 'Joint pain',
        'Vomissement (Vomiting)': 'Vomiting',
        'Diarrhée  (Diarrhea)': 'Diarrhea',
        'Douleur abdominale (stomac pain)': 'Abdominal pain',
        'Nausée (Nausea)': 'Nausea',
        'Convulsions généralisées ou focales (Generalised or focal convulsion)':
            'Convulsions',
        'Prostration': 'Prostration',
        'Ictère (Icterus)': 'Jaundice (Icterus)',
        'Saignement/ Manifestations hémorragiques (Bleeding)':
            'Bleeding / hemorrhagic signs',
        'Vertige (Dizzy)': 'Dizziness',
        'Détresse respiratoire (Respiratory distress)':
            'Respiratory distress',
        'Pâleur cutanéo muqueuse ou Anémie (Mucosal skin pallor or Anemia)':
            'Pallor / Anemia',
        'Troubles de la conscience (Consciousness trouble)':
            'Consciousness trouble',
    }

    sym_cols = st.columns(3)
    sym_vals = {}
    sym_list = list(symptoms.items())
    n_per    = len(sym_list) // 3 + 1
    for ci, col in enumerate(sym_cols):
        with col:
            for orig, label in sym_list[ci*n_per:(ci+1)*n_per]:
                sym_vals[orig] = st.checkbox(
                    label, key=f"sym_{ci}_{hash(orig)}")

    st.markdown(
        "<div class='section-header'>"
        "Lab Values (optional)</div>",
        unsafe_allow_html=True)
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        platelet = st.number_input(
            "Platelet (×10³/µL)",
            min_value=0, max_value=1000,
            value=150, step=1)
        wbc = st.number_input(
            "WBC (cells/µL)",
            min_value=0, max_value=50000,
            value=7000, step=100)
    with lc2:
        temp_val = st.number_input(
            "Temperature (°C)",
            min_value=35.0, max_value=42.0,
            value=37.5, step=0.1)
        pulse = st.number_input(
            "Pulse (bpm)",
            min_value=40, max_value=200,
            value=80, step=1)
    with lc3:
        weight = st.number_input(
            "Weight (kg)",
            min_value=1, max_value=150,
            value=60, step=1)

    if st.button(
            "⚡ Generate Assessment",
            type="primary",
            use_container_width=True,
            key="btn_manual"):

        row = {col: 0.0 for col in FEATURE_NAMES}

        for orig_col, val in sym_vals.items():
            clean = clean_colname(orig_col)
            if clean in row:
                row[clean] = float(int(val))

        lab_map = {
            'Numération_plaquettaire_Platelet_count'   : float(platelet),
            'Nombre_de_globules_blancs_cellulesML_Whi' : float(wbc),
            'Température_axillaire_médiane_IQR_C_Axil' : float(temp_val),
            'Fréquence_du_pouls_battementsm_in_SD_Pul' : float(pulse),
            'Poids_Weight'                              : float(weight),
            'Âge_Age'                                   : float(age),
            'Genre_Gender'    : 1.0 if gender == 'Male' else 0.0,
            'Centre_de_santé' : 0.0 if faskes == 'CMA de DO' else 1.0,
        }
        for k, v in lab_map.items():
            if k in row:
                row[k] = v

        msk_score = float(sum([
            int(sym_vals.get(
                'Douleur articulaire (Joint pain)', False)),
            int(sym_vals.get(
                'Douleur musculaire ( Muscle pain)', False)),
        ]))
        gi_score = float(sum([
            int(sym_vals.get(
                'Vomissement (Vomiting)', False)),
            int(sym_vals.get(
                'Diarrhée  (Diarrhea)', False)),
            int(sym_vals.get(
                'Douleur abdominale (stomac pain)', False)),
            int(sym_vals.get('Nausée (Nausea)', False)),
        ]))
        resp_score = float(sum([
            int(sym_vals.get('Toux (Cough)', False)),
            int(sym_vals.get(
                'Détresse respiratoire (Respiratory distress)',
                False)),
        ]))
        neuro_score = float(sum([
            int(sym_vals.get(
                'Convulsions généralisées ou focales (Generalised or focal convulsion)',
                False)),
            int(sym_vals.get('Prostration', False)),
            int(sym_vals.get(
                'Troubles de la conscience (Consciousness trouble)',
                False)),
        ]))
        fever_score = float(sum([
            int(sym_vals.get(
                'Haute température.(temperature, Hyperpyrexia)',
                False)),
            int(sym_vals.get(
                'Fièvre depuis 48 heures(Fever 48 hrs)',
                False)),
            int(sym_vals.get(
                'Fièvre au cours des 7 derniers jours (Fever in the last 7 days)',
                False)),
        ]))

        row['FE_msk_score']        = msk_score
        row['FE_gi_score']         = gi_score
        row['FE_resp_score']       = resp_score
        row['FE_neuro_score']      = neuro_score
        row['FE_fever_score']      = fever_score
        row['FE_has_msk']          = float(msk_score   >= 2)
        row['FE_has_gi']           = float(gi_score    >= 2)
        row['FE_has_resp']         = float(resp_score  >= 2)
        row['FE_has_neuro']        = float(neuro_score >= 2)
        row['FE_dengue_pattern']   = msk_score * gi_score
        row['FE_typhoid_pattern']  = gi_score * (4 - msk_score)
        row['FE_others_pattern']   = resp_score * 2 - msk_score
        row['FE_is_child']         = float(age < 18)
        row['FE_is_young_adult']   = float(18 <= age < 35)
        row['FE_is_elderly']       = float(age >= 60)
        row['FE_persistent_fever'] = float(
            sym_vals.get(
                'Fièvre depuis 48 heures(Fever 48 hrs)',
                False) and
            sym_vals.get(
                'Fièvre au cours des 7 derniers jours (Fever in the last 7 days)',
                False))

        X_manual = np.array(
            [[row.get(f, 0.0) for f in FEATURE_NAMES]],
            dtype=np.float32)

        st.session_state['X_manual_tab1'] = X_manual
        st.session_state['pred_tab1']     = \
            predict_new_patient(X_manual)
        st.session_state['pinfo_tab1']    = {
            'id'    : 'Manual Input',
            'age'   : f"{age} years",
            'gender': gender,
            'faskes': faskes,
        }
        st.session_state['chat_history_tab1']     = []
        st.session_state['chat_context_set_tab1'] = False

    if 'pred_tab1' in st.session_state:
        st.markdown("---")
        render_report(
            st.session_state['pred_tab1'],
            X_arr=st.session_state['X_manual_tab1'],
            show_shap=True,
            patient_info=st.session_state['pinfo_tab1'])
        st.markdown("---")
        render_ai_chat(
            st.session_state['pred_tab1'],
            st.session_state['X_manual_tab1'],
            st.session_state['pinfo_tab1'],
            chat_key="tab1")

# ── TAB 2: UPLOAD CSV ─────────────────────────────────────
with tab2:
    st.markdown(
        "<div class='section-header'>"
        "Upload Patient CSV</div>",
        unsafe_allow_html=True)
    st.caption(
        "Upload a CSV file with French column names matching "
        "the original dataset format. "
        "Missing columns will be filled with 0.")

    uploaded = st.file_uploader(
        "Choose CSV file", type=['csv'])

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(
                f"✓ Loaded **{len(df_up)} patients**, "
                f"{len(df_up.columns)} columns")

            with st.expander("Preview data", expanded=False):
                st.dataframe(df_up.head(5),
                             use_container_width=True)

            if st.button(
                    "⚡ Run Batch Assessment",
                    type="primary",
                    use_container_width=True,
                    key="btn_csv"):

                with st.spinner(
                        "Preprocessing & predicting..."):
                    X_batch  = preprocess_df(df_up).values
                    probas_b = np.column_stack([
                        m.predict_proba(X_batch)[:, 1]
                        for m in models
                    ])
                    probas_b = apply_yf_rules(
                        X_batch, probas_b, FEATURE_NAMES)

                st.session_state['X_batch_tab2']  = X_batch
                st.session_state['probas_b_tab2'] = probas_b
                st.session_state['df_up_tab2']    = df_up

            if 'X_batch_tab2' in st.session_state:
                X_batch  = st.session_state['X_batch_tab2']
                probas_b = st.session_state['probas_b_tab2']
                df_up    = st.session_state['df_up_tab2']

                st.markdown(
                    "<div class='section-header'>"
                    "Batch Results Summary</div>",
                    unsafe_allow_html=True)

                results = pd.DataFrame(
                    (probas_b >= THRESHOLDS).astype(int),
                    columns=LABELS)
                results.insert(0, 'Patient',
                               range(len(df_up)))
                results['n_diseases'] = \
                    results[LABELS].sum(axis=1)
                results['top_label'] = [
                    LABELS[np.argmax(probas_b[i])]
                    for i in range(len(probas_b))]
                results['top_prob'] = \
                    probas_b.max(axis=1).round(3)

                # Summary metrics
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                metrics = [
                    ("Total", len(df_up), ""),
                    ("Co-infections",
                     (results['n_diseases'] >= 2).sum(),
                     f"{(results['n_diseases']>=2).mean()*100:.1f}%"),
                    ("Malaria",
                     results['Malaria'].sum(), ""),
                    ("Dengue",
                     results['Dengue'].sum(), ""),
                    ("Yellow Fever",
                     results['Yellow Fever'].sum(), ""),
                ]
                for col, (lbl, val, delta) in zip(
                        [mc1,mc2,mc3,mc4,mc5], metrics):
                    with col:
                        st.metric(lbl, val,
                                  delta if delta else None)

                # Population radar
                pop_mean = probas_b.mean(axis=0)
                st.plotly_chart(
                    plot_radar(pop_mean),
                    use_container_width=True,
                    config={'displayModeBar': False})
                st.caption(
                    "Population average probability "
                    "across all uploaded patients")

                # Results table
                st.dataframe(
                    results.style.background_gradient(
                        subset=LABELS, cmap='RdYlGn',
                        vmin=0, vmax=1),
                    use_container_width=True)

                # Per-patient detail
                st.markdown(
                    "<div class='section-header'>"
                    "Per-Patient Detail</div>",
                    unsafe_allow_html=True)
                pat_sel = st.selectbox(
                    "Select patient for detailed report",
                    range(len(df_up)),
                    format_func=lambda x: f"Patient #{x}",
                    key="pat_sel_tab2")

                pred_sel = predict_new_patient(
                    X_batch[pat_sel:pat_sel+1])

                render_report(
                    pred_sel,
                    X_arr=X_batch[pat_sel:pat_sel+1],
                    show_shap=True,
                    patient_info={
                        'id'    : f'#{pat_sel}',
                        'age'   : 'From CSV',
                        'gender': 'From CSV',
                        'faskes': 'From CSV',
                    })

                st.markdown("---")
                render_ai_chat(
                    pred_sel,
                    X_batch[pat_sel:pat_sel+1],
                    {'age'   : 'From CSV',
                     'gender': 'From CSV',
                     'faskes': 'From CSV'},
                    chat_key=f"tab2_{pat_sel}")

                st.download_button(
                    "⬇️ Download Results CSV",
                    data=results.to_csv(index=False),
                    file_name="vbd_cdss_results.csv",
                    mime="text/csv",
                    use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
            st.caption(
                "Ensure CSV format matches the original "
                "dataset (French column names).")

# ── TAB 3: DATASET LOOKUP ─────────────────────────────────
with tab3:
    st.markdown(
        "<div class='section-header'>"
        "Dataset Patient Lookup</div>",
        unsafe_allow_html=True)
    st.caption(
        "Look up a patient from the original 300-patient "
        "dataset. Uses exact bootstrap CI from notebook "
        "(N=50 iterations) — identical to notebook output.")

    pat_idx = st.number_input(
        "Patient index (0–299)",
        min_value=0, max_value=299,
        value=16, step=1)

    if st.button(
            "🔍 Lookup Patient",
            type="primary",
            use_container_width=True,
            key="btn_lookup"):

        pred = predict_dataset_patient(int(pat_idx))
        st.session_state['pred_tab3']   = pred
        st.session_state['patidx_tab3'] = int(pat_idx)
        st.session_state['chat_history_tab3']     = []
        st.session_state['chat_context_set_tab3'] = False

    if 'pred_tab3' in st.session_state:
        idx_shown = st.session_state['patidx_tab3']
        st.caption(
            f"Bootstrap CI: exact from notebook (N=50).")

        render_report(
            st.session_state['pred_tab3'],
            X_arr=None,
            show_shap=False,
            patient_info={
                'id'    : f'#{idx_shown}',
                'age'   : 'From dataset',
                'gender': 'From dataset',
                'faskes': 'From dataset',
            })

        st.markdown("---")
        render_ai_chat(
            st.session_state['pred_tab3'],
            None,
            {'age'   : f"Patient #{idx_shown}",
             'gender': 'From dataset',
             'faskes': 'From dataset'},
            chat_key="tab3")
