
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import shap
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="VBD Clinical Decision Support",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    return models, meta, feat_names

models, META, FEATURE_NAMES = load_artifacts()

LABELS     = META['target_labels']
THRESHOLDS = np.array(META['cv_thresholds'])
UNC_P75    = META['unc_p75']
N_BOOT     = 30  # lebih cepat di cloud vs 50 di notebook

# ============================================================
# FEATURE ENGINEERING — identik dengan notebook
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
            df[f'FE_has_{name}'] = (df[f'FE_{name}_score'] >= 2).astype(int)

    df['FE_persistent_fever'] = (
        (df.get('Fièvre depuis 48 heures(Fever 48 hrs)',
                pd.Series(0, index=df.index)) == 1) &
        (df.get('Fièvre au cours des 7 derniers jours (Fever in the last 7 days)',
                pd.Series(0, index=df.index)) == 1)
    ).astype(int)

    df['Âge (Age)'] = pd.to_numeric(df['Âge (Age)'], errors='coerce')
    df['FE_is_child']       = (df['Âge (Age)'] < 18).astype(int)
    df['FE_is_young_adult'] = ((df['Âge (Age)'] >= 18) &
                                (df['Âge (Age)'] < 35)).astype(int)
    df['FE_is_elderly']     = (df['Âge (Age)'] >= 60).astype(int)
    df['FE_dengue_pattern']  = df['FE_msk_score'] * df['FE_gi_score']
    df['FE_typhoid_pattern'] = df['FE_gi_score'] * (4 - df['FE_msk_score'])
    df['FE_others_pattern']  = df['FE_resp_score'] * 2 - df['FE_msk_score']
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
    muscle_col = [c for c in feature_names
                  if 'Muscle' in c and 'pain' in c.lower()]
    cough_col  = [c for c in feature_names
                  if 'Cough' in c or 'Toux' in c]
    conv_col   = [c for c in feature_names
                  if 'Generalised' in c]
    mask = np.zeros(len(X_val), dtype=bool)
    fn   = list(feature_names)
    if muscle_col:
        mask |= (X_val[:, fn.index(muscle_col[0])] == 0)
    if cough_col:
        mask |= (X_val[:, fn.index(cough_col[0])]  == 1)
    if conv_col:
        mask |= (X_val[:, fn.index(conv_col[0])]   == 1)
    probas[mask, 2] = 0.0
    return probas

def preprocess_df(df):
    """Full preprocessing pipeline — identik dengan notebook."""
    df = build_fe_batch1(df)
    drop_cols = [c for c in df.columns
                 if c in META['target_cols'] + ['Dengue (Dengua)', 'n_diseases']]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = fix_cols(df)
    # Align ke feature names dari training
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_NAMES]
    return df.fillna(0).astype(np.float32)

# ============================================================
# PREDICTION ENGINE
# ============================================================
def predict_with_bootstrap(X_arr, n_boot=N_BOOT, seed=456):
    """Bootstrap CI — identik dengan notebook."""
    np.random.seed(seed)
    boot_probas = np.zeros((n_boot, len(X_arr), 5))
    for b in range(n_boot):
        boot_idx = np.random.choice(len(X_arr), len(X_arr), replace=True)
        X_boot   = X_arr[boot_idx]
        for i, m in enumerate(models):
            m_b = RandomForestClassifier(
                **{k: v for k, v in META['rf_params'].items()},
                class_weight='balanced',
                random_state=seed + b,
                n_jobs=-1
            )
            m_b.fit(X_boot,
                    m.predict(X_arr))
            boot_probas[b, :, i] = m_b.predict_proba(X_arr)[:, 1]

    mean_p  = boot_probas.mean(axis=0)
    std_p   = boot_probas.std(axis=0)
    lo_p    = np.percentile(boot_probas, 2.5,  axis=0)
    hi_p    = np.percentile(boot_probas, 97.5, axis=0)

    mean_p = apply_yf_rules(X_arr, mean_p, FEATURE_NAMES)
    lo_p   = apply_yf_rules(X_arr, lo_p,   FEATURE_NAMES)
    hi_p   = apply_yf_rules(X_arr, hi_p,   FEATURE_NAMES)

    return mean_p, std_p, lo_p, hi_p

def predict_single(X_arr):
    """Fast single prediction tanpa bootstrap."""
    probas = np.column_stack([
        m.predict_proba(X_arr)[:, 1] for m in models
    ])
    return apply_yf_rules(X_arr, probas, FEATURE_NAMES)

def get_shap_top3(X_arr, label_idx):
    """SHAP top 3 features untuk satu label."""
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
# REPORT RENDERER
# ============================================================
COLORS = ['#ef4444', '#3b82f6', '#f59e0b', '#10b981', '#8b5cf6']
ACTIONS = {
    'Malaria'      : 'Confirm with RDT / blood smear',
    'Dengue'       : 'Observe, check platelet count',
    'Yellow Fever' : 'Isolate & notify health authority',
    'Typhoid'      : 'Blood culture & empirical antibiotics',
    'Others'       : 'Investigate further',
}

def render_report(mean_p, std_p, lo_p, hi_p, pat_idx=0,
                  X_arr=None, show_shap=True):
    probs  = mean_p[pat_idx]
    lo     = lo_p[pat_idx]
    hi     = hi_p[pat_idx]
    std    = std_p[pat_idx].mean()
    unc_flag = std > UNC_P75

    # Uncertainty banner
    if unc_flag:
        st.warning(
            f"⚠️ High uncertainty (std={std:.4f} > P75={UNC_P75:.4f}) "
            f"— second opinion recommended")
    else:
        st.success(
            f"✓ Confidence within normal range (std={std:.4f})")

    # Disease probability table
    st.markdown("**Disease probability assessment** "
                "(95% Bootstrap CI, N=30)")
    sorted_idx = np.argsort(probs)[::-1]

    for i in sorted_idx:
        label  = LABELS[i]
        prob   = probs[i]
        above  = prob >= THRESHOLDS[i]
        marker = "●" if above else "○"
        color  = COLORS[i]
        action = ACTIONS[label] if above else "Monitor"

        col1, col2, col3 = st.columns([2, 3, 3])
        with col1:
            st.markdown(
                f"<span style='color:{color};font-weight:600'>"
                f"{marker} {label}</span>",
                unsafe_allow_html=True)
        with col2:
            st.progress(float(prob))
            st.caption(
                f"{prob*100:.1f}% [{lo[i]*100:.1f}–{hi[i]*100:.1f}%]")
        with col3:
            if above:
                st.error(f"→ {action}")
            else:
                st.info("→ Monitor")

    st.caption(
        f"Threshold per label: "
        + " | ".join([f"{l[:3]}={t:.2f}"
                      for l, t in zip(LABELS, THRESHOLDS)]))

    # SHAP
    if show_shap and X_arr is not None:
        above_labels = [i for i in range(5)
                        if probs[i] >= THRESHOLDS[i]]
        if above_labels:
            st.markdown("---")
            st.markdown("**Top contributing factors** "
                        "(labels above threshold)")
            for i in above_labels[:3]:
                with st.expander(
                        f"{LABELS[i]} — {probs[i]*100:.1f}%",
                        expanded=True):
                    top3 = get_shap_top3(X_arr, i)
                    for feat, sv, fv in top3:
                        direction = "↑ increases" if sv > 0 \
                                    else "↓ decreases"
                        color_sv  = "red" if sv > 0 else "blue"
                        st.markdown(
                            f"`{feat}` &nbsp; val=**{fv:.2f}** &nbsp; "
                            f"<span style='color:{color_sv}'>"
                            f"SHAP={sv:+.4f} {direction}</span>",
                            unsafe_allow_html=True)

    st.caption(
        "⚕️ This output is for clinical decision support only. "
        "Final diagnosis must be confirmed by a licensed medical "
        "professional.")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚕️ VBD CDSS")
    st.caption("Vector-Borne Disease\nClinical Decision Support")
    st.markdown("---")
    st.markdown("**Model info**")
    st.caption(f"Algorithm: Random Forest (Binary Relevance)")
    st.caption(f"CV F1 Macro: 0.6272 ± 0.0376")
    st.caption(f"Dataset: N=300, Burkina Faso")
    st.caption(f"Labels: {', '.join(LABELS)}")
    st.markdown("---")
    st.markdown("**Thresholds (median 5-fold CV)**")
    for l, t in zip(LABELS, THRESHOLDS):
        st.caption(f"{l}: {t:.2f}")
    st.markdown("---")
    use_bootstrap = st.checkbox(
        "Enable bootstrap CI (slower)", value=False)
    st.caption(
        "Bootstrap N=30 iterations. "
        "Disable for faster predictions.")

# ============================================================
# MAIN TABS
# ============================================================
st.title("Vector-Borne Disease")
st.subheader("Clinical Decision Support System")
st.caption(
    "Multi-label prediction of co-infections: "
    "Malaria, Dengue, Yellow Fever, Typhoid, Others")
st.markdown("---")

tab1, tab2 = st.tabs(["📋 Manual input", "📁 Upload CSV"])

# ── TAB 1: MANUAL INPUT ───────────────────────────────────
with tab1:
    st.markdown("### Patient information")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input(
            "Age (years)", min_value=0, max_value=120,
            value=25, step=1)
    with col2:
        gender = st.selectbox(
            "Gender", ["Male", "Female"], index=0)
    with col3:
        faskes = st.selectbox(
            "Facility", ["CMA de DO", "CMA de DAFRA"], index=0)

    st.markdown("### Clinical symptoms")
    st.caption("Check all symptoms present in this patient")

    # Symptom groups — sesuai dengan dataset
    symptom_cols = st.columns(3)
    symptoms = {
        'Fièvre depuis 48 heures(Fever 48 hrs)':
            'Fever ≥ 48h',
        'Fièvre au cours des 7 derniers jours (Fever in the last 7 days)':
            'Fever last 7 days',
        'Haute température.(temperature, Hyperpyrexia)':
            'High temperature',
        'Toux (Cough)':
            'Cough',
        'Douleur musculaire ( Muscle pain)':
            'Muscle pain',
        'Douleur articulaire (Joint pain)':
            'Joint pain',
        'Vomissement (Vomiting)':
            'Vomiting',
        'Diarrhée  (Diarrhea)':
            'Diarrhea',
        'Douleur abdominale (stomac pain)':
            'Abdominal pain',
        'Nausée (Nausea)':
            'Nausea',
        'Convulsions généralisées ou focales (Generalised or focal convulsion)':
            'Convulsions',
        'Prostration':
            'Prostration',
        'Ictère (Icterus)':
            'Jaundice',
        'Saignement/ Manifestations hémorragiques (Bleeding)':
            'Bleeding',
        'Vertige (Dizzy)':
            'Dizziness',
        'Détresse respiratoire (Respiratory distress)':
            'Respiratory distress',
        'Pâleur cutanéo muqueuse ou Anémie (Mucosal skin pallor or Anemia)':
            'Pallor / Anemia',
        'Troubles de la conscience (Consciousness trouble)':
            'Consciousness trouble',
    }

    symptom_vals = {}
    sym_list     = list(symptoms.items())
    n_per_col    = len(sym_list) // 3 + 1
    for ci, col in enumerate(symptom_cols):
        with col:
            for orig, label in sym_list[ci*n_per_col:(ci+1)*n_per_col]:
                symptom_vals[orig] = st.checkbox(label, key=f"sym_{orig}")

    st.markdown("### Lab values (optional)")
    lab_col1, lab_col2, lab_col3 = st.columns(3)
    with lab_col1:
        platelet = st.number_input(
            "Platelet count (×10³/µL)",
            min_value=0, max_value=1000,
            value=150, step=1)
        wbc = st.number_input(
            "WBC count (cells/µL)",
            min_value=0, max_value=50000,
            value=7000, step=100)
    with lab_col2:
        temp = st.number_input(
            "Axillary temperature (°C)",
            min_value=35.0, max_value=42.0,
            value=37.5, step=0.1)
        pulse = st.number_input(
            "Pulse rate (bpm)",
            min_value=40, max_value=200,
            value=80, step=1)
    with lab_col3:
        weight = st.number_input(
            "Weight (kg)",
            min_value=1, max_value=150,
            value=60, step=1)

    if st.button("Generate assessment", type="primary",
                 use_container_width=True):

        # Build row dari input manual
        row = {col: 0 for col in FEATURE_NAMES}

        # Isi symptom values
        for orig_col, val in symptom_vals.items():
            clean = clean_colname(orig_col)
            if clean in row:
                row[clean] = int(val)

        # Isi lab values
        lab_map = {
            'Numération_plaquettaire_Platelet_count'    : platelet,
            'Nombre_de_globules_blancs_cellulesML_Whi'  : wbc,
            'Température_axillaire_médiane_IQR_C_Axil'  : temp,
            'Fréquence_du_pouls_battementsm_in_SD_Pul'  : pulse,
            'Poids_Weight'                               : weight,
            'Âge_Age'                                    : age,
            'Genre_Gender'                               : 1 if gender == 'Male' else 0,
            'Centre_de_santé'                            : 0 if faskes == 'CMA de DO' else 1,
        }
        for k, v in lab_map.items():
            if k in row:
                row[k] = float(v)

        # FE scores manual
        neuro_feats = [
            'Convulsions_généralisées_ou_focales_Gen',
            'Prostration',
        ]
        row['FE_msk_score'] = sum([
            int(symptom_vals.get(
                'Douleur articulaire (Joint pain)', False)),
            int(symptom_vals.get(
                'Douleur musculaire ( Muscle pain)', False)),
        ])
        row['FE_gi_score'] = sum([
            int(symptom_vals.get('Vomissement (Vomiting)', False)),
            int(symptom_vals.get('Diarrhée  (Diarrhea)', False)),
            int(symptom_vals.get(
                'Douleur abdominale (stomac pain)', False)),
            int(symptom_vals.get('Nausée (Nausea)', False)),
        ])
        row['FE_resp_score'] = sum([
            int(symptom_vals.get('Toux (Cough)', False)),
            int(symptom_vals.get(
                'Détresse respiratoire (Respiratory distress)', False)),
        ])
        row['FE_fever_score'] = sum([
            int(symptom_vals.get(
                'Haute température.(temperature, Hyperpyrexia)', False)),
            int(symptom_vals.get(
                'Fièvre depuis 48 heures(Fever 48 hrs)', False)),
            int(symptom_vals.get(
                'Fièvre au cours des 7 derniers jours (Fever in the last 7 days)',
                False)),
        ])
        row['FE_dengue_pattern'] = (row['FE_msk_score'] *
                                    row['FE_gi_score'])
        row['FE_typhoid_pattern'] = (row['FE_gi_score'] *
                                     (4 - row['FE_msk_score']))
        row['FE_others_pattern']  = (row['FE_resp_score'] * 2 -
                                     row['FE_msk_score'])
        row['FE_is_child']        = int(age < 18)
        row['FE_is_young_adult']  = int(18 <= age < 35)
        row['FE_is_elderly']      = int(age >= 60)
        row['FE_persistent_fever'] = int(
            symptom_vals.get(
                'Fièvre depuis 48 heures(Fever 48 hrs)', False) and
            symptom_vals.get(
                'Fièvre au cours des 7 derniers jours (Fever in the last 7 days)',
                False))
        row['FE_has_msk']  = int(row['FE_msk_score']  >= 2)
        row['FE_has_gi']   = int(row['FE_gi_score']   >= 2)
        row['FE_has_resp'] = int(row['FE_resp_score']  >= 2)
        row['FE_has_neuro'] = int(sum([
            int(symptom_vals.get(
                'Convulsions généralisées ou focales (Generalised or focal convulsion)',
                False)),
            int(symptom_vals.get('Prostration', False)),
        ]) >= 2)

        X_manual = np.array(
            [[row.get(f, 0) for f in FEATURE_NAMES]],
            dtype=np.float32)

        st.markdown("---")
        st.markdown("### Assessment result")

        with st.spinner("Computing prediction..."):
            if use_bootstrap:
                with st.spinner(
                        f"Running bootstrap CI ({N_BOOT} iter)..."):
                    mean_p, std_p, lo_p, hi_p = \
                        predict_with_bootstrap(X_manual)
            else:
                probas = predict_single(X_manual)
                mean_p = probas
                std_p  = np.zeros_like(probas)
                lo_p   = probas
                hi_p   = probas

        render_report(
            mean_p, std_p, lo_p, hi_p,
            pat_idx=0, X_arr=X_manual,
            show_shap=True)

# ── TAB 2: UPLOAD CSV ─────────────────────────────────────
with tab2:
    st.markdown("### Upload patient CSV")
    st.caption(
        "Upload a CSV file with the same column format as the "
        "original dataset (French column names). "
        "Missing columns will be filled with 0.")

    uploaded = st.file_uploader(
        "Choose CSV file", type=['csv'])

    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            st.success(
                f"Loaded {len(df_upload)} patients, "
                f"{len(df_upload.columns)} columns")

            with st.expander("Preview data", expanded=False):
                st.dataframe(df_upload.head(5))

            if st.button("Run batch assessment",
                         type="primary",
                         use_container_width=True):

                with st.spinner("Preprocessing..."):
                    X_batch = preprocess_df(df_upload).values

                with st.spinner("Predicting..."):
                    probas_batch = predict_single(X_batch)
                    mean_p = probas_batch
                    std_p  = np.zeros_like(probas_batch)
                    lo_p   = probas_batch
                    hi_p   = probas_batch

                # Summary table
                st.markdown("### Batch results summary")
                results = pd.DataFrame(
                    (mean_p >= THRESHOLDS).astype(int),
                    columns=LABELS)
                results.insert(0, 'Patient', range(len(df_upload)))
                results['n_diseases'] = results[LABELS].sum(axis=1)
                results['max_prob_label'] = [
                    LABELS[np.argmax(mean_p[i])]
                    for i in range(len(mean_p))]
                results['max_prob'] = mean_p.max(axis=1).round(3)

                st.dataframe(results, use_container_width=True)

                # Summary stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total patients", len(df_upload))
                with col2:
                    n_co = (results['n_diseases'] >= 2).sum()
                    st.metric("Co-infections", n_co,
                              f"{n_co/len(df_upload)*100:.1f}%")
                with col3:
                    n_mal = results['Malaria'].sum()
                    st.metric("Malaria predicted", n_mal)
                with col4:
                    n_yf = results['Yellow Fever'].sum()
                    st.metric("Yellow Fever predicted", n_yf)

                # Per-patient detail
                st.markdown("### Per-patient detail")
                pat_sel = st.selectbox(
                    "Select patient for detailed report",
                    options=range(len(df_upload)),
                    format_func=lambda x: f"Patient #{x}")

                render_report(
                    mean_p, std_p, lo_p, hi_p,
                    pat_idx=pat_sel,
                    X_arr=X_batch[pat_sel:pat_sel+1],
                    show_shap=True)

                # Download results
                csv_out = results.to_csv(index=False)
                st.download_button(
                    "Download results CSV",
                    data=csv_out,
                    file_name="vbd_cdss_results.csv",
                    mime="text/csv",
                    use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.caption(
                "Pastikan format CSV sesuai dengan dataset asli "
                "(French column names).")
