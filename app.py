

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import shap

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
    boot_stats = joblib.load('artifacts/bootstrap_stats.pkl')
    return models, meta, feat_names, boot_stats

models, META, FEATURE_NAMES, BOOT_STATS = load_artifacts()

LABELS        = META['target_labels']
THRESHOLDS    = np.array(META['cv_thresholds'])
UNC_P75       = META['unc_p75']
POP_MEAN_STD  = BOOT_STATS['pop_mean_std']
POP_STD_LABEL = np.array(BOOT_STATS['pop_std_per_label'])

# ============================================================
# PREPROCESSING — identik dengan notebook
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
    df['Âge (Age)'] = pd.to_numeric(df['Âge (Age)'], errors='coerce')
    df['FE_is_child']       = (df['Âge (Age)'] < 18).astype(int)
    df['FE_is_young_adult'] = (
        (df['Âge (Age)'] >= 18) &
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
    fn = list(feature_names)
    muscle_col = [c for c in fn if 'Muscle' in c and 'pain' in c.lower()]
    cough_col  = [c for c in fn if 'Cough' in c or 'Toux' in c]
    conv_col   = [c for c in fn if 'Generalised' in c]
    mask = np.zeros(len(X_val), dtype=bool)
    if muscle_col: mask |= (X_val[:, fn.index(muscle_col[0])] == 0)
    if cough_col:  mask |= (X_val[:, fn.index(cough_col[0])]  == 1)
    if conv_col:   mask |= (X_val[:, fn.index(conv_col[0])]   == 1)
    probas[mask, 2] = 0.0
    return probas

def preprocess_df(df):
    """
    Full preprocessing pipeline — identik dengan notebook.
    Target cols di-drop SEBELUM FE dan fix_cols
    untuk menghindari potential leakage.
    """
    # Drop target cols dan leakage cols SEBELUM apapun
    drop_before = [c for c in
                   META['target_cols'] +
                   ['Dengue (Dengua)', 'n_diseases']
                   if c in df.columns]
    df = df.drop(columns=drop_before)

    # Feature engineering
    df = build_fe_batch1(df)

    # Clean column names
    df = fix_cols(df)

    # Align ke feature names dari training
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    return df[FEATURE_NAMES].fillna(0).astype(np.float32)

# ============================================================
# PREDICTION ENGINE
# ============================================================
def predict_dataset_patient(patient_idx):
    """
    Pasien dari dataset (0-299):
    pakai bootstrap stats exact dari notebook.
    Identik dengan output notebook — tidak ada recompute.
    """
    return {
        'mean'  : BOOT_STATS['mean'][patient_idx],
        'std'   : float(BOOT_STATS['std'][patient_idx].mean()),
        'lower' : BOOT_STATS['lower'][patient_idx],
        'upper' : BOOT_STATS['upper'][patient_idx],
        'source': 'bootstrap_notebook',
    }

def predict_new_patient(X_arr):
    """
    Pasien baru (manual input / upload CSV):
    - Point estimate dari model RF (identik dengan notebook)
    - YF rules diterapkan (identik dengan notebook)
    - CI = point_estimate ± 1.96 × pop_std_per_label
      (population-level proxy dari bootstrap notebook)
    - std = pop_mean_std sebagai proxy uncertainty

    Limitation: CI adalah population-level estimate,
    bukan individual bootstrap CI. Data training tidak
    di-ship ke app untuk menjaga privasi data klinis.
    """
    probas = np.column_stack([
        m.predict_proba(X_arr)[:, 1] for m in models
    ])
    probas = apply_yf_rules(X_arr, probas, FEATURE_NAMES)

    lower = np.clip(probas - 1.96 * POP_STD_LABEL, 0, 1)
    upper = np.clip(probas + 1.96 * POP_STD_LABEL, 0, 1)

    return {
        'mean'  : probas[0],
        'std'   : float(POP_MEAN_STD),
        'lower' : lower[0],
        'upper' : upper[0],
        'source': 'population_estimate',
    }

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

def render_report(pred, X_arr=None, show_shap=True,
                  true_dx=None):
    mean_p   = pred['mean']
    lo       = pred['lower']
    hi       = pred['upper']
    std      = pred['std']
    source   = pred['source']
    unc_flag = std > UNC_P75

    # CI source note untuk pasien baru
    if source == 'population_estimate':
        st.info(
            "ℹ️ Confidence interval shown is a population-level "
            "estimate (± 1.96 × population std from training "
            "bootstrap, N=50). Individual bootstrap CI is only "
            "available for dataset patients (Tab 3).")

    # Uncertainty banner
    if unc_flag:
        st.warning(
            f"⚠️ High uncertainty "
            f"(std={std:.4f} > P75={UNC_P75:.4f}) "
            f"— second opinion recommended")
    else:
        st.success(
            f"✓ Confidence within normal population range "
            f"(std={std:.4f})")

    # Disease probability rows
    st.markdown("**Disease probability assessment** "
                "| Threshold: median 5-fold CV")
    sorted_idx = np.argsort(mean_p)[::-1]

    for i in sorted_idx:
        label  = LABELS[i]
        prob   = float(mean_p[i])
        above  = prob >= THRESHOLDS[i]
        marker = "●" if above else "○"
        color  = COLORS[i]
        action = ACTIONS[label] if above else "Monitor"

        col1, col2, col3 = st.columns([2, 3, 3])
        with col1:
            st.markdown(
                f"<span style='color:{color};"
                f"font-weight:600'>{marker} {label}</span>",
                unsafe_allow_html=True)
        with col2:
            st.progress(prob)
            st.caption(
                f"{prob*100:.1f}% "
                f"[{float(lo[i])*100:.1f}–"
                f"{float(hi[i])*100:.1f}%]")
        with col3:
            if above:
                st.error(f"→ {action}")
            else:
                st.info("→ Monitor")

    st.caption(
        "● above threshold → action | ○ below → monitor | "
        + " | ".join([f"{l[:3]}={t:.2f}"
                      for l, t in zip(LABELS, THRESHOLDS)]))

    # SHAP
    if show_shap and X_arr is not None:
        above_labels = [i for i in range(5)
                        if float(mean_p[i]) >= THRESHOLDS[i]]
        if above_labels:
            st.markdown("---")
            st.markdown("**Top contributing factors** "
                        "(labels above threshold, via SHAP)")
            for i in above_labels[:3]:
                with st.expander(
                        f"{LABELS[i]} — "
                        f"{float(mean_p[i])*100:.1f}%",
                        expanded=True):
                    top3 = get_shap_top3(X_arr, i)
                    for feat, sv, fv in top3:
                        direction = ("↑ increases" if sv > 0
                                     else "↓ decreases")
                        color_sv  = ("red" if sv > 0
                                     else "blue")
                        st.markdown(
                            f"`{feat}` &nbsp; "
                            f"val=**{fv:.2f}** &nbsp; "
                            f"<span style='color:{color_sv}'>"
                            f"SHAP={sv:+.4f} "
                            f"{direction}</span>",
                            unsafe_allow_html=True)

    # Ground truth jika tersedia
    if true_dx is not None:
        st.markdown("---")
        st.caption(
            f"[Evaluation only — not available in deployment] "
            f"True diagnosis: "
            f"{', '.join(true_dx) if true_dx else 'None'}")

    st.markdown("---")
    st.caption(
        "⚕️ This output is for clinical decision support only. "
        "Final diagnosis must be confirmed by a licensed "
        "medical professional. Not a substitute for clinical "
        "examination or laboratory testing.")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚕️ VBD CDSS")
    st.caption(
        "Vector-Borne Disease\n"
        "Clinical Decision Support System")
    st.markdown("---")
    st.markdown("**Model**")
    st.caption("Random Forest (Binary Relevance)")
    st.caption("CV F1 Macro: 0.6272 ± 0.0376")
    st.caption("N=300 patients, Burkina Faso")
    st.caption("Seed: 456")
    st.markdown("---")
    st.markdown("**Thresholds (median 5-fold CV)**")
    for l, t in zip(LABELS, THRESHOLDS):
        st.caption(f"{l}: {t:.2f}")
    st.markdown("---")
    st.markdown("**Uncertainty**")
    st.caption(f"P75 threshold : {UNC_P75:.4f}")
    st.caption(f"Pop mean std  : {POP_MEAN_STD:.4f}")
    st.markdown("---")
    st.markdown("**Per-label pop std**")
    for l, s in zip(LABELS, POP_STD_LABEL):
        st.caption(f"{l}: {s:.4f}")

# ============================================================
# MAIN
# ============================================================
st.title("Vector-Borne Disease")
st.subheader("Clinical Decision Support System")
st.caption(
    "Multi-label prediction · "
    "Malaria · Dengue · Yellow Fever · Typhoid · Others")
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "📋 Manual input",
    "📁 Upload CSV",
    "🔍 Dataset patient lookup",
])

# ── TAB 1: MANUAL INPUT ───────────────────────────────────
with tab1:
    st.markdown("### Patient information")

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

    st.markdown("### Clinical symptoms")
    st.caption("Check all symptoms present")

    symptoms = {
        'Fièvre depuis 48 heures(Fever 48 hrs)':
            'Fever ≥ 48h',
        'Fièvre au cours des 7 derniers jours (Fever in the last 7 days)':
            'Fever last 7 days',
        'Haute température.(temperature, Hyperpyrexia)':
            'High temperature / hyperpyrexia',
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
            'Jaundice (Icterus)',
        'Saignement/ Manifestations hémorragiques (Bleeding)':
            'Bleeding / hemorrhagic signs',
        'Vertige (Dizzy)':
            'Dizziness',
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

    st.markdown("### Lab values (optional)")
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        platelet = st.number_input(
            "Platelet count (×10³/µL)",
            min_value=0, max_value=1000,
            value=150, step=1)
        wbc = st.number_input(
            "WBC count (cells/µL)",
            min_value=0, max_value=50000,
            value=7000, step=100)
    with lc2:
        temp_val = st.number_input(
            "Axillary temperature (°C)",
            min_value=35.0, max_value=42.0,
            value=37.5, step=0.1)
        pulse = st.number_input(
            "Pulse rate (bpm)",
            min_value=40, max_value=200,
            value=80, step=1)
    with lc3:
        weight = st.number_input(
            "Weight (kg)",
            min_value=1, max_value=150,
            value=60, step=1)

    if st.button("Generate assessment",
                 type="primary",
                 use_container_width=True):

        # Build feature vector
        row = {col: 0.0 for col in FEATURE_NAMES}

        # Symptom values
        for orig_col, val in sym_vals.items():
            clean = clean_colname(orig_col)
            if clean in row:
                row[clean] = float(int(val))

        # Lab values
        lab_map = {
            'Numération_plaquettaire_Platelet_count'   : float(platelet),
            'Nombre_de_globules_blancs_cellulesML_Whi' : float(wbc),
            'Température_axillaire_médiane_IQR_C_Axil' : float(temp_val),
            'Fréquence_du_pouls_battementsm_in_SD_Pul' : float(pulse),
            'Poids_Weight'                              : float(weight),
            'Âge_Age'                                   : float(age),
            'Genre_Gender'                              : 1.0 if gender == 'Male' else 0.0,
            'Centre_de_santé'                           : 0.0 if faskes == 'CMA de DO' else 1.0,
        }
        for k, v in lab_map.items():
            if k in row:
                row[k] = v

        # FE scores — identik dengan build_fe_batch1
        msk_score = float(sum([
            int(sym_vals.get('Douleur articulaire (Joint pain)', False)),
            int(sym_vals.get('Douleur musculaire ( Muscle pain)', False)),
        ]))
        gi_score = float(sum([
            int(sym_vals.get('Vomissement (Vomiting)', False)),
            int(sym_vals.get('Diarrhée  (Diarrhea)', False)),
            int(sym_vals.get('Douleur abdominale (stomac pain)', False)),
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

        row['FE_msk_score']       = msk_score
        row['FE_gi_score']        = gi_score
        row['FE_resp_score']      = resp_score
        row['FE_neuro_score']     = neuro_score
        row['FE_has_msk']         = float(msk_score  >= 2)
        row['FE_has_gi']          = float(gi_score   >= 2)
        row['FE_has_resp']        = float(resp_score  >= 2)
        row['FE_has_neuro']       = float(neuro_score >= 2)
        row['FE_dengue_pattern']  = msk_score * gi_score
        row['FE_typhoid_pattern'] = gi_score * (4 - msk_score)
        row['FE_others_pattern']  = resp_score * 2 - msk_score
        row['FE_is_child']        = float(age < 18)
        row['FE_is_young_adult']  = float(18 <= age < 35)
        row['FE_is_elderly']      = float(age >= 60)
        row['FE_persistent_fever'] = float(
            sym_vals.get(
                'Fièvre depuis 48 heures(Fever 48 hrs)',
                False) and
            sym_vals.get(
                'Fièvre au cours des 7 derniers jours (Fever in the last 7 days)',
                False))

        # fever_score untuk FE_fever_score
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
        row['FE_fever_score'] = fever_score

        X_manual = np.array(
            [[row.get(f, 0.0) for f in FEATURE_NAMES]],
            dtype=np.float32)

        st.markdown("---")
        st.markdown("### Assessment result")
        with st.spinner("Computing prediction..."):
            pred = predict_new_patient(X_manual)
        render_report(pred, X_arr=X_manual, show_shap=True)

# ── TAB 2: UPLOAD CSV ─────────────────────────────────────
with tab2:
    st.markdown("### Upload patient CSV")
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
                f"Loaded {len(df_up)} patients, "
                f"{len(df_up.columns)} columns")

            with st.expander("Preview data", expanded=False):
                st.dataframe(df_up.head(5))

            if st.button("Run batch assessment",
                         type="primary",
                         use_container_width=True):

                with st.spinner(
                        "Preprocessing & predicting..."):
                    X_batch  = preprocess_df(df_up).values
                    probas_b = np.column_stack([
                        m.predict_proba(X_batch)[:, 1]
                        for m in models
                    ])
                    probas_b = apply_yf_rules(
                        X_batch, probas_b, FEATURE_NAMES)

                # Summary table
                st.markdown("### Batch results summary")
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

                st.dataframe(results,
                             use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total patients",
                              len(df_up))
                with c2:
                    n_co = (results['n_diseases'] >= 2).sum()
                    st.metric("Co-infections", n_co,
                              f"{n_co/len(df_up)*100:.1f}%")
                with c3:
                    st.metric("Malaria predicted",
                              results['Malaria'].sum())
                with c4:
                    st.metric("Yellow Fever predicted",
                              results['Yellow Fever'].sum())

                # Per-patient detail
                st.markdown("### Per-patient detail")
                pat_sel = st.selectbox(
                    "Select patient for detailed report",
                    range(len(df_up)),
                    format_func=lambda x: f"Patient #{x}")

                pred_sel = predict_new_patient(
                    X_batch[pat_sel:pat_sel+1])
                render_report(
                    pred_sel,
                    X_arr=X_batch[pat_sel:pat_sel+1],
                    show_shap=True)

                # Download
                st.download_button(
                    "Download results CSV",
                    data=results.to_csv(index=False),
                    file_name="vbd_cdss_results.csv",
                    mime="text/csv",
                    use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.caption(
                "Ensure CSV format matches the original "
                "dataset (French column names).")

# ── TAB 3: DATASET PATIENT LOOKUP ─────────────────────────
with tab3:
    st.markdown("### Dataset patient lookup")
    st.caption(
        "Look up a patient from the original 300-patient "
        "dataset. Uses exact bootstrap CI computed in "
        "notebook (N=50 iterations) — identical to "
        "notebook output.")

    pat_idx = st.number_input(
        "Patient index (0–299)",
        min_value=0, max_value=299,
        value=16, step=1)

    if st.button("Lookup patient",
                 type="primary",
                 use_container_width=True):

        pred = predict_dataset_patient(int(pat_idx))
        st.markdown(f"### Report — Patient #{pat_idx}")
        st.caption(
            "Bootstrap CI: exact from notebook (N=50). "
            "SHAP not shown — use Manual Input tab for "
            "SHAP explanations on a specific patient.")
        render_report(
            pred,
            X_arr=None,
            show_shap=False,
            true_dx=None)
