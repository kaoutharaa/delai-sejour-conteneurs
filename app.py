# -*- coding: utf-8 -*-
"""Application Streamlit - Prediction du delai de sejour des conteneurs (Port de Casablanca)."""
import datetime as dt

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Délai de Séjour — Port de Casablanca', layout='wide',
                    initial_sidebar_state='collapsed')

# ============================================================= THEME =============================================================
LIGHT = dict(
    INK='#1C2B3A', MUTED='#5C6B7A', BG='#F5F8FC', SURFACE='#FFFFFF', BORDER='#DCE3EC',
    ACCENT='#1F5C9E', ACCENT_DARK='#153F6E', ACCENT_SOFT='#E8F1FA',
    WARN='#8A5A12', WARN_SOFT='#FBF0DD', BAD='#A5352B', BAD_SOFT='#FBEAE8', OK='#2E6B4F', OK_SOFT='#E7F2EC',
    PLOT_COLORS=['#1F5C9E', '#5C6B7A', '#4A90C4', '#153F6E', '#8AAFD1', '#2E6B4F'],
)
DARK = dict(
    INK='#E6ECF3', MUTED='#93A3B5', BG='#0F1620', SURFACE='#17202C', BORDER='#2A3A4C',
    ACCENT='#5B9BD8', ACCENT_DARK='#8FC1EC', ACCENT_SOFT='#1B2E42',
    WARN='#D9A441', WARN_SOFT='#2E2410', BAD='#E48A7E', BAD_SOFT='#2E1917', OK='#69C79A', OK_SOFT='#142A20',
    PLOT_COLORS=['#5B9BD8', '#93A3B5', '#8FC1EC', '#3A6E9E', '#4A6B8A', '#69C79A'],
)

# le bouton bascule la session_state AVANT qu'on choisisse la palette, donc l'etat est deja
# a jour des la premiere ligne du script (Streamlit restaure session_state avant de rejouer le script)
IS_DARK = st.session_state.get('dark_mode', False)
_T = DARK if IS_DARK else LIGHT
INK, MUTED, BG, SURFACE, BORDER = _T['INK'], _T['MUTED'], _T['BG'], _T['SURFACE'], _T['BORDER']
ACCENT, ACCENT_DARK, ACCENT_SOFT = _T['ACCENT'], _T['ACCENT_DARK'], _T['ACCENT_SOFT']
WARN, WARN_SOFT, BAD, BAD_SOFT, OK, OK_SOFT = _T['WARN'], _T['WARN_SOFT'], _T['BAD'], _T['BAD_SOFT'], _T['OK'], _T['OK_SOFT']
PLOT_COLORS = _T['PLOT_COLORS']

_CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  #MainMenu {{visibility: hidden;}}
  footer {{visibility: hidden;}}
  header[data-testid="stHeader"] {{background: transparent;}}
  html, body, [class*="css"] {{
      font-family: 'IBM Plex Sans', system-ui, sans-serif;
  }}
  .stApp {{
      background: {BG};
      color: {INK};
  }}
  .block-container {{
      padding-top: 1.6rem;
      max-width: 1180px;
  }}
  .mono {{ font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; }}
  /* header band */
  .app-header {{
      display:flex; align-items:center; gap:14px;
      padding-bottom: 18px; margin-bottom: 6px;
      border-bottom: 1px solid {BORDER};
  }}
  .app-badge {{
      width:46px; height:46px; border-radius:9px;
      background: {ACCENT}; color:white; display:flex; align-items:center; justify-content:center;
      font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:15px; letter-spacing:.02em; flex:none;
  }}
  .app-title {{ font-size:1.55rem; font-weight:700; margin:0; line-height:1.15; color:{INK}; }}
  .app-sub {{ font-size:.92rem; color:{MUTED}; margin:2px 0 0; }}
  .theme-toggle {{ padding-top:10px; }}
  .theme-toggle label p {{ font-size:.78rem !important; color:{MUTED} !important; }}
  /* cards */
  .card {{
      background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px;
      padding:16px 18px; box-shadow: 0 1px 2px rgba(21,33,40,.05), 0 6px 16px -8px rgba(21,33,40,.10);
      height:100%;
  }}
  .card h4 {{ margin:0 0 6px; font-size:.95rem; font-weight:700; color:{INK}; }}
  .card p {{ margin:0; font-size:.85rem; color:{MUTED}; line-height:1.5; }}
  .stat-label {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.06em; color:{MUTED}; font-weight:600; }}
  .stat-value {{ font-family:'IBM Plex Mono',monospace; font-size:1.5rem; font-weight:600; color:{INK}; margin-top:4px; }}
  .stat-sub {{ font-size:.74rem; color:{MUTED}; font-family:'IBM Plex Mono',monospace; margin-top:2px; }}
  .pred-result {{
      background: linear-gradient(135deg, {ACCENT_SOFT} 0%, {SURFACE} 70%);
      border:1px solid {ACCENT}; border-radius:16px; padding:28px 30px;
      text-align:center;
  }}
  .pred-value {{ font-family:'IBM Plex Mono',monospace; font-size:3.2rem; font-weight:700; color:{ACCENT_DARK}; line-height:1; }}
  .pred-caption {{ font-size:.85rem; color:{MUTED}; margin-top:6px; }}
  .callout {{ border-radius:12px; padding:14px 18px; font-size:.87rem; line-height:1.55; }}
  .callout-warn {{ background:{WARN_SOFT}; border:1px solid {WARN}; color:{INK}; }}
  .callout-ok {{ background:{OK_SOFT}; border:1px solid {OK}; color:{INK}; }}
  /* tabs */
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {BORDER}; }}
  .stTabs [data-baseweb="tab"] {{
      height:42px; background:transparent; border-radius:8px 8px 0 0; color:{MUTED};
      font-weight:600; font-size:.92rem; padding:0 16px;
  }}
  .stTabs [aria-selected="true"] {{ color:{ACCENT_DARK} !important; border-bottom:2.5px solid {ACCENT} !important; }}
  /* buttons */
  .stButton>button {{
      background:{ACCENT}; color:white; border:none; border-radius:9px; font-weight:600;
      padding:.55rem 1.3rem; font-size:.92rem;
  }}
  .stButton>button:hover {{ background:{ACCENT_DARK}; color:white; }}
  section[data-testid="stSidebar"] {{ background:{SURFACE}; border-right:1px solid {BORDER}; }}
  hr {{ border-color: {BORDER}; }}
</style>
"""
# Streamlit's markdown renderer treats a raw HTML block as ended by the first blank line
# (CommonMark rule) -- any blank line inside <style> makes the rest render as literal text.
_CSS = '\n'.join(line for line in _CSS.split('\n') if line.strip() != '')
st.markdown(_CSS, unsafe_allow_html=True)


# ============================================================= DATA =============================================================
@st.cache_resource
def load_bundle():
    return joblib.load('model_bundle.joblib')


bundle = load_bundle()
MODEL = bundle['model']
FULL_FEATURES = bundle['full_features']
PROFILE_COLS = bundle['profile_cols']
RECENCY_COLS = bundle['recency_cols']
DIM_SNAPSHOT = bundle['dim_snapshot']
CATEGORY_LISTS = bundle['category_lists']
EXPLO = bundle['explo']
GLOBAL_MEAN = bundle['global_mean']
RECENCY_MEDIAN = bundle['recency_median']
DOC_METRICS = bundle['documented_metrics']

CAT_MAP = {
    'ID_TERMINAL': 'terminal', 'ID_IMPORTATEUR': 'importateur', 'ID_MARCHANDISE': 'marchandise',
    'ID_PORT_CHARGEMENT': 'port_chargement', 'ID_PAYS_CHARGEMENT': 'pays_chargement',
}

FIXED_HOLIDAYS_MD = {(1, 1), (1, 11), (5, 1), (7, 30), (8, 14), (8, 20), (8, 21), (11, 6), (11, 18)}
RELIGIOUS_HOLIDAYS = {
    2020: [(5, 24), (5, 25), (7, 31), (8, 1), (8, 20), (10, 29), (10, 30)],
    2021: [(5, 13), (5, 14), (5, 15), (7, 20), (7, 21), (8, 10), (10, 19), (10, 20)],
    2022: [(5, 2), (5, 3), (5, 4), (7, 9), (7, 10), (7, 11), (7, 31), (10, 9), (10, 10)],
    2023: [(4, 22), (4, 23), (4, 24), (6, 29), (6, 30), (7, 19), (9, 28), (9, 29)],
    2024: [(4, 10), (4, 11), (6, 16), (6, 17), (7, 7), (9, 15), (9, 16)],
    2025: [(3, 30), (3, 31), (6, 6), (6, 7), (6, 26), (9, 4), (9, 5)],
    2026: [(3, 20), (3, 21), (5, 27), (5, 28), (6, 16), (8, 24), (8, 25)],
}
RAMADAN_PERIODS = {
    2020: ('2020-04-25', '2020-05-23'), 2021: ('2021-04-13', '2021-05-12'),
    2022: ('2022-04-03', '2022-05-01'), 2023: ('2023-03-23', '2023-04-21'),
    2024: ('2024-03-11', '2024-04-09'), 2025: ('2025-03-01', '2025-03-30'),
    2026: ('2026-02-18', '2026-03-19'),
}


def is_ferie(date):
    return int((date.month, date.day) in FIXED_HOLIDAYS_MD or (date.month, date.day) in RELIGIOUS_HOLIDAYS.get(date.year, []))


def is_ramadan(date):
    if date.year not in RAMADAN_PERIODS:
        return 0
    start, end = (pd.Timestamp(x) for x in RAMADAN_PERIODS[date.year])
    return int(start <= pd.Timestamp(date) <= end)


def category_options(col):
    return CATEGORY_LISTS[col]


def get_profile(col, selected_id):
    prefix = CAT_MAP[col]
    snap = DIM_SNAPSHOT[col]
    if selected_id is None or selected_id not in snap:
        return {
            f'{prefix}_count': 0.0, f'{prefix}_mean_w': GLOBAL_MEAN, f'{prefix}_std_w': 0.0,
            f'{prefix}_mean_all': GLOBAL_MEAN, f'{prefix}_trend': 0.0, f'{prefix}_high_delay_rate': 0.25,
        }, None
    row = snap[selected_id]
    return {
        f'{prefix}_count': row['count'], f'{prefix}_mean_w': row['mean_w'], f'{prefix}_std_w': row['std_w'],
        f'{prefix}_mean_all': row['mean_all'], f'{prefix}_trend': row['trend'],
        f'{prefix}_high_delay_rate': row['high_delay_rate'],
    }, row['last_date']


def build_feature_vector(terminal, importateur, marchandise, port, pays, date, delai_pointage, lmd_lms):
    feats = {}
    feats['ID_TERMINAL'] = terminal
    feats['ANNEE'] = date.year
    feats['month'] = date.month
    feats['weekday'] = date.weekday()
    feats['quarter'] = (date.month - 1) // 3 + 1
    feats['DELAI_POINTAGE_BAD'] = delai_pointage
    feats['LMD_LMS'] = int(lmd_lms)
    feats['abs_pointage'] = abs(delai_pointage)
    feats['pointage_positif'] = int(delai_pointage > 0)
    feats['month_sin'] = np.sin(2 * np.pi * feats['month'] / 12)
    feats['month_cos'] = np.cos(2 * np.pi * feats['month'] / 12)
    feats['weekday_sin'] = np.sin(2 * np.pi * feats['weekday'] / 7)
    feats['weekday_cos'] = np.cos(2 * np.pi * feats['weekday'] / 7)
    feats['est_ferie'] = is_ferie(date)
    feats['est_ramadan'] = is_ramadan(date)

    selections = {'ID_TERMINAL': terminal, 'ID_IMPORTATEUR': importateur, 'ID_MARCHANDISE': marchandise,
                  'ID_PORT_CHARGEMENT': port, 'ID_PAYS_CHARGEMENT': pays}
    unknown_cats = []
    for col, sel_id in selections.items():
        prof, last_date = get_profile(col, sel_id)
        feats.update(prof)
        prefix = CAT_MAP[col]
        dsl_col = f'{prefix}_days_since_last'
        if last_date is not None:
            feats[dsl_col] = max(0, (pd.Timestamp(date) - pd.Timestamp(last_date)).days)
        else:
            feats[dsl_col] = RECENCY_MEDIAN[dsl_col]
            unknown_cats.append(col)

    row = pd.DataFrame([feats])[FULL_FEATURES]
    return row, unknown_cats


def plotly_layout(fig, height=340, title=None):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=42 if title else 10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans', color=INK, size=12),
        title=dict(text=title, font=dict(size=13, color=INK)) if title else None,
        xaxis=dict(gridcolor=BORDER, zeroline=False), yaxis=dict(gridcolor=BORDER, zeroline=False),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
    )
    return fig


# ============================================================= HEADER =============================================================
head_col, toggle_col = st.columns([7, 1])
with head_col:
    st.markdown(f"""
    <div class="app-header">
      <div class="app-badge">PS</div>
      <div>
        <p class="app-title">Délai de séjour des conteneurs</p>
        <p class="app-sub">Port de Casablanca — outil de prédiction et tableau de bord du projet</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
with toggle_col:
    st.markdown('<div class="theme-toggle">', unsafe_allow_html=True)
    st.toggle('Mode sombre', value=IS_DARK, key='dark_mode')
    st.markdown('</div>', unsafe_allow_html=True)

tab_pred, tab_apercu, tab_explo, tab_limites = st.tabs(['Prédiction', 'Aperçu du projet', 'Exploration', 'Limites du modèle'])

# ============================================================= TAB PREDICTION =============================================================
with tab_pred:
    left, right = st.columns([1.1, 1], gap='large')

    with left:
        st.markdown("##### Caractéristiques du connaissement (BL)")

        c1, c2 = st.columns(2)
        with c1:
            terminal = st.selectbox('Terminal', options=[9, 10, 58], format_func=lambda x: f'Terminal {x}')
            date_val = st.date_input('Date de validation', value=dt.date(2024, 6, 15),
                                      min_value=dt.date(2020, 1, 1), max_value=dt.date(2026, 12, 31))
        with c2:
            delai_pointage = st.slider('Délai de pointage (jours)', -10.0, 10.0, 1.0, 0.1)
            lmd_lms = st.checkbox('Marchandise LMD/LMS (dangereuse / spécifique)', value=False)

        def cat_selector(label, col):
            opts = category_options(col)
            labels = ['— Nouveau / inconnu —'] + [
                f"#{o[col]} — {int(o['volume'])} envois, {o['delai_moyen']:.1f} j en moyenne" for o in opts
            ]
            ids = [None] + [o[col] for o in opts]
            idx = st.selectbox(label, options=list(range(len(labels))), format_func=lambda i: labels[i], key=col)
            return ids[idx]

        importateur = cat_selector('Importateur', 'ID_IMPORTATEUR')
        marchandise = cat_selector('Marchandise', 'ID_MARCHANDISE')
        port = cat_selector('Port de chargement', 'ID_PORT_CHARGEMENT')
        pays = cat_selector('Pays de chargement', 'ID_PAYS_CHARGEMENT')

        predict_clicked = st.button('Calculer la prédiction', width='stretch')

    with right:
        st.markdown("##### Résultat")
        if predict_clicked:
            row, unknown = build_feature_vector(terminal, importateur, marchandise, port, pays,
                                                 date_val, delai_pointage, lmd_lms)
            pred = float(MODEL.predict(row)[0])
            pred = max(0.0, pred)
            delta = pred - EXPLO['target_mean']
            st.markdown(f"""
            <div class="pred-result">
              <div class="stat-label">Délai de séjour prédit</div>
              <div class="pred-value">{pred:.1f} <span style="font-size:1.3rem;">jours</span></div>
              <div class="pred-caption">{'+' if delta >= 0 else ''}{delta:.1f} j par rapport à la moyenne du port ({EXPLO['target_mean']:.1f} j)</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="callout callout-warn" style="margin-top:14px;">
            <b>Marge d'erreur à garder à l'esprit :</b> ce modèle atteint un MAE d'environ {DOC_METRICS['stack_test_mae']} jours en validation temporelle honnête —
            considérez cette prédiction comme un ordre de grandeur (± {DOC_METRICS['stack_test_mae']:.0f}-{DOC_METRICS['stack_test_mae']+1:.0f} jours), pas une valeur exacte.
            Voir l'onglet <i>Limites & diagnostic</i>.
            </div>
            """, unsafe_allow_html=True)
            if unknown:
                noms = ', '.join(CAT_MAP[c] for c in unknown)
                st.markdown(f"""
                <div class="callout callout-ok" style="margin-top:10px;">
                Catégorie(s) inconnue(s) dans l'historique : <b>{noms}</b> — la prédiction utilise un profil moyen de repli pour ces catégories.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card">
            <h4>Comment ça marche</h4>
            <p>Renseignez les caractéristiques d'un connaissement (BL) à gauche, puis cliquez sur « Calculer la prédiction ».
            Le modèle combine les caractéristiques saisies avec l'historique récent connu de l'importateur, de la marchandise,
            du port et du pays sélectionnés (moyennes glissantes causales, tendance, taux de retard passé) pour estimer
            le nombre de jours que le conteneur passera au port.</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================= TAB APERCU =============================================================
with tab_apercu:
    st.markdown("##### Résultat retenu")
    m1, m2, m3, m4 = st.columns(4)
    for col, label, value, sub in [
        (m1, 'Modèle déployé (CatBoost tuné)', f"R² {DOC_METRICS['test_r2']:.3f}", f"MAE {DOC_METRICS['test_mae']:.2f} j"),
        (m2, 'Meilleur résultat (stacking, notebook complémentaire)', f"R² {DOC_METRICS['stack_test_r2']:.3f}", f"MAE {DOC_METRICS['stack_test_mae']:.2f} j"),
        (m3, 'Pipeline principal (4 modèles + embeddings)', f"R² {DOC_METRICS['main_notebook_test_r2']:.3f}", 'notebook.ipynb'),
        (m4, 'Plafond absolu (test de fuite oracle)', f"R² {DOC_METRICS['oracle_test_r2']:.3f}", f"MAE {DOC_METRICS['oracle_test_mae']:.2f} j"),
    ]:
        with col:
            st.markdown(f"""<div class="card"><div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div><div class="stat-sub">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cA, cB = st.columns([1.3, 1], gap='large')
    with cA:
        stages = ['Baseline', '+ tables\nde dimension', '+ calendrier\n/ récence', 'LightGBM\ntuné', 'CatBoost\ntuné', 'Stacking\n(retenu)']
        values = [0.132, 0.287, 0.291, 0.306, 0.312, 0.315]
        colors = [PLOT_COLORS[0]] * 5 + [ACCENT]
        fig = go.Figure(go.Bar(x=stages, y=values, marker_color=colors, text=[f'{v:.3f}' for v in values], textposition='outside'))
        fig = plotly_layout(fig, height=360, title='Progression du R² (test 2024) — approche tables de dimension')
        fig.update_yaxes(range=[0, 0.38])
        st.plotly_chart(fig, width='stretch')
    with cB:
        st.markdown("""
        <div class="card">
        <h4>Démarche</h4>
        <p>Deux notebooks indépendants, deux méthodologies de feature engineering différentes, convergent vers le même
        plateau de performance (R² ≈ 0,31-0,32) :</p>
        <p style="margin-top:8px;"><b>notebook.ipynb</b> — pipeline incrémental : calendrier, moyennes glissantes causales,
        target encoding, tuning, deep learning (MLP, embeddings de catégories, Transformer), stacking final à 4 modèles.</p>
        <p style="margin-top:8px;"><b>notebook_tables_dimension.ipynb</b> — schéma en étoile : une table de dimension par
        colonne catégorielle (6 statistiques causales chacune), tuning et stacking. C'est ce modèle qui est déployé ici.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================= TAB EXPLORATION =============================================================
with tab_explo:
    e1, e2, e3 = st.columns(3)
    e1.markdown(f"""<div class="card"><div class="stat-label">Connaissements (BL)</div>
    <div class="stat-value">{EXPLO['n_bl']:,}</div></div>""".replace(',', ' '), unsafe_allow_html=True)
    e2.markdown(f"""<div class="card"><div class="stat-label">Lots</div>
    <div class="stat-value">{EXPLO['n_lots']:,}</div></div>""".replace(',', ' '), unsafe_allow_html=True)
    e3.markdown(f"""<div class="card"><div class="stat-label">Délai moyen</div>
    <div class="stat-value">{EXPLO['target_mean']:.2f} j</div><div class="stat-sub">écart-type {EXPLO['target_std']:.2f} j</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap='large')
    with c1:
        bins = EXPLO['delai_sejour_bins']
        centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        fig = go.Figure(go.Bar(x=centers, y=EXPLO['delai_sejour_hist'], marker_color=ACCENT))
        fig = plotly_layout(fig, title='Distribution du délai de séjour (0-45 j)')
        fig.update_xaxes(title='jours')
        st.plotly_chart(fig, width='stretch')
    with c2:
        term = EXPLO['terminal_volume']
        fig = go.Figure(go.Bar(x=[f'Terminal {k}' for k in term.keys()], y=list(term.values()),
                                marker_color=PLOT_COLORS[1]))
        fig = plotly_layout(fig, title='Volume de BL par terminal')
        st.plotly_chart(fig, width='stretch')

    c3, c4 = st.columns(2, gap='large')
    with c3:
        ma = EXPLO['month_avg']
        fig = go.Figure(go.Scatter(x=list(ma.keys()), y=list(ma.values()), mode='lines+markers',
                                    line=dict(color=ACCENT, width=2.5), marker=dict(size=7)))
        fig = plotly_layout(fig, title='Délai moyen par mois')
        fig.update_xaxes(title='mois', dtick=1)
        st.plotly_chart(fig, width='stretch')
    with c4:
        wa = EXPLO['weekday_avg']
        labels = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
        fig = go.Figure(go.Bar(x=[labels[int(k)] for k in wa.keys()], y=list(wa.values()), marker_color=PLOT_COLORS[2]))
        fig = plotly_layout(fig, title='Délai moyen par jour de semaine')
        st.plotly_chart(fig, width='stretch')

# ============================================================= TAB LIMITES =============================================================
with tab_limites:
    st.markdown(f"""
    <div class="callout callout-warn">
    <b>Pourquoi la prédiction ci-dessus reste un ordre de grandeur, pas une valeur exacte :</b>
    ce projet a documenté un plafond robuste de R² ≈ 0,31-0,32, confirmé par plus de 8 approches de modélisation
    indépendantes (arbres de décision, réseaux de neurones, embeddings de catégories, Transformer, tables de dimension) —
    et par un test de fuite de données volontaire qui en fait un plafond absolu.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap='large')
    with c1:
        st.markdown("""
        <div class="card">
        <h4>Test de fuite volontaire (oracle)</h4>
        <p>Les statistiques par catégorie ont été recalculées volontairement sur train+validation+test réunis
        (une fuite de données totale, simulant un modèle qui connaîtrait déjà la vraie moyenne de chaque catégorie,
        y compris sur le test). Résultat : R² = 0,320, MAE = 3,07 jours seulement.</p>
        <p style="margin-top:8px;">Même avec une triche complète, l'identité catégorielle (importateur, terminal,
        marchandise, port, pays) ne peut pas expliquer plus de ~32 % de la variance du délai de séjour. Le reste
        n'est pas contenu dans ces colonnes, fuite ou pas.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
        <h4>Ce qui manque pour aller plus loin</h4>
        <p>Les véritables déterminants du délai ne sont pas dans les données disponibles :</p>
        <p>• Statut de paiement des droits de douane<br>
        • Disponibilité du transitaire<br>
        • Planning des navires suivants<br>
        • Congestion ponctuelle du terminal le jour J</p>
        </div>
        """, unsafe_allow_html=True)


st.markdown(f"""
<div style="margin-top:28px; padding-top:14px; border-top:1px solid {BORDER}; display:flex; justify-content:space-between;
     font-size:.76rem; color:{MUTED}; font-family:'IBM Plex Mono',monospace;">
  <span>notebook.ipynb · notebook_tables_dimension.ipynb · README.md</span>
  <span>Modèle : CatBoost tuné, entraîné sur l'ensemble des données filtrées</span>
</div>
""", unsafe_allow_html=True)
