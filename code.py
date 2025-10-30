import os
import unicodedata
import collections
import pandas as pd
import geopandas as gpd
import plotly.express as px
import streamlit as st

# ==========================
# Config Streamlit
# ==========================
st.set_page_config(page_title="Burkina Faso — Indices climatiques (2024/2025)", layout="wide")
# ==========================
# Fichiers attendus
# ==========================
GEOJSON_PATH = "gadm41_BFA_3.json"
YEAR_FILES = {
    2024: "data 2024 for monitoring.xlsx",
    2025: "data 2025 for monitoring.xlsx",
}

# ==========================
# Aides
# ==========================
@st.cache_data
def _norm(s: str) -> str:
    if s is None:
        return s
    return (
        str(s)
        .strip()
        .replace("\u00A0", " ")
        .replace("  ", " ")
        .lower()
    )

EXPECTED_COLS = {
    "NAME_3": ["name_3", "name3", "name 3", "nom_departement", "departement"],
    "Moderate Drought Index Triggered": [
        "moderate drought index triggered", "moderate drought index trigerred",
        "secheresse moderee", "secheresse modérée (1/0)", "drought_moderee"
    ],
    "Severe Drought Index Triggered": [
        "severe drought index triggered", "severe drought index trigerred",
        "secheresse severe", "secheresse sévère (1/0)", "drought_severe"
    ],
    "Moderate Flood Index Trigerred": [
        "moderate flood index trigerred", "moderate flood index triggered",
        "inondation moderee", "inondation modérée (1/0)", "flood_moderee"
    ],
    "Severe Flood Index Trigerred": [
        "severe flood index trigerred", "severe flood index triggered",
        "inondation severe", "inondation sévère (1/0)", "flood_severe"
    ],
    "Exposure at Risk": [
        "exposure at risk", "exposition", "montant exposé", "exposure"
    ],
    "Expected Losses": [
        "expected losses", "pertes attendues", "losses"
    ],
}

EVENT_COLS = [
    "Moderate Drought Index Triggered",
    "Severe Drought Index Triggered",
    "Moderate Flood Index Trigerred",
    "Severe Flood Index Trigerred",
]

# ==========================
# COULEURS — EXACTEMENT TES TEINTES VALIDÉES
# ==========================
COLOR_MAP = {
    "Sécheresse modérée": "rgba(255,55,86,1)",   # rouge clair saturé
    "Sécheresse sévère": "rgba(207,0,3,1)",      # rouge foncé profond
    "Inondation modérée": "rgba(12,210,250,1)",  # bleu clair vif
    "Inondation sévère": "rgba(18,54,196,1)",    # bleu foncé intense
    "Aucun événement": "rgba(0,0,0,0)",          # transparent
}

CATEGORY_ORDER = [
    "Inondation sévère",
    "Inondation modérée",
    "Sécheresse sévère",
    "Sécheresse modérée",
    "Aucun événement",
]

PRIORITY = [
    ("Severe Flood Index Trigerred", "Inondation sévère"),
    ("Moderate Flood Index Trigerred", "Inondation modérée"),
    ("Severe Drought Index Triggered", "Sécheresse sévère"),
    ("Moderate Drought Index Triggered", "Sécheresse modérée"),
]


CANONICALS = {
    "secheresse moderee": "Sécheresse modérée",
    "secheresse modéré": "Sécheresse modérée",
    "secheresse moderee (1/0)": "Sécheresse modérée",
    "secheresse severe":  "Sécheresse sévère",
    "secheresse sévère":  "Sécheresse sévère",
    "inondation moderee": "Inondation modérée",
    "inondation modérée": "Inondation modérée",
    "inondation severe":  "Inondation sévère",
    "inondation sévère":  "Inondation sévère",
    "aucun evenement":    "Aucun événement",
    "aucun événement":    "Aucun événement",
}
def _canonize_label(x: str) -> str:
    if x is None:
        return "Aucun événement"
    s = str(x).replace("\u00A0", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    s_key = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
    return CANONICALS.get(s_key, s)

def _coerce_indicator(series: pd.Series) -> pd.Series:
    mapping = {
        "1": 1, 1: 1, 1.0: 1, True: 1, "true": 1, "oui": 1, "y": 1, "yes": 1,
        "0": 0, 0: 0, 0.0: 0, False: 0, "false": 0, "non": 0, "n": 0, "no": 0
    }
    if series.dtype == object:
        out = series.astype(str).str.lower().map(mapping)
        mask_missing = out.isna()
        if mask_missing.any():
            out2 = pd.to_numeric(series, errors="coerce").fillna(0).round().clip(0,1).astype(int)
            out = out.fillna(out2)
    else:
        out = pd.to_numeric(series, errors="coerce").fillna(0).round().clip(0,1).astype(int)
    return out

def _coerce_money(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        cleaned = (series.astype(str)
                   .str.replace("\u00A0", " ")
                   .str.replace(" ", "")
                   .str.replace(",", ".")
        )
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    norm_to_orig = {_norm(c): c for c in df.columns}
    for canonical, aliases in EXPECTED_COLS.items():
        if canonical == "NAME_3":
            if "name_3" in norm_to_orig:
                col_map[norm_to_orig["name_3"]] = "NAME_3"
            else:
                for alias in aliases:
                    if alias in norm_to_orig:
                        col_map[norm_to_orig[alias]] = "NAME_3"
                        break
        else:
            if _norm(canonical) in norm_to_orig:
                col_map[norm_to_orig[_norm(canonical)]] = canonical
            else:
                for alias in aliases:
                    if alias in norm_to_orig:
                        col_map[norm_to_orig[alias]] = canonical
                        break
    df = df.rename(columns=col_map)
    keep = [c for c in EXPECTED_COLS.keys() if c in df.columns]
    df = df[keep].copy()

    if "NAME_3" in df.columns:
        df["NAME_3"] = df["NAME_3"].astype(str).str.strip()

    for c in EVENT_COLS:
        if c in df.columns:
            df[c] = _coerce_indicator(df[c])
    if "Exposure at Risk" in df.columns:
        df["Exposure at Risk"] = _coerce_money(df["Exposure at Risk"])
    if "Expected Losses" in df.columns:
        df["Expected Losses"] = _coerce_money(df["Expected Losses"])
    return df

def _choose_event(row: pd.Series) -> str:
    for col, label in PRIORITY:
        if col in row and pd.notna(row[col]) and int(row[col]) == 1:
            return label
    return "Aucun événement"

@st.cache_data
def load_geojson(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    # Correction demandée
    try:
        if len(gdf) > 195:
            gdf.loc[195, "NAME_3"] = "Fada N'gourma"
    except Exception:
        pass
    if "NAME_3" not in gdf.columns:
        for c in ["name_3", "Name_3", "NAME3", "NAME_3"]:
            if c in gdf.columns:
                gdf = gdf.rename(columns={c: "NAME_3"})
                break
    gdf["NAME_3"] = gdf["NAME_3"].astype(str).str.strip()
    return gdf

@st.cache_resource
def load_excel(path: str) -> pd.ExcelFile:
    return pd.ExcelFile(path)

@st.cache_data
def parse_sheet(_excel_file: pd.ExcelFile, sheet_name: str, excel_key: str) -> pd.DataFrame:
    df = _excel_file.parse(sheet_name=sheet_name, dtype=object)
    df.columns = df.columns.astype(str).str.strip()
    df = _standardize_columns(df)
    for c in EVENT_COLS:
        if c not in df.columns:
            df[c] = 0
    if "Exposure at Risk" not in df.columns:
        df["Exposure at Risk"] = pd.NA
    if "Expected Losses" not in df.columns:
        df["Expected Losses"] = pd.NA
    return df

# ==========================
# Chargement données
# ==========================
with st.spinner("📂 Chargement des données..."):
    try:
        gdf = load_geojson(GEOJSON_PATH)
    except Exception as e:
        st.error(f"Erreur de lecture du GeoJSON '{GEOJSON_PATH}': {e}")
        st.stop()

available_years = [year for year, path in YEAR_FILES.items() if os.path.exists(path)]
if not available_years:
    st.error("Aucun fichier Excel trouvé. Placez au moins l'un des fichiers suivants : "
             + ", ".join(YEAR_FILES.values()))
    st.stop()
available_years = sorted(available_years)

# ==========================
# UI — Sélection Année & Mois
# ==========================
st.sidebar.header("🗓️ Sélection des données")
selected_year = st.sidebar.selectbox(
    "Choisir l'année",
    options=available_years,
    index=available_years.index(2025) if 2025 in available_years else 0
)
# ✅ Titre dynamique
st.title(f"🌍 Burkina Faso — Évènements climatiques {selected_year}")

excel_path = YEAR_FILES[selected_year]
with st.spinner(f"Ouverture du fichier Excel {selected_year}..."):
    try:
        xls = load_excel(excel_path)
        sheet_names = xls.sheet_names
    except Exception as e:
        st.error(f"Erreur de lecture du fichier Excel '{excel_path}': {e}")
        st.stop()

selected_sheet = st.sidebar.selectbox(
    "Choisir le mois (onglet Excel)",
    options=sheet_names,
    index=0,
    key=f"sheet_{selected_year}"
)

# ==========================
# Préparation des données du mois
# ==========================
df_month = parse_sheet(xls, selected_sheet, excel_key=excel_path)

# Jointure Geo + Données
merged = gdf.merge(df_month, on="NAME_3", how="left")

# Étiquette d'événement -> canonicalisation stricte
merged["Événement"] = merged.apply(_choose_event, axis=1).map(_canonize_label)

# Petit diagnostic pour repérer les labels parasites
counts = collections.Counter(merged["Événement"])
st.sidebar.caption("Catégories détectées : " + ", ".join(f"{k} ({v})" for k, v in counts.items()))

# ==========================
# Carte
# ==========================
map_center, zoom_level = {"lat": 12.5, "lon": -1.5}, 5.5

merged["Exposure_fmt"] = merged["Exposure at Risk"].apply(
    lambda x: "" if pd.isna(x) else f"{float(x):,.0f}".replace(",", " ")
)
merged["Losses_fmt"] = merged["Expected Losses"].apply(
    lambda x: "" if pd.isna(x) else f"{float(x):,.0f}".replace(",", " ")
)

hover_cols = ["NAME_3", "Événement", "Exposure_fmt", "Losses_fmt"]
custom_data = merged[hover_cols]

fig = px.choropleth_mapbox(
    merged,
    geojson=merged.geometry,
    locations=merged.index,
    color="Événement",
    color_discrete_map=COLOR_MAP,
    category_orders={"Événement": CATEGORY_ORDER},
    mapbox_style="carto-positron",
    zoom=zoom_level,
    center=map_center,
    opacity=0.85,
    custom_data=custom_data,
)

fig.update_traces(
    hovertemplate="<b>%{customdata[0]}</b><br>" +
                  "Phénomène : %{customdata[1]}<br>" +
                  "Montant exposé : %{customdata[2]}<br>" +
                  "Pertes attendues : %{customdata[3]}<extra></extra>"
)
fig.update_layout(
    margin={"t": 0, "b": 0, "l": 0, "r": 0},
    legend_title_text=f"Événement observé — {selected_year}"
)

st.plotly_chart(fig, use_container_width=True)

# ==========================
# Légende explicite
# ==========================
st.markdown("### 🗂️ Légende")
legend_items = [
    ("Sécheresse modérée", COLOR_MAP["Sécheresse modérée"]),
    ("Sécheresse sévère", COLOR_MAP["Sécheresse sévère"]),
    ("Inondation modérée", COLOR_MAP["Inondation modérée"]),
    ("Inondation sévère", COLOR_MAP["Inondation sévère"]),
    ("Aucun événement (non coloré)", COLOR_MAP["Aucun événement"]),
]
for label, rgba in legend_items:
    st.markdown(
        f'<div style="display:flex;align-items:center;margin:4px 0;">'
        f'<div style="width:16px;height:16px;background:{rgba};border:1px solid #aaa;margin-right:8px;"></div>'
        f'<div>{label}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# ==========================
# Tableau détaillé (optionnel)
# ==========================
with st.expander(f"Voir le tableau détaillé — {selected_year} / {selected_sheet}"):
    display_cols = ["NAME_3", "Événement"] + EVENT_COLS + ["Exposure at Risk", "Expected Losses"]
    show_df = merged[display_cols].copy()
    st.dataframe(show_df, use_container_width=True)

# ==========================
# Notes
# ==========================
