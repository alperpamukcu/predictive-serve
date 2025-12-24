# streamlit_app.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.utils.config import PROCESSED_DIR
from src.analysis.metrics import compute_overall_metrics


st.set_page_config(
    page_title="Predictive Serve Dashboard",
    layout="wide",
)


VAL_PRED_PATH = PROCESSED_DIR / "val_predictions.csv"
ALL_PRED_PATH = PROCESSED_DIR / "all_predictions.csv"


# ------------------------------
# 1) Dil seçimi & label helper
# ------------------------------
LANG = st.sidebar.selectbox("Dil / Language", ["Türkçe", "English"])
IS_TR = LANG == "Türkçe"


def L(tr: str, en: str) -> str:
    return tr if IS_TR else en


# ------------------------------
# 2) Veri yükleme
# ------------------------------
@st.cache_data
def load_predictions() -> pd.DataFrame:
    """Önce all_predictions varsa onu, yoksa val_predictions'ı yükler."""
    if ALL_PRED_PATH.exists():
        path = ALL_PRED_PATH
    else:
        path = VAL_PRED_PATH

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Edge yoksa hesapla
    if "p_model" in df.columns and "pA_market" in df.columns and "edge" not in df.columns:
        df["edge"] = df["p_model"] - df["pA_market"]

    # Doğruluk kolonları
    if "correct_model" not in df.columns and "p_model" in df.columns:
        df["correct_model"] = (
            (df["p_model"] >= 0.5).astype(int) == df["y"].astype(int)
        )
    if "pA_market" in df.columns and "correct_market" not in df.columns:
        df["correct_market"] = (
            (df["pA_market"] >= 0.5).astype(int) == df["y"].astype(int)
        )

    return df


df = load_predictions()

if ALL_PRED_PATH.exists():
    data_info = L(
        "Veri kaynağı: all_predictions (tüm yıllar, train + validation).",
        "Data source: all_predictions (all years, train + validation).",
    )
else:
    data_info = L(
        "Veri kaynağı: val_predictions (sadece 2022+ validation dönemi).",
        "Data source: val_predictions (only 2022+ validation period).",
    )

st.markdown(
    f"""
    # Predictive Serve Dashboard  

    {data_info}
    """
)

# ------------------------------
# 3) Sidebar filtreleri
# ------------------------------
st.sidebar.header(L("Filtreler", "Filters"))

min_year = int(df["date"].dt.year.min())
max_year = int(df["date"].dt.year.max())

year_range = st.sidebar.slider(
    L("Yıl aralığı", "Year range"),
    min_value=min_year,
    max_value=max_year,
    value=(max(min_year, 2015), max_year),
    step=1,
)

surfaces = sorted(df["surface"].dropna().unique().tolist())
surface_selected = st.sidebar.multiselect(
    L("Zemin (surface)", "Surface"),
    options=surfaces,
    default=surfaces,
)

p_model_min, p_model_max = st.sidebar.slider(
    L("Model olasılık aralığı (p_model)", "Model probability range (p_model)"),
    0.0,
    1.0,
    (0.0, 1.0),
    step=0.05,
)

min_edge = st.sidebar.slider(
    L("Minimum edge (Model − Market)", "Minimum edge (Model − Market)"),
    -0.3,
    0.3,
    0.0,
    step=0.01,
)

# ------------------------------
# 4) Filtre uygulama
# ------------------------------
filt_df = df.copy()

filt_df = filt_df[
    (filt_df["date"].dt.year >= year_range[0])
    & (filt_df["date"].dt.year <= year_range[1])
]

if surface_selected:
    filt_df = filt_df[filt_df["surface"].isin(surface_selected)]

if "p_model" in filt_df.columns:
    filt_df = filt_df[
        (filt_df["p_model"] >= p_model_min)
        & (filt_df["p_model"] <= p_model_max)
    ]

if "edge" in filt_df.columns and "pA_market" in filt_df.columns:
    filt_df = filt_df[filt_df["edge"] >= min_edge]

st.write(
    L(
        f"Filtre sonrası maç sayısı: **{len(filt_df):,}**",
        f"Number of matches after filters: **{len(filt_df):,}**",
    )
)

if filt_df.empty:
    st.warning(
        L(
            "Bu filtrelerle eşleşen maç bulunamadı. Filtreleri gevşetmeyi deneyebilirsin.",
            "No matches for these filters. Try relaxing the filters.",
        )
    )
    st.stop()

# ------------------------------
# 5) Genel metrik kartları + açıklamalar
# ------------------------------
overall = compute_overall_metrics(filt_df)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    L("Maç sayısı", "Number of matches"),
    f"{overall.n_matches:,}",
)

col2.metric(
    L("Model isabet oranı", "Model accuracy"),
    f"{overall.model_acc*100:.2f} %",
)

if overall.market_acc is not None:
    col3.metric(
        L("Bahis şirketi isabet oranı", "Bookmaker accuracy"),
        f"{overall.market_acc*100:.2f} %",
    )

col4.metric(
    L("Accuracy farkı (Model − Market)", "Accuracy diff (Model − Market)"),
    f"{(overall.model_acc - (overall.market_acc or 0))*100:.2f} p.p.",
)

with st.expander(L("Metrik açıklamaları", "Metric explanations")):
    if IS_TR:
        st.markdown(
            """
            - **Logloss**: Modelin doğru sonuca ne kadar “olasılık” verdiğini ölçer.  
              Düşük logloss = daha iyi kalibre edilmiş model.
            - **Brier skoru**: Olasılık tahminlerinin karesel hatası. Yine düşük olması iyidir.
            - **Accuracy (isabet oranı)**: Kaç maçta doğru tarafı seçtiğimiz (p ≥ 0.5) yüzdesi.
            - **Edge**: Model olasılığı − bahis şirketi olasılığı. Pozitif edge, teorik olarak avantajlı bahsi gösterir.
            """
        )
    else:
        st.markdown(
            """
            - **Logloss**: Measures how much probability the model assigns to the true outcome.  
              Lower logloss = better calibrated model.
            - **Brier score**: Squared error of probability forecasts. Lower is better.
            - **Accuracy**: Percentage of matches where the predicted side (p ≥ 0.5) is correct.
            - **Edge**: Model probability − bookmaker probability. Positive edge suggests theoretical value.
            """
        )

st.divider()

# ------------------------------
# 6) Sekmeler: Overview, Leaderboard, Match Explorer
# ------------------------------
tab_overview, tab_leaderboard, tab_match = st.tabs(
    [
        L("Genel görünüm", "Overview"),
        L("Oyuncu liderlik tablosu", "Player leaderboard"),
        L("Maç seç / detay", "Match explorer"),
    ]
)

# ---- TAB 1: OVERVIEW ----
with tab_overview:
    st.subheader(L("Model vs Bahis şirketi – özet metrikler",
                   "Model vs Bookmaker – summary metrics"))

    metrics_df = pd.DataFrame(
        {
            L("Logloss", "Logloss"): [overall.model_logloss, overall.market_logloss],
            L("Brier skoru", "Brier score"): [overall.model_brier, overall.market_brier],
            L("İsabet oranı", "Accuracy"): [overall.model_acc, overall.market_acc],
        },
        index=[L("Model", "Model"), L("Bahis şirketi", "Bookmaker")],
    )

    st.dataframe(
        metrics_df.style.format(
            {
                L("Logloss", "Logloss"): "{:.6f}",
                L("Brier skoru", "Brier score"): "{:.6f}",
                L("İsabet oranı", "Accuracy"): "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown(L("### Edge’e göre en iyi maçlar", "### Top value bets by edge"))

    if "edge" in filt_df.columns and "pA_market" in filt_df.columns:
        top_n = st.slider(
            L("Kaç maç gösterilsin?", "How many matches to show?"),
            10,
            200,
            50,
            step=10,
        )
        top_df = (
            filt_df.sort_values("edge", ascending=False)
            .head(top_n)
            .loc[
                :,
                [
                    "date",
                    "surface",
                    "playerA",
                    "playerB",
                    "p_model",
                    "pA_market",
                    "edge",
                    "y",
                    "correct_model",
                    "correct_market",
                ],
            ]
        )

        rename_cols = {
            "date": L("Tarih", "Date"),
            "surface": L("Zemin", "Surface"),
            "playerA": L("Oyuncu A", "Player A"),
            "playerB": L("Oyuncu B", "Player B"),
            "p_model": L("Model olasılığı", "Model prob."),
            "pA_market": L("Bahis olasılığı", "Bookmaker prob."),
            "edge": L("Edge (M−B)", "Edge (M−B)"),
            "y": L("Gerçek sonuç (A kazandı mı?)", "True outcome (did A win?)"),
            "correct_model": L("Model doğru mu?", "Model correct?"),
            "correct_market": L("Bahis doğru mu?", "Bookmaker correct?"),
        }

        st.dataframe(
            top_df.rename(columns=rename_cols).style.format(
                {
                    rename_cols["date"]: lambda d: d.strftime("%Y-%m-%d"),
                    rename_cols["p_model"]: "{:.3f}",
                    rename_cols["pA_market"]: "{:.3f}",
                    rename_cols["edge"]: "{:+.3f}",
                }
            ),
            use_container_width=True,
            height=500,
        )
    else:
        st.info(
            L(
                "Edge bilgisi için pA_market kolonu gerekli.",
                "Edge requires pA_market column.",
            )
        )

# ---- TAB 2: LEADERBOARD ----
with tab_leaderboard:
    st.subheader(L("Oyuncu liderlik tablosu", "Player leaderboard"))

    perf_df = filt_df[
        [
            "date",
            "playerA",
            "y",
            "correct_model",
            "correct_market",
            "edge",
        ]
    ].copy()
    perf_df["year"] = perf_df["date"].dt.year

    grouped = perf_df.groupby("playerA").agg(
        n_matches=("y", "size"),
        winrate=("y", "mean"),
        model_acc=("correct_model", "mean"),
        market_acc=("correct_market", "mean"),
        avg_edge=("edge", "mean"),
    )

    min_matches = st.slider(
        L("Minimum maç sayısı", "Minimum number of matches"),
        10,
        200,
        30,
        step=5,
    )
    grouped = grouped[grouped["n_matches"] >= min_matches]

    sort_by = st.selectbox(
        L("Sıralama kriteri", "Sort by"),
        options=[
            "winrate",
            "model_acc",
            "market_acc",
            "avg_edge",
            "n_matches",
        ],
        index=0,
    )
    ascending = st.checkbox(
        L("Artan sırala", "Sort ascending"), value=False
    )

    grouped = grouped.sort_values(sort_by, ascending=ascending)

    display = grouped.copy()
    display["winrate"] = display["winrate"] * 100
    display["model_acc"] = display["model_acc"] * 100
    display["market_acc"] = display["market_acc"] * 100

    display = display.rename(
        columns={
            "n_matches": L("Maç sayısı", "Matches"),
            "winrate": L("Kazanma oranı (%)", "Win rate (%)"),
            "model_acc": L("Model isabet (%)", "Model acc (%)"),
            "market_acc": L("Bahis isabet (%)", "Bookmaker acc (%)"),
            "avg_edge": L("Ortalama edge", "Average edge"),
        }
    )

    st.dataframe(
        display.style.format(
            {
                L("Maç sayısı", "Matches"): "{:d}",
                L("Kazanma oranı (%)", "Win rate (%)"): "{:.1f}",
                L("Model isabet (%)", "Model acc (%)"): "{:.1f}",
                L("Bahis isabet (%)", "Bookmaker acc (%)"): "{:.1f}",
                L("Ortalama edge", "Average edge"): "{:+.3f}",
            }
        ),
        use_container_width=True,
        height=400,
    )

    st.markdown(L("### Zaman içinde sıralama grafiği", "### Ranking over time"))

    # Zaman çözünürlüğü (şimdilik yıl bazlı)
    time_resolution = st.selectbox(
        L("Zaman çözünürlüğü", "Time resolution"),
        [L("Yıl", "Year")],  # istersek ileride "Ay" ekleriz
    )

    # Yıl bazlı ranking
    yearly = perf_df.groupby(["year", "playerA"]).agg(
        n_matches=("y", "size"),
        winrate=("y", "mean"),
    )

    # En çok maça çıkan ilk N oyuncu
    overall_counts = yearly.groupby("playerA")["n_matches"].sum()
    top_k = st.slider(
        L("Grafikte kaç oyuncu gösterilsin?", "How many players in the chart?"),
        3,
        15,
        5,
    )
    top_players = overall_counts.sort_values(ascending=False).head(top_k).index.tolist()

    yearly_top = yearly.reset_index()
    yearly_top = yearly_top[yearly_top["playerA"].isin(top_players)]

    # Her yıl için rank (1 en yüksek winrate)
    yearly_top["rank"] = (
        yearly_top.groupby("year")["winrate"].rank(ascending=False, method="min")
    )

    if yearly_top.empty:
        st.info(
            L(
                "Bu filtreler için liderlik grafiği oluşturulamadı.",
                "No data to build ranking chart with these filters.",
            )
        )
    else:
        # rank ekseni ters olsun diye -rank kullanıyoruz
        chart_data = yearly_top.pivot_table(
            index="year",
            columns="playerA",
            values="rank",
        )

        st.line_chart(
            -chart_data,  # -rank: 1. sıra en yukarı
            height=350,
        )

# ---- TAB 3: MATCH EXPLORER ----
with tab_match:
    st.subheader(L("Maç seç ve detayını incele", "Select a match and inspect details"))

    # Tarihe göre sıralayıp son yılları öne çıkaralım
    match_df = filt_df.sort_values("date", ascending=False).copy()
    match_df["label"] = (
        match_df["date"].dt.strftime("%Y-%m-%d")
        + " – "
        + match_df["playerA"]
        + " vs "
        + match_df["playerB"]
        + " ("
        + match_df["surface"]
        + ")"
    )

    options = match_df["label"].tolist()
    selected_label = st.selectbox(
        L("Bir maç seç", "Select a match"), options=options
    )

    row = match_df[match_df["label"] == selected_label].iloc[0]

    st.markdown("### " + selected_label)

    c1, c2 = st.columns(2)

    with c1:
        st.write(L("Model tahmini", "Model prediction"))
        st.write(L("A kazanma olasılığı", "Prob. A wins"), f"{row['p_model']:.3f}")
        st.write(
            L("Gerçek sonuç (A kazandı mı?)", "True outcome (did A win?)"),
            int(row["y"]),
        )
        st.write(
            L("Model doğru mu?", "Model correct?"),
            bool(row["correct_model"]),
        )

    with c2:
        if not np.isnan(row.get("pA_market", np.nan)):
            st.write(L("Bahis şirketi tahmini", "Bookmaker prediction"))
            st.write(
                L("A kazanma olasılığı (market)", "Prob. A wins (market)"),
                f"{row['pA_market']:.3f}",
            )
            st.write(L("Edge (Model − Market)", "Edge (Model − Market)"),
                     f"{row['edge']:+.3f}")
            st.write(
                L("Bahis doğru mu?", "Bookmaker correct?"),
                bool(row["correct_market"]),
            )

    with st.expander(L("Ham satırı göster", "Show raw row")):
        st.json(row.to_dict())
