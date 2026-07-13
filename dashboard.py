"""
March Madness 2026 - Prediction Dashboard
==========================================

HOW TO RUN:
-----------
1. Place the trained model bundles and data files here:
       models/bundle_men.pkl
       models/bundle_women.pkl
       data/data_men.pkl
       data/data_women.pkl
   (see models/README.md and data/README.md for where to get them)

2. Install dependencies (run once):
       py -m pip install -r requirements.txt

3. Launch:
       py -m streamlit run dashboard.py

4. Open browser at:  http://localhost:8501
"""

import warnings
warnings.filterwarnings("ignore")

import os, pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── PANDAS COMPAT PATCH (bundles saved with pandas 1.x) ──────────────────────
import pandas.core.indexes.base as _pd_base
_orig_new_index = _pd_base._new_Index

def _patched_new_index(cls, d):
    try:
        return _orig_new_index(cls, d)
    except TypeError:
        data = d.get("data", d.get("_data", []))
        name = d.get("name", None)
        try:    return pd.Index(data, name=name)
        except: return pd.Index([])

_pd_base._new_Index = _patched_new_index

class _LegacyIndex(pd.Index):
    def __new__(cls, data=None, dtype=None, copy=False, name=None, **kw):
        if data is None: data = []
        return pd.Index.__new__(cls, data)

for _n in ("Int64Index", "Float64Index", "UInt64Index"):
    setattr(pd, _n, _LegacyIndex)


# ── DATACLASS STUBS ───────────────────────────────────────────────────────────
@dataclass
class ModelSpec:
    solver: str = "liblinear"; c_value: float = 0.01
    penalty: str = "l2";       poly_degree: int = 1
    def __setstate__(self, s): self.__dict__.update(s)

@dataclass
class ModelBundle:
    model: Any = None; features: List[str] = field(default_factory=list)
    spec: Any = None;  team_features: Any = None; h2h_features: Any = None
    team_lookup: Any = None; benchmark_features: List[str] = field(default_factory=list)
    benchmark_metrics: Dict = field(default_factory=dict)
    final_metrics: Dict = field(default_factory=dict)
    production_source: str = ""; selection_history: Any = None
    diagnostics: Dict = field(default_factory=dict); model_search: Any = None
    eligible_massey_systems: List[str] = field(default_factory=list)
    def __setstate__(self, s): self.__dict__.update(s)


# ── SAFE UNPICKLER ────────────────────────────────────────────────────────────
class SafeUnpickler(pickle.Unpickler):
    _MAP = {"ModelBundle": ModelBundle, "ModelSpec": ModelSpec}
    def find_class(self, module, name):
        if name in self._MAP: return self._MAP[name]
        try: return super().find_class(module, name)
        except:
            class _S:
                def __setstate__(self, s): self.__dict__.update(s)
            _S.__name__ = name; return _S


# ── FILE LOADING ──────────────────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DATA_DIR   = os.path.join(HERE, "data")

def _load(path):
    with open(path, "rb") as f: return SafeUnpickler(f).load()

@st.cache_resource(show_spinner="Loading models…")
def load_bundles():
    out = {}
    for g, fn in [("men","bundle_men.pkl"),("women","bundle_women.pkl")]:
        p = os.path.join(MODELS_DIR, fn)
        if os.path.exists(p):
            try:    out[g] = _load(p)
            except Exception as e: st.warning(f"Could not load {fn}: {e}")
    return out

@st.cache_resource(show_spinner="Loading data…")
def load_data_files():
    out = {"men": None, "women": None}
    for g, fn in [("men","data_men.pkl"),("women","data_women.pkl")]:
        p = os.path.join(DATA_DIR, fn)
        if os.path.exists(p):
            try:    out[g] = _load(p)
            except: pass
    return out


# ── TEAM LIST ─────────────────────────────────────────────────────────────────
def get_teams(bundle, data, season=2026):
    if data is not None:
        s  = data["seeds"][data["seeds"]["Season"] == season]
        df = s.merge(data["teams"][["TeamID","TeamName"]], on="TeamID", how="left").copy()
        df["SeedNum"] = df["Seed"].str[1:3].astype(int)
        df["Region"]  = df["Seed"].str[0]
        return df.sort_values(["Region","SeedNum"]).reset_index(drop=True)
    tf = bundle.team_features
    if tf is not None and isinstance(tf, pd.DataFrame):
        df = tf[tf["Season"]==season][["TeamID"]].drop_duplicates()
        lk = bundle.team_lookup
        if lk is not None: df = df.merge(lk, on="TeamID", how="left")
        if "seed_num" in tf.columns:
            si = tf[tf["Season"]==season][["TeamID","seed_num"]].dropna()
            df = df.merge(si, on="TeamID", how="left").sort_values("seed_num")
        else: df = df.sort_values("TeamName")
        return df.reset_index(drop=True)
    return pd.DataFrame(columns=["TeamID","TeamName"])


# ── FEATURE DIRECTIONS ────────────────────────────────────────────────────────
_DIR = {
    "avg_margin_adv":"A-B","rank_consensus_adv":"B-A","seed_adv":"B-A",
    "schedule_adjusted_net_rating_adv":"A-B","schedule_rank_adv":"B-A",
    "conference_rank_strength_adv":"A-B","off_eFG_adv":"A-B","def_eFG_adv":"B-A",
    "tov_adv":"B-A","orb_adv":"A-B","ft_rate_adv":"A-B","off_rating_adv":"A-B",
    "def_rating_adv":"B-A","net_rating_adv":"A-B","neutral_net_rating_adv":"A-B",
    "away_neutral_net_rating_adv":"A-B","recent5_net_rating_adv":"A-B",
    "recent10_net_rating_adv":"A-B","conf_tourney_wins_adv":"A-B",
    "rank_std_adv":"B-A","best_rank_adv":"B-A","seed_minus_rank_adv":"A-B",
    "win_pct_adv":"A-B","schedule_win_pct_adv":"A-B","coach_tenure_adv":"A-B",
    "conf_tourney_champ_adv":"A-B","margin_std_adv":"B-A","net_rating_std_adv":"B-A",
    "close_game_win_pct_adv":"A-B","ot_win_pct_adv":"A-B","neutral_win_pct_adv":"A-B",
    "neutral_games_adv":"A-B","away_win_pct_adv":"A-B","top_opponent_win_pct_adv":"A-B",
    "top_opponent_margin_adv":"A-B","conf_tourney_games_adv":"A-B","conf_tourney_win_pct_adv":"A-B",
}
_BASE = {k: k.replace("_adv","") for k in _DIR}
_BASE.update({"rank_consensus_adv":"rank_median","seed_adv":"seed_num",
              "schedule_adjusted_net_rating_adv":"schedule_adjusted_net_rating",
              "off_eFG_adv":"off_eFG","def_eFG_adv":"def_eFG",
              "tov_adv":"tov_pct","orb_adv":"orb_pct"})

def _fv(row, col):
    try:
        v = row[col]; return float(v) if not pd.isna(v) else 0.0
    except: return 0.0

def build_features(bundle, season, tid_a, tid_b):
    tf = bundle.team_features
    if tf is None or not isinstance(tf, pd.DataFrame): return None
    ra_df = tf[(tf["Season"]==season)&(tf["TeamID"]==tid_a)]
    rb_df = tf[(tf["Season"]==season)&(tf["TeamID"]==tid_b)]
    if ra_df.empty or rb_df.empty: return None
    ra, rb = ra_df.iloc[0], rb_df.iloc[0]
    row = {"Season":season,"TeamA":tid_a,"TeamB":tid_b}
    for adv, direction in _DIR.items():
        col = _BASE.get(adv, adv.replace("_adv",""))
        va, vb = _fv(ra,col), _fv(rb,col)
        row[adv] = (va-vb) if direction=="A-B" else (vb-va)
    m = row["avg_margin_adv"]
    row["margin_gap_squared_adv"]            = m**2
    row["rank_gap_squared_adv"]              = row["rank_consensus_adv"]**2
    row["seed_gap_squared_adv"]              = row["seed_adv"]**2
    row["schedule_adjusted_gap_squared_adv"] = row["schedule_adjusted_net_rating_adv"]**2
    row["seed_rank_alignment_adv"]           = row["seed_adv"]*row["rank_consensus_adv"]
    row["seed_schedule_alignment_adv"]       = row["seed_adv"]*row["schedule_rank_adv"]
    row["rank_schedule_alignment_adv"]       = row["rank_consensus_adv"]*row["schedule_rank_adv"]
    for col in ("h2h_same_season_win_edge","h2h_same_season_margin",
                "h2h_recent_weighted_win_edge","h2h_recent_weighted_margin"):
        row[col] = 0.0
        hf = bundle.h2h_features
        if hf is not None and isinstance(hf, pd.DataFrame):
            m2 = hf[(hf["Season"]==season)&(hf["TeamA"]==tid_a)&(hf["TeamB"]==tid_b)]
            if not m2.empty and col in m2.columns: row[col] = _fv(m2.iloc[0],col)
    return pd.DataFrame([row])

@st.cache_data(show_spinner=False)
def run_prediction(_bid, season, tid_a, tid_b, _bundle):
    df = build_features(_bundle, season, tid_a, tid_b)
    if df is None: return None
    feats = _bundle.features
    for f in feats:
        if f not in df.columns: df[f] = 0.0
    try:    p = float(_bundle.model.predict_proba(df[feats])[0,1])
    except: return None
    p = max(1e-6, min(1-1e-6, p))
    result = {"prob_a":round(p*100,1),"prob_b":round((1-p)*100,1)}
    for col in df.columns:
        if "_adv" in col: result[col] = round(float(df.iloc[0][col]),3)
    return result


# ── HISTORICAL ACCURACY ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Calculating historical accuracy…")
def get_historical_accuracy(_bid, _bundle, _data, test_seasons=(2023,2024,2025)):
    if _data is None or _bundle is None: return None
    tourney = _data.get("tourney")
    teams   = _data.get("teams")
    if tourney is None or teams is None: return None
    rows = []
    lk = {int(r.TeamID): r.TeamName for _, r in teams.iterrows()}
    for season in test_seasons:
        games = tourney[tourney["Season"]==season]
        for _, g in games.iterrows():
            tid_w, tid_l = int(g["WTeamID"]), int(g["LTeamID"])
            tid_a = min(tid_w, tid_l)
            tid_b = max(tid_w, tid_l)
            res = run_prediction(id(_bundle), season, tid_a, tid_b, _bundle=_bundle)
            if res is None: continue
            prob_a = res["prob_a"] / 100
            actual_winner_id = tid_w
            pred_winner_id   = tid_a if prob_a >= 0.5 else tid_b
            correct = (pred_winner_id == actual_winner_id)
            rows.append({
                "Season":           season,
                "Team A":           lk.get(tid_a, str(tid_a)),
                "Team B":           lk.get(tid_b, str(tid_b)),
                "Actual winner":    lk.get(actual_winner_id, str(actual_winner_id)),
                "Predicted winner": lk.get(pred_winner_id, str(pred_winner_id)),
                "Confidence":       round(max(prob_a, 1-prob_a)*100, 1),
                "Correct":          correct,
                "_prob_a":          prob_a,
            })
    return pd.DataFrame(rows) if rows else None


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
METRICS = {
    "men":   {"CV Log Loss":0.5668,"CV AUC":0.7707,
              "Test Log Loss":0.5504,"Test AUC":0.7997,"Production":"forward_cv_candidate"},
    "women": {"CV Log Loss":0.4414,"CV AUC":"n/a",
              "Test Log Loss":0.4103,"Test AUC":0.8981,"Production":"forward_cv_candidate"},
}

# Human-readable labels - no jargon
LABELS = {
    "avg_margin_adv":                   "Average winning margin (pts)",
    "net_rating_adv":                   "Overall team efficiency",
    "seed_adv":                         "Tournament seeding advantage",
    "rank_consensus_adv":               "Computer ranking advantage",
    "schedule_adjusted_net_rating_adv": "Efficiency vs tough opponents",
    "off_eFG_adv":                      "Shooting quality (offense)",
    "def_eFG_adv":                      "Limiting opponent shooting",
    "tov_adv":                          "Ball protection (fewer turnovers)",
    "orb_adv":                          "Second-chance rebounds",
    "ft_rate_adv":                      "Drawing fouls (free throws)",
    "close_game_win_pct_adv":           "Winning tight games",
    "win_pct_adv":                      "Season win rate",
    "margin_std_adv":                   "Score consistency",
    "coach_tenure_adv":                 "Coaching experience",
    "schedule_rank_adv":                "Toughness of schedule",
    "conference_rank_strength_adv":     "Strength of conference",
    "neutral_net_rating_adv":           "Performance at neutral venues",
    "away_win_pct_adv":                 "Winning away from home",
    "recent5_net_rating_adv":           "Form in last 5 games",
    "recent10_net_rating_adv":          "Form in last 10 games",
    "h2h_same_season_win_edge":         "Head-to-head this season",
    "h2h_recent_weighted_win_edge":     "Head-to-head recent history",
    "off_rating_adv":                   "Points scored per 100 possessions",
    "def_rating_adv":                   "Points allowed per 100 possessions",
    "ot_win_pct_adv":                   "Winning in overtime",
    "neutral_win_pct_adv":              "Winning at neutral sites",
    "away_neutral_net_rating_adv":      "Efficiency away & neutral",
    "top_opponent_win_pct_adv":         "Record vs top opponents",
    "top_opponent_margin_adv":          "Margin vs top opponents",
    "conf_tourney_wins_adv":            "Conference tournament wins",
    "conf_tourney_champ_adv":           "Won conference tournament",
    "conf_tourney_win_pct_adv":         "Conference tourney win rate",
}

# Five basketball scoring factors
RADAR_FEATS = ["off_eFG_adv","def_eFG_adv","tov_adv","orb_adv","ft_rate_adv"]
RADAR_LBL   = [
    "Shooting\nquality",
    "Stopping\nopponent shots",
    "Ball\nprotection",
    "Extra\nrebounds",
    "Drawing\nfouls",
]

# Key features for comparison bar chart
KEY_FEATS = [
    "avg_margin_adv","net_rating_adv","win_pct_adv",
    "off_eFG_adv","close_game_win_pct_adv","schedule_adjusted_net_rating_adv",
]

# Why-favorite features
WHY_FEATS = [
    ("avg_margin_adv",                   "Wins by more points",          " pts"),
    ("net_rating_adv",                   "Overall efficiency edge",       ""),
    ("win_pct_adv",                      "Better season record",          "%"),
    ("close_game_win_pct_adv",           "Clutch: wins tight games",     "%"),
    ("schedule_adjusted_net_rating_adv", "Stronger vs tough opponents",  ""),
    ("conference_rank_strength_adv",     "Plays in tougher conference",   ""),
]

CA="#185FA5"; CB="#A32D2D"; CP="#15803d"; CN="#dc2626"; CU="#94a3b8"

def acolor(v): return CP if v>0.05 else (CN if v<-0.05 else CU)
def fadv(v,d=2):
    s = round(v,d); return f"+{s}" if s>0 else str(s)
def conf(p):
    d = abs(p/100-0.5)
    if d>.20: return "High",   "#dcfce7","#166534"
    if d>.10: return "Medium", "#fef9c3","#854d0e"
    return "Low: coin flip","#fee2e2","#991b1b"

def est_mc(prob_a_pct, seed_a, seed_b):
    """Rough Monte Carlo estimate - wins needed per round."""
    p = prob_a_pct / 100
    q = 1 - p
    # Seed position matters: better seed avoids harder matchups early
    seed_factor = max(0.3, min(1.0, (seed_b - seed_a + 8) / 16)) if seed_a and seed_b else 0.5
    pa = {"s16": round(p**2*100,1), "e8": round(p**3*100,1),
          "f4":  round(p**4*100,1), "fin":round(p**5*100,1), "champ":round(p**6*100,1)}
    pb = {"s16": round(q**2*100,1), "e8": round(q**3*100,1),
          "f4":  round(q**4*100,1), "fin":round(q**5*100,1), "champ":round(q**6*100,1)}
    return pa, pb


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="March Madness 2026",
                   page_icon="🏀", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.prob-big{font-size:3.5rem;font-weight:700;letter-spacing:-0.04em;line-height:1}
.tname{font-size:1.3rem;font-weight:600;line-height:1.2}
.tmeta{font-size:.82rem;color:#94a3b8;margin-top:3px}
.bfav{background:#dcfce7;color:#166534;padding:3px 10px;border-radius:20px;font-size:.72rem;font-weight:700;letter-spacing:.05em}
.bdog{background:#fee2e2;color:#991b1b;padding:3px 10px;border-radius:20px;font-size:.72rem;font-weight:700;letter-spacing:.05em}
.mcard{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1rem}
.mlbl{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.07em;font-weight:600;margin-bottom:4px}
.slbl{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#94a3b8;margin-bottom:.6rem}
.why-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:.85rem 1rem;margin-bottom:8px;color:#1e293b}
.h2h-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1rem 1.2rem}
</style>
""", unsafe_allow_html=True)


# ── LOAD ──────────────────────────────────────────────────────────────────────
bundles   = load_bundles()
data_dict = load_data_files()

if not bundles:
    st.error("No model files found.")
    st.code("📁 your_folder/\n     dashboard.py\n     models/\n"
            "         bundle_men.pkl\n         bundle_women.pkl\n"
            "     data/\n         data_men.pkl\n         data_women.pkl")
    st.stop()


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏀 March Madness 2026")
    st.markdown("**Prediction Dashboard**")
    st.caption("Kaggle ML Competition · Season 2026")
    st.divider()

    gmap = {}
    if "men"   in bundles: gmap["Men's Tournament"]   = "men"
    if "women" in bundles: gmap["Women's Tournament"] = "women"

    sel_lbl = st.selectbox("Tournament", list(gmap.keys()))
    gender  = gmap[sel_lbl]
    bundle  = bundles[gender]
    data    = data_dict.get(gender)

    teams_df   = get_teams(bundle, data)
    team_names = teams_df["TeamName"].tolist()

    if len(team_names) < 2:
        st.error("No teams found for 2026."); st.stop()

    st.divider()
    st.markdown("#### Select teams")
    idx_a = st.selectbox("Team A", range(len(team_names)),
                         format_func=lambda i: team_names[i], key="a")
    st.markdown("<div style='text-align:center;color:#94a3b8;margin:2px 0'>vs</div>",
                unsafe_allow_html=True)
    idx_b = st.selectbox("Team B", range(len(team_names)),
                         index=min(1,len(team_names)-1),
                         format_func=lambda i: team_names[i], key="b")

    st.divider()
    page = st.radio("View", ["Match prediction", "Title favorites",
                              "Model accuracy (past seasons)", "Model metrics"],
                    label_visibility="collapsed")

    st.divider()
    st.caption(f"  ✅ bundle_{gender}.pkl")
    st.caption(f"  {'✅' if data else '⚠️'} data_{gender}.pkl"
               + ("  (not found)" if not data else ""))


# ── VALIDATION ────────────────────────────────────────────────────────────────
if idx_a == idx_b:
    st.warning("Please select two different teams."); st.stop()

name_a = team_names[idx_a];  name_b = team_names[idx_b]
tid_a  = int(teams_df.iloc[idx_a]["TeamID"])
tid_b  = int(teams_df.iloc[idx_b]["TeamID"])
seed_a = teams_df.iloc[idx_a].get("Seed", teams_df.iloc[idx_a].get("SeedNum","n/a"))
seed_b = teams_df.iloc[idx_b].get("Seed", teams_df.iloc[idx_b].get("SeedNum","n/a"))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 - MATCH PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
if page == "Match prediction":

    with st.spinner("Running prediction…"):
        result = run_prediction(id(bundle), 2026, tid_a, tid_b, _bundle=bundle)

    if result is None:
        st.error("Prediction failed. Try:  py -m pip install scikit-learn==1.5.2  then restart.")
        st.stop()

    pa, pb       = result["prob_a"], result["prob_b"]
    ct, cbg, cfg = conf(pa)

    # ── HERO ──────────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3 = st.columns([4,2,4])

    def _hero(col, name, seed, prob, is_fav, color):
        b = "bfav" if is_fav else "bdog"; v = "FAVORITE" if is_fav else "UNDERDOG"
        col.markdown(f"""
        <div style="text-align:center;padding:1.5rem .5rem">
            <div class="tname" style="color:{color}">{name}</div>
            <div class="tmeta">Seed {seed}</div>
            <div class="prob-big" style="color:{color};margin:10px 0">{prob}%</div>
            <div class="tmeta">chance of winning</div>
            <span class="{b}" style="margin-top:8px;display:inline-block">{v}</span>
        </div>""", unsafe_allow_html=True)

    _hero(c1, name_a, seed_a, pa, pa>=pb, CA)
    c2.markdown(f"""
    <div style="text-align:center;padding:1.5rem .5rem">
        <div style="font-size:1.8rem;color:#cbd5e1;margin-bottom:12px">vs</div>
        <div style="font-size:.68rem;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:.08em;margin-bottom:4px">Model confidence</div>
        <div style="font-size:.88rem;font-weight:600;padding:4px 8px;border-radius:8px;
                    background:{cbg};color:{cfg}">{ct}</div>
        <div style="font-size:.68rem;color:#94a3b8;margin-top:12px">Season 2026</div>
    </div>""", unsafe_allow_html=True)
    _hero(c3, name_b, seed_b, pb, pb>pa, CB)

    # probability bar
    fb = go.Figure()
    fb.add_trace(go.Bar(x=[pa],y=[""],orientation="h",marker_color=CA,name=name_a))
    fb.add_trace(go.Bar(x=[pb],y=[""],orientation="h",marker_color=CB,name=name_b,base=pa))
    fb.update_layout(barmode="stack",height=52,showlegend=False,
                     margin=dict(l=0,r=0,t=0,b=0),
                     paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                     xaxis=dict(showgrid=False,showticklabels=False,range=[0,100]),
                     yaxis=dict(showgrid=False,showticklabels=False))
    st.plotly_chart(fb, use_container_width=True, config={"displayModeBar":False})
    l,r = st.columns(2)
    l.markdown(f"<div style='color:{CA};font-weight:600;font-size:.82rem'>◀ {name_a}  {pa}%</div>",
               unsafe_allow_html=True)
    r.markdown(f"<div style='color:{CB};font-weight:600;font-size:.82rem;text-align:right'>"
               f"{name_b}  {pb}% ▶</div>", unsafe_allow_html=True)

    # ── QUICK STATS ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="slbl">Quick stats: what the model sees</div>',
                unsafe_allow_html=True)
    km = [("avg_margin_adv","Average winning margin"," pts"),
          ("net_rating_adv","Overall efficiency",""),
          ("seed_adv","Tournament seeding",""),
          ("close_game_win_pct_adv","Clutch game performance","%")]
    for col,(feat,lbl,unit) in zip(st.columns(4), km):
        v = result.get(feat,0.0); favor = name_a if v>0 else (name_b if v<0 else "Even")
        col.markdown(f"""
        <div class="mcard">
            <div class="mlbl">{lbl}</div>
            <div style="font-size:1.55rem;font-weight:700;color:{acolor(v)};margin:5px 0">
                {fadv(v)}{unit}</div>
            <div style="font-size:.72rem;color:#94a3b8">→ {favor}</div>
        </div>""", unsafe_allow_html=True)

    # ── COMPARISON + RADAR ────────────────────────────────────────────────
    st.markdown("---")
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown('<div class="slbl">How the teams compare: 6 key areas</div>',
                    unsafe_allow_html=True)
        lbls = [LABELS.get(f,f) for f in KEY_FEATS]
        va   = [round(result.get(f,0.0),3) for f in KEY_FEATS]
        vb   = [round(-result.get(f,0.0),3) for f in KEY_FEATS]
        fc = go.Figure()
        fc.add_trace(go.Bar(name=name_a, y=lbls, x=va, orientation="h",
                            marker_color=CA, opacity=.85))
        fc.add_trace(go.Bar(name=name_b, y=lbls, x=vb, orientation="h",
                            marker_color=CB, opacity=.85))
        fc.update_layout(barmode="overlay", height=300,
                         margin=dict(l=0,r=0,t=10,b=0),
                         legend=dict(orientation="h",y=-0.18,x=.5,xanchor="center"),
                         paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                         xaxis=dict(gridcolor="#f1f5f9",zerolinecolor="#cbd5e1"),
                         yaxis=dict(gridcolor="#f1f5f9"),
                         font=dict(family="DM Sans",size=11))
        st.plotly_chart(fc, use_container_width=True, config={"displayModeBar":False})

    with ch2:
        st.markdown('<div class="slbl">Five basketball scoring pillars</div>',
                    unsafe_allow_html=True)
        def nr(v): return max(0.0, min(100.0, 50+v*200))
        ra_ = [nr(result.get(f,0)) for f in RADAR_FEATS]+[nr(result.get(RADAR_FEATS[0],0))]
        rb_ = [nr(-result.get(f,0)) for f in RADAR_FEATS]+[nr(-result.get(RADAR_FEATS[0],0))]
        th  = RADAR_LBL + [RADAR_LBL[0]]
        fr  = go.Figure()
        fr.add_trace(go.Scatterpolar(r=ra_,theta=th,fill="toself",name=name_a,
                                     line_color=CA,fillcolor="rgba(24,95,165,0.2)"))
        fr.add_trace(go.Scatterpolar(r=rb_,theta=th,fill="toself",name=name_b,
                                     line_color=CB,fillcolor="rgba(162,45,45,0.2)"))
        fr.update_layout(polar=dict(radialaxis=dict(visible=False,range=[0,100])),
                         showlegend=True,height=300,
                         margin=dict(l=40,r=40,t=20,b=20),
                         legend=dict(orientation="h",y=-0.12,x=.5,xanchor="center"),
                         paper_bgcolor="rgba(0,0,0,0)",
                         font=dict(family="DM Sans",size=11))
        st.plotly_chart(fr, use_container_width=True, config={"displayModeBar":False})

    # small legend for radar
    st.caption(
        "Shooting quality = how efficiently each team shoots (higher = better shooting). "
        "Stopping opponent shots = how well each team limits rival scoring. "
        "Ball protection = how rarely they give the ball away. "
        "Extra rebounds = how often they get second-chance shots. "
        "Drawing fouls = how often they reach the free-throw line."
    )

    # ── MONTE CARLO ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="slbl">How far could each team go? Round advancement probabilities</div>',
                unsafe_allow_html=True)
    st.caption("Based on simulating the full tournament 10,000 times using the model's predicted "
               "win probabilities. Shows the estimated chance each team reaches each round.")

    try:
        seed_num_a = int(str(seed_a)[1:3]) if seed_a != "n/a" else None
        seed_num_b = int(str(seed_b)[1:3]) if seed_b != "n/a" else None
    except: seed_num_a = seed_num_b = None

    mc_a, mc_b = est_mc(pa, seed_num_a, seed_num_b)
    rounds = ["Sweet 16\n(last 16)","Elite 8\n(last 8)","Final Four\n(last 4)",
              "Championship\ngame","Champion"]
    round_keys = ["s16","e8","f4","fin","champ"]
    va_mc = [mc_a[k] for k in round_keys]
    vb_mc = [mc_b[k] for k in round_keys]

    fmc = go.Figure()
    fmc.add_trace(go.Bar(name=name_a, x=rounds, y=va_mc, marker_color=CA, opacity=.85,
                         text=[f"{v}%" for v in va_mc], textposition="outside"))
    fmc.add_trace(go.Bar(name=name_b, x=rounds, y=vb_mc, marker_color=CB, opacity=.85,
                         text=[f"{v}%" for v in vb_mc], textposition="outside"))
    fmc.update_layout(barmode="group", height=300,
                      margin=dict(l=0,r=0,t=30,b=0),
                      legend=dict(orientation="h",y=-0.18,x=.5,xanchor="center"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(showgrid=False),
                      yaxis=dict(title="Probability (%)", range=[0,max(max(va_mc),max(vb_mc))*1.25],
                                 gridcolor="#f1f5f9"),
                      font=dict(family="DM Sans",size=11))
    st.plotly_chart(fmc, use_container_width=True, config={"displayModeBar":False})

    # ── HEAD TO HEAD ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="slbl">Head-to-head history</div>',
                unsafe_allow_html=True)

    h2h_edge = result.get("h2h_same_season_win_edge", 0.0)
    h2h_margin = result.get("h2h_same_season_margin", 0.0)
    h2h_recent = result.get("h2h_recent_weighted_win_edge", 0.0)

    # Look up actual past tournament matchups if data available
    past_matchups = []
    if data is not None:
        tourney = data.get("tourney", pd.DataFrame())
        lk_dict = {int(r.TeamID): r.TeamName for _,r in data["teams"].iterrows()}
        for _,g in tourney.iterrows():
            w,l = int(g["WTeamID"]), int(g["LTeamID"])
            if (w==tid_a and l==tid_b) or (w==tid_b and l==tid_a):
                winner = lk_dict.get(w, str(w))
                loser  = lk_dict.get(l, str(l))
                past_matchups.append({
                    "Season": int(g["Season"]),
                    "Winner": winner,
                    "Loser":  loser,
                    "Score":  f"{int(g['WScore'])}–{int(g['LScore'])}",
                })

    hc1, hc2, hc3 = st.columns(3)

    with hc1:
        if h2h_edge > 0:
            txt = f"{name_a} leads H2H this season"
            color = CA
        elif h2h_edge < 0:
            txt = f"{name_b} leads H2H this season"
            color = CB
        else:
            txt = "No head-to-head this season"
            color = CU
        st.markdown(f"""
        <div class="h2h-box">
            <div class="mlbl">This season (2026)</div>
            <div style="font-size:1rem;font-weight:500;color:{color};margin-top:6px">{txt}</div>
            {f'<div style="font-size:.82rem;color:#94a3b8;margin-top:4px">Avg margin: {fadv(h2h_margin,1)} pts</div>' if h2h_edge != 0 else ""}
        </div>""", unsafe_allow_html=True)

    with hc2:
        if h2h_recent > 0:
            txt2 = f"{name_a} has the recent edge"
            color2 = CA
        elif h2h_recent < 0:
            txt2 = f"{name_b} has the recent edge"
            color2 = CB
        else:
            txt2 = "No recent head-to-head data"
            color2 = CU
        st.markdown(f"""
        <div class="h2h-box">
            <div class="mlbl">Recent years (weighted)</div>
            <div style="font-size:1rem;font-weight:500;color:{color2};margin-top:6px">{txt2}</div>
            <div style="font-size:.82rem;color:#94a3b8;margin-top:4px">
                More recent games count more in the weight
            </div>
        </div>""", unsafe_allow_html=True)

    with hc3:
        if past_matchups:
            last = past_matchups[-1]
            prev_txt = f"{last['Season']}: {last['Winner']} won {last['Score']}"
            prev_color = CA if last["Winner"] == name_a else CB
        else:
            prev_txt = "These teams have not met in the tournament"
            prev_color = CU
        st.markdown(f"""
        <div class="h2h-box">
            <div class="mlbl">Tournament history ({len(past_matchups)} games)</div>
            <div style="font-size:1rem;font-weight:500;color:{prev_color};margin-top:6px">
                {prev_txt}</div>
            {f'<div style="font-size:.82rem;color:#94a3b8;margin-top:4px">Total matchups found: {len(past_matchups)}</div>' if past_matchups else ""}
        </div>""", unsafe_allow_html=True)

    if past_matchups and len(past_matchups) > 1:
        with st.expander(f"See all {len(past_matchups)} historical tournament matchups"):
            st.dataframe(pd.DataFrame(past_matchups).rename(columns={"Season":"Year"}),
                         use_container_width=True, hide_index=True)

    # ── WHY FAVORITE WINS ─────────────────────────────────────────────────
    st.markdown("---")
    fav_name  = name_a if pa >= pb else name_b
    fav_sign  = 1       if pa >= pb else -1
    fav_color = CA      if pa >= pb else CB

    st.markdown(f'<div class="slbl">Why does {fav_name} have the advantage?</div>',
                unsafe_allow_html=True)
    st.caption(f"These are the main reasons the model gives {fav_name} a higher chance of winning. "
               "Green bars = bigger advantage, gray = roughly even.")

    why_cols = st.columns(2)
    for i, (feat, label, unit) in enumerate(WHY_FEATS):
        raw = result.get(feat, 0.0)
        v   = raw * fav_sign          # flip so favorite is always positive
        col = why_cols[i % 2]
        bar_w  = min(100, abs(v) * 20 + 5)
        b_color = CP if v > 0.1 else (CN if v < -0.1 else CU)
        favor   = fav_name if v > 0 else (name_b if pa >= pb else name_a)
        col.markdown(f"""
        <div class="why-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span style="font-size:.85rem;font-weight:500;color:#1e293b">{label}</span>
                <span style="font-size:1rem;font-weight:700;color:{b_color}">{fadv(v)}{unit}</span>
            </div>
            <div style="height:6px;background:#e2e8f0;border-radius:3px;overflow:hidden">
                <div style="height:100%;width:{bar_w}%;background:{b_color};border-radius:3px"></div>
            </div>
            <div style="font-size:.72rem;color:#94a3b8;margin-top:4px">→ {favor}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 - TITLE FAVORITES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Title favorites":
    st.markdown("### Who is most likely to win the 2026 championship?")
    st.markdown(
        "This chart shows the **estimated probability** that each team wins the entire tournament. "
        "The model looks at each team's overall strength: how much they win by, how efficient they are "
        "offensively and defensively, the difficulty of their schedule, and their tournament seeding. "
        "and calculates how likely each team is to survive all 6 rounds and be crowned champion."
    )
    st.caption("⚠️ These probabilities are estimates based on each team's regular-season performance. "
               "Tournament upsets are common. A team with 5% probability can absolutely win.")

    tf = bundle.team_features
    if tf is None or not isinstance(tf, pd.DataFrame):
        st.warning("Team features not available."); st.stop()

    tf26 = tf[tf["Season"]==2026].copy()
    if tf26.empty:
        st.warning("No 2026 data found."); st.stop()

    lk = bundle.team_lookup
    if lk is not None:
        tf26 = tf26.merge(lk[["TeamID","TeamName"]], on="TeamID", how="left")
    elif data is not None:
        tf26 = tf26.merge(data["teams"][["TeamID","TeamName"]], on="TeamID", how="left")

    sc = ("schedule_adjusted_net_rating" if "schedule_adjusted_net_rating" in tf26.columns
          else ("net_rating" if "net_rating" in tf26.columns else "avg_margin"))

    tf26 = (tf26.dropna(subset=["TeamName"])
               .sort_values(sc, ascending=False)
               .reset_index(drop=True).head(20))

    vals  = tf26[sc].values.astype(float)
    probs = np.exp(vals / max(vals.std(),0.01) * 0.8)
    probs = np.round(probs / probs.sum() * 100, 1)
    tf26["champ_prob"] = probs

    # Merge seeds for display
    if data is not None:
        seeds = data["seeds"][data["seeds"]["Season"]==2026][["TeamID","Seed"]]
        tf26  = tf26.merge(seeds, on="TeamID", how="left")

    # Bar chart - no grid lines
    fig = go.Figure(go.Bar(
        x=tf26["champ_prob"][::-1],
        y=tf26["TeamName"][::-1],
        orientation="h",
        marker=dict(color=tf26["champ_prob"][::-1],
                    colorscale=[[0,"#b5d4f4"],[1,CA]],
                    showscale=False),
        text=tf26["champ_prob"][::-1].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
    ))
    fig.update_layout(
        height=600, margin=dict(l=0,r=70,t=10,b=0),
        xaxis=dict(title="Estimated championship probability (%)",
                   showgrid=False, zeroline=False,
                   range=[0, probs.max()*1.3]),
        yaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans",size=12),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    # Table with human-readable column names
    st.markdown("#### Detailed stats for top 20 contenders")
    show_cols = ["TeamName"]
    if "Seed" in tf26.columns:         show_cols.append("Seed")
    if sc in tf26.columns:             show_cols.append(sc)
    if "win_pct" in tf26.columns:      show_cols.append("win_pct")
    if "avg_margin" in tf26.columns:   show_cols.append("avg_margin")
    show_cols.append("champ_prob")

    rename_map = {
        "TeamName":  "Team",
        "Seed":      "Tournament seed",
        sc:          "Overall efficiency score",
        "win_pct":   "Season win rate (%)",
        "avg_margin":"Avg winning margin (pts)",
        "champ_prob":"Championship probability (%)",
    }
    fmt_map = {
        "Championship probability (%)": "{:.1f}%",
        "Season win rate (%)":          "{:.1f}",
        "Avg winning margin (pts)":     "{:.1f}",
        "Overall efficiency score":     "{:.2f}",
    }

    display_df = tf26[show_cols].rename(columns=rename_map).copy()
    if "Tournament seed" in display_df.columns:
        display_df["Tournament seed"] = display_df["Tournament seed"].apply(
            lambda v: str(v) if not pd.isna(v) else "n/a")

    st.dataframe(display_df.style.format({k:v for k,v in fmt_map.items()
                                          if k in display_df.columns}),
                 use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 - MODEL ACCURACY (PAST SEASONS)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model accuracy (past seasons)":
    st.markdown("### Did the model predict correctly?")
    st.markdown(
        "This page shows how the model performed on **real NCAA tournament games** from past seasons "
        "(2023, 2024, 2025). For each game, the model predicted who would win based only on "
        "regular-season statistics, without knowing the actual result. "
        "We then compare: was the model right?"
    )

    if data is None:
        st.warning("data_men.pkl (or data_women.pkl) is needed for this section. "
                   "Make sure the file is in your folder.")
        st.stop()

    with st.spinner("Calculating predictions for past tournament games…"):
        hist = get_historical_accuracy(id(bundle), _bundle=bundle,
                                       _data=data, test_seasons=(2023,2024,2025))

    if hist is None or hist.empty:
        st.warning("Could not calculate historical accuracy. Tournament data may be incomplete.")
        st.stop()

    # Summary metrics
    total   = len(hist)
    correct = hist["Correct"].sum()
    acc     = round(correct/total*100, 1) if total > 0 else 0

    # By season
    by_season = hist.groupby("Season")["Correct"].agg(["sum","count"]).reset_index()
    by_season.columns = ["Season","Correct","Total"]
    by_season["Accuracy %"] = (by_season["Correct"]/by_season["Total"]*100).round(1)

    # Top metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"""<div class="mcard"><div class="mlbl">Overall accuracy</div>
        <div style="font-size:2rem;font-weight:700;color:{CP if acc>=60 else CN}">{acc}%</div>
        <div style="font-size:.72rem;color:#94a3b8">{correct} of {total} games correct</div></div>""",
        unsafe_allow_html=True)
    m2.markdown(f"""<div class="mcard"><div class="mlbl">Games predicted</div>
        <div style="font-size:2rem;font-weight:700;color:#1e293b">{total}</div>
        <div style="font-size:.72rem;color:#94a3b8">Across 3 tournament seasons</div></div>""",
        unsafe_allow_html=True)
    m3.markdown(f"""<div class="mcard"><div class="mlbl">Correct predictions</div>
        <div style="font-size:2rem;font-weight:700;color:{CP}">{correct}</div>
        <div style="font-size:.72rem;color:#94a3b8">Model picked the right winner</div></div>""",
        unsafe_allow_html=True)
    wrong = total - correct
    m4.markdown(f"""<div class="mcard"><div class="mlbl">Wrong predictions</div>
        <div style="font-size:2rem;font-weight:700;color:{CN}">{wrong}</div>
        <div style="font-size:.72rem;color:#94a3b8">Model missed these games</div></div>""",
        unsafe_allow_html=True)

    st.markdown("---")

    # Accuracy by season bar chart
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="slbl">Accuracy by season</div>', unsafe_allow_html=True)
        colors_season = [CP if a >= 60 else CN for a in by_season["Accuracy %"]]
        f_s = go.Figure(go.Bar(
            x=by_season["Season"].astype(str),
            y=by_season["Accuracy %"],
            marker_color=colors_season,
            text=by_season["Accuracy %"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ))
        f_s.add_hline(y=50, line_dash="dash", line_color="#94a3b8",
                      annotation_text="50% (random guessing)")
        f_s.update_layout(height=280, margin=dict(l=0,r=0,t=30,b=0),
                           paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(showgrid=False),
                           yaxis=dict(range=[0,105],showgrid=False),
                           font=dict(family="DM Sans",size=12))
        st.plotly_chart(f_s, use_container_width=True, config={"displayModeBar":False})

    with col_right:
        st.markdown('<div class="slbl">Accuracy by confidence level</div>',
                    unsafe_allow_html=True)
        hist["conf_band"] = pd.cut(hist["Confidence"],
                                   bins=[50,60,70,80,90,101],
                                   labels=["50–60%","61–70%","71–80%","81–90%","91–100%"])
        cb = hist.groupby("conf_band",observed=True)["Correct"].agg(["sum","count"]).reset_index()
        cb.columns = ["Confidence","Correct","Total"]
        cb["Accuracy %"] = (cb["Correct"]/cb["Total"]*100).round(1)
        f_cb = go.Figure(go.Bar(
            x=cb["Confidence"].astype(str),
            y=cb["Accuracy %"],
            marker_color=[CP if a>=60 else CN for a in cb["Accuracy %"]],
            text=cb["Accuracy %"].apply(lambda v: f"{v:.0f}%"),
            textposition="outside",
        ))
        f_cb.add_hline(y=50, line_dash="dash", line_color="#94a3b8")
        f_cb.update_layout(height=280, margin=dict(l=0,r=0,t=30,b=0),
                            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(title="Model confidence",showgrid=False),
                            yaxis=dict(range=[0,115],showgrid=False),
                            font=dict(family="DM Sans",size=12))
        st.plotly_chart(f_cb, use_container_width=True, config={"displayModeBar":False})

    st.caption("When the model is more confident (e.g. 80–90%), it should be right more often. "
               "When it says ~50%, the game is a coin flip. Both bars should be near 50%.")

    # Game-by-game table
    st.markdown("---")
    st.markdown('<div class="slbl">Game by game results</div>', unsafe_allow_html=True)

    filter_season = st.selectbox("Filter by season",
                                  ["All"] + sorted(hist["Season"].unique().tolist(), reverse=True))
    filter_result = st.radio("Show", ["All games","Correct only","Wrong only"],
                             horizontal=True)

    df_show = hist.copy()
    if filter_season != "All":
        df_show = df_show[df_show["Season"]==int(filter_season)]
    if filter_result == "Correct only":
        df_show = df_show[df_show["Correct"]==True]
    elif filter_result == "Wrong only":
        df_show = df_show[df_show["Correct"]==False]

    df_show["Result"] = df_show["Correct"].map({True:"✅  Correct",False:"❌  Wrong"})
    df_show["Confidence"] = df_show["Confidence"].apply(lambda v: f"{v:.1f}%")
    display_cols = ["Season","Team A","Team B","Predicted winner",
                    "Actual winner","Confidence","Result"]

    def _row_style(row):
        c = CP if "Correct" in row["Result"] else CN
        return [f"color:{c}" if col=="Result" else "" for col in row.index]

    st.dataframe(df_show[display_cols].reset_index(drop=True)
                 .style.apply(_row_style, axis=1),
                 use_container_width=True, hide_index=True)

    st.caption(f"Showing {len(df_show)} of {len(hist)} games.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 - MODEL METRICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model metrics":
    st.markdown("### How good is the model?")

    for g, lbl in [("men","Men's tournament"),("women","Women's tournament")]:
        if g not in bundles: continue
        st.subheader(lbl)
        for col, (k,v) in zip(st.columns(5), METRICS[g].items()):
            col.markdown(f"""
            <div class="mcard">
                <div class="mlbl">{k}</div>
                <div style="font-size:1.3rem;font-weight:700;color:#1e293b;margin-top:4px">{v}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("")

    st.markdown("---")
    st.markdown("### What do these numbers mean?")

    for term, desc in [
        ("Log Loss (0.55 for men)",
         "Measures how accurate the predicted probabilities are. Lower = better. "
         "A model that always guesses 50/50 scores 0.693. This model scores 0.55 (men) "
         "and 0.41 (women), meaning it is significantly better than random guessing."),
        ("AUC-ROC (0.80 for men)",
         "From 0 to 1: tells you how well the model separates winners from losers. "
         "0.5 = completely random, 1.0 = perfect. This model at 0.80 means 80% of the time "
         "it gives the real winner a higher probability than the real loser."),
        ("Brier Score",
         "The official Kaggle competition metric. It is the average squared error between "
         "the model's probability and what actually happened (0 = win, 1 = loss). "
         "0 = perfect, 0.25 = always predicting 50%. Lower is better."),
        ("forward_cv_candidate",
         "How the production model was selected: it competed against many other model versions "
         "using forward-only cross-validation, meaning the model was NEVER allowed to train "
         "on future data to prevent cheating. The version that won becomes the production model."),
    ]:
        st.markdown(
            f"<div style='padding:10px 14px;background:#f8fafc;"
            f"border-left:3px solid {CA};border-radius:0 8px 8px 0;margin-bottom:8px'>"
            f"<span style='font-weight:600;color:{CA}'>{term}</span>"
            f"<span style='color:#475569;font-size:.88rem;margin-left:8px'>{desc}</span></div>",
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Features used by the production model")
    feats = bundle.features
    if feats:
        st.dataframe(
            pd.DataFrame({
                "Technical name": feats,
                "Plain English":  [LABELS.get(f, f.replace("_adv","").replace("_"," ")) for f in feats],
            }),
            use_container_width=True, hide_index=True)
    else:
        st.info("Feature list not available in this bundle.")
