"""Generate the static prediction website at web/index.html.

Reads:
  - predictions/*.md
  - data/processed/matches.csv
  - data/processed/deliveries.csv
  - data/processed/features.csv
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "predictions"
WEB_DIR = ROOT / "web"
PROC = ROOT / "data" / "processed"

TEAM_INITIALS = {
    "Mumbai Indians": "MI", "Chennai Super Kings": "CSK",
    "Royal Challengers Bengaluru": "RCB", "Royal Challengers Bangalore": "RCB",
    "Kolkata Knight Riders": "KKR", "Sunrisers Hyderabad": "SRH",
    "Delhi Capitals": "DC", "Punjab Kings": "PBKS",
    "Rajasthan Royals": "RR", "Lucknow Super Giants": "LSG",
    "Gujarat Titans": "GT",
}

TEAM_COLORS = {
    "MI":  ("#004BA0", "#D1AB3E"),
    "CSK": ("#FBC02D", "#1565C0"),
    "RCB": ("#D81920", "#1A1A1A"),
    "KKR": ("#3A225D", "#F2C94C"),
    "SRH": ("#F26522", "#1A1A1A"),
    "DC":  ("#17449B", "#EF1B23"),
    "PBKS":("#D71920", "#A2A8AB"),
    "RR":  ("#E91E96", "#254AA5"),
    "LSG": ("#0057A2", "#F36F21"),
    "GT":  ("#1B2951", "#B7A26C"),
}


# ---------------------------------------------------------------------------
# Parsing & data assembly
# ---------------------------------------------------------------------------

def parse_prediction(md_path: Path) -> dict:
    text = md_path.read_text()
    def find(pat, default=""):
        m = re.search(pat, text, re.M)
        return m.group(1).strip() if m else default

    match = find(r"\*\*Match\*\*:\s*(.+)$")
    # Strip parentheticals like ("El Clasico") that break team-name lookups
    match = re.sub(r"\s*\([^)]*\)", "", match).strip()
    venue = find(r"\*\*Venue\*\*:\s*(.+)$")
    teams = match.split(" vs ") if " vs " in match else ["Team 1", "Team 2"]
    team1, team2 = (teams + ["",""])[:2]
    team1, team2 = team1.strip(), team2.strip()

    winner = find(r"\*\*Winner\*\*\s*\|\s*(.+?)\s*\|")
    probs = re.findall(r"P\(.+?\)\s*\|\s*([\d.]+)\s*\|", text)
    p1 = probs[0] if probs else "0.5"
    p2 = probs[1] if len(probs) > 1 else f"{1 - float(p1):.2f}"

    score = find(r"Predicted 1st-innings total\s*\|\s*(.+?)\s*\|")
    sixes = find(r"Predicted total sixes \(combined\)\s*\|\s*(.+?)\s*\|")
    fours = find(r"Predicted total fours \(combined\)\s*\|\s*(.+?)\s*\|")

    pitch = {
        "first_innings": find(r"Avg 1st innings:\s*(\d+)"),
        "runs_per_wicket": find(r"Runs/wicket:\s*([\d.]+)"),
        "sixes": find(r"Avg sixes/match:\s*([\d.]+)"),
        "fours": find(r"Avg fours/match:\s*([\d.]+)"),
        "wickets": find(r"Avg wickets/match:\s*([\d.]+)"),
    }

    actual_winner = find(r"\| Winner \| (.+?) \|")
    if actual_winner.lower() in ("tbd", ""):
        actual_winner = ""

    date_str = re.search(r"(\d{4}-\d{2}-\d{2})", md_path.stem)
    date = date_str.group(1) if date_str else ""

    return {
        "filename": md_path.name, "date": date,
        "team1": team1, "team2": team2,
        "team1_init": TEAM_INITIALS.get(team1, team1[:3].upper()),
        "team2_init": TEAM_INITIALS.get(team2, team2[:3].upper()),
        "venue": venue, "winner": winner, "p1": p1, "p2": p2,
        "predicted_score": score, "predicted_sixes": sixes, "predicted_fours": fours,
        "pitch": pitch, "actual_winner": actual_winner,
    }


def gather_predictions() -> list[dict]:
    if not PRED_DIR.exists():
        return []
    return [parse_prediction(p) for p in sorted(PRED_DIR.glob("*.md"))]


def team_form_2026(matches: pd.DataFrame, team: str, n: int = 5) -> list[dict]:
    games = matches[(matches["season"].astype(str) == "2026") &
                    ((matches["team1"] == team) | (matches["team2"] == team))]
    games = games.sort_values("date").tail(n)
    out = []
    for _, g in games.iterrows():
        opp = g["team2"] if g["team1"] == team else g["team1"]
        won = g["winner"] == team
        out.append({"won": bool(won), "opp": TEAM_INITIALS.get(opp, opp[:3].upper())})
    return out


def team_scoring_2026(deliveries: pd.DataFrame, matches: pd.DataFrame, team: str) -> dict:
    m26 = matches[matches["season"].astype(str) == "2026"]
    d26 = deliveries[deliveries["match_id"].isin(m26["match_id"])]
    bat = d26[d26["batting_team"] == team]
    if bat.empty:
        return {"scored": 0, "conceded": 0, "sixes": 0, "fours": 0}
    inn = bat.groupby("match_id").agg(
        runs=("runs_total","sum"),
        sixes=("runs_batter", lambda s: (s == 6).sum()),
        fours=("runs_batter", lambda s: (s == 4).sum()),
    )
    team_matches = m26[(m26["team1"] == team) | (m26["team2"] == team)]["match_id"]
    conceded = d26[(d26["match_id"].isin(team_matches)) & (d26["batting_team"] != team)]
    return {
        "scored": float(inn["runs"].mean()),
        "conceded": float(conceded.groupby("match_id")["runs_total"].sum().mean()),
        "sixes": float(inn["sixes"].mean()),
        "fours": float(inn["fours"].mean()),
    }


def top_batters_2026(deliveries: pd.DataFrame, matches: pd.DataFrame, team: str, n: int = 3) -> list[dict]:
    m26 = matches[matches["season"].astype(str) == "2026"]
    d26 = deliveries[deliveries["match_id"].isin(m26["match_id"])]
    bat = d26[d26["batting_team"] == team]
    if bat.empty:
        return []
    g = bat.groupby("batter").agg(
        runs=("runs_batter","sum"),
        balls=("runs_batter","count"),
        innings=("match_id","nunique"),
    )
    g = g[g["innings"] >= 3]
    g["sr"] = (g["runs"] / g["balls"] * 100).round(1)
    g = g.sort_values("runs", ascending=False).head(n).reset_index()
    return g.to_dict("records")


def top_bowlers_2026(deliveries: pd.DataFrame, matches: pd.DataFrame, team: str, n: int = 3) -> list[dict]:
    m26 = matches[matches["season"].astype(str) == "2026"]
    d26 = deliveries[deliveries["match_id"].isin(m26["match_id"])]
    team_matches = m26[(m26["team1"] == team) | (m26["team2"] == team)]["match_id"]
    bowled = d26[(d26["match_id"].isin(team_matches)) & (d26["batting_team"] != team)]
    if bowled.empty:
        return []
    g = bowled.groupby("bowler").agg(
        runs=("runs_total","sum"),
        balls=("runs_total","count"),
        wickets=("wicket","sum"),
        innings=("match_id","nunique"),
    )
    g = g[g["innings"] >= 3]
    g["er"] = (g["runs"] / (g["balls"]/6)).round(2)
    g = g.sort_values(["wickets","er"], ascending=[False, True]).head(n).reset_index()
    return g.to_dict("records")


def h2h_2026(matches: pd.DataFrame, t1: str, t2: str) -> dict:
    m26 = matches[matches["season"].astype(str) == "2026"]
    games = m26[((m26["team1"] == t1) & (m26["team2"] == t2)) |
                ((m26["team1"] == t2) & (m26["team2"] == t1))]
    if games.empty:
        return {"played": 0, "t1_wins": 0, "t2_wins": 0}
    t1w = (games["winner"] == t1).sum()
    return {"played": int(len(games)), "t1_wins": int(t1w), "t2_wins": int(len(games) - t1w)}


def compute_backtest(n_test: int = 10) -> dict:
    df = pd.read_csv(PROC / "features.csv", parse_dates=["date"])
    df = df.dropna(subset=["target"]).sort_values("date").reset_index(drop=True)
    matches = pd.read_csv(PROC / "matches.csv", parse_dates=["date"])

    NUMERIC = ["t1_form","t2_form","t1_avg_scored","t2_avg_scored","t1_avg_conceded",
               "t2_avg_conceded","t1_venue_winrate","t2_venue_winrate","h2h_t1_winrate",
               "t1_toss_winrate_10","t2_toss_winrate_10","toss_winner_is_t1","toss_decision_bat"]
    CATS = ["team1","team2","venue"]
    for c in CATS:
        le = LabelEncoder(); le.fit(df[c].astype(str))
        df[c+"_enc"] = le.transform(df[c].astype(str))
    feat_cols = NUMERIC + [c+"_enc" for c in CATS]

    completed = df[df["season"].astype(str) == "2026"].sort_values("date")
    test = completed.tail(n_test)

    rows, correct = [], 0
    for _, r in test.iterrows():
        train = df[df["date"] < r["date"]]
        m = GradientBoostingClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42)
        m.fit(train[feat_cols], train["target"])
        X = pd.DataFrame([r[feat_cols].values], columns=feat_cols)
        p1 = float(m.predict_proba(X)[0,1])
        pred = r["team1"] if p1 >= 0.5 else r["team2"]
        actual = matches.loc[matches["match_id"]==r["match_id"], "winner"].iloc[0]
        ok = pred == actual
        if ok: correct += 1
        rows.append({
            "date": str(r["date"].date()),
            "team1": r["team1"], "team2": r["team2"],
            "team1_init": TEAM_INITIALS.get(r["team1"], r["team1"][:3].upper()),
            "team2_init": TEAM_INITIALS.get(r["team2"], r["team2"][:3].upper()),
            "predicted": pred, "actual": actual, "p1": round(p1, 2), "ok": ok,
        })
    return {"accuracy": correct/len(test), "n": len(test), "correct": correct, "rows": rows}


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def kpi_strip(predictions, backtest):
    today = predictions[-1] if predictions else None
    accuracy_pct = int(backtest["accuracy"] * 100)
    items = [
        ("Predictions", str(len(predictions)), "Cards saved"),
        ("Hit rate", f"{accuracy_pct}%", f"Last {backtest['n']} matches"),
        ("Seasons", "3+", "IPL 2023 → 2026"),
        ("Today", today["winner"] if today and today.get("winner") else "—",
         today["team1_init"] + " vs " + today["team2_init"] if today else ""),
    ]
    out = []
    for i, (label, value, sub) in enumerate(items):
        out.append(f"""
        <div class="kpi-card glass-soft rounded-2xl p-5 anim-slide" style="animation-delay:{i*120}ms">
          <div class="text-[10px] uppercase tracking-[0.25em] text-white/40">{label}</div>
          <div class="font-display text-3xl font-bold mt-2 truncate">{value}</div>
          <div class="text-xs text-white/50 mt-1 truncate">{sub}</div>
        </div>""")
    return "\n".join(out)


def form_chips(form: list[dict]) -> str:
    if not form:
        return '<div class="text-white/30 text-xs">—</div>'
    chips = []
    for f in form:
        cls = "bg-mint/15 text-mint border-mint/30" if f["won"] else "bg-crimson/15 text-crimson border-crimson/30"
        letter = "W" if f["won"] else "L"
        chips.append(f'<div class="form-chip {cls} border rounded-lg w-9 h-9 flex flex-col items-center justify-center"><span class="font-display font-bold leading-none">{letter}</span><span class="text-[8px] opacity-70 mt-0.5">{f["opp"]}</span></div>')
    return f'<div class="flex gap-1.5 flex-wrap justify-center">{"".join(chips)}</div>'


def probability_donut(p1: float, t1_init: str, t2_init: str, t1_color: str, t2_color: str) -> str:
    # SVG donut chart
    radius = 72
    circumference = 2 * math.pi * radius
    p1_arc = circumference * p1
    return f"""
    <div class="relative w-52 h-52">
      <svg class="w-full h-full -rotate-90" viewBox="0 0 200 200">
        <defs>
          <linearGradient id="grad-t1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="{t1_color}"/><stop offset="100%" stop-color="#ffd17a"/>
          </linearGradient>
          <linearGradient id="grad-t2" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="rgba(255,255,255,0.18)"/><stop offset="100%" stop-color="rgba(255,255,255,0.08)"/>
          </linearGradient>
        </defs>
        <circle cx="100" cy="100" r="{radius}" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="14"/>
        <circle cx="100" cy="100" r="{radius}" fill="none" stroke="url(#grad-t2)" stroke-width="14"
                stroke-dasharray="{circumference}" stroke-dashoffset="{p1_arc}" stroke-linecap="round"/>
        <circle cx="100" cy="100" r="{radius}" fill="none" stroke="url(#grad-t1)" stroke-width="14"
                stroke-dasharray="{p1_arc} {circumference}" stroke-linecap="round"
                style="filter: drop-shadow(0 0 12px rgba(245,180,65,0.5))">
          <animate attributeName="stroke-dasharray" from="0 {circumference}" to="{p1_arc} {circumference}" dur="1.4s" fill="freeze" calcMode="spline" keySplines="0.2 0.8 0.2 1"/>
        </circle>
      </svg>
      <div class="absolute inset-0 flex flex-col items-center justify-center">
        <div class="text-[10px] uppercase tracking-widest text-white/40">Win prob</div>
        <div class="font-display font-bold text-4xl leading-none mt-1 gradient-text">{int(p1*100)}%</div>
        <div class="text-xs text-white/50 mt-1">{t1_init} vs {t2_init}</div>
      </div>
    </div>"""


def six_visual(n: int, color: str = "#f5b441") -> str:
    n = int(round(n))
    n_show = min(n, 26)
    balls = "".join([f'<div class="six-dot anim-fade" style="animation-delay:{i*40}ms; background: radial-gradient(circle at 30% 30%, {color}, #b8862e)"></div>' for i in range(n_show)])
    rest = f'<span class="text-white/40 text-xs ml-2">+{n - n_show}</span>' if n > n_show else ''
    return f'<div class="flex flex-wrap gap-1.5 items-center">{balls}{rest}</div>'


def boundary_visual(n: int, color: str) -> str:
    n = int(round(n))
    n_show = min(n, 36)
    bars = "".join([f'<span class="boundary-bar anim-fade" style="animation-delay:{i*30}ms; background:{color}"></span>' for i in range(n_show)])
    rest = f'<span class="text-white/40 text-xs ml-2">+{n - n_show}</span>' if n > n_show else ''
    return f'<div class="flex flex-wrap gap-[3px] items-end">{bars}{rest}</div>'


def pitch_heatmap(rpw: float, sixes: float) -> str:
    """Generate a 5x5 pitch heatmap based on runs per wicket and sixes/match."""
    cells = []
    intensity = max(0, min(1, (rpw - 18) / 22))  # 18 (bowling) -> 40 (very batting)
    six_intensity = max(0, min(1, sixes / 30))
    for i in range(5):
        for j in range(5):
            # heat increases towards center
            cx, cy = 2, 2
            d = ((i-cx)**2 + (j-cy)**2) ** 0.5
            heat = (1 - d/3.5) * intensity + six_intensity * 0.3
            heat = max(0, min(1, heat))
            r = int(245 * heat + 30 * (1-heat))
            g = int(180 * heat + 40 * (1-heat))
            b = int(70 * heat + 60 * (1-heat))
            cells.append(f'<div style="background:rgba({r},{g},{b},{0.25 + heat*0.55})" class="pitch-cell"></div>')
    return f'<div class="pitch-grid">{"".join(cells)}</div>'


def player_row(name: str, val1: str, val2: str) -> str:
    return f"""
    <div class="flex items-center justify-between py-1.5 border-b border-white/5 last:border-b-0">
      <div class="text-sm truncate">{name}</div>
      <div class="text-xs text-white/50 tabular-nums">{val1} <span class="text-white/30">·</span> {val2}</div>
    </div>"""


def featured_card(p: dict, matches: pd.DataFrame, deliveries: pd.DataFrame) -> str:
    if not p:
        return '<div class="glass rounded-3xl p-12 text-center text-white/40">No predictions saved yet.</div>'
    c1 = TEAM_COLORS.get(p["team1_init"], ("#444","#888"))
    c2 = TEAM_COLORS.get(p["team2_init"], ("#444","#888"))
    p1 = float(p["p1"])
    p2 = float(p["p2"])

    form1 = team_form_2026(matches, p["team1"])
    form2 = team_form_2026(matches, p["team2"])
    s1 = team_scoring_2026(deliveries, matches, p["team1"])
    s2 = team_scoring_2026(deliveries, matches, p["team2"])
    t1_bat = top_batters_2026(deliveries, matches, p["team1"])
    t2_bat = top_batters_2026(deliveries, matches, p["team2"])
    t1_bowl = top_bowlers_2026(deliveries, matches, p["team1"])
    t2_bowl = top_bowlers_2026(deliveries, matches, p["team2"])
    h2h = h2h_2026(matches, p["team1"], p["team2"])

    # H2H bar
    h2h_total = max(h2h["played"], 1)
    h2h_bar = f"""
    <div class="flex h-2 rounded-full overflow-hidden bg-white/5">
      <div style="width:{h2h['t1_wins']/h2h_total*100:.0f}%; background:linear-gradient(90deg,{c1[0]},{c1[1]})"></div>
      <div style="width:{h2h['t2_wins']/h2h_total*100:.0f}%; background:linear-gradient(90deg,{c2[1]},{c2[0]})"></div>
    </div>
    <div class="flex justify-between text-[10px] text-white/50 mt-1.5"><span>{p['team1_init']} {h2h['t1_wins']}</span><span class="text-white/30">{h2h['played']} games · 2026</span><span>{p['team2_init']} {h2h['t2_wins']}</span></div>"""

    # Score range visualization
    score_range = "—"
    range_match = re.search(r"(\d+)[–-](\d+)\s*\(range\s*(\d+)[–-](\d+)\)", p["predicted_score"])
    range_html = ""
    if range_match:
        lo_p, hi_p, lo, hi = (int(x) for x in range_match.groups())
        span = max(hi - lo, 1)
        marker_lo = (lo_p - lo) / span * 100
        marker_hi = (hi_p - lo) / span * 100
        range_html = f"""
        <div class="mt-4 relative h-12 mx-2">
          <div class="absolute left-0 right-0 top-7 h-1 rounded-full bg-white/5"></div>
          <div class="absolute top-7 h-1 rounded-full" style="left:{marker_lo:.1f}%; width:{marker_hi-marker_lo:.1f}%; background:linear-gradient(90deg,{c1[0]},#ffd17a)"></div>
          <div class="absolute top-9 text-[10px] text-white/40" style="left:0">{lo}</div>
          <div class="absolute top-9 text-[10px] text-white/40 right-0">{hi}</div>
          <div class="absolute top-0 text-[10px] gradient-text font-semibold whitespace-nowrap" style="left:{marker_lo:.1f}%; transform: translateX(-50%)">{lo_p}</div>
          <div class="absolute top-0 text-[10px] gradient-text font-semibold whitespace-nowrap" style="left:{marker_hi:.1f}%; transform: translateX(-50%)">{hi_p}</div>
        </div>"""

    # Top players HTML
    def render_players(items, key1, key2, label1="r", label2="sr"):
        if not items:
            return '<div class="text-white/30 text-xs py-2">No data</div>'
        return "".join([
            player_row(it.get("batter") or it.get("bowler"),
                       f"{int(it[key1])}{label1}", f"{it[key2]}{label2}")
            for it in items
        ])

    pitch_class = "Batting paradise" if float(p["pitch"]["runs_per_wicket"] or 0) > 30 else \
                  "Balanced track" if float(p["pitch"]["runs_per_wicket"] or 0) > 22 else "Bowler-friendly"

    sixes_n = int(re.search(r"(\d+)", p["predicted_sixes"]).group(1)) if p["predicted_sixes"] else 0
    fours_n = int(re.search(r"(\d+)", p["predicted_fours"]).group(1)) if p["predicted_fours"] else 0

    return f"""
    <div class="anim-slide grid lg:grid-cols-[1.3fr_1fr] gap-6">
      <!-- LEFT: main battle card -->
      <div class="featured-card relative rounded-3xl border border-white/10 p-6 sm:p-9"
           style="background:
              radial-gradient(ellipse at 0% 0%, {c1[0]}40, transparent 55%),
              radial-gradient(ellipse at 100% 100%, {c2[0]}40, transparent 55%),
              linear-gradient(180deg, #0a0d18, #06080f);">
        <div class="absolute inset-0 -z-0 pointer-events-none opacity-50"
             style="background-image: radial-gradient(rgba(255,255,255,0.08) 1px, transparent 1px); background-size: 22px 22px;"></div>

        <!-- Tag row -->
        <div class="relative flex items-center justify-between">
          <div class="flex items-center gap-2 text-[11px] uppercase tracking-[0.25em] text-white/60">
            <span class="w-1.5 h-1.5 rounded-full bg-accent ring-pulse"></span>
            Live · {p['date']}
          </div>
          <div class="text-[11px] uppercase tracking-[0.25em] text-white/40">Match preview</div>
        </div>

        <!-- VS layout -->
        <div class="relative mt-6 grid grid-cols-1 sm:grid-cols-[1fr_auto_1fr] items-center gap-6">
          <!-- Team 1 -->
          <div class="text-center min-w-0">
            <div class="team-orb mx-auto" style="--c1:{c1[0]}; --c2:{c1[1]}">
              <span>{p['team1_init']}</span>
            </div>
            <div class="font-display text-sm sm:text-base mt-3 truncate px-2">{p['team1']}</div>
            <div class="text-xs text-white/50 mt-1">P · <span class="font-semibold gradient-text">{p['p1']}</span></div>
            <div class="mt-3 flex justify-center">{form_chips(form1)}</div>
          </div>

          <!-- Probability donut -->
          <div class="flex items-center justify-center order-first sm:order-none">
            {probability_donut(p1, p['team1_init'], p['team2_init'], c1[0], c2[0])}
          </div>

          <!-- Team 2 -->
          <div class="text-center min-w-0">
            <div class="team-orb mx-auto" style="--c1:{c2[0]}; --c2:{c2[1]}">
              <span>{p['team2_init']}</span>
            </div>
            <div class="font-display text-sm sm:text-base mt-3 truncate px-2">{p['team2']}</div>
            <div class="text-xs text-white/50 mt-1">P · <span class="font-semibold">{p['p2']}</span></div>
            <div class="mt-3 flex justify-center">{form_chips(form2)}</div>
          </div>
        </div>

        <!-- Verdict row -->
        <div class="mt-7 flex items-center justify-between gap-3 flex-wrap">
          <div>
            <div class="text-[11px] uppercase tracking-widest text-white/40">Predicted winner</div>
            <div class="font-display text-2xl mt-1"><span class="gradient-text">{p['winner']}</span></div>
          </div>
          <a href="https://www.espncricinfo.com/series/ipl-2026-1510719" target="_blank" rel="noopener"
             class="watch-btn group inline-flex items-center gap-2 px-4 py-2.5 rounded-full text-sm font-semibold transition"
             style="background: linear-gradient(135deg, var(--t1), var(--t2)); color: white;
                    box-shadow: 0 0 36px -10px color-mix(in srgb, var(--t1) 60%, transparent);">
            <span class="relative flex h-2 w-2">
              <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
              <span class="relative inline-flex rounded-full h-2 w-2 bg-white"></span>
            </span>
            Watch live
            <svg class="w-3.5 h-3.5 transition-transform group-hover:translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
          </a>
          <div class="text-right">
            <div class="text-[11px] uppercase tracking-widest text-white/40">Venue</div>
            <div class="text-xs text-white/70 mt-1 truncate max-w-xs">{p['venue']}</div>
          </div>
        </div>

        <div class="seam mt-6"></div>

        <!-- Predicted score range -->
        <div class="mt-5 px-4">
          <div class="flex items-center justify-between text-[11px] uppercase tracking-widest text-white/40 -mx-4">
            <span>1st-innings projection</span>
            <span class="text-white/60 normal-case tracking-normal text-xs">{p['predicted_score']}</span>
          </div>
          {range_html}
        </div>

        <!-- Boundaries -->
        <div class="mt-6 grid sm:grid-cols-2 gap-5">
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="text-[11px] uppercase tracking-widest text-white/40">Sixes (combined)</span>
              <span class="font-display text-lg gradient-text">{p['predicted_sixes']}</span>
            </div>
            {six_visual(sixes_n)}
          </div>
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="text-[11px] uppercase tracking-widest text-white/40">Fours (combined)</span>
              <span class="font-display text-lg gradient-text">{p['predicted_fours']}</span>
            </div>
            {boundary_visual(fours_n, "rgba(255,255,255,0.55)")}
          </div>
        </div>

        <!-- H2H bar -->
        <div class="mt-7">
          <div class="text-[11px] uppercase tracking-widest text-white/40 mb-2">Head-to-head · IPL 2026</div>
          {h2h_bar}
        </div>
      </div>

      <!-- RIGHT: stacked panels -->
      <div class="space-y-4">
        <!-- Pitch -->
        <div class="glass rounded-3xl p-6">
          <div class="flex items-center justify-between">
            <div class="text-[11px] uppercase tracking-widest text-white/40">Pitch read</div>
            <div class="text-[10px] px-2 py-0.5 rounded-full bg-accent/15 text-accent">{pitch_class}</div>
          </div>
          <div class="mt-4 grid grid-cols-[1fr_auto] items-center gap-5">
            <div class="grid grid-cols-2 gap-3 text-sm">
              <div><div class="text-white/40 text-[10px] uppercase tracking-widest">1st inn</div><div class="font-display text-2xl">{p['pitch']['first_innings']}</div></div>
              <div><div class="text-white/40 text-[10px] uppercase tracking-widest">R/wkt</div><div class="font-display text-2xl">{p['pitch']['runs_per_wicket']}</div></div>
              <div><div class="text-white/40 text-[10px] uppercase tracking-widest">6s/match</div><div class="font-display text-2xl">{p['pitch']['sixes']}</div></div>
              <div><div class="text-white/40 text-[10px] uppercase tracking-widest">4s/match</div><div class="font-display text-2xl">{p['pitch']['fours']}</div></div>
            </div>
            <div>{pitch_heatmap(float(p['pitch']['runs_per_wicket'] or 22), float(p['pitch']['sixes'] or 14))}</div>
          </div>
        </div>

        <!-- Team comparison -->
        <div class="glass rounded-3xl p-6">
          <div class="text-[11px] uppercase tracking-widest text-white/40 mb-4">Form · IPL 2026</div>
          <div class="space-y-4 text-sm">
            {compare_row("Avg scored", s1["scored"], s2["scored"], c1[0], c2[0], suffix="")}
            {compare_row("Avg conceded", s1["conceded"], s2["conceded"], c1[0], c2[0], suffix="", invert=True)}
            {compare_row("Avg sixes (inn)", s1["sixes"], s2["sixes"], c1[0], c2[0])}
            {compare_row("Avg fours (inn)", s1["fours"], s2["fours"], c1[0], c2[0])}
          </div>
        </div>

        <!-- Top players -->
        <div class="glass rounded-3xl p-6">
          <div class="text-[11px] uppercase tracking-widest text-white/40 mb-3">Top performers · 2026</div>
          <div class="grid grid-cols-2 gap-5 text-sm">
            <div>
              <div class="text-[10px] uppercase tracking-widest mb-2" style="color: color-mix(in srgb, {c1[0]} 65%, #ffd17a);">{p['team1_init']} batters</div>
              {render_players(t1_bat, "runs", "sr", " r", " sr")}
            </div>
            <div>
              <div class="text-[10px] uppercase tracking-widest mb-2" style="color: color-mix(in srgb, {c2[0]} 65%, #ffd17a);">{p['team2_init']} batters</div>
              {render_players(t2_bat, "runs", "sr", " r", " sr")}
            </div>
            <div>
              <div class="text-[10px] uppercase tracking-widest mb-2 mt-2" style="color: color-mix(in srgb, {c1[0]} 65%, #ffd17a);">{p['team1_init']} bowlers</div>
              {render_players(t1_bowl, "wickets", "er", "w", " er")}
            </div>
            <div>
              <div class="text-[10px] uppercase tracking-widest mb-2 mt-2" style="color: color-mix(in srgb, {c2[0]} 65%, #ffd17a);">{p['team2_init']} bowlers</div>
              {render_players(t2_bowl, "wickets", "er", "w", " er")}
            </div>
          </div>
        </div>
      </div>
    </div>"""


def compare_row(label: str, v1: float, v2: float, c1: str, c2: str, suffix: str = "", invert: bool = False) -> str:
    mx = max(v1, v2, 1)
    w1 = v1 / mx * 100
    w2 = v2 / mx * 100
    leader_t1 = (v1 > v2) ^ invert
    return f"""
    <div>
      <div class="flex items-center justify-between text-[11px] text-white/50 mb-1.5">
        <span class="tabular-nums {'text-mint' if leader_t1 else ''}">{v1:.1f}{suffix}</span>
        <span class="uppercase tracking-widest text-[10px] text-white/40">{label}</span>
        <span class="tabular-nums {'text-mint' if not leader_t1 else ''}">{v2:.1f}{suffix}</span>
      </div>
      <div class="grid grid-cols-2 gap-1">
        <div class="h-1.5 rounded-full bg-white/5 overflow-hidden flex justify-end"><div style="width:{w1}%; background:linear-gradient(90deg,transparent,{c1})" class="h-full"></div></div>
        <div class="h-1.5 rounded-full bg-white/5 overflow-hidden"><div style="width:{w2}%; background:linear-gradient(90deg,{c2},transparent)" class="h-full"></div></div>
      </div>
    </div>"""


def streak_grid(rows: list[dict]) -> str:
    cells = []
    for r in rows:
        cls = "bg-mint/80" if r["ok"] else "bg-crimson/70"
        title = f"{r['date']} · {r['team1_init']} vs {r['team2_init']}: predicted {r['predicted'][:3].upper()}, actual {r['actual'][:3].upper()}"
        cells.append(f'<div class="streak-cell {cls}" title="{title}"></div>')
    return f'<div class="flex gap-1 flex-wrap">{"".join(cells)}</div>'


def backtest_table(rows: list[dict]) -> str:
    out = []
    for m in reversed(rows):
        cls = "text-mint" if m["ok"] else "text-crimson"
        sym = "✓" if m["ok"] else "✗"
        bg = "before:bg-mint/40" if m["ok"] else "before:bg-crimson/40"
        c1c = TEAM_COLORS.get(m["team1_init"], ("#888","#888"))
        c2c = TEAM_COLORS.get(m["team2_init"], ("#888","#888"))
        out.append(f"""
        <tr class="border-b border-white/5 hover:bg-white/[0.03] transition relative before:content-[''] before:absolute before:left-0 before:top-1/2 before:-translate-y-1/2 before:w-0.5 before:h-6 {bg}">
          <td class="p-4 text-white/60 tabular-nums">{m['date']}</td>
          <td class="p-4">
            <span class="inline-flex items-center gap-2">
              <span class="px-2 py-0.5 rounded-md font-bold text-white text-xs"
                    style="background: linear-gradient(135deg, {c1c[0]}, {c1c[1]}); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.15);">{m['team1_init']}</span>
              <span class="text-white/40 text-xs">vs</span>
              <span class="px-2 py-0.5 rounded-md font-bold text-white text-xs"
                    style="background: linear-gradient(135deg, {c2c[0]}, {c2c[1]}); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.15);">{m['team2_init']}</span>
            </span>
          </td>
          <td class="p-4 text-white/80 tabular-nums">{m['p1']:.2f}</td>
          <td class="p-4 text-white font-semibold">{TEAM_INITIALS.get(m['predicted'], m['predicted'])}</td>
          <td class="p-4 text-white/85 font-semibold">{TEAM_INITIALS.get(m['actual'], m['actual'])}</td>
          <td class="p-4 text-center text-xl {cls}">{sym}</td>
        </tr>""")
    return "\n".join(out)


def archive_cards(predictions: list[dict]) -> str:
    if not predictions:
        return '<div class="text-white/40 text-sm">No saved predictions yet.</div>'
    out = []
    for p in reversed(predictions):
        c1 = TEAM_COLORS.get(p["team1_init"], ("#444","#888"))
        c2 = TEAM_COLORS.get(p["team2_init"], ("#444","#888"))
        actual_html = ""
        if p.get("actual_winner"):
            ok = p["actual_winner"] == p["winner"]
            badge = "bg-mint/20 text-mint" if ok else "bg-crimson/20 text-crimson"
            actual_html = f'<span class="px-2 py-0.5 rounded-full text-[10px] {badge}">Actual · {p["actual_winner"]}</span>'
        out.append(f"""
        <details class="archive-card glass rounded-3xl p-6 group">
          <summary class="cursor-pointer flex items-center justify-between list-none gap-4">
            <div class="flex items-center gap-4 min-w-0">
              <div class="flex -space-x-2 shrink-0">
                <div class="w-10 h-10 rounded-full flex items-center justify-center font-display font-bold text-sm ring-2 ring-ink-950"
                     style="background: linear-gradient(135deg,{c1[0]},{c1[1]})">{p['team1_init']}</div>
                <div class="w-10 h-10 rounded-full flex items-center justify-center font-display font-bold text-sm ring-2 ring-ink-950"
                     style="background: linear-gradient(135deg,{c2[0]},{c2[1]})">{p['team2_init']}</div>
              </div>
              <div class="min-w-0">
                <div class="text-xs text-white/40">{p['date']}</div>
                <div class="font-display text-base font-semibold mt-0.5">{p['team1_init']} vs {p['team2_init']}</div>
                <div class="text-xs text-white/50 mt-0.5 truncate max-w-xs">{p['venue']}</div>
              </div>
            </div>
            <div class="text-right shrink-0">
              <div class="font-display text-base font-semibold gradient-text">{p['winner'] or '—'}</div>
              <div class="text-[10px] text-white/40 mt-0.5">P {p['p1']} / {p['p2']}</div>
              <div class="mt-1">{actual_html}</div>
            </div>
          </summary>
          <div class="mt-5 pt-5 border-t border-white/5 grid grid-cols-3 gap-3 text-center text-sm">
            <div><div class="text-white/40 text-[10px] uppercase tracking-widest">Score</div><div class="font-display text-base mt-1">{p['predicted_score']}</div></div>
            <div><div class="text-white/40 text-[10px] uppercase tracking-widest">Sixes</div><div class="font-display text-base mt-1">{p['predicted_sixes']}</div></div>
            <div><div class="text-white/40 text-[10px] uppercase tracking-widest">Fours</div><div class="font-display text-base mt-1">{p['predicted_fours']}</div></div>
          </div>
        </details>""")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="en" class="dark" style="--t1:__T1_COLOR__; --t1-2:__T1_COLOR2__; --t2:__T2_COLOR__; --t2-2:__T2_COLOR2__;">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Pitchwise · __MATCH_TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com"></script>
<script>
  tailwind.config = {
    theme: {
      extend: {
        fontFamily: { sans: ['Inter','sans-serif'], display: ['Space Grotesk','sans-serif'] },
        colors: {
          ink: { 950:'#04060c', 900:'#070a14', 850:'#0b0e1a', 800:'#13161f', 700:'#1c2030' },
          accent: { DEFAULT:'#f5b441', glow:'#ffd17a' },
          mint: '#7dd3a3', crimson:'#ff5d6c',
        },
      }
    }
  }
</script>
<style>
  body { background:
    radial-gradient(ellipse 80% 50% at 20% 0%, color-mix(in srgb, var(--t1) 25%, transparent) 0%, transparent 60%),
    radial-gradient(ellipse 80% 50% at 80% 0%, color-mix(in srgb, var(--t2) 25%, transparent) 0%, transparent 60%),
    radial-gradient(ellipse at top, #0c0f1c 0%, #04060c 60%);
    overflow-x:hidden; min-height: 100vh; cursor: default;
  }
  /* Floating cricket ball that drifts */
  @media (hover: none) { .float-ball { display: none; } }
  .float-ball {
    position: fixed; pointer-events: none; z-index: 0;
    width: 120px; height: 120px; border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, var(--t1-2), var(--t1) 60%, var(--t2) 100%);
    filter: blur(0.5px) drop-shadow(0 20px 40px color-mix(in srgb, var(--t1) 40%, transparent));
    opacity: 0.18; animation: drift 20s ease-in-out infinite;
  }
  .float-ball::before, .float-ball::after { content:''; position:absolute; left:8%; right:8%; height:1.5px;
    background: rgba(255,255,255,0.5); border-radius:1px; opacity:0.6; }
  .float-ball::before { top: 32%; }
  .float-ball::after { bottom: 32%; }
  @keyframes drift {
    0%,100% { transform: translate(80vw, 30vh) rotate(0deg); }
    50%     { transform: translate(20vw, 60vh) rotate(180deg); }
  }
  .float-ball.b2 { width: 80px; height: 80px; opacity: 0.12; animation-duration: 28s; animation-delay: -7s; }

  /* Confetti on the winner pill */
  .winner-pill { position: relative; overflow: hidden; }
  .winner-pill::after {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(circle at 50% 50%, color-mix(in srgb, var(--t1) 40%, transparent), transparent 70%);
    opacity: 0; animation: pop 4s ease-in-out infinite;
  }
  @keyframes pop { 0%,100% { opacity: 0; transform: scale(.9);} 50% { opacity: 1; transform: scale(1.1);} }

  /* Type animation */
  .typing { display: inline-block; overflow: hidden; white-space: nowrap;
            border-right: 2px solid currentColor; animation: typewrite 2.4s steps(20) 0.4s 1 normal both, blink 0.8s steps(2) infinite; }
  @keyframes typewrite { from { width: 0 } to { width: 100% } }
  @keyframes blink { 50% { border-color: transparent } }

  /* Tilt cards */
  .tilt { transform-style: preserve-3d; perspective: 1000px; transition: transform .3s ease; }
  .tilt:hover { transform: translateY(-4px); }

  /* Pulsing scoreboard light */
  .live-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
              background: var(--t1); box-shadow: 0 0 0 0 var(--t1); animation: live 1.5s ease-in-out infinite; }
  @keyframes live { 0%,100% { box-shadow: 0 0 0 0 var(--t1) } 70% { box-shadow: 0 0 0 8px transparent } }
  .glass { backdrop-filter: blur(22px) saturate(125%); background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.02)); border: 1px solid rgba(255,255,255,0.08); }
  .glass-soft { background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.015)); border: 1px solid rgba(255,255,255,0.06); }
  .grain::before { content:''; position:fixed; inset:0; background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence baseFrequency='0.92' numOctaves='3' /%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.06'/%3E%3C/svg%3E"); pointer-events:none; z-index:1; }
  .seam { background-image: linear-gradient(90deg, transparent 50%, color-mix(in srgb, var(--t1) 60%, transparent) 50%); background-size: 8px 1px; height:1px; }
  /* Force a bright midpoint so text never goes too dark on the deep bg */
  .gradient-text { background: linear-gradient(135deg, #ffd17a 0%, color-mix(in srgb, var(--t1) 65%, #ffd17a) 50%, #ff8e72 100%); -webkit-background-clip: text; background-clip: text; color: transparent; text-shadow: 0 0 24px rgba(255,209,122,0.15); }
  .gradient-text-mix { background: linear-gradient(135deg, color-mix(in srgb, var(--t1) 60%, #ffd17a) 0%, color-mix(in srgb, var(--t2) 60%, #ff8e72) 100%); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .anim-fade { animation: fade .8s ease both; }
  @keyframes fade { from { opacity:0; transform: translateY(8px) } to { opacity:1; transform: none } }
  .anim-slide { animation: slide 1s cubic-bezier(.2,.8,.2,1) both; }
  @keyframes slide { from { opacity:0; transform: translateY(28px) } to { opacity:1; transform: none } }
  .ring-pulse { animation: pulse 2.4s ease-in-out infinite; box-shadow: 0 0 0 0 color-mix(in srgb, var(--t1) 50%, transparent); }
  @keyframes pulse { 50% { box-shadow: 0 0 0 14px color-mix(in srgb, var(--t1) 0%, transparent) } }
  details summary::-webkit-details-marker { display:none; }

  .team-orb {
    width: 76px; height: 76px; border-radius: 50%;
    font-size: 1.3rem;
    background: radial-gradient(circle at 30% 30%, var(--c2), var(--c1));
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    color: white; letter-spacing: 0.02em; position: relative; isolation: isolate;
    box-shadow: 0 18px 40px -10px rgba(0,0,0,.6), inset 0 -8px 20px rgba(0,0,0,0.4), inset 0 6px 14px rgba(255,255,255,0.18);
    transition: transform .4s cubic-bezier(.2,.8,.2,1);
  }
  @media (min-width: 640px) {
    .team-orb { width: 96px; height: 96px; font-size: 1.6rem; }
  }
  .team-orb::before {
    content:''; position:absolute; inset:0; border-radius:50%;
    background: conic-gradient(from 240deg, transparent, rgba(255,255,255,0.18), transparent 60%);
    mask: radial-gradient(circle, transparent 56%, black 58%);
    animation: spin 8s linear infinite;
  }
  .team-orb:hover { transform: translateY(-3px) scale(1.04); }
  @keyframes spin { to { transform: rotate(360deg) } }

  .form-chip { transition: transform .25s ease; font-family: 'Space Grotesk'; }
  .form-chip:hover { transform: translateY(-2px); }

  .six-dot { width: 14px; height: 14px; border-radius: 50%; box-shadow: 0 2px 6px rgba(0,0,0,.3), inset 0 -2px 3px rgba(0,0,0,.4), inset 0 1px 1.5px rgba(255,255,255,.4); }
  .boundary-bar { display:inline-block; width: 4px; height: 18px; border-radius: 2px; opacity:.85; }

  .pitch-grid { display: grid; grid-template-columns: repeat(5, 14px); gap: 2px; padding: 4px; border-radius: 8px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); }
  .pitch-cell { width: 14px; height: 18px; border-radius: 2px; transition: opacity .3s ease; }

  .streak-cell { width: 22px; height: 22px; border-radius: 6px; transition: transform .2s ease; box-shadow: 0 2px 6px rgba(0,0,0,.3); cursor: default; }
  .streak-cell:hover { transform: scale(1.2); }

  .scroll-prog { position: fixed; left:0; top:0; height:2px; background: linear-gradient(90deg,#f5b441,#ff8e72); z-index: 60; width: 0; transition: width .12s linear; }
  .archive-card { transition: transform .35s ease, border-color .35s ease; }
  .archive-card[open], .archive-card:hover { border-color: rgba(245,180,65,0.25); transform: translateY(-2px); }

  .featured-card { transition: transform .5s cubic-bezier(.2,.8,.2,1); }
  .featured-card:hover { transform: translateY(-4px); }

  .marquee { display:flex; gap: 3rem; animation: marq 40s linear infinite; white-space: nowrap; }
  @keyframes marq { from { transform: translateX(0) } to { transform: translateX(-50%) } }

  .kpi-card { position: relative; overflow: hidden; transition: transform .4s ease, border-color .4s ease; }
  .kpi-card:hover { transform: translateY(-3px); border-color: rgba(245,180,65,0.22); }
  .kpi-card::after { content:''; position: absolute; inset: -1px; background: radial-gradient(160px 90px at var(--mx,50%) var(--my,50%), color-mix(in srgb, var(--t1) 35%, transparent), transparent 60%); pointer-events: none; opacity: 0; transition: opacity .3s ease; }
  .kpi-card:hover::after { opacity: 1; }
  .kpi-card:hover { border-color: color-mix(in srgb, var(--t1) 30%, transparent); }
  .archive-card[open], .archive-card:hover { border-color: color-mix(in srgb, var(--t1) 25%, transparent); }
  .accent-btn { background: linear-gradient(135deg, var(--t1), var(--t2)); color: white; box-shadow: 0 0 60px -12px color-mix(in srgb, var(--t1) 60%, transparent); }
  .accent-btn:hover { filter: brightness(1.15); }
  .team-stripe { height: 3px; background: linear-gradient(90deg, var(--t1) 0%, var(--t1) 50%, var(--t2) 50%, var(--t2) 100%); }

  /* number counter */
  .count-up { display: inline-block; min-width: 1.5em; }
</style>
</head>
<body class="font-sans text-white antialiased grain selection:bg-accent/30 selection:text-white">
<div class="scroll-prog" id="prog"></div>
<div class="float-ball" aria-hidden="true"></div>
<div class="float-ball b2" aria-hidden="true"></div>

<!-- NAV -->
<header class="sticky top-0 z-30 backdrop-blur-xl bg-ink-950/70 border-b border-white/5">
  <div class="team-stripe"></div>
  <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <div class="relative">
        <svg width="34" height="34" viewBox="0 0 60 60" class="ring-pulse rounded-full">
          <defs>
            <radialGradient id="ball" cx="35%" cy="35%">
              <stop offset="0%" stop-color="var(--t1-2)"/>
              <stop offset="60%" stop-color="var(--t1)"/>
              <stop offset="100%" stop-color="var(--t2)"/>
            </radialGradient>
          </defs>
          <circle cx="30" cy="30" r="26" fill="url(#ball)"/>
          <path d="M 14 22 Q 30 30 46 22" stroke="white" stroke-width="0.8" fill="none" stroke-dasharray="2 2"/>
          <path d="M 14 38 Q 30 30 46 38" stroke="white" stroke-width="0.8" fill="none" stroke-dasharray="2 2"/>
        </svg>
      </div>
      <div>
        <div class="font-display font-bold tracking-tight text-base">Pitchwise</div>
        <div class="text-[10px] text-white/40 -mt-0.5 uppercase tracking-widest">IPL 2026 · ML predictions</div>
      </div>
    </div>
    <nav class="hidden md:flex gap-7 text-sm text-white/70">
      <a href="#today" class="hover:text-white">Today</a>
      <a href="#track" class="hover:text-white">Track record</a>
      <a href="#archive" class="hover:text-white">Archive</a>
      <a href="#how" class="hover:text-white">How it works</a>
    </nav>
    <div class="hidden sm:flex items-center gap-2 text-xs text-white/60 px-3 py-1.5 rounded-full glass">
      <span class="w-1.5 h-1.5 rounded-full bg-mint animate-pulse"></span>
      Model live
    </div>
  </div>
</header>

<!-- HERO -->
<section class="relative overflow-hidden">
  <div class="absolute inset-0 -z-0 opacity-70" aria-hidden="true">
    <div class="absolute -top-40 left-[10%] w-[55vw] h-[55vw] rounded-full blur-3xl" style="background: color-mix(in srgb, var(--t1) 30%, transparent)"></div>
    <div class="absolute top-10 right-[5%] w-[50vw] h-[50vw] rounded-full blur-3xl" style="background: color-mix(in srgb, var(--t2) 30%, transparent)"></div>
    <div class="absolute bottom-0 left-1/3 w-[40vw] h-[40vw] rounded-full blur-3xl" style="background: color-mix(in srgb, var(--t1-2) 18%, transparent)"></div>
  </div>
  <div class="relative max-w-7xl mx-auto px-6 pt-20 pb-16">
    <div class="flex items-center gap-2 text-[11px] uppercase tracking-[0.3em] gradient-text-mix font-semibold anim-fade">
      <span class="seam w-10"></span> Match-day intelligence
    </div>
    <h1 class="mt-6 font-display text-4xl sm:text-6xl md:text-7xl lg:text-8xl font-bold leading-[0.95] max-w-5xl anim-fade">
      Reading the <span class="gradient-text typing">pitch</span><br/> before the toss.
    </h1>
    <p class="mt-6 max-w-2xl text-base sm:text-lg text-white/70 leading-relaxed anim-fade">
      An ML model trained on three seasons of ball-by-ball IPL data. Live form, venue tendencies,
      head-to-head and toss history feed every prediction. Honest accuracy — no vibes.
    </p>
    <div class="mt-10 flex flex-wrap gap-4 anim-fade">
      <a href="#today" class="accent-btn px-6 py-3 rounded-full font-semibold transition">See today's call →</a>
      <a href="#track" class="px-6 py-3 rounded-full glass text-white/90 hover:bg-white/10 transition">Track record</a>
    </div>

    <!-- KPI strip -->
    <div class="mt-16 grid grid-cols-2 md:grid-cols-4 gap-4">__KPI_STRIP__</div>
  </div>

  <!-- Marquee -->
  <div class="relative border-y border-white/5 py-5 overflow-hidden glass-soft">
    <div class="marquee text-white/30 font-display uppercase tracking-[0.3em] text-sm">
      <span>Mumbai Indians · Chennai Super Kings · Royal Challengers Bengaluru · Kolkata Knight Riders · Sunrisers Hyderabad · Delhi Capitals · Punjab Kings · Rajasthan Royals · Lucknow Super Giants · Gujarat Titans · </span>
      <span>Mumbai Indians · Chennai Super Kings · Royal Challengers Bengaluru · Kolkata Knight Riders · Sunrisers Hyderabad · Delhi Capitals · Punjab Kings · Rajasthan Royals · Lucknow Super Giants · Gujarat Titans · </span>
    </div>
  </div>
</section>

<!-- TODAY -->
<section id="today" class="relative py-20">
  <div class="max-w-7xl mx-auto px-6">
    <div class="flex items-end justify-between mb-10">
      <div>
        <div class="text-[11px] uppercase tracking-[0.3em] gradient-text-mix font-semibold">Featured prediction</div>
        <h2 class="font-display text-3xl sm:text-5xl font-bold mt-2 leading-tight">Today on the pitch.</h2>
        <div class="mt-3 flex items-center gap-2 text-sm text-white/60">
          <span class="live-dot"></span>
          <span>Live model output · refreshed __GENERATED_AT__</span>
        </div>
      </div>
      <div class="text-sm text-white/50 text-right">
        <div>__TODAY_DATE__</div>
        <div class="text-[10px] uppercase tracking-widest text-white/30 mt-1">Match day</div>
      </div>
    </div>
    __FEATURED__
  </div>
</section>

<!-- TRACK RECORD -->
<section id="track" class="py-20 bg-gradient-to-b from-transparent via-ink-900/30 to-transparent border-y border-white/5">
  <div class="max-w-7xl mx-auto px-6">
    <div class="grid lg:grid-cols-[1.1fr_1fr] gap-10 items-end mb-10">
      <div>
        <div class="text-[11px] uppercase tracking-[0.3em] gradient-text-mix font-semibold">Track record</div>
        <h2 class="font-display text-3xl sm:text-5xl font-bold mt-2 leading-tight">Last <span class="gradient-text">__BACKTEST_N__</span> matches,<br/>called blind.</h2>
        <p class="text-white/60 mt-4 max-w-lg">Walk-forward back-test: for each match the model retrains using only data from before that day. No cheating, no peeking ahead.</p>
      </div>
      <!-- Big accuracy gauge + streak grid -->
      <div class="grid grid-cols-2 gap-4">
        <div class="glass rounded-3xl p-6 flex flex-col items-center justify-center">
          <svg viewBox="0 0 120 120" class="w-32 h-32 -rotate-90">
            <circle cx="60" cy="60" r="50" stroke="rgba(255,255,255,0.06)" stroke-width="10" fill="none"/>
            <circle cx="60" cy="60" r="50" stroke="url(#gauge)" stroke-width="10" fill="none"
                    stroke-linecap="round" stroke-dasharray="__GAUGE_DASH__" style="filter:drop-shadow(0 0 12px rgba(245,180,65,0.5))"/>
            <defs>
              <linearGradient id="gauge" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="var(--t1-2)"/><stop offset="100%" stop-color="var(--t2)"/>
              </linearGradient>
            </defs>
          </svg>
          <div class="-mt-24 text-center">
            <div class="font-display text-4xl font-bold gradient-text">__BACKTEST_PCT__%</div>
            <div class="text-[10px] uppercase tracking-widest text-white/50 mt-1">Hit rate</div>
          </div>
          <div class="mt-20 text-xs text-white/50">__BACKTEST_CORRECT__ of __BACKTEST_N__ correct</div>
        </div>
        <div class="glass rounded-3xl p-6">
          <div class="text-[10px] uppercase tracking-widest text-white/40">Streak (oldest → newest)</div>
          <div class="mt-4">__STREAK_GRID__</div>
          <div class="seam mt-5 mb-3"></div>
          <div class="grid grid-cols-3 gap-2 text-center">
            <div><div class="text-[10px] uppercase text-white/40 tracking-widest">Wins</div><div class="font-display text-2xl text-mint mt-1">__WINS__</div></div>
            <div><div class="text-[10px] uppercase text-white/40 tracking-widest">Losses</div><div class="font-display text-2xl text-crimson mt-1">__LOSSES__</div></div>
            <div><div class="text-[10px] uppercase text-white/40 tracking-widest">Avg P</div><div class="font-display text-2xl mt-1 gradient-text">__AVG_P__</div></div>
          </div>
        </div>
      </div>
    </div>

    <div class="overflow-x-auto rounded-3xl glass">
      <table class="w-full text-sm">
        <thead class="text-white/40 text-[10px] uppercase tracking-widest">
          <tr class="border-b border-white/5">
            <th class="text-left p-4">Date</th>
            <th class="text-left p-4">Match</th>
            <th class="text-left p-4">P(t1)</th>
            <th class="text-left p-4">Predicted</th>
            <th class="text-left p-4">Actual</th>
            <th class="text-center p-4">Result</th>
          </tr>
        </thead>
        <tbody>__BACKTEST_ROWS__</tbody>
      </table>
    </div>
  </div>
</section>

<!-- ARCHIVE -->
<section id="archive" class="py-20">
  <div class="max-w-7xl mx-auto px-6">
    <div class="text-[11px] uppercase tracking-[0.3em] gradient-text-mix font-semibold">All predictions</div>
    <h2 class="font-display text-3xl sm:text-5xl font-bold mt-2 mb-10">Archive.</h2>
    <div class="grid md:grid-cols-2 gap-5">__ARCHIVE__</div>
  </div>
</section>

<!-- HOW -->
<section id="how" class="py-20 border-t border-white/5">
  <div class="max-w-6xl mx-auto px-6">
    <div class="text-[11px] uppercase tracking-[0.3em] gradient-text-mix font-semibold">Under the hood</div>
    <h2 class="font-display text-3xl sm:text-5xl font-bold mt-2 mb-12">How the model thinks.</h2>
    <div class="grid md:grid-cols-2 gap-5">
      <div class="glass rounded-3xl p-7"><div class="text-accent text-3xl font-display font-bold">01</div><div class="font-semibold mt-3 text-lg">Three years of ball-by-ball data</div><p class="text-white/60 mt-2 leading-relaxed">Cricsheet IPL 2023 → 2026. Every delivery, every wicket, every toss — 60k+ deliveries across 250+ matches.</p></div>
      <div class="glass rounded-3xl p-7"><div class="text-accent text-3xl font-display font-bold">02</div><div class="font-semibold mt-3 text-lg">Sixteen leak-free features</div><p class="text-white/60 mt-2 leading-relaxed">Rolling form, venue win-rate, H2H, scoring &amp; conceding rates, toss history — computed using only data from before each match.</p></div>
      <div class="glass rounded-3xl p-7"><div class="text-accent text-3xl font-display font-bold">03</div><div class="font-semibold mt-3 text-lg">Gradient-boosted classifier</div><p class="text-white/60 mt-2 leading-relaxed">Calibrated probability for the winner. Score &amp; boundaries blend team trends with venue norms.</p></div>
      <div class="glass rounded-3xl p-7"><div class="text-accent text-3xl font-display font-bold">04</div><div class="font-semibold mt-3 text-lg">Walk-forward validation</div><p class="text-white/60 mt-2 leading-relaxed">No future leakage. Every back-tested match is predicted using only the past. The hit rate is what we'd actually have hit, live.</p></div>
    </div>
  </div>
</section>

<footer class="py-12 border-t border-white/5 mt-10">
  <div class="max-w-7xl mx-auto px-6 flex flex-wrap items-center justify-between gap-4 text-xs text-white/40">
    <div>Built with Cricsheet · scikit-learn · Tailwind. A college project.</div>
    <div>Generated __GENERATED_AT__</div>
  </div>
</footer>

<script>
  // scroll progress
  const prog = document.getElementById('prog');
  window.addEventListener('scroll', () => {
    const h = document.documentElement;
    const pct = (h.scrollTop / (h.scrollHeight - h.clientHeight)) * 100;
    prog.style.width = pct + '%';
  });

  // KPI cursor glow
  document.querySelectorAll('.kpi-card').forEach(card => {
    card.addEventListener('mousemove', e => {
      const rect = card.getBoundingClientRect();
      card.style.setProperty('--mx', (e.clientX - rect.left) + 'px');
      card.style.setProperty('--my', (e.clientY - rect.top) + 'px');
    });
  });

  // intersection-observer reveal
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('anim-slide'); });
  }, { threshold: 0.12 });
  document.querySelectorAll('section, .archive-card').forEach(el => io.observe(el));

  // Tilt-on-mouse for team orbs
  document.querySelectorAll('.team-orb').forEach(orb => {
    orb.addEventListener('mousemove', e => {
      const rect = orb.getBoundingClientRect();
      const x = (e.clientX - rect.left - rect.width/2) / rect.width;
      const y = (e.clientY - rect.top - rect.height/2) / rect.height;
      orb.style.transform = `translateY(-3px) scale(1.04) rotateY(${x*15}deg) rotateX(${-y*15}deg)`;
    });
    orb.addEventListener('mouseleave', () => { orb.style.transform = ''; });
  });

  // Cursor-tracking floating ball
  const ball = document.querySelector('.float-ball');
  if (ball) {
    let mx = window.innerWidth*0.5, my = window.innerHeight*0.4;
    let cx = mx, cy = my;
    document.addEventListener('mousemove', e => { mx = e.clientX; my = e.clientY; });
    function frame() {
      cx += (mx - cx) * 0.02; cy += (my - cy) * 0.02;
      ball.style.transform = `translate(${cx - 60}px, ${cy - 60}px)`;
      ball.style.animation = 'none';  // override the drift on first cursor move
      requestAnimationFrame(frame);
    }
    let started = false;
    document.addEventListener('mousemove', () => { if (!started) { started = true; requestAnimationFrame(frame); } }, { once: true });
  }

  // Confetti burst when clicking the winner
  function spawnConfetti(x, y) {
    const colors = [getComputedStyle(document.documentElement).getPropertyValue('--t1'),
                    getComputedStyle(document.documentElement).getPropertyValue('--t1-2'),
                    getComputedStyle(document.documentElement).getPropertyValue('--t2'),
                    '#ffd17a'];
    for (let i = 0; i < 40; i++) {
      const c = document.createElement('span');
      const ang = Math.random() * Math.PI * 2;
      const vel = 4 + Math.random() * 6;
      c.style.cssText = `position:fixed;left:${x}px;top:${y}px;width:6px;height:10px;background:${colors[i%4]};z-index:100;pointer-events:none;border-radius:1px;`;
      document.body.appendChild(c);
      let vx = Math.cos(ang)*vel, vy = Math.sin(ang)*vel - 3, gx = 0, gy = 0;
      const start = Date.now();
      function step() {
        const t = (Date.now() - start) / 1000;
        gx += vx; gy += vy; vy += 0.25;
        c.style.transform = `translate(${gx}px, ${gy}px) rotate(${t*720}deg)`;
        c.style.opacity = String(Math.max(0, 1 - t/1.5));
        if (t < 1.5) requestAnimationFrame(step); else c.remove();
      }
      step();
    }
  }
  document.querySelectorAll('.featured-card .gradient-text').forEach(el => {
    el.addEventListener('click', e => {
      const r = e.target.getBoundingClientRect();
      spawnConfetti(r.left + r.width/2, r.top + r.height/2);
    });
    el.style.cursor = 'pointer'; el.title = 'Click for some hype';
  });

  // Make stat cards "wiggle" when hovered
  document.querySelectorAll('.featured-card .glass, .featured-card [class*="rounded-xl"]').forEach(card => {
    card.addEventListener('mouseenter', () => card.style.transition = 'transform 0.3s cubic-bezier(.2,.8,.2,1)');
  });
</script>
</body>
</html>
"""


def main() -> None:
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    predictions = gather_predictions()
    matches = pd.read_csv(PROC / "matches.csv", parse_dates=["date"])
    deliveries = pd.read_csv(PROC / "deliveries.csv")
    backtest = compute_backtest(10)

    today = predictions[-1] if predictions else None
    today_date = today["date"] if today else ""

    # gauge math
    circumference = 2 * math.pi * 50
    gauge_dash = f"{circumference * backtest['accuracy']:.2f} {circumference:.2f}"
    wins = sum(1 for r in backtest["rows"] if r["ok"])
    losses = backtest["n"] - wins
    avg_p = sum(r["p1"] for r in backtest["rows"]) / max(backtest["n"], 1)

    # Pull team theme colors from the featured prediction; fall back to amber/red
    if today:
        c1 = TEAM_COLORS.get(today["team1_init"], ("#f5b441", "#ffd17a"))
        c2 = TEAM_COLORS.get(today["team2_init"], ("#ff8e72", "#ffd17a"))
        match_title = f"{today['team1_init']} vs {today['team2_init']} · IPL 2026"
    else:
        c1 = ("#f5b441", "#ffd17a"); c2 = ("#ff8e72", "#ffd17a")
        match_title = "IPL 2026 Predictions"

    html = HTML
    html = html.replace("__T1_COLOR__", c1[0])
    html = html.replace("__T1_COLOR2__", c1[1])
    html = html.replace("__T2_COLOR__", c2[0])
    html = html.replace("__T2_COLOR2__", c2[1])
    html = html.replace("__MATCH_TITLE__", match_title)
    html = html.replace("__KPI_STRIP__", kpi_strip(predictions, backtest))
    html = html.replace("__TODAY_DATE__", today_date)
    html = html.replace("__FEATURED__", featured_card(today, matches, deliveries))
    html = html.replace("__BACKTEST_PCT__", str(int(backtest["accuracy"] * 100)))
    html = html.replace("__BACKTEST_CORRECT__", str(backtest["correct"]))
    html = html.replace("__BACKTEST_N__", str(backtest["n"]))
    html = html.replace("__GAUGE_DASH__", gauge_dash)
    html = html.replace("__STREAK_GRID__", streak_grid(backtest["rows"]))
    html = html.replace("__WINS__", str(wins))
    html = html.replace("__LOSSES__", str(losses))
    html = html.replace("__AVG_P__", f"{avg_p:.2f}")
    html = html.replace("__BACKTEST_ROWS__", backtest_table(backtest["rows"]))
    html = html.replace("__ARCHIVE__", archive_cards(predictions))
    html = html.replace("__GENERATED_AT__", datetime.now().strftime("%Y-%m-%d %H:%M"))

    out = WEB_DIR / "index.html"
    out.write_text(html)
    print(f"Wrote {out}  ({len(html):,} bytes, {len(predictions)} predictions, {backtest['n']} backtest)")


if __name__ == "__main__":
    main()
