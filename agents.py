"""Multi-agent match-intelligence pipeline.

Agents (each with one job) coordinate through a shared state dict:

  Analyst  -> computes stats, momentum, and key moments (with the *why*)
  Narrator -> writes live commentary for each key moment
  Recap    -> writes the full-time recap in the chosen persona
  Checker  -> verifies the recap against the Analyst's numbers

If the Checker rejects a recap, it hands feedback back to the Recap agent
(up to MAX_RETRIES). If the LLM is unavailable or keeps failing, the
pipeline falls back to a deterministic template that is correct by
construction, so the app always produces a verified result.

LLM: Azure OpenAI (Microsoft Foundry deployment). Set AZURE_OPENAI_ENDPOINT,
AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT to enable. Otherwise the app
runs fully offline with template writers.
"""
from __future__ import annotations

import os
import re
import time
from collections import defaultdict

MAX_RETRIES = 2
PERSONAS = {
    "Fan": "an excitable supporter writing for a fan site: warm, vivid, short sentences",
    "Broadcaster": "a polished TV match reporter: authoritative, concise, scene-setting",
    "Analyst": "a tactical analyst: precise, evidence-led, references xG and momentum",
}


# ----------------------------------------------------------------- LLM layer
def llm_available() -> bool:
    return all(os.getenv(k) for k in
               ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"))


def call_llm(system: str, user: str, max_tokens: int = 400) -> str:
    from openai import AzureOpenAI  # imported lazily so offline mode needs no key

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
    resp = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.7,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# ------------------------------------------------------------------- Analyst
class AnalystAgent:
    name = "Analyst"

    def run(self, state: dict) -> None:
        m = state["match"]
        home, away, events = m["home"], m["away"], m["events"]
        teams = (home, away)

        stats = {t: defaultdict(float) for t in teams}
        for e in events:
            s = stats[e["team"]]
            s["events"] += 1
            if e["type"] == "shot":
                s["shots"] += 1
                s["xg"] += e["xg"]
                if e["outcome"] in ("goal", "saved"):
                    s["on_target"] += 1
                if e["outcome"] == "goal":
                    s["goals"] += 1
            elif e["type"] in ("pass_into_box", "corner", "tackle", "foul"):
                s[e["type"]] += 1
            elif e["type"] == "card":
                s["reds" if e["outcome"] == "red" else "yellows"] += 1

        total = sum(stats[t]["events"] for t in teams) or 1
        for t in teams:
            stats[t]["territory"] = round(100 * stats[t]["events"] / total)
            stats[t]["xg"] = round(stats[t]["xg"], 2)
            stats[t] = {k: (int(v) if k != "xg" else v) for k, v in stats[t].items()}

        # Momentum: attacking threat per 5-minute window, home minus away.
        last = max(e["minute"] for e in events)
        momentum = []
        for start in range(1, last + 1, 5):
            threat = {home: 0.0, away: 0.0}
            for e in events:
                if start <= e["minute"] < start + 5:
                    threat[e["team"]] += {"pass_into_box": 0.4, "corner": 0.5, "shot": 1.0 + 8 * e["xg"]}.get(e["type"], 0)
            momentum.append({"window": f"{start}-{start + 4}", "net": round(threat[home] - threat[away], 2)})

        # Key moments, each with an explanation of why it matters.
        moments, home_g, away_g = [], 0, 0
        for i, e in enumerate(events):
            why = None
            if e["type"] == "shot" and e["outcome"] == "goal":
                before = (home_g, away_g)
                home_g, away_g = (home_g + 1, away_g) if e["team"] == home else (home_g, away_g + 1)
                mine, theirs = (home_g, away_g) if e["team"] == home else (away_g, home_g)
                if mine == theirs:
                    ctx = "levelled the match"
                elif mine - theirs == 1 and before[0] == before[1]:
                    ctx = "broke the deadlock"
                elif mine > theirs:
                    ctx = "extended the lead" if mine - theirs > 1 or before != (0, 0) else "took the lead"
                else:
                    ctx = "pulled one back"
                kind = "goal"
                why = f"{ctx} (chance quality: {e['xg']:.2f} xG; {'a long-range effort' if e['xg'] < 0.10 else 'a decent chance' if e['xg'] < 0.25 else 'a high-value chance'})"
            elif e["type"] == "card" and e["outcome"] == "red":
                kind, why = "red_card", "sending-off changes the balance of the match"
            elif e["type"] == "shot" and e["xg"] >= 0.30 and e["outcome"] != "goal":
                kind, why = "big_chance", f"a {e['xg']:.2f} xG chance that was {e['outcome'].replace('_', ' ')}: a real missed opportunity"
            else:
                continue
            moments.append({**e, "kind": kind, "why": why,
                            "score_at_time": e["score_after"]})

        state["stats"] = stats
        state["momentum"] = momentum
        state["moments"] = moments
        state["final_score"] = f"{stats[home].get('goals', 0)}-{stats[away].get('goals', 0)}"
        state["scorers"] = [(e["player"], e["team"], e["minute"]) for e in moments if e["kind"] == "goal"]


# ------------------------------------------------------------------ Narrator
class NarratorAgent:
    name = "Narrator"

    def _template(self, m: dict) -> str:
        who = f"{m['player']} ({m['team']})"
        if m["kind"] == "goal":
            return f"GOAL! {who} scores. {m['why'].split(' (')[0].capitalize()}. It's {m['score_at_time']}."
        if m["kind"] == "red_card":
            return f"RED CARD! {who} is sent off. {m['why'].capitalize()}."
        return f"{who} misses a big chance ({m['xg']:.2f} xG), {m['outcome'].replace('_', ' ')}."

    def run(self, state: dict, persona: str, use_llm: bool) -> bool:
        used_llm = False
        for m in state["moments"]:
            text = None
            if use_llm:
                try:
                    text = call_llm(
                        f"You are {PERSONAS[persona]}. Write ONE live-commentary line (max 25 words). "
                        "Use only the facts given. Do not invent names, numbers or events.",
                        f"Minute {m['minute']}: {m['kind']} by {m['player']} of {m['team']}. "
                        f"Context: {m['why']}. Score after: {m['score_at_time']}.",
                        max_tokens=80)
                    used_llm = True
                except Exception:
                    text = None
            m["commentary"] = text or self._template(m)
        return used_llm


# --------------------------------------------------------------------- Recap
class RecapAgent:
    name = "Recap"

    def facts(self, state: dict) -> str:
        m, s = state["match"], state["stats"]
        h, a = m["home"], m["away"]
        lines = [f"Final score: {h} {state['final_score']} {a}",
                 f"Shots: {h} {s[h].get('shots', 0)}, {a} {s[a].get('shots', 0)}",
                 f"xG: {h} {s[h]['xg']}, {a} {s[a]['xg']}",
                 f"Territorial share: {h} {s[h]['territory']}%, {a} {s[a]['territory']}%"]
        lines += [f"Goal: {p} ({t}) {mn}'" for p, t, mn in state["scorers"]]
        lines += [f"Red card: {x['player']} ({x['team']}) {x['minute']}'" for x in state["moments"] if x["kind"] == "red_card"]
        return "\n".join(lines)

    def write_llm(self, state: dict, persona: str, feedback: str | None) -> str:
        prompt = "Write a full-time match recap (90-130 words) using ONLY these facts:\n" + self.facts(state)
        prompt += "\nWrite the scoreline as 'H-A' with a plain hyphen. Mention every goalscorer by surname."
        if feedback:
            prompt += f"\n\nYour previous draft was rejected by the fact-checker:\n{feedback}\nFix these issues."
        return call_llm(f"You are {PERSONAS[persona]}. Never invent facts.", prompt, max_tokens=300)

    def write_template(self, state: dict, persona: str) -> str:
        m, s = state["match"], state["stats"]
        h, a = m["home"], m["away"]
        hg, ag = (int(x) for x in state["final_score"].split("-"))
        if hg == ag:
            result = f"{h} and {a} share the spoils, {state['final_score']}"
        else:
            w, l = (h, a) if hg > ag else (a, h)
            result = f"{w} beat {l} {max(hg, ag)}-{min(hg, ag)}"
        scorers = ", ".join(f"{p.split()[-1]} ({mn}')" for p, _, mn in state["scorers"]) or "no one found the net"
        edge = h if s[h]["xg"] > s[a]["xg"] else a
        body = {
            "Fan": f"{result}! What a game. Goals: {scorers}. {edge} created the better chances "
                   f"({s[edge]['xg']} xG) and it showed. {h} had {s[h]['shots']} shots, {a} had {s[a]['shots']}.",
            "Broadcaster": f"Full time: {result}. Scorers: {scorers}. {edge} carried the greater threat, "
                           f"with {s[edge]['shots']} shots and {s[edge]['xg']} xG, with territory split {s[h]['territory']}% to {s[a]['territory']}%.",
            "Analyst": f"{result}. Expected goals favoured {edge} ({s[h]['xg']} vs {s[a]['xg']}). "
                       f"Shot volume: {h} {s[h]['shots']}, {a} {s[a]['shots']}. Scorers: {scorers}. "
                       f"Territorial share split {s[h]['territory']}% to {s[a]['territory']}%.",
        }
        return body[persona]


# ------------------------------------------------------------------- Checker
class FactCheckerAgent:
    name = "Checker"

    def check(self, text: str, state: dict) -> list[str]:
        issues = []
        norm = text.replace("–", "-").replace("—", "-")
        scores = set(re.findall(r"\b(\d+)\s*-\s*(\d+)\b(?!\s*%)", norm))
        actual = tuple(state["final_score"].split("-"))
        reversed_ = (actual[1], actual[0])
        if not scores:
            issues.append(f"No scoreline found; must state {state['final_score']}.")
        for sc in scores:
            if sc not in (actual, reversed_):
                issues.append(f"Scoreline {sc[0]}-{sc[1]} is wrong; the final score is {state['final_score']}.")
        for player, _, _ in state["scorers"]:
            if player.split()[-1].lower() not in text.lower():
                issues.append(f"Goalscorer {player} is not mentioned.")
        shot_totals = {str(int(v.get("shots", 0))) for v in state["stats"].values()}
        for n in re.findall(r"(\d+)\s+shots", text):
            if n not in shot_totals:
                issues.append(f"'{n} shots' does not match the data ({', '.join(sorted(shot_totals))}).")
        return issues


# -------------------------------------------------------------- Orchestrator
def run_pipeline(match: dict, persona: str = "Broadcaster", use_llm: bool = False) -> dict:
    state: dict = {"match": match, "trace": []}
    use_llm = use_llm and llm_available()

    def log(agent, action, status, detail=""):
        state["trace"].append({"agent": agent, "action": action, "status": status, "detail": detail})

    t = time.time()
    AnalystAgent().run(state)
    log("Analyst", "Computed stats, momentum and key moments", "ok",
        f"{len(state['moments'])} key moments, final score {state['final_score']} ({(time.time() - t) * 1000:.0f} ms)")

    used = NarratorAgent().run(state, persona, use_llm)
    log("Narrator", "Wrote live commentary", "ok",
        f"{len(state['moments'])} lines via {'Azure OpenAI' if used else 'template writer'}")

    recap_agent, checker = RecapAgent(), FactCheckerAgent()
    feedback, recap, source = None, None, "template"
    attempts = (MAX_RETRIES + 1) if use_llm else 0
    for attempt in range(1, attempts + 1):
        try:
            draft = recap_agent.write_llm(state, persona, feedback)
        except Exception as exc:  # network / quota / config failure
            log("Recap", f"Draft {attempt}", "error", f"LLM call failed: {type(exc).__name__}")
            break
        issues = checker.check(draft, state)
        if not issues:
            log("Recap", f"Draft {attempt}", "ok", "Written via Azure OpenAI")
            log("Checker", f"Verified draft {attempt}", "ok", "Score, scorers and shot counts match the data")
            recap, source = draft, "Azure OpenAI"
            break
        log("Recap", f"Draft {attempt}", "ok", "Written via Azure OpenAI")
        log("Checker", f"Rejected draft {attempt}", "rejected", " | ".join(issues))
        feedback = "\n".join(f"- {i}" for i in issues)  # hand-off back to Recap

    if recap is None:
        recap = recap_agent.write_template(state, persona)
        issues = checker.check(recap, state)
        why = "LLM not configured" if not use_llm else "LLM drafts failed verification, recovered with template"
        log("Recap", "Fallback template", "recovered" if use_llm else "ok", why)
        log("Checker", "Verified template recap", "ok" if not issues else "rejected",
            "Score, scorers and shot counts match the data" if not issues else " | ".join(issues))

    state["recap"], state["recap_source"] = recap, source
    return state
