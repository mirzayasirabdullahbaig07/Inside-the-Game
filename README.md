# ⚽ Inside the Game

Multi-agent match intelligence built on **synthetic** football event data
(no real Premier League data used). Built for the *Inside the Game: Developer Hackathon*.

## What it does
Turns raw match events (passes into the box, shots, cards, corners) into:
- **Explainable key moments**: every goal or big chance comes with a "why this matters" note (score context, xG).
- **Live narrative**: scrub through the match and get commentary in a fan, broadcaster or analyst voice.
- **Automated full-time recap**, fact-checked against the data before it is shown.
- **Momentum and stats views** for analysts.

## Agents
| Agent | Role |
|---|---|
| Analyst | Computes stats, momentum, key moments and their explanations |
| Narrator | Writes live commentary per key moment |
| Recap | Writes the full-time recap in the chosen voice |
| Checker | Verifies score, scorers and shot counts; rejects bad drafts with feedback |

Hand-offs go through a shared state. A rejected draft is sent back to Recap with
feedback (up to 2 retries). If the LLM is unavailable or keeps failing, a
deterministic template recovers, so the result is always verified. The **Agent trace**
tab shows all of this.

## Microsoft technology
- **Azure OpenAI via Microsoft Foundry** for commentary and recaps (optional; app runs offline without it)
- Streamlit app deployable on Azure App Service / Container Apps (see below)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Enable the LLM (optional)
Set these as environment variables, or in Streamlit Cloud under *Settings → Secrets*:
```toml
AZURE_OPENAI_ENDPOINT = "https://<your-resource>.openai.azure.com/"
AZURE_OPENAI_API_KEY = "<key>"
AZURE_OPENAI_DEPLOYMENT = "<your-deployment-name>"
```
Never commit keys to the repo.

## Deploy on Streamlit Community Cloud
1. Push this folder to a **public** GitHub repo.
2. Go to share.streamlit.io, click *New app*, choose the repo, set main file to `app.py`.
3. (Optional) Add the secrets above.

## Data
`data.py` generates fictional clubs and players with a seeded random model, so the
same seed always yields the same match.
