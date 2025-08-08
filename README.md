# Med Consensus (Multispecialty Hospital)

An AI‑assisted, multispecialty medical consensus tool. It reuses the Zou Group Virtual Lab multi‑agent framework to coordinate specialists (Emergency Medicine, Cardiology, Pharmacotherapy, Ethics, etc.) in a structured team discussion and produce an evidence‑aware consensus plan. UI built with Streamlit.

Reference framework: Zou Group Virtual Lab `virtual-lab` — see repo: https://github.com/zou-group/virtual-lab

## Features
- Multi‑agent “team meeting” to analyze a case and synthesize a consensus plan
- Transparent artifacts: full transcript (.md) and raw messages (.json)
- Streamlit UI for agenda, questions, rules, and team configuration
- Clarity Assistant: auto‑suggests clarifying questions; your answers guide the agents
- Optional PubMed search (via the underlying framework)

## Safety & Scope
- Educational/prototype use only; not medical advice
- Avoid PHI; comply with institutional policies
- Human oversight and guideline verification required

## Quickstart
1) Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2) API key
```bash
cp .env.example .env
# edit .env and set: OPENAI_API_KEY=sk-...
```

3) Run UI
```bash
streamlit run app.py
# open http://localhost:8501
```

## Hospital Workflow
1) Enter a patient case in “Case Description (Agenda)”
2) Click “Suggest Questions” to generate clarifying questions; answer them to remove ambiguity
3) Adjust “Agenda Questions” and “Rules/Guardrails” (e.g., dosing specificity, citations)
4) Configure team roles (Attending, Differential Diagnostician, Evidence, Pharmacotherapy, Ethics)
5) Run Team Meeting → review tabs: Consensus Summary, Transcript, Raw JSON
6) Saved artifacts: `medical_meetings/<session>.md` and `<session>.json`

## Configuration Tips
- Models: select per‑agent model in UI (e.g., gpt‑4o, gpt‑4o‑mini)
- Temperature: 0.1–0.3 for consistent consensus
- Rounds: increase for deeper discussion
- PubMed: toggle in sidebar to include literature search

## CLI Example
```bash
python medical_consensus.py
```
Edit `AGENDA`, `AGENDA_QUESTIONS`, and `AGENDA_RULES` in the script as needed.

## Deploy
- Local: Streamlit run as above
- Container: build a minimal Python image and expose port 8501
- Internal use: protect behind SSO/reverse proxy; store `.env` securely (not in git)

## License
- App code: MIT (adjust as needed)
- Virtual Lab: see license in the upstream repository
