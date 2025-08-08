import os
from pathlib import Path
import streamlit as st
import json
from dotenv import load_dotenv
from virtual_lab.agent import Agent
from virtual_lab.run_meeting import run_meeting

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(page_title="Medical Multi‑Agent Consensus", layout="wide")
st.title("Medical Multi‑Agent Consensus")

with st.sidebar:
    st.header("API & Run Settings")
    api_key = st.text_input("OpenAI API Key", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    num_rounds = st.slider("Discussion Rounds", 1, 5, 2, 1)
    pubmed = st.checkbox("Enable PubMed Search", value=False)
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0)
    save_name = st.text_input("Session name", value="web_case", help="Used to name saved artifacts (.md/.json)")
    
    # Loader for previous sessions
    st.markdown("---")
    st.subheader("Load Previous Session")
    sessions_dir = BASE_DIR / "medical_meetings"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted({p.stem for p in sessions_dir.glob("*.md")} | {p.stem for p in sessions_dir.glob("*.json")})
    selected_session = st.selectbox("Select a session to view", [""] + existing, index=0)
    load_btn = st.button("Load Session")

st.subheader("Case Description (Agenda)")
agenda = st.text_area(
    "Describe the case", 
    placeholder="e.g., A 58-year-old with chest pain...",
    height=120,
)

st.subheader("Agenda Questions")
def_questions = [
    "What are the top differential diagnoses with estimated probabilities?",
    "What immediate red flags or stabilization steps are required?",
    "What diagnostic tests will most efficiently discriminate the leading diagnoses?",
    "What is the initial evidence-based management plan and monitoring?",
    "What risks, contraindications, and patient preferences should be considered?",
]
questions = st.text_area(
    "One question per line",
    value="\n".join(def_questions),
    height=140,
)

st.subheader("Rules / Guardrails")
def_rules = [
    "Do not make unverifiable clinical claims; cite guidelines when possible.",
    "Provide specific doses, intervals, and monitoring when recommending treatments.",
    "Call out uncertainty and alternatives explicitly.",
    "Assume this is an educational exercise, not medical advice.",
]
rules = st.text_area(
    "One rule per line",
    value="\n".join(def_rules),
    height=120,
)

st.subheader("Team Configuration")
lead_title = st.text_input("Team Lead Title", value="Attending Physician")
lead_expertise = st.text_input("Team Lead Expertise", value="internal medicine, evidence-based guidelines, diagnostic reasoning")

col1, col2 = st.columns(2)
with col1:
    m1_title = st.text_input("Member 1 Title", value="Differential Diagnostician")
    m1_exp = st.text_input("Member 1 Expertise", value="diagnostic heuristics, Bayesian reasoning, pattern recognition")
with col2:
    m2_title = st.text_input("Member 2 Title", value="Evidence Synthesis Specialist")
    m2_exp = st.text_input("Member 2 Expertise", value="clinical trials, guidelines (e.g., ACC/AHA, IDSA), systematic reviews")

col3, col4 = st.columns(2)
with col3:
    m3_title = st.text_input("Member 3 Title", value="Pharmacotherapy Expert")
    m3_exp = st.text_input("Member 3 Expertise", value="drug selection, dosing, contraindications, interactions, renal/hepatic adjustments")
with col4:
    m4_title = st.text_input("Member 4 Title", value="Risk and Ethics Officer")
    m4_exp = st.text_input("Member 4 Expertise", value="patient safety, ethics, shared decision-making, risk stratification")

run_btn = st.button("Run Team Meeting", type="primary")

output_container = st.container()

# Helper to render artifacts for a given session name
def render_session_artifacts(session_name: str):
    save_dir = BASE_DIR / "medical_meetings"
    md_path = save_dir / f"{session_name}.md"
    json_path = save_dir / f"{session_name}.json"

    st.subheader("Consensus Summary (from transcript)")
    if md_path.exists():
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            # Heuristic: show the last "### Recommendation" + below when available
            summary_start = md_content.find("### Recommendation")
            if summary_start != -1:
                st.markdown(md_content[summary_start:])
            else:
                st.markdown(md_content)
        except Exception:
            st.info("Unable to parse summary from transcript; showing full transcript below.")
    else:
        st.info("Transcript (.md) not found.")

    st.divider()
    st.subheader("Discussion Transcript")
    if md_path.exists():
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        with st.expander("Show full markdown transcript", expanded=True):
            st.markdown(md_content)
        st.download_button(
            label="Download transcript (.md)",
            data=md_content,
            file_name=f"{session_name}.md",
            mime="text/markdown",
        )
    else:
        st.info("Transcript (.md) not found.")

    st.subheader("Raw Messages (JSON)")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            json_content = f.read()
        try:
            import json as _json
            messages = _json.loads(json_content)
            with st.expander("Show raw messages JSON", expanded=False):
                st.json(messages)
        except Exception:
            st.code(json_content, language="json")
        st.download_button(
            label="Download messages (.json)",
            data=json_content,
            file_name=f"{session_name}.json",
            mime="application/json",
        )
    else:
        st.info("Messages (.json) not found.")

if run_btn:
    if not api_key:
        st.error("Please provide an OpenAI API key.")
    elif not agenda.strip():
        st.error("Please provide a case description (agenda).")
    else:
        os.environ["OPENAI_API_KEY"] = api_key
        team_lead = Agent(
            title=lead_title,
            expertise=lead_expertise,
            goal="coordinate the team to analyze the case, resolve disagreements, and deliver a concise, actionable consensus plan",
            role="team lead and final arbiter",
            model=model,
        )
        team_members = (
            Agent(title=m1_title, expertise=m1_exp, goal="propose and prioritize differential diagnoses with probabilities and justifications", role="generate differentials and evaluate likelihoods", model=model),
            Agent(title=m2_title, expertise=m2_exp, goal="summarize best available evidence relevant to diagnosis and management", role="evidence reviewer", model=model),
            Agent(title=m3_title, expertise=m3_exp, goal="recommend safe and effective medication plan and monitoring", role="therapeutics recommender", model=model),
            Agent(title=m4_title, expertise=m4_exp, goal="identify risks, contraindications, and consent considerations; ensure patient-centered plan", role="risk mitigation and ethics", model=model),
        )
        agenda_qs = tuple([q.strip() for q in questions.splitlines() if q.strip()])
        agenda_rules = tuple([r.strip() for r in rules.splitlines() if r.strip()])
        save_dir = BASE_DIR / "medical_meetings"
        save_dir.mkdir(parents=True, exist_ok=True)
        with st.spinner("Running team meeting... this may take a few minutes"):
            try:
                summary = run_meeting(
                    meeting_type="team",
                    agenda=agenda,
                    save_dir=save_dir,
                    save_name=save_name,
                    team_lead=team_lead,
                    team_members=team_members,
                    agenda_questions=agenda_qs,
                    agenda_rules=agenda_rules,
                    num_rounds=st.session_state.get("num_rounds_override", None) or int(num_rounds),
                    temperature=float(temperature),
                    pubmed_search=bool(pubmed),
                    return_summary=True,
                )
                output_container.subheader("Consensus Summary")
                output_container.markdown(summary)

                # Transparency: render artifacts for this session
                render_session_artifacts(save_name)
            except Exception as e:
                st.exception(e)

# Load previously saved session without rerun
if load_btn and selected_session:
    render_session_artifacts(selected_session)
