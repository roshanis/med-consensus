import os
from pathlib import Path
import streamlit as st
import json
from typing import List, Dict
from openai import OpenAI
import streamlit.components.v1 as components
from virtual_lab.agent import Agent
from virtual_lab.run_meeting import run_meeting

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Medical Multi‑Agent Consensus", page_icon="🏥", layout="wide")

"""
Minimal CSS to remove top white band and keep a clean, native look.
"""
st.markdown(
    """
    <style>
    [data-testid="stHeader"] { display: none; }
    [data-testid="stToolbar"] { display: none !important; }
    .block-container { padding-top: 0.5rem !important; }
    .card { padding: 1rem 1.2rem; border-radius: 8px; border: 1px solid #e8ebf3; margin-bottom: 1rem; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; background:#eef3ff; color:#2952ff; font-size: 12px; margin-right:8px; }
    .hero-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem; }
    .hero-subtitle { color: #6b7280; margin-top: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero header
left_h, right_h = st.columns([3, 1])
with left_h:
    st.markdown("<div class='hero-title'>🏥 Multispecialty Medical Consensus</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Coordinate cardiology, emergency medicine, pharmacotherapy, and ethics specialists to deliver a clear, evidence‑based plan.</div>",
        unsafe_allow_html=True,
    )
with right_h:
    st.markdown("<span class='badge'>Interactive</span><span class='badge'>Transparent</span><span class='badge'>Evidence‑aware</span>", unsafe_allow_html=True)

# Initialize session state containers
if "clarifying_questions" not in st.session_state:
    st.session_state.clarifying_questions = []
if "clarifying_answers" not in st.session_state:
    st.session_state.clarifying_answers = {}

with st.sidebar:
    # Prefer Streamlit secrets if present, otherwise env (no manual input)
    # Only look in Streamlit secrets (ignore .env and process env)
    _default_api_key = ""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            _default_api_key = st.secrets["OPENAI_API_KEY"] or ""
    except Exception:
        _default_api_key = ""
    num_rounds = st.slider("Discussion Rounds", 1, 5, 2, 1)
    pubmed = st.checkbox("Enable PubMed Search", value=False)
    model = st.selectbox("Model", ["gpt-5", "gpt-5-nano", "gpt-4o", "gpt-4o-mini"], index=0)
    st.caption("Sessions are auto-numbered (web_00001, web_00002, …) and only the latest 5 are kept.")
    
    # Loader for previous sessions
    st.markdown("---")
    st.subheader("Load Previous Session")
    sessions_dir = BASE_DIR / "medical_meetings"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted({p.stem for p in sessions_dir.glob("*.md")} | {p.stem for p in sessions_dir.glob("*.json")})
    selected_session = st.selectbox("Select a session to view", [""] + existing, index=0)
    load_btn = st.button("Load Session")

col_case, col_clarity = st.columns([3, 2], vertical_alignment="top")
with col_case:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Case Description (Agenda)")
    agenda = st.text_area(
        "Describe the case",
        placeholder="e.g., A 58-year-old with chest pain...",
        height=140,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_clarity:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Clarity Assistant")
    st.caption("Let the AI ask clarifying questions if the case description is ambiguous or missing key details.")
    col_ca1, col_ca2 = st.columns([1, 1])
    with col_ca1:
        suggest_btn = st.button("Suggest Questions")
    with col_ca2:
        max_q = st.slider("Max Qs", min_value=3, max_value=10, value=5)
    st.markdown("</div>", unsafe_allow_html=True)

"""
All model calls use provider defaults; no temperature control.
"""


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def generate_clarifying_questions(case_text: str, max_questions: int, model_name: str) -> List[str]:
    client = OpenAI()
    system = (
        "You are a clinical intake assistant. Read the user's case description and draft concise clarifying "
        "questions to remove ambiguity and fill missing critical details (history, timing, risk factors, red flags, "
        "medications, vitals, pertinent negatives). Do not answer; only ask necessary questions."
    )
    user = f"Case description:\n\n{case_text}\n\nReturn {max_questions} numbered clarifying questions."
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content if resp.choices else ""
    # Parse lines that look like numbered items
    questions: List[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering like "1. ", "- ", "• "
        if line[0].isdigit():
            # remove leading number and dot
            q = line.split(".", 1)
            line = q[1].strip() if len(q) > 1 else line
        if line.startswith(("- ", "• ")):
            line = line[2:].strip()
        questions.append(line)
    # Deduplicate and trim to max
    uniq: List[str] = []
    for q in questions:
        if q and q not in uniq:
            uniq.append(q)
    return uniq[:max_questions]


# ----- Full meeting caching -----
def _serialize_agent(agent: Agent) -> Dict[str, str]:
    return {
        "title": agent.title,
        "expertise": agent.expertise,
        "goal": agent.goal,
        "role": agent.role,
        "model": agent.model,
    }


def _deserialize_agent(data: Dict[str, str]) -> Agent:
    return Agent(
        title=data["title"],
        expertise=data["expertise"],
        goal=data["goal"],
        role=data["role"],
        model=data["model"],
    )


@st.cache_data(show_spinner=True, ttl=60 * 60 * 24)
def run_meeting_cached(
    agenda: str,
    agenda_questions: tuple[str, ...],
    agenda_rules: tuple[str, ...],
    contexts: tuple[str, ...],
    num_rounds: int,
    pubmed_search: bool,
    team_lead_data: Dict[str, str],
    team_members_data: tuple[Dict[str, str], ...],
    save_name: str,
) -> str:
    save_dir = BASE_DIR / "medical_meetings"
    save_dir.mkdir(parents=True, exist_ok=True)
    team_lead = _deserialize_agent(team_lead_data)
    team_members = tuple(_deserialize_agent(d) for d in team_members_data)

    # Map unsupported Assistants models (e.g., gpt-5 family) to a supported one (gpt-4o)
    def to_assistants_model(name: str) -> str:
        n = (name or "").lower()
        if n.startswith("gpt-5"):
            return "gpt-4o"
        return name

    team_lead.model = to_assistants_model(team_lead.model)
    team_members = tuple(
        Agent(m.title, m.expertise, m.goal, m.role, to_assistants_model(m.model)) for m in team_members
    )
    summary = run_meeting(
        meeting_type="team",
        agenda=agenda,
        save_dir=save_dir,
        save_name=save_name,
        team_lead=team_lead,
        team_members=team_members,
        agenda_questions=agenda_questions,
        agenda_rules=agenda_rules,
        contexts=contexts,
        num_rounds=num_rounds,
        temperature=1.0,
        pubmed_search=pubmed_search,
        return_summary=True,
    )
    return summary

if suggest_btn:
    # Use sidebar key if provided
    if _default_api_key:
        os.environ["OPENAI_API_KEY"] = _default_api_key
    if not _default_api_key:
        st.error("Please set your OpenAI API key in the sidebar or .env before generating questions.")
    elif not agenda.strip():
        st.error("Please enter a case description first.")
    else:
        try:
            questions_list = generate_clarifying_questions(agenda, max_q, model)
            st.session_state.clarifying_questions = questions_list
            # Initialize answer slots for new questions
            for q in questions_list:
                st.session_state.clarifying_answers.setdefault(q, "")
            st.success("Clarifying questions generated.")
        except Exception as e:
            st.exception(e)

# Optional cache controls
with st.sidebar:
    if st.button("Clear prompt cache"):
        generate_clarifying_questions.clear()
        st.success("Prompt cache cleared.")

if st.session_state.clarifying_questions:
    with st.expander("Clarifying Questions (answer to improve precision)", expanded=True):
        for q in st.session_state.clarifying_questions:
            st.session_state.clarifying_answers[q] = st.text_area(q, value=st.session_state.clarifying_answers.get(q, ""), height=70)

st.markdown("<div class='card'>", unsafe_allow_html=True)
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
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
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
st.markdown("</div>", unsafe_allow_html=True)

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


# ----- Web session naming and pruning helpers -----
def _get_web_session_basenames(save_dir: Path) -> list[str]:
    names = set()
    for p in list(save_dir.glob("web_*.md")) + list(save_dir.glob("web_*.json")):
        names.add(p.stem)
    return sorted(names)


def _make_next_web_session_name(save_dir: Path) -> str:
    import re
    max_idx = 0
    for stem in _get_web_session_basenames(save_dir):
        m = re.match(r"web_(\d+)$", stem)
        if m:
            try:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass
    return f"web_{max_idx + 1:05d}"


def _prune_web_sessions(save_dir: Path, max_sessions: int = 5) -> None:
    from typing import Tuple
    stems = _get_web_session_basenames(save_dir)
    if len(stems) <= max_sessions:
        return
    # Compute last modified time per session (max of md/json)
    def mtime_for(stem: str) -> float:
        md = save_dir / f"{stem}.md"
        js = save_dir / f"{stem}.json"
        times: list[float] = []
        if md.exists():
            times.append(md.stat().st_mtime)
        if js.exists():
            times.append(js.stat().st_mtime)
        return max(times) if times else 0.0

    stems_sorted = sorted(stems, key=mtime_for, reverse=True)
    to_delete = stems_sorted[max_sessions:]
    for stem in to_delete:
        for ext in (".md", ".json"):
            path = save_dir / f"{stem}{ext}"
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass

if run_btn:
    if not _default_api_key:
        st.error("OpenAI API key missing. Set it in .streamlit/secrets.toml or the environment.")
    elif not agenda.strip():
        st.error("Please provide a case description (agenda).")
    else:
        if _default_api_key:
            os.environ["OPENAI_API_KEY"] = _default_api_key
        # Build clarifications context
        clarifications_text = ""
        if st.session_state.clarifying_questions:
            qa_lines = ["Clarifications provided by user:"]
            for cq in st.session_state.clarifying_questions:
                ans = st.session_state.clarifying_answers.get(cq, "").strip()
                if ans:
                    qa_lines.append(f"- {cq}\n  Answer: {ans}")
                else:
                    qa_lines.append(f"- {cq}\n  Answer: (not provided)")
            clarifications_text = "\n".join(qa_lines)
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
                # Auto-number session name and prune to keep the latest 5 web_* sessions
                auto_save_name = _make_next_web_session_name(save_dir)
                summary = run_meeting_cached(
                    agenda=agenda,
                    agenda_questions=agenda_qs,
                    agenda_rules=agenda_rules,
                    contexts=(clarifications_text,) if clarifications_text else (),
                    num_rounds=st.session_state.get("num_rounds_override", None) or int(num_rounds),
                    pubmed_search=bool(pubmed),
                    team_lead_data=_serialize_agent(team_lead),
                    team_members_data=tuple(_serialize_agent(m) for m in team_members),
                    save_name=auto_save_name,
                )
                _prune_web_sessions(save_dir, max_sessions=5)
                tabs = st.tabs(["🧭 Consensus Summary", "🗒️ Transcript", "🧱 Raw JSON"]) 
                with tabs[0]:
                    st.markdown('<div id="consensus-summary-anchor"></div>', unsafe_allow_html=True)
                    output_container.subheader("Consensus Summary")
                    output_container.markdown(summary)
                    # Auto-scroll to summary when ready
                    components.html(
                        """
                        <script>
                        setTimeout(function(){
                          var el = parent.document.querySelector('#consensus-summary-anchor');
                          if (el && el.scrollIntoView) {
                            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                          }
                        }, 100);
                        </script>
                        """,
                        height=0,
                    )
                with tabs[1]:
                    # Transcript
                    render_session_artifacts(auto_save_name)
                with tabs[2]:
                    # Only render JSON section
                    save_dir = BASE_DIR / "medical_meetings"
                    json_path = save_dir / f"{auto_save_name}.json"
                    if json_path.exists():
                        with open(json_path, "r", encoding="utf-8") as f:
                            json_content = f.read()
                        try:
                            messages = json.loads(json_content)
                            st.json(messages)
                        except Exception:
                            st.code(json_content, language="json")
                    else:
                        st.info("Messages (.json) not found.")
            except Exception as e:
                st.exception(e)

# Load previously saved session without rerun
if load_btn and selected_session:
    render_session_artifacts(selected_session)
