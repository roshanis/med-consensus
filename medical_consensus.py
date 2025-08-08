import os
from pathlib import Path
from virtual_lab.agent import Agent
from virtual_lab.run_meeting import run_meeting
from dotenv import load_dotenv

# Load environment variables from .env next to this script
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Configuration: ensure OPENAI_API_KEY is set in environment
if not os.environ.get('OPENAI_API_KEY'):
    raise SystemExit('Please set OPENAI_API_KEY in your environment before running.')

# Define medical multi-agent team
TEAM_LEAD = Agent(
    title="Attending Physician",
    expertise="internal medicine, evidence-based guidelines, diagnostic reasoning",
    goal="coordinate the team to analyze the case, resolve disagreements, and deliver a concise, actionable consensus plan",
    role="team lead and final arbiter",
    model="gpt-4o"
)

TEAM_MEMBERS = (
    Agent(
        title="Differential Diagnostician",
        expertise="diagnostic heuristics, Bayesian reasoning, pattern recognition",
        goal="propose and prioritize differential diagnoses with probabilities and justifications",
        role="generate differentials and evaluate likelihoods",
        model="gpt-4o"
    ),
    Agent(
        title="Evidence Synthesis Specialist",
        expertise="clinical trials, guidelines (e.g., ACC/AHA, IDSA), systematic reviews",
        goal="summarize best available evidence relevant to diagnosis and management",
        role="evidence reviewer",
        model="gpt-4o"
    ),
    Agent(
        title="Pharmacotherapy Expert",
        expertise="drug selection, dosing, contraindications, interactions, renal/hepatic adjustments",
        goal="recommend safe and effective medication plan and monitoring",
        role="therapeutics recommender",
        model="gpt-4o"
    ),
    Agent(
        title="Risk and Ethics Officer",
        expertise="patient safety, ethics, shared decision-making, risk stratification",
        goal="identify risks, contraindications, and consent considerations; ensure patient-centered plan",
        role="risk mitigation and ethics",
        model="gpt-4o"
    ),
)

# Example agenda: replace with your case
AGENDA = (
    "A 58-year-old with chest pain. Analyze likely etiologies, initial workup, and management plan."
)

# Structured agenda questions
AGENDA_QUESTIONS = (
    "What are the top 5 differential diagnoses with estimated probabilities?",
    "What immediate red flags or stabilization steps are required?",
    "What diagnostic tests will most efficiently discriminate the leading diagnoses?",
    "What is the initial evidence-based management plan and monitoring?",
    "What risks, contraindications, and patient preferences should be considered?",
)

# Guardrails/rules tailored for medical use
AGENDA_RULES = (
    "Do not make unverifiable clinical claims; cite guideline class/level when possible.",
    "Prefer specificity with doses, intervals, and monitoring parameters when recommending treatments.",
    "Call out uncertainty and alternative paths explicitly.",
    "Assume this is a hypothetical educational exercise and not real medical advice.",
)

SAVE_DIR = Path("./medical_meetings")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    consensus = run_meeting(
        meeting_type="team",
        agenda=AGENDA,
        save_dir=SAVE_DIR,
        save_name="chest_pain_case",
        team_lead=TEAM_LEAD,
        team_members=TEAM_MEMBERS,
        agenda_questions=AGENDA_QUESTIONS,
        agenda_rules=AGENDA_RULES,
        num_rounds=2,
        temperature=0.2,
        pubmed_search=False,
        return_summary=True,
    )
    print("\n===== CONSENSUS SUMMARY =====\n")
    print(consensus)
