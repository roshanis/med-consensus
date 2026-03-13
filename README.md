# Med Consensus

Med Consensus is a multi-agent clinical discussion tool for complex medical cases. It coordinates a panel of specialists, captures the full discussion, and produces a structured consensus summary for review.

## What it does

- Runs a team-style case discussion across specialist roles
- Collects a readable consensus summary plus raw meeting artifacts
- Uses a Streamlit interface for agenda setup, guardrails, and team configuration
- Supports clarifying questions before the main run
- Can include literature lookup through the underlying framework

## Intended use

- Educational and prototype clinical reasoning support
- Internal case discussion experiments
- Workflow exploration for multispecialty review

This is not medical advice and should not be used without human oversight.

## Quick start

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
cp .env.example .env
```

Add `OPENAI_API_KEY` to `.env`.

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## Typical workflow

1. Enter the case in the agenda field.
2. Generate and answer clarifying questions.
3. Adjust rules and team roles.
4. Run the consensus meeting.
5. Review the summary, transcript, and JSON output.

Saved artifacts land in `medical_meetings/`.

## Repository layout

- `app.py`: Streamlit interface
- `medical_consensus.py`: CLI-oriented flow
- `medical_meetings/`: generated outputs
- `requirements.txt`: Python dependencies

## Status

Current status: active prototype for multispecialty medical case review.

## License

MIT for this app code. Upstream framework licensing remains with the original project.
