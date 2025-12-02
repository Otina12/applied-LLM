# No-SQL Agent

This project implements a simple HR assistant that manages employees stored in a SQLite database. The assistant uses OpenAI function calling to add, delete, and query employees using natural language.

## Installation

1. Move into the project folder.

2. Create a virtual environment:
```
python -m venv .venv
```

3. Activate it.

Windows:
```
.venv\Scripts\activate
```

macOS or Linux:
```
source .venv/bin/activate
```

4. Install dependencies:
```
pip install -r requirements.txt
```

5. Create a .env file with your OpenAI API key:
```
OPENAI_API_KEY=your_key_here
```

## Running the agent

Open the notebook `src/agent.ipynb` and run it.

## Running tests

Tests require the .env file to be present.

Run:
```
pytest -q
```

## Notes

Some tests may rarely fail because responses from the LLM are nondeterministic.
