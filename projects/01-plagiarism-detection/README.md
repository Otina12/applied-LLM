# Plagiarism Detection Project

This project contains notebooks and scripts for experimenting with plagiarism detection using embeddings, BM25, FAISS, and model based evaluation.

## Setup

1. Move into the project folder.
2. Create a virtual environment.

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

4. Install all required packages.

   ```
   pip install -r requirements.txt
   ```

5. Create a .env file with your OpenAI API key:

   ```
   OPENAI_API_KEY=your_key_here
   ```

## Running

Open the notebooks in Jupyter or VS Code and run the cells. All steps for downloading data, building indexes, and running evaluations are included inside the notebooks.
