# Resume Optimizer (LaTeX Segment Rewriter)

This project takes a LaTeX resume, extracts resume text, extracts important terms from a job description, finds marked segments in the resume, and rewrites only those segments to better match the job description. The output is an updated .tex file and a compiled PDF.

Typical workflow:

1. Start with a LaTeX resume template.
2. Mark rewriteable sections as segments.
3. Provide a job description.
4. Rewrite segments using OpenAI.
5. Compile the updated LaTeX into a PDF.

## Segment marker format

A segment is delimited by comment markers like this:

```tex
%<RESUME_SEGMENT:EXPERIENCE_1:BEGIN>
... content to rewrite ...
%<RESUME_SEGMENT:EXPERIENCE_1:END>
```

Only the content between BEGIN and END is replaced, and the tokens themselves.

## Requirements

### Python

Python 3.10 or newer is recommended.

### System tools

You need a LaTeX distribution and latexmk to compile .tex to PDF.

- Windows: MiKTeX or TeX Live
- macOS: MacTeX
- Linux: TeX Live

latexmk is usually included with TeX Live and MacTeX. On MiKTeX it is included, but it may require Perl and sometimes extra packages.

## Setup

### 1. Clone the repo and enter it

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create and activate a virtual environment

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then activate again.

#### macOS or Linux (bash or zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## OpenAI configuration

This project uses the OpenAI Python SDK.

Set an environment variable called OPENAI_API_KEY.

```
OPENAI_API_KEY=your_api_key
```

## Running the pipeline

Your repo may have a different entry point, but a typical flow is:

1. Put your input resume in `input/resumes/` directory.
2. Put the job description text in `input/jobs/` directory.
3. Modify file names in `optimize_resume.ipynb` and run the notebook.

Expected output should include:
- an optimized .tex file in an output folder (`output/optimized-<resume_name>/optimized-<resume_name>.tex`)
- a compiled PDF (`output/optimized-<resume_name>/optimized-<resume_name>.pdf`)

You can compare existing sample resumes before and after optimization:
- [resume-1](input/resumes/pdfs/resume-1.pdf) and [optimized-resume-1](output/optimized-resume-1/optimized-resume-1.pdf)
- [resume-2](input/resumes/pdfs/resume-2.pdf) and [optimized-resume-2](output/optimized-resume-2/optimized-resume-2.pdf)

Solution is not perfect and can have many edge cases.