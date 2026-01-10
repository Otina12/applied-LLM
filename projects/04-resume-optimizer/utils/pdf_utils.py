import subprocess
from pathlib import Path
import fitz # PyMuPDF

@staticmethod
def compile_tex_to_pdf(tex_path, output_dir) -> Path:
    if not tex_path.exists():
        raise FileNotFoundError(f'TeX file not found: {tex_path}')

    output_dir.mkdir(parents = True, exist_ok = True)

    generate_pdf_cmd = [
        'latexmk',
        '-pdf',
        '-interaction=nonstopmode',
        '-halt-on-error',
        f'-outdir={output_dir}',
        str(tex_path)
    ]

    result = subprocess.run(
        generate_pdf_cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True
    )

    if result.returncode != 0:
        raise RuntimeError('LaTeX compilation failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}')
    
    clean_aux_files_cmd = [
        'latexmk',
        '-c',
        f'-outdir={output_dir}',
        str(tex_path)
    ]

    subprocess.run(
        clean_aux_files_cmd,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL
    )

    pdf_path = output_dir / f'{tex_path.stem}.pdf'
    if not pdf_path.exists():
        raise RuntimeError('PDF was not generated.')

    return pdf_path

@staticmethod
def extract_text_from_pdf(pdf_path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f'PDF not found: {pdf_path}')

    text_parts = []

    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text()
            if text:
                text_parts.append(text)

    return '\n'.join(text_parts)