import os
import json
from datetime import datetime

def generate_final_report(cleaning_report, engineering_report, training_report):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cleaning_section = _format_cleaning_section(cleaning_report)
    engineering_section = _format_engineering_section(engineering_report)
    training_section = _format_training_section(training_report)

    md = []
    md.append('# Final AutoML Report')
    md.append(f'Generated at {timestamp}')
    md.append('')

    md.append('## 1. Data Cleaning')
    md.append(cleaning_section)
    md.append('')

    md.append('## 2. Feature Engineering')
    md.append(engineering_section)
    md.append('')

    md.append('## 3. Model Training')
    md.append(training_section)
    md.append('')

    md.append('---')
    md.append('Report generation complete.')

    return '\n'.join(md)

def save_report(report_text, output_path):
    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok = True)

    with open(output_path, 'w', encoding = 'utf8') as f:
        f.write(report_text)

def print_summary(cleaning_report, engineering_report, training_report):
    print('')
    print('==============================')
    print('FINAL REPORT SUMMARY')
    print('==============================')
    print('')

    print('Data Cleaning:')
    print(f'  From {cleaning_report["original_shape"]} to {cleaning_report["cleaned_shape"]}')
    print('')

    print('Feature Engineering:')
    print(f'  Target column: {engineering_report["target_column"]}')
    print(f'  Output shape: {engineering_report["output_shape"]}')
    print(f'  Features created: {engineering_report["features_created"]}')
    print('')

    print('Model Training:')
    print(f'  Iterations: {training_report["total_iterations"]}')
    print(f'  Best metrics: {json.dumps(training_report["best_metrics"], indent = 2)}')
    print('')

def _format_cleaning_section(report):
    return (
        f'**Original shape:** {report["original_shape"]}\n'
        f'**Cleaned shape:** {report["cleaned_shape"]}\n\n'
        f'**Summary of actions:**\n'
        f'{report["summary"]}\n\n'
        f'**Output file:** `{report["output_file"]}`'
    )

def _format_engineering_section(report):
    created = report.get('feature_creation_details', [])

    lines = []
    for item in created:
        lines.append(f'- `{item["feature"]}` from `{item["expression"]}`')

    features_block = '\n'.join(lines) if lines else 'No interaction features created.'

    return (
        f'**Input shape:** {report["input_shape"]}\n'
        f'**Output shape:** {report["output_shape"]}\n\n'
        f'**Target column:** `{report["target_column"]}`\n'
        f'**Task type:** {report["task_type"]}\n\n'
        f'**Features created:** {report["features_created"]}\n'
        f'{features_block}\n\n'
        f'**Final feature set:**\n'
        f'{", ".join(report["final_features"])}\n\n'
        f'**Summary:**\n'
        f'{report["summary"]}\n\n'
        f'**Output file:** `{report["output_file"]}`'
    )

def _format_training_section(report):
    best = report.get('best_metrics', {})
    best_json = json.dumps(best, indent = 2) if best else 'No metrics recorded.'

    return (
        f'**Total iterations:** {report["total_iterations"]}\n\n'
        f'**Best metrics:**\n'
        f'{best_json}\n\n'
        f'**Summary:**\n'
        f'{report["summary"]}'
    )
