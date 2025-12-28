import os
import sys
import json
import subprocess
import tempfile
from openai import OpenAI

class ModelTrainerAgent:
    def __init__(self, api_key, logger):
        self.client = OpenAI(api_key = api_key)
        self.model = 'gpt-4o-mini'
        self.logger = logger
        self.conversation_history = []
        self.tools = self._get_tool_definitions()
        self.training_iterations = []
        self.engineering_report = None

    def train_model(self, engineering_report):
        self.engineering_report = engineering_report
        self.training_iterations = []

        os.makedirs('data', exist_ok = True)

        self.conversation_history = [
            {'role': 'system', 'content': self._get_system_prompt()},
            {'role': 'user', 'content': self._get_user_prompt(engineering_report)}
        ]

        self.logger.log('ModelTrainer', 'start', 'Starting model training')
        print('\n=== Model Trainer Starting ===\n')

        max_iterations = 25
        cur_iteration = 0
        finalized = False

        while cur_iteration < max_iterations:
            cur_iteration += 1
            print(f'\n--- Iteration {cur_iteration} ---\n')

            try:
                response = self.client.chat.completions.create(
                    model = self.model,
                    messages = self.conversation_history,
                    tools = self.tools,
                    tool_choice = 'auto'
                )
            except Exception as e:
                error_text = f'The last completion failed with: {str(e)}'
                print(error_text)
                self.logger.log('ModelTrainer', 'api_error', error_text)

                self.conversation_history.append({
                    'role': 'user',
                    'content': error_text + ' Continue.'
                })
                continue

            message = response.choices[0].message

            if message.content:
                print('Model reasoning:\n', message.content)
                self.logger.log('ModelTrainer', 'reasoning', message.content)

            if not message.tool_calls:
                print('No tool call found. Asking model to continue.')
                self.conversation_history.append({
                    'role': 'user',
                    'content': (
                        'You must call execute_python_code or finalize_training. Continue.'
                    )
                })
                continue

            self.conversation_history.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                raw_args = tool_call.function.arguments
                print(f'Model requested tool: {tool_name}')

                try:
                    tool_args = json.loads(raw_args)
                except Exception as e:
                    err = f'Invalid JSON tool arguments: {str(e)}'
                    print(err)
                    self.conversation_history.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': err
                    })
                    continue

                result = self._execute_tool(tool_name, tool_args)

                print('\nTool result:\n')
                print(result[:800])

                self.conversation_history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': result
                })

                if tool_name == 'execute_python_code':
                    if 'Error:' in result or 'Traceback' in result:
                        follow = (
                            'The script failed. Fix the error and call execute_python_code again.'
                        )
                    else:
                        follow = (
                            'Script ran. Read metrics. Tune hyperparameters or finalize_training.'
                        )

                    self.conversation_history.append({
                        'role': 'user',
                        'content': follow
                    })

                elif tool_name == 'finalize_training':
                    print('\n=== Training Finalized by LLM ===\n')
                    finalized = True
                    break

            if finalized:
                break

        if not finalized:
            print('\n=== Auto Finalization Triggered ===\n')

            auto_summary = (
                'The model did not call finalize_training. Auto-finalizing.'
            )
            fallback_metrics = {
                'note': 'LLM did not provide metrics.'
            }

            self._finalize_training(auto_summary, fallback_metrics)

        report_path = 'data/training_report.json'
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)

            report['total_iterations'] = len(self.training_iterations)
            return report

        raise Exception('No training_report.json produced')

    def _execute_python_code(self, code, description):
        print(f'\nExecuting training script: {description}\n')

        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix = '.py', text = True)
            os.close(tmp_fd)

            with open(tmp_path, 'w', encoding = 'utf-8') as f:
                f.write(code)

            completed = subprocess.run(
                [sys.executable, tmp_path],
                capture_output = True,
                text = True,
                timeout = 300
            )

            stdout = completed.stdout or ''
            stderr = completed.stderr or ''
            exit_code = completed.returncode

            print('STDOUT:\n', stdout)
            print('STDERR:\n', stderr)

            self.training_iterations.append({
                'description': description,
                'exit_code': exit_code,
                'stdout': stdout,
                'stderr': stderr
            })

            if exit_code != 0:
                return f'Error: Training script failed.\nSTDERR:\n{stderr}'

            return stdout + '\n' + stderr

        except Exception as e:
            msg = f'Error executing python code: {str(e)}'
            print(msg)
            return msg
        finally:
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except:
                pass

    def _execute_tool(self, tool_name, tool_args):
        if tool_name == 'execute_python_code':
            return self._execute_python_code(
                tool_args['code'],
                tool_args.get('description', 'No description')
            )
        elif tool_name == 'finalize_training':
            return self._finalize_training(
                tool_args['final_summary'],
                tool_args.get('best_metrics', {})
            )
        else:
            return f'Error: Unknown tool {tool_name}'

    def _finalize_training(self, final_summary, best_metrics):
        report = {
            'agent': 'Model Trainer',
            'input_file': self.engineering_report.get('output_file'),
            'task_type': self.engineering_report.get('task_type'),
            'target_column': self.engineering_report.get('target_column'),
            'best_metrics': best_metrics,
            'iterations': self.training_iterations,
            'summary': final_summary,
            'total_iterations': len(self.training_iterations)
        }

        with open('data/training_report.json', 'w') as f:
            json.dump(report, f, indent = 2)

        print('\nSaved training_report.json\n')
        return 'OK'

    def _get_system_prompt(self):
        return '''
You are a Senior Machine Learning Engineer whose responsibility is to write
complete Python training scripts and refine them based on feedback from
previous executions.

You operate in an iterative feedback loop:
1. You generate a full Python script for training and evaluating an XGBoost model.
2. The script is executed in a clean Python environment.
3. You receive the printed metrics, logs, and error messages.
4. You decide whether training is acceptable or if you must improve the script.
5. If improvements are possible, you generate a new revised script.
6. When the results are strong, you finalize training.

============================================================
MODEL TRAINING REQUIREMENTS
============================================================

• You must ALWAYS output a complete runnable Python script whenever you produce code.
  - It must include every import.
  - It must not rely on anything outside the script.
  - It must run from top to bottom without modification.

• You must ALWAYS use train_test_split(test_size=0.2, random_state=42).

• MODEL TYPE:
  - For classification tasks, use XGBClassifier.
  - For regression tasks, use XGBRegressor.

• REQUIRED METRICS:
  Classification:
    - Accuracy
    - Precision
    - Recall
    - F1-Score

  Regression:
    - RMSE (computed using rmse = sqrt(mean_squared_error))
    - MAE
    - R2

• METRICS MUST BE PRINTED CLEARLY so they can be parsed unambiguously.

============================================================
ITERATIVE IMPROVEMENT LOOP
============================================================

After each execution result:
• If the script failed with any Python error, you must:
  - Read the exact error message.
  - Fix the root cause.
  - Produce a corrected full script.
  - Never repeat the same mistake twice.

• If the script ran successfully:
  - Read the metrics.
  - Decide whether performance can improve.
  - If improvement seems possible, modify only a few hyperparameters
    and generate a better script.
  - Do not make random changes. Adjust with purpose.

============================================================
FINALIZATION
============================================================

When you judge that the model performs well enough:
• Produce a short natural-language summary.
• Include the actual best metric values.
• Finalize training.

You must NEVER stop without either:
• Generating a new full Python script for the next iteration, OR
• Finalizing training with the best metrics.

Your decisions must always be based on:
• The engineering report
• The training script output
• The observed metrics
• Any runtime errors

Be intelligent, disciplined, and iterative.
'''

    def _get_user_prompt(self, engineering_report):
        return f'''
You are now beginning the training phase.

DATA INFORMATION:
• Task type: {engineering_report['task_type']}
• CSV path: {engineering_report['output_file']}
• Target column: {engineering_report['target_column']}

YOUR JOB:
1. Write a complete Python script that loads the data from the CSV path.
2. Select all columns except the target as features.
3. Split into train and test sets (20 percent test, random_state=42).
4. Train an XGBoost model:
    - If classification: XGBClassifier.
    - If regression: XGBRegressor.
5. Compute and print all required metrics:
    Classification:
        Accuracy, Precision, Recall, F1
    Regression:
        RMSE, MAE, R2
6. Print metrics clearly to standard output.

AFTER EXECUTION:
You will receive the script output including metrics or Python errors.

YOUR NEXT ACTION MUST BE:
• If the script produced an error:
    - Analyze it.
    - Fix it.
    - Produce a completely new working script.

• If the script succeeded:
    - Evaluate the metrics.
    - Decide whether you should improve them.
    - If improvement is possible, write a new script with tuned hyperparameters.
    - If performance is strong enough, finalize training.

IMPORTANT:
• You must make your own decisions.
• Do not follow fixed recipes.
• Only modify hyperparameters intentionally based on results.
• Always output a fully standalone Python script when producing code.

Begin by analyzing the engineered dataset information above and producing
your first baseline training script.
'''

    def _get_tool_definitions(self):
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'execute_python_code',
                    'description': (
                        "Run a complete standalone Python training script. "
                        "Use this whenever you want to train a model, evaluate metrics, "
                        "test hyperparameters, or verify that your code works. "
                        "Every call must include a FULL runnable script with ALL imports, "
                        "feature selection, data loading, train test split, model training, "
                        "metric printing, and any saving logic. "
                        "After execution, read the printed metrics or error messages "
                        "and decide the next step in training."
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'code': {
                                'type': 'string',
                                'description': (
                                    "A fully self contained Python script. "
                                    "Must include all necessary imports and run top to bottom. "
                                    "This script is executed exactly as provided."
                                )
                            },
                            'description': {
                                'type': 'string',
                                'description': (
                                    "A one line explanation of this training attempt. "
                                    "Example: 'Baseline model', 'Tuned depth', 'Improved learning rate', etc."
                                )
                            }
                        },
                        'required': ['code', 'description']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'finalize_training',
                    'description': (
                        "Call this when you have finished the iterative training process. "
                        "Provide a final natural language summary of what you achieved "
                        "and the BEST metrics obtained during all iterations. "
                        "Only call this once you decide the model is sufficiently good."
                    ),
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'final_summary': {
                                'type': 'string',
                                'description': (
                                    "A concise explanation of the full training journey. "
                                    "Describe which models were tried, how metrics evolved, "
                                    "and why the final model was selected."
                                )
                            },
                            'best_metrics': {
                                'type': 'object',
                                'description': (
                                    "A dictionary containing the best metrics observed. "
                                    "Classification should include accuracy, precision, recall, f1. "
                                    "Regression should include rmse, mae, r2. "
                                    "Use real metric values, not placeholders."
                                )
                            }
                        },
                        'required': ['final_summary', 'best_metrics']
                    }
                }
            }
        ]
