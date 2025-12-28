import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from openai import OpenAI

class FeatureEngineerAgent:
    def __init__(self, api_key, logger):
        self.client = OpenAI(api_key = api_key)
        self.model = 'gpt-4o-mini'
        self.logger = logger
        self.conversation_history = []
        self.tools = self._get_tool_definitions()
        self.feature_creation_log = []

    def engineer_features(self, input_path, cleaning_report, output_path = 'data/engineered_data.csv'):
        self.df = pd.read_csv(input_path)
        self.original_shape = self.df.shape
        self._infer_target_info()

        self.conversation_history = [
            {'role': 'system', 'content': self._get_system_prompt()},
            {'role': 'user', 'content': self._get_user_prompt(cleaning_report, output_path)}
        ]
        
        max_iterations = 25
        cur_iteration = 0
        
        self.logger.log('FeatureEngineer', 'start', 'Starting feature engineering')
        
        while cur_iteration < max_iterations:
            cur_iteration += 1
            
            try:
                response = self.client.chat.completions.create(
                    model = self.model,
                    messages = self.conversation_history,
                    tools = self.tools,
                    tool_choice = 'auto'
                )
            except Exception as e:
                self.conversation_history.append({
                    'role': 'user',
                    'content': f'The last completion failed with: {str(e)}. Continue from the last step.'
                })
                continue
            
            message = response.choices[0].message
            
            if message.content:
                self.logger.log('FeatureEngineer', 'reasoning', message.content)
            
            if not message.tool_calls:
                break
            
            self.conversation_history.append(message)
            
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                raw_args = tool_call.function.arguments

                try:
                    tool_args = json.loads(raw_args)
                except Exception as e:
                    self.conversation_history.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': f'Error: invalid JSON arguments. Details: {str(e)}'
                    })
                    self.conversation_history.append({
                        'role': 'user',
                        'content': 'Your last tool arguments were not valid JSON. Call the tool again with correct JSON.'
                    })
                    continue
                
                self.logger.log('FeatureEngineer', 'tool_call', f'Calling tool {tool_name}', tool_args)
                
                result = self._execute_tool(tool_name, tool_args)
                
                self.logger.log('FeatureEngineer', 'tool_result', f'Tool {tool_name} returned', {'result_preview': result[:500]})
                
                self.conversation_history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': result
                })

                if 'Error' in result:
                    self.conversation_history.append({
                        'role': 'user',
                        'content': (
                            'The previous tool call produced an error. '
                            'Review your reasoning, fix the issue, and call a corrected tool such as create_interaction, encode_categorical, correlation_analysis, select_top_features, or finalize_engineering.'
                        )
                    })
                    continue
                
                if tool_name == 'finalize_engineering':
                    self.logger.log('FeatureEngineer', 'finish', 'Feature engineering completed')
                    
                    with open('data/engineering_report.json', 'r') as f:
                        return json.load(f)
                
                self.conversation_history.append({
                    'role': 'user',
                    'content': (
                        'Tool executed successfully. Continue feature engineering. '
                        'If you are finished, call finalize_engineering tool.'
                    )
                })
        
        auto_summary = 'Feature engineering completed without an explicit finalize_engineering call. Finalizing with the current dataframe.'
        finalize_result = self._finalize_engineering(output_path, auto_summary)
        self.logger.log('FeatureEngineer', 'auto_finalize', finalize_result)

        if os.path.exists('data/engineering_report.json'):
            with open('data/engineering_report.json', 'r') as f:
                return json.load(f)

        raise Exception('Agent did not finalize engineering within iteration limit')

    def _infer_target_info(self):
        if not hasattr(self, 'df'):
            raise Exception('Dataframe not loaded. Load dataset before target inference.')

        columns_info = {}

        for col in self.df.columns:
            col_values = self.df[col].dropna()
            sample_vals = col_values.head(5).tolist()
            dtype = str(self.df[col].dtype)
            unique_count = self.df[col].nunique()

            columns_info[col] = {
                "dtype": dtype,
                "unique_count": int(unique_count),
                "sample_values": sample_vals
            }

        prompt = {
            "instruction": "Identify the correct target column and the correct machine learning task type",
            "rules": (
                "Return a JSON object with fields target_column and task_type "
                "Task types allowed are classification or regression "
                "If the target represents categories then use classification "
                "If the target is continuous then use regression "
                "Think carefully about the meaning of the column and its values "
                "Never return placeholders. Always return real values."
            ),
            "columns": columns_info
        }

        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {"role": "system", "content": "You analyze dataset columns and decide the true target column and task type. Always return valid JSON only."},
                {"role": "user", "content": json.dumps(prompt)}
            ],
            response_format = { # don't have time for Pydantic ;(
                "type": "json_schema",
                "json_schema": {
                    "name": "target_info_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "target_column": {"type": "string"},
                            "task_type": {"type": "string", "enum": ["classification", "regression"]}
                        },
                        "required": ["target_column", "task_type"]
                    }
                }
            }
        )

        raw = response.choices[0].message.content

        try:
            data = json.loads(raw)
        except Exception:
            raise Exception(f'Model returned invalid JSON: {raw}')

        chosen = data.get('target_column')
        task_type = data.get('task_type')

        if chosen not in self.df.columns:
            raise Exception(f'Model selected {chosen} but it is not a dataframe column.')

        self.target_column = chosen
        self.task_type = task_type

        cols = [c for c in self.df.columns if c != self.target_column] + [self.target_column]
        self.df = self.df[cols]

    def _create_interaction(self, new_column_name, expression, reasoning):
        try:
            self.df[new_column_name] = eval(expression, {'df': self.df, 'np': np, 'pd': pd})
            
            self.feature_creation_log.append({
                'feature': new_column_name,
                'expression': expression,
                'reasoning': reasoning
            })
            
            new_col = self.df[new_column_name]
            stats = {
                'created': new_column_name,
                'dtype': str(new_col.dtype),
                'null_count': int(new_col.isnull().sum()),
                'null_percentage': f'{new_col.isnull().sum() / len(new_col) * 100:.2f}%'
            }
            
            if pd.api.types.is_numeric_dtype(new_col):
                stats.update({
                    'mean': float(new_col.mean()) if not new_col.isnull().all() else None,
                    'std': float(new_col.std()) if not new_col.isnull().all() else None,
                    'min': float(new_col.min()) if not new_col.isnull().all() else None,
                    'max': float(new_col.max()) if not new_col.isnull().all() else None
                })
            
            stats['sample_values'] = new_col.head(5).tolist()
            
            return json.dumps(stats, indent = 2)
            
        except Exception as e:
            return f'Error creating interaction feature: {str(e)}. Check your expression syntax.'

    def _encode_categorical(self, column_name, encoding_type):
        if column_name not in self.df.columns:
            return f'Error: Column "{column_name}" not found'
        
        original_shape = self.df.shape
        
        if encoding_type == 'onehot':
            dummies = pd.get_dummies(self.df[column_name], prefix = column_name, drop_first = True)
            self.df = pd.concat([self.df, dummies], axis = 1)
            self.df.drop(columns = [column_name], inplace = True)
            
            result = {
                'encoding_type': 'one-hot',
                'original_column': column_name,
                'new_columns': list(dummies.columns),
                'columns_created': len(dummies.columns),
                'original_shape': original_shape,
                'new_shape': self.df.shape
            }
            
            return json.dumps(result, indent = 2)
            
        elif encoding_type == 'label':
            try:
                self.df[column_name] = self.df[column_name].astype('category').cat.codes
            except:
                return 'Error: Could not apply label encoding'
            
            result = {
                'encoding_type': 'label',
                'column': column_name,
                'encoded_values': f'0 to {self.df[column_name].max()}',
                'sample_mapping': self.df[column_name].value_counts().head(5).to_dict()
            }
            
            return json.dumps(result, indent = 2)

    def _correlation_analysis(self):
        if not hasattr(self, 'target_column'):
            return 'Error: Target column not set. Call set_target_column first before correlation analysis.'
        
        feature_cols = [col for col in self.df.columns if col != self.target_column]
        numeric_features = self.df[feature_cols].select_dtypes(include = [np.number]).columns.tolist()
        
        if len(numeric_features) == 0:
            return 'Error: No numeric features available for correlation analysis. Encode categorical features first.'
        
        X = self.df[numeric_features].fillna(0)
        y = self.df[self.target_column]
        
        try:
            if self.task_type == 'classification':
                y = pd.factorize(y)[0]
                mi_scores = mutual_info_classif(X, y, random_state = 42)
            else:
                mi_scores = mutual_info_regression(X, y, random_state = 42)
        except Exception as e:
            return f'Error: Failed to compute mutual information: {str(e)}'
        
        mi_scores = pd.Series(mi_scores, index=numeric_features).sort_values(ascending = False)
        
        result = {
            'task_type': self.task_type,
            'target_column': self.target_column,
            'total_features_analyzed': len(numeric_features),
            'top_10_features': mi_scores.head(10).to_dict(),
            'bottom_5_features': mi_scores.tail(5).to_dict(),
            'mean_mi_score': float(mi_scores.mean()),
            'max_mi_score': float(mi_scores.max()),
            'min_mi_score': float(mi_scores.min())
        }
        
        return json.dumps(result, indent = 2)

    def _select_top_features(self, k):
        if not hasattr(self, 'target_column'):
            return 'Error: Target column not set. Cannot select features without knowing the target.'
        
        feature_cols = [col for col in self.df.columns if col != self.target_column]
        numeric_features = self.df[feature_cols].select_dtypes(include = [np.number]).columns.tolist()
        
        if len(numeric_features) <= k:
            return f'Already have {len(numeric_features)} numeric features, which is <= {k}. No selection needed.'
        
        X = self.df[numeric_features].fillna(0)
        y = self.df[self.target_column]
        
        try:
            if self.task_type == 'classification':
                y = pd.factorize(y)[0]
                mi_scores = mutual_info_classif(X, y, random_state = 42)
            else:
                mi_scores = mutual_info_regression(X, y, random_state = 42)
        except Exception as e:
            return f'Error: Could not compute feature selection: {str(e)}'
        
        mi_scores = pd.Series(mi_scores, index = numeric_features).sort_values(ascending = False)
        selected_features = mi_scores.head(k).index.tolist()
        dropped_features = [f for f in numeric_features if f not in selected_features]
        
        self.df = self.df[selected_features + [self.target_column]]
        
        result = {
            'k': k,
            'selected_features': selected_features,
            'dropped_features': dropped_features[:10],
            'dropped_count': len(dropped_features),
            'new_shape': self.df.shape,
            'feature_scores': mi_scores.head(k).to_dict()
        }
        
        return json.dumps(result, indent=2)

    def _finalize_engineering(self, output_path, summary):
        try:
            self.df.to_csv(output_path, index = False)
        except Exception as e:
            return f'Error: Could not save engineered dataset: {str(e)}'
        
        report = {
            'agent': 'Feature Engineer',
            'input_shape': self.original_shape,
            'output_shape': self.df.shape,
            'target_column': self.target_column,
            'task_type': self.task_type,
            'features_created': len(self.feature_creation_log),
            'feature_creation_details': self.feature_creation_log,
            'final_features': list(self.df.columns),
            'summary': summary,
            'output_file': output_path
        }
        
        try:
            with open('data/engineering_report.json', 'w') as f:
                json.dump(report, f, indent = 2)
        except Exception as e:
            return f'Error: Could not write engineering report: {str(e)}'
        
        return f'Feature engineering completed successfully. Saved {self.df.shape[0]} rows × {self.df.shape[1]} columns to {output_path}. Report saved to data/engineering_report.json'

    def _execute_tool(self, tool_name, tool_args):
        try:
            if tool_name == 'create_interaction':
                return self._create_interaction(tool_args['new_column_name'], tool_args['expression'], tool_args['reasoning'])
            elif tool_name == 'encode_categorical':
                return self._encode_categorical(tool_args['column_name'], tool_args['encoding_type'])
            elif tool_name == 'correlation_analysis':
                return self._correlation_analysis()
            elif tool_name == 'select_top_features':
                return self._select_top_features(tool_args['k'])
            elif tool_name == 'finalize_engineering':
                return self._finalize_engineering(tool_args['output_path'], tool_args['summary'])
            else:
                return f'Error: Unknown tool {tool_name}'
        except Exception as e:
            return f'Error executing {tool_name}: {str(e)}'

    def _get_system_prompt(self):
        return '''
You are an expert Feature Architect and Data Scientist. Your mission is to maximize the predictive power of the dataset through intelligent feature engineering.

Your responsibilities:
1. Understand the data structure and relationships
2. Create meaningful interaction features using domain knowledge
3. Encode categorical variables appropriately
4. Analyze feature importance with correlation analysis
5. Select the most relevant features to optimize model performance

Process and best practices:
1. Start by analyzing the data structure to understand what you are working with
2. Think about domain logic. Create interaction features that make sense for the data
3. For categorical variables:
   Use one hot encoding for low cardinality
   Use label encoding for higher cardinality
4. Run correlation analysis to understand which features matter for prediction
5. Select top features based on mutual information
6. Finalize with a summary of decisions and reasoning

Make decisions based on the data.
'''

    def _get_user_prompt(self, cleaning_report, output_path):
        return f'''
Context from Data Cleaner Agent:
{json.dumps(cleaning_report, indent = 2)}

Your tasks:
1. Analyze the data and understand available features
2. Create two to four or more interaction features based on domain logic
3. Encode categorical variables as needed
4. Run correlation analysis
5. Select the best features
6. Save the engineered dataset to: {output_path}

Be creative with logic but make sure it fits the data.
'''

    def _get_tool_definitions(self):
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'create_interaction',
                    'description': 'Create a new feature from mathematical operations on existing columns.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'new_column_name': {'type': 'string'},
                            'expression': {'type': 'string'},
                            'reasoning': {'type': 'string'}
                        },
                        'required': ['new_column_name', 'expression', 'reasoning']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'encode_categorical',
                    'description': 'Encode a categorical column using one hot or label encoding.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'column_name': {'type': 'string'},
                            'encoding_type': {'type': 'string', 'enum': ['onehot', 'label']}
                        },
                        'required': ['column_name', 'encoding_type']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'correlation_analysis',
                    'description': 'Analyze correlation between numeric features and the target.',
                    'parameters': {
                        'type': 'object',
                        'properties': {}
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'select_top_features',
                    'description': 'Select the k most important features based on mutual information.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'k': {'type': 'integer', 'minimum': 1}
                        },
                        'required': ['k']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'finalize_engineering',
                    'description': 'Save the engineered dataset and write a summary.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'output_path': {'type': 'string'},
                            'summary': {'type': 'string'}
                        },
                        'required': ['output_path', 'summary']
                    }
                }
            }
        ]
