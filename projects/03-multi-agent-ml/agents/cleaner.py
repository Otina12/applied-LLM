import json
import pandas as pd
from openai import OpenAI

class DataCleanerAgent:
    def __init__(self, api_key, logger):
        self.client = OpenAI(api_key = api_key)
        self.model = 'gpt-4o-mini'
        self.logger = logger
        self.conversation_history = []
        self.tools = self._get_tool_definitions()

    def clean_data(self, input_path, output_path = 'data/clean_data.csv'):
        self.conversation_history = [
            {'role': 'system', 'content': self._get_system_prompt()},
            {'role': 'user', 'content': self._get_user_prompt(input_path, output_path)}
        ]
        
        max_iterations = 25
        cur_iteration = 0
        
        self.logger.log('DataCleaner', 'start', 'Starting data cleaning', {'input': input_path})
        
        while cur_iteration < max_iterations:
            cur_iteration += 1
            
            response = self.client.chat.completions.create(
                model = self.model,
                messages = self.conversation_history,
                tools = self.tools,
                tool_choice = 'auto'
            )
            
            message = response.choices[0].message
            
            if message.content:
                self.logger.log('DataCleaner', 'reasoning', message.content)
            
            if not message.tool_calls:
                break
            
            self.conversation_history.append(message)
            
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                self.logger.log('DataCleaner', 'tool_call', f'Calling tool {tool_name}', tool_args)

                result = self._execute_tool(tool_name, tool_args)
                
                self.logger.log('DataCleaner', 'tool_result', f'Tool {tool_name} returned', {'result_preview': result[:500]})
                
                self.conversation_history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': result
                })
                
                if tool_name == 'finalize_cleaning':
                    self.logger.log('DataCleaner', 'finish', 'Cleaning completed')

                    with open('data/cleaning_report.json', 'r') as f:
                        return json.load(f)
        
        raise Exception('Agent did not finalize cleaning within iteration limit')
    
    def _inspect_metadata(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        self.original_shape = self.df.shape
        
        info = {
            'shape': f'{self.df.shape[0]} rows × {self.df.shape[1]} columns',
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'null_counts': self.df.isnull().sum().to_dict(),
            'null_percentages': (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict(),
            'memory_usage': f'{self.df.memory_usage(deep = True).sum() / 1024:.2f} KB'
        }
        
        return json.dumps(info, indent = 2)
    
    def _get_column_stats(self, column_name):
        if column_name not in self.df.columns:
            return f'Error: Column "{column_name}" not found'
        
        col = self.df[column_name]
        stats = {
            'column': column_name,
            'dtype': str(col.dtype),
            'null_count': int(col.isnull().sum()),
            'null_percentage': f'{col.isnull().sum() / len(col) * 100:.2f}%',
            'unique_count': int(col.nunique()),
            'unique_percentage': f'{col.nunique() / len(col) * 100:.2f}%'
        }
        
        if pd.api.types.is_numeric_dtype(col):
            stats.update({
                'mean': float(col.mean()) if not col.isnull().all() else None,
                'median': float(col.median()) if not col.isnull().all() else None,
                'std': float(col.std()) if not col.isnull().all() else None,
                'min': float(col.min()) if not col.isnull().all() else None,
                'max': float(col.max()) if not col.isnull().all() else None,
                'quartiles': col.quantile([0.25, 0.5, 0.75]).to_dict() if not col.isnull().all() else None
            })
        else:
            value_counts = col.value_counts().head(10).to_dict()
            stats['top_10_values'] = {str(k): int(v) for k, v in value_counts.items()}
        
        return json.dumps(stats, indent = 2)
    
    def _impute_missing(self, column_name, strategy, fill_value = None):
        if column_name not in self.df.columns:
            return f'Error: Column "{column_name}" not found'
        
        missing_before = self.df[column_name].isnull().sum()
        
        if missing_before == 0:
            return f'No missing values in column "{column_name}"'
        
        if strategy == 'mean':
            self.df[column_name].fillna(self.df[column_name].mean(), inplace = True)
        elif strategy == 'median':
            self.df[column_name].fillna(self.df[column_name].median(), inplace = True)
        elif strategy == 'mode':
            mode_val = self.df[column_name].mode()[0] if not self.df[column_name].mode().empty else None
            if mode_val is not None:
                self.df[column_name].fillna(mode_val, inplace = True)
        elif strategy == 'constant':
            self.df[column_name].fillna(fill_value, inplace = True)
        
        missing_after = self.df[column_name].isnull().sum()
        
        return f'Imputed {missing_before - missing_after} missing values in "{column_name}" using {strategy} strategy'
    
    def _drop_column(self, column_name, reason):
        if column_name not in self.df.columns:
            return f'Error: Column "{column_name}" not found'
        
        self.df.drop(columns = [column_name], inplace = True)
        return f'Dropped column "{column_name}". Reason: {reason}'
    
    def _convert_dtype(self, column_name, target_dtype):
        if column_name not in self.df.columns:
            return f'Error: Column "{column_name}" not found'
        
        try:
            if target_dtype == 'int':
                self.df[column_name] = pd.to_numeric(self.df[column_name], errors = 'coerce').astype('Int64')
            elif target_dtype == 'float':
                self.df[column_name] = pd.to_numeric(self.df[column_name], errors = 'coerce')
            elif target_dtype == 'string':
                self.df[column_name] = self.df[column_name].astype(str)
            elif target_dtype == 'datetime':
                self.df[column_name] = pd.to_datetime(self.df[column_name], errors = 'coerce')
            elif target_dtype == 'category':
                self.df[column_name] = self.df[column_name].astype('category')
            
            return f'Converted column "{column_name}" to {target_dtype}'
        except Exception as e:
            return f'Error converting column "{column_name}": {str(e)}'
    
    def _finalize_cleaning(self, output_path, summary):
        self.df.to_csv(output_path, index = False)
        
        report = {
            'agent': 'Data Cleaner',
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'summary': summary,
            'output_file': output_path
        }
        
        with open('data/cleaning_report.json', 'w') as f:
            json.dump(report, f, indent = 2)
        
        return f'Cleaning completed. Saved to {output_path}. Report saved to data/cleaning_report.json'
    
    def _execute_tool(self, tool_name, tool_args):
        try:
            if tool_name == 'inspect_metadata':
                return self._inspect_metadata(tool_args['dataset_path'])
            elif tool_name == 'get_column_stats':
                return self._get_column_stats(tool_args['column_name'])
            elif tool_name == 'impute_missing':
                return self._impute_missing(
                    tool_args['column_name'],
                    tool_args['strategy'],
                    tool_args.get('fill_value')
                )
            elif tool_name == 'drop_column':
                return self._drop_column(tool_args['column_name'], tool_args['reason'])
            elif tool_name == 'convert_dtype':
                return self._convert_dtype(tool_args['column_name'], tool_args['target_dtype'])
            elif tool_name == 'finalize_cleaning':
                return self._finalize_cleaning(tool_args['output_path'], tool_args['summary'])
            else:
                return f'Error: Unknown tool {tool_name}'
        except Exception as e:
            return f'Error executing {tool_name}: {str(e)}'

    def _get_system_prompt(self):
        return '''
You are an expert Data Auditor and Cleaner. Your mission is to ensure data quality.

Your responsibilities:
1. Inspect the dataset thoroughly using available tools
2. Identify and fix data quality issues:
   - Missing values (decide best imputation strategy per column)
   - Wrong data types (convert as needed)
   - Useless columns (IDs, high cardinality, too many nulls)
   - Outliers and anomalies
3. Make intelligent decisions based on the data, not hardcoded rules
4. Document every action and reasoning

Process:
1. Start with inspect_metadata to understand the dataset
2. Use get_column_stats for columns that need investigation
3. Apply cleaning operations (impute, drop, convert) as needed
4. When satisfied, call finalize_cleaning with a comprehensive summary

Be thorough but efficient. Make data-driven decisions.
'''

    def _get_user_prompt(self, input_path, output_path):
        return f'''
Please clean the dataset located at: {input_path}

Analyze the data carefully and apply appropriate cleaning operations. The output should be saved to: {output_path}

Remember:
- Examine all columns and their characteristics
- Make intelligent decisions about handling missing values
- Remove columns that won't help in modeling
- Ensure data types are correct
- Provide clear reasoning for every action
'''

    def _get_tool_definitions(self):
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'inspect_metadata',
                    'description': 'Inspects the dataset metadata including shape, column names, data types, null counts, and basic statistics. Use this first to understand the dataset structure.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'dataset_path': {
                                'type': 'string',
                                'description': 'Path to the CSV file to inspect'
                            }
                        },
                        'required': ['dataset_path']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_column_stats',
                    'description': 'Get detailed statistics for a specific column including distribution, unique values, and patterns. Use this to understand individual columns better.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'column_name': {
                                'type': 'string',
                                'description': 'Name of the column to analyze'
                            }
                        },
                        'required': ['column_name']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'impute_missing',
                    'description': 'Fill missing values in a column using the specified strategy. Choose the strategy based on data type and distribution.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'column_name': {
                                'type': 'string',
                                'description': 'Name of the column to impute'
                            },
                            'strategy': {
                                'type': 'string',
                                'enum': ['mean', 'median', 'mode', 'constant'],
                                'description': 'Imputation strategy: mean (numeric), median (numeric), mode (categorical), or constant'
                            },
                            'fill_value': {
                                'type': 'string',
                                'description': 'Value to use when strategy is "constant"'
                            }
                        },
                        'required': ['column_name', 'strategy']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'drop_column',
                    'description': 'Remove a column from the dataset. Use this for columns that are not useful (e.g., IDs, duplicates, or too many missing values).',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'column_name': {
                                'type': 'string',
                                'description': 'Name of the column to drop'
                            },
                            'reason': {
                                'type': 'string',
                                'description': 'Reason for dropping the column'
                            }
                        },
                        'required': ['column_name', 'reason']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'convert_dtype',
                    'description': 'Convert a column to a different data type. Use this to fix wrong data types.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'column_name': {
                                'type': 'string',
                                'description': 'Name of the column to convert'
                            },
                            'target_dtype': {
                                'type': 'string',
                                'enum': ['int', 'float', 'string', 'datetime', 'category'],
                                'description': 'Target data type'
                            }
                        },
                        'required': ['column_name', 'target_dtype']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'finalize_cleaning',
                    'description': 'Save the cleaned dataset and generate a summary report. Call this when you are satisfied with the data quality.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'output_path': {
                                'type': 'string',
                                'description': 'Path where to save the cleaned data'
                            },
                            'summary': {
                                'type': 'string',
                                'description': 'A comprehensive summary of all cleaning actions taken and reasoning'
                            }
                        },
                        'required': ['output_path', 'summary']
                    }
                }
            }
        ]