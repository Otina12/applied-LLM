def get_system_prompt():
    return """
You are an HR assistant that manages a SQLite employee database.
The user speaks in everyday language and you decide when a database action is needed.
You never write SQL. NEVER. You rely on the provided tools to change or read data.
If you cannot perform an action using tools, do not write SQL and say that you're unable to perform that operation.
Use a tool only when the user asks for an action that affects the employee records.
"""

def get_tools_schema():
    return [
        {
            "type": "function",
            "function": {
                "name": "add_employee",
                "description": "Create a new employee record. Use this when the user says they hired someone or wants someone added to the system.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                        "salary": {"type": "integer"}
                    },
                    "required": ["name", "role", "salary"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "delete_employee",
                "description": "Remove an employee by name. Use this when the user asks to fire someone or delete them from the records.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_employee",
                "description": "Look up one employee by name. Use this when the user wants information such as their salary, role, or existence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_by_role",
                "description": "List all employees with a specific role. Use this when the user asks for people in a department or job category.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"}
                    },
                    "required": ["role"]
                }
            }
        }
    ]
