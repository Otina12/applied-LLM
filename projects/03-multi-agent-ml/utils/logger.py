import json
from datetime import datetime
import os

class AgentLogger:
    def __init__(self, path):
        self.path = path
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

    def clear(self):
        open(self.path, 'w').close()

    def log(self, agent, event, message = None, params = None):
        entry = {
            'agent': agent,
            'event': event,
            'message': message,
            'params': params,
            'time': datetime.utcnow().isoformat()
        }
        with open(self.path, 'a', encoding = 'utf8') as f:
            f.write(json.dumps(entry) + '\n')
