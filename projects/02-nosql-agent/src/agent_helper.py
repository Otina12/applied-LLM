import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from src.tools import add_employee, delete_employee, find_employee, find_by_role
from db.db import init_db


class HRChatAgent:
    def __init__(self, system_prompt, tools_schema):
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("OPENAI_API_KEY was not found.")

        self.client = OpenAI(api_key=api_key)
        self.system_prompt = system_prompt
        self.tools_schema = tools_schema

        init_db()

    def call_tool(self, name, args):
        if name == "add_employee":
            return add_employee(**args)
        if name == "delete_employee":
            return delete_employee(**args)
        if name == "find_employee":
            return find_employee(**args)
        if name == "find_by_role":
            return find_by_role(**args)

        return {"error": "unknown tool"}

    def chat(self, user_text):
        msg = self.client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text}
            ],
            tools = self.tools_schema
        )

        reply = msg.choices[0].message

        if reply.tool_calls:
            call = reply.tool_calls[0]
            tool_name = call.function.name
            args = json.loads(call.function.arguments)

            result = self.call_tool(tool_name, args)

            final_msg = self.client.chat.completions.create(
                model = "gpt-4o-mini",
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an HR assistant. Summarize the tool result for "
                            "the user in natural language. Do not mention JSON or technical "
                            "details. Confirm what happened in a clear, simple sentence."
                        )
                    },
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": f"Tool result: {json.dumps(result)}"}
                ]
            )

            return final_msg.choices[0].message.content

        return reply.content
