import json
from typing import List
from pydantic import BaseModel, Field

class Segment(BaseModel):
    id: str
    reason: str
    latex: str

class SegmentFinderResult(BaseModel):
    segments: List[Segment] = Field(default_factory = list)

class SegmentFinder:
    def __init__(self, client, model = 'gpt-4o-mini'):
        self.client = client
        self.model = model

    def call_segment_finder(self, resume_tex, job_description) -> List[Segment]:
        payload = {
            'resume_tex': resume_tex,
            'job_description': job_description
        }

        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': USER_PROMPT},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii = False)}
        ]

        response = self.client.chat.completions.create(
            model = self.model,
            messages = messages,
            response_format = {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'segment_finder_result',
                    'strict': True,
                    'schema': SegmentFinderResult.model_json_schema()
                }
            }
        )

        content = response.choices[0].message.content
        if not content:
            return []

        result = SegmentFinderResult.model_validate_json(content)
        return result.segments


SYSTEM_PROMPT = r'''
You are a resume LaTeX segment finder.

Goal:
Given a full resume in LaTeX and a job description, select specific LaTeX snippets that should be improved for the target job. You must return exact substrings copied from the resume input, so the caller can locate and wrap them.

Hard rules:
1. Output must be valid JSON and must match the required schema.
2. Every segment.latex must be copied exactly from the provided resume_tex, character for character.
3. Do not rewrite anything. Do not propose edits. Only select segments and explain why.
4. Keep segments small and focused. Prefer single bullet items, single summary sentences, single skills lines, single project bullets.
5. Avoid selecting large blocks like an entire section, an entire environment, or many bullets at once.
6. Do not include LaTeX that contains the segment markers, assume the resume has no markers yet.
7. Do not select segments that are already tightly aligned with the job description unless there is a clear missing keyword or missing responsibility match.
8. Do not select personal data fields like name, email, phone, address, links, or education dates unless the job requires a specific credential that is missing from an existing line.
9. Do not invent facts. Use only what exists in the resume.
10. If you cannot find any safe and useful segments, return an empty list.

Selection priorities:
A. Experience bullets that describe impact, ownership, scope, tools, and outcomes.
B. Project bullets that match the job responsibilities.
C. Summary or profile lines that can be targeted.
D. Skills lines that can be reorganized or modified to better match keywords.

Reason guidelines:
For each segment, provide a short reason describing what is missing relative to the job description, such as missing keywords, missing scope, missing outcomes, missing tools, unclear ownership, or weak phrasing.

Output schema:
{
  "segments": [
    {
      "id": "s1",
      "reason": "string",
      "latex": "string"
    }
  ]
}

Id rules:
Use stable ids like s1, s2, s3, in the same order they appear in the resume_tex.

Quality checks before returning:
1. Verify each latex value is an exact substring of resume_tex.
2. Verify segments appear in resume order.
3. Verify there are no duplicates.
4. Verify JSON is parseable.

Return only JSON, no extra text.
'''


USER_PROMPT = r'''
You will receive two inputs as JSON:
1. resume_tex, a full LaTeX resume.
2. job_description, the target role description.

Task:
Return JSON with a list of segments to optimize, following the system rules.

What to select:
Pick up to 12 segments total.
Each segment must be an exact copy of a snippet from resume_tex.
Prefer bullets and short lines.
Each selected segment must have a clear reason tied to job_description.

What not to do:
Do not rewrite or suggest changes.
Do not add new text.
Do not include anything that is not present in resume_tex.

Return only JSON, no extra text.

Example output:
{
  "segments": [
    {
      "id": "s1",
      "reason": "Mentions APIs but misses authentication and security keywords present in the job description.",
      "latex": "\\item Built APIs for payments."
    }
  ]
}
'''