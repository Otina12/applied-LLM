from typing import List
from pydantic import BaseModel

class Segment(BaseModel):
    model_config = {'extra': 'forbid'}

    id: str
    reason: str
    latex: str

class SegmentFinderResult(BaseModel):
    model_config = {'extra': 'forbid'}

    segments: List[Segment]

class SegmentFinder:
    def __init__(self, client, model = 'gpt-4o-mini'):
        self.client = client
        self.model = model

    def call_segment_finder(self, resume_tex, job_description) -> List[Segment]:
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': USER_PROMPT},
            {'role': 'user', 'content': 'RESUME_TEX:\n' + resume_tex},
            {'role': 'user', 'content': 'JOB_DESCRIPTION:\n' + job_description},
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
Given a full resume written in LaTeX and a job description, select specific LaTeX substrings from the resume that should be improved for the target job. You must return exact substrings copied from RESUME_TEX so the caller can locate and wrap them safely.

Hard rules:
1. Output must be valid JSON and must match the required schema.
2. Every segment.latex must be copied exactly from RESUME_TEX, character for character.
3. Never copy text from JOB_DESCRIPTION into segment.latex.
4. Do not rewrite anything. Do not propose edits. Only select segments and explain why.
5. When selecting content from structured entries like \item or \resumeItem, do not include the command name.
6. For \resumeItem and similar commands that contain a descriptive second argument, select the entire second argument including its surrounding braces.
7. For \item bullets, select the entire item content including its surrounding braces if braces exist in the resume source. If braces do not exist, select the full item text but do not include "\item".
8. segment.latex must not include leading indentation or trailing whitespace.
9. Avoid selecting large blocks such as whole sections, environments, or multiple bullets at once.
10. Do not select personal data such as name, email, phone, address, links, or dates unless a required credential is missing.
11. Do not invent facts. Use only what exists in RESUME_TEX.
12. If you cannot find any safe and useful segments, return an empty list.
13. Do not select brace blocks that contain multiple LaTeX commands, formatting macros, or layout commands.
14. Never select grouped formatting blocks such as skills lists that contain commands like \textbf, \hfill, \emph, or similar.
15. For skills sections, do not select the entire skills block. Select only plain descriptive text blocks, not formatted lists.
16. If a brace wrapped block contains more than one LaTeX command inside it, it must NOT be selected.
17. If no clean, single purpose description exists inside a skills item, skip it entirely.


Rules for brace wrapped description selection:
A. If the resume has a pattern like:
   \resumeItem{Title}{Description}
   then segment.latex must be exactly:
   {Description}
B. The braces must be included in segment.latex.
C. The description must be complete, not partial.
D. The substring must be contiguous and must match exactly as written in RESUME_TEX.
E. Do not include any characters from the command name or its first argument.
F. If the description spans multiple lines in the LaTeX source, you may include those newline characters only if they are inside the braces in RESUME_TEX. Do not add or remove whitespace.
G. A valid segment must represent a single logical description, not a layout or formatting group.
H. If the content inside braces is primarily formatting or label based (for example language lists or technology lists), do not select it.

Selection priorities:
A. Experience descriptions that relate to responsibilities in the job description.
B. Project descriptions that match required tools and scope.
C. Summary statements that can be targeted for keywords.
D. Skills lines that miss important keywords.

Reason guidelines:
For each segment, provide a short reason explaining what is missing relative to the job description, such as missing keywords, missing scope, missing outcomes, missing tools, unclear ownership, or weak phrasing.

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
Use stable ids such as s1, s2, s3, in the same order as they appear in RESUME_TEX.

Quality checks before returning:
1. Each latex value is an exact substring of RESUME_TEX.
2. Each latex value starts with "{" and ends with "}" for command description selections.
3. No latex value contains any job description text that is not present in RESUME_TEX.
4. Segments appear in resume order.
5. There are no duplicates.
6. The JSON output is parseable.
7. Each latex value must not contain multiple LaTeX commands such as \textbf, \hfill, or similar formatting macros.

Example:

LaTeX source:
\resumeItem{Notifications}
          {Service for sending email, push and in app notifications. Involved in features such as delivery time optimization, tracking, queuing and A/B testing. Built an internal app to run batch campaigns for marketing etc.}

Correct selected segment:
{Service for sending email, push and in app notifications. Involved in features such as delivery time optimization, tracking, queuing and A/B testing. Built an internal app to run batch campaigns for marketing etc.}

Incorrect selections:
1. \resumeItem{Notifications}
2. Service for sending email, push and in app notifications.
3. \resumeItem{Notifications}{Service for sending email, push and in app notifications. ...}

Return only JSON, no extra text.
'''

USER_PROMPT = r'''
You will receive two messages:
1) RESUME_TEX
2) JOB_DESCRIPTION

Task:
Return JSON with a list of segments to optimize.

Hard rules:
1) segment.latex must be copied from RESUME_TEX only.
2) Never copy from JOB_DESCRIPTION.
3) Every segment.latex must be an exact substring of RESUME_TEX.
4) When selecting a \resumeItem description, return the full second argument including braces, like {Description}.
5) Do not return partial phrases for brace wrapped descriptions.

Selection rules:
Pick up to 12 segments total.
Prefer descriptions of experience and projects that should be improved to match JOB_DESCRIPTION.
Keep the number of segments small and focused.

Return only JSON, no extra text.
'''