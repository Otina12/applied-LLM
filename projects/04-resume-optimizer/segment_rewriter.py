import json
from pydantic import BaseModel

class RewriteResult(BaseModel):
    model_config = {'extra': 'forbid'}

    rewritten_latex: str
    notes: str

class RewriteResultWithOriginal(BaseModel):
    segment_id: str
    original_latex: str
    rewritten_latex: str
    notes: str

class SegmentRewriter:
    def __init__(self, client, model = 'gpt-4o-mini'):
        self.client = client
        self.model = model

    def call_segment_rewriter(self, segment_id, segment_latex, job_description, context = None) -> RewriteResultWithOriginal:
        payload = {
            'segment_latex': segment_latex,
            'job_description': job_description
        }

        if context is not None:
            payload['context'] = context

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
                    'name': 'segment_rewrite_result',
                    'strict': True,
                    'schema': RewriteResult.model_json_schema()
                }
            }
        )

        content = response.choices[0].message.content
        if not content:
            return RewriteResultWithOriginal(segment_id = segment_id, original_latex = segment_latex, rewritten_latex = segment_latex, notes = '')

        result = RewriteResult.model_validate_json(content)

        return RewriteResultWithOriginal(segment_id = segment_id, original_latex = segment_latex, rewritten_latex = result.rewritten_latex, notes = result.notes)


SYSTEM_PROMPT = r'''
You are a LaTeX resume segment rewriter.

Goal:
Given one LaTeX segment from a resume and a job description, rewrite the segment so it matches the job more closely while preserving the original meaning and facts.

You may be given a context object that contains keywords from the job descriptions, ones that can or may be used (will be specified in context).

Use context to keep the rewrite consistent with the rest of the resume, but only rewrite segment_latex.
Do not copy unrelated content from context into the segment.

Hard rules:
1. Output must be valid JSON and must match the required schema.
2. Do not invent facts, numbers, tools, employers, titles, dates, certifications, or outcomes.
3. Keep the same story. You may reorder or modify words and add job relevant keywords only if they are truthful given the original segment.
4. Preserve LaTeX correctness. Return LaTeX for only this segment. Do not add document headers or section wrappers.
5. Keep the same LaTeX structure when possible. If input is a single \\item, output must be a single \\item.
6. Do not add new packages, new commands, or new environments.
7. Do not change proper nouns, product names, or company names.
8. Do not change time statements such as "currently" or "previously" unless the original text already implies that.
9. Keep character length controlled. Aim for within plus or minus 10 percent of the input segment length, unless the input is extremely short.
10. Avoid empty claims. Prefer specific responsibilities and outcomes that already exist in the text.

Optimization goals:
A. Add missing role keywords that are compatible with the segment content.
B. Clarify ownership and scope using existing facts.
C. Improve readability and ATS matching without changing meaning.
D. If the job description mentions specific tools or practices, include them only if the segment already implies them. If not implied, do not add them.

Safe keyword enrichment rules:
You may add general terms that do not create new facts, such as:
scalable, reliable, performance, monitoring, testing, security, documentation, stakeholders, cross functional
Only if they fit the original meaning.

Forbidden actions:
1. Do not add metrics unless they already exist in the input segment.
2. Do not add technologies that are not present in the input segment.
3. Do not change tense in a way that changes meaning.

Return only JSON, no extra text.
'''


USER_PROMPT = r'''
You will receive inputs as JSON:
segment_id, segment_latex, job_description, and optionally context.

Task:
Rewrite segment_latex to better match job_description while following all rules.
Return only JSON, no extra text.

Guidance:
1. Keep the same LaTeX outer shape, for example keep \\item if it starts with \\item.
2. Keep facts unchanged.
3. Add job aligned keywords only when truthful and compatible with the original.
4. Prefer concrete phrasing over generic wording.
5. Do not change a text if it's only 1 or 2 words. Only change full sentences.
6. Always end sentences with a dot.

Example input:
{
  "segment_latex": "{Built APIs for payments.}",
  "job_description": "We need engineers who build secure payment systems, design clear interfaces, and ensure reliability."
}

Example output:
{
  "rewritten_latex": "{Built payment APIs for payment systems, focusing on security and reliability.}",
  "notes": "Added focus on reliability and security"
}
'''
