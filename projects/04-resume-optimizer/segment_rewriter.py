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

You may be given a context object that contains:
- resume_outline: high level structure
- segment_location: where this segment lives
- local_context_before and local_context_after: nearby LaTeX
- top_relevant_snippets: other lines from the resume that matter

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
9. Keep length controlled. Aim for within plus or minus 25 percent of the input segment length, unless the input is extremely short.
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

Example input:
{
  "segment_id": "s1",
  "segment_latex": "\\item Built APIs for payments.",
  "job_description": "We need engineers who build secure payment systems, design clear interfaces, and ensure reliability."
}

Example output:
{
  "segment_id": "s1",
  "rewritten_latex": "\\item Built payment APIs with clear interfaces, focusing on secure access and reliable integrations.",
  "notes": ""
}
'''




# import json
# from typing import Any, Dict, List, Optional
# from pydantic import BaseModel, Field

# class SegmentInput(BaseModel):
#     segment_id: str
#     segment_latex: str

# class RewriteResult(BaseModel):
#     segment_id: str
#     rewritten_latex: str
#     notes: Optional[str]

# class BatchRewriteResult(BaseModel):
#     rewrites: List[RewriteResult]

# class RewriteWithOriginal(BaseModel):
#     segment_id: str
#     original_latex: str
#     rewritten_latex: str
#     notes: Optional[str]

# class SegmentRewriter:
#     def __init__(self, client, model = 'gpt-4o-mini'):
#         self.client = client
#         self.model = model

#     def call_segment_rewriter(self, segments: List[SegmentInput], job_description, context = None) -> List[RewriteWithOriginal]:
#         payload: Dict[str, Any] = {
#             'segments': [s.model_dump() for s in segments],
#             'job_description': job_description
#         }

#         if context is not None:
#             payload['context'] = context

#         messages = [
#             {'role': 'system', 'content': SYSTEM_PROMPT},
#             {'role': 'user', 'content': USER_PROMPT},
#             {'role': 'user', 'content': json.dumps(payload, ensure_ascii = False)}
#         ]

#         response = self.client.chat.completions.create(
#             model = self.model,
#             messages = messages,
#             response_format = {
#                 'type': 'json_schema',
#                 'json_schema': {
#                     'name': 'segment_rewrite_batch_result',
#                     'strict': True,
#                     'schema': BatchRewriteResult.model_json_schema()
#                 }
#             }
#         )

#         content = response.choices[0].message.content
#         if not content:
#             return [
#                 RewriteWithOriginal(segment_id = s.segment_id, original_latex = s.segment_latex, rewritten_latex = s.segment_latex, notes = None)
#                 for s in segments
#             ]

#         result = BatchRewriteResult.model_validate_json(content)

#         input_by_id = {s.segment_id: s.segment_latex for s in segments}

#         output_by_id = {}
#         for r in result.rewrites:
#             if r.segment_id in output_by_id:
#                 raise ValueError(f'Duplicate segment_id in output: {r.segment_id}')
#             output_by_id[r.segment_id] = r

#         missing_ids = [s.segment_id for s in segments if s.segment_id not in output_by_id]
#         extra_ids = [rid for rid in output_by_id.keys() if rid not in input_by_id]

#         if missing_ids:
#             raise ValueError(f'Missing segment_ids in output: {missing_ids}')
#         if extra_ids:
#             raise ValueError(f'Unexpected segment_ids in output: {extra_ids}')

#         combined = []
#         for s in segments:
#             r = output_by_id[s.segment_id]
#             combined.append(RewriteWithOriginal(segment_id = s.segment_id, original_latex = s.segment_latex, rewritten_latex = r.rewritten_latex, notes = r.notes))

#         return combined


# SYSTEM_PROMPT = r'''
# You are a LaTeX resume segment rewriter.

# Goal:
# Given multiple LaTeX segments from a resume and a job description, rewrite each segment so it matches the job more closely while preserving the original meaning and facts.

# You may be given a context object that contains:
# - resume_outline
# - segment_location
# - local_context_before and local_context_after
# - top_relevant_snippets

# Use context to keep the rewrites consistent with the rest of the resume, but only rewrite the provided segment_latex values.
# Do not copy unrelated content from context into a segment.

# Hard rules:
# 1. Output must be valid JSON and must match the required schema.
# 2. Do not invent facts, numbers, tools, employers, titles, dates, certifications, or outcomes.
# 3. Keep the same story. You may reorder or adjust wording and add job aligned keywords only if they are true given the original segment.
# 4. Preserve LaTeX correctness. Return LaTeX for each rewritten segment only.
# 5. Keep the same LaTeX outer shape when possible. If input starts with \item, output must start with \item.
# 6. Do not add new packages, new commands, or new environments.
# 7. Do not change proper nouns, product names, or company names.
# 8. Keep length controlled, aim for within plus or minus 25 percent of the input segment length, unless the input is extremely short.
# 9. Avoid empty claims. Prefer specific responsibilities and outcomes that already exist in the text.

# Batch rules:
# 1. You will receive a list of segments. Rewrite each one.
# 2. Return exactly one output item per input segment.
# 3. Keep segment_id unchanged and unique.
# 4. If a segment does not need changes, return it unchanged.

# Output format:
# Return only JSON, no extra text.
# '''


# USER_PROMPT = r'''
# Input JSON contains:
# - segments: array of objects with segment_id and segment_latex
# - job_description: string
# - context: optional object

# Task:
# Rewrite each segment_latex to better match job_description while following all rules.

# Output JSON schema:
# {
#   "rewrites": [
#     {
#       "segment_id": "s1",
#       "rewritten_latex": "string",
#       "notes": null
#     }
#   ]
# }

# Rules:
# 1. rewrites length must equal segments length.
# 2. Every segment_id in rewrites must match an input segment_id.
# 3. Do not add or remove segments.
# 4. notes must always be present, use null when there are no notes.

# Example input:
# {
#   "segments": [
#     { "segment_id": "s1", "segment_latex": "{Built APIs for payments.}" },
#     { "segment_id": "s2", "segment_latex": "{Improved database queries.}" }
#   ],
#   "job_description": "We need engineers who build secure payment systems and improve performance."
# }

# Example output:
# {
#   "rewrites": [
#     {
#       "segment_id": "s1",
#       "rewritten_latex": "{Built payment APIs with clear interfaces, focusing on secure access and reliable integrations.}",
#       "notes": null
#     },
#     {
#       "segment_id": "s2",
#       "rewritten_latex": "{Improved database queries to reduce latency and support reliable performance.}",
#       "notes": null
#     }
#   ]
# }

# Return only JSON, no extra text.
# '''
