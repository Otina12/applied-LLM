import re

@staticmethod
def make_segment_begin(segment_id):
    return f'%<RESUME_SEGMENT:{segment_id}:BEGIN>'

@staticmethod
def make_segment_end(segment_id):
    return f'%<RESUME_SEGMENT:{segment_id}:END>'

@staticmethod
def wrap_segment(segment_id, latex):
    return '\n'.join([make_segment_begin(segment_id), latex, make_segment_end(segment_id)])

def extract_marked_segments(tex):
    pattern = re.compile(
        r'%<RESUME_SEGMENT:(?P<id>[^:]+):BEGIN>\s*\n(?P<body>.*?)\n%<RESUME_SEGMENT:(?P=id):END>',
        flags = re.DOTALL
    )
    out = {}
    for m in pattern.finditer(tex):
        out[m.group('id')] = m.group('body')
    return out

def replace_marked_segment(tex, segment_id, new_body):
    pattern = re.compile(
        rf'(%<RESUME_SEGMENT:{re.escape(segment_id)}:BEGIN>\s*\n)(.*?)(\n%<RESUME_SEGMENT:{re.escape(segment_id)}:END>)',
        flags = re.DOTALL
    )

    def repl(match: re.Match):
        return match.group(1) + new_body + match.group(3)

    updated_tex, n = pattern.subn(repl, tex, count = 1)
    if n != 1:
        raise ValueError(f'Could not replace segment {segment_id}, matches found: {n}')
    
    return updated_tex

def inject_segment_markers(resume_tex, segments):
    updated_resume_tex = resume_tex

    for segment in segments:
        if segment.latex not in updated_resume_tex:
            raise ValueError(f'Segment latex not found for segment {segment.segment_id}. Make sure the segment finder returns exact snippets from the resume.')
        updated_resume_tex = updated_resume_tex.replace(segment.latex, wrap_segment(segment.segment_id, segment.latex), 1)
        
    return updated_resume_tex