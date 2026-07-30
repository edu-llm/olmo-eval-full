import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import csv
import json
import os
import time

import pyarrow.parquet as pq

# Allow very large fields (prompt content can be long).
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

DATA_DIR = r'C:\Users\Avane\UT-Austin\AlphaIA\LLM-From-Scratch\olmo-eval-full\eduLLM-Evals\data\dr-sci'

BASE_COLUMNS = [
    'row_index', 'subject', 'from', 'difficulty', 'match_rule',
    'question', 'ground_truth', 'reference_answer', 'style',
    'data_source', 'prompt_content',
]
OPEN_EXTRA_COLUMNS = ['num_criteria', 'rubric_json']

FORMULA_PREFIXES = ('=', '+', '-', '@')


def sanitize(value, stats):
    """Stringify a cell value and guard against CSV formula injection.

    Returns the safe string. Increments stats['sanitized'] when a guard fires.
    """
    if value is None:
        return ''
    if isinstance(value, bool):
        s = 'true' if value else 'false'
    else:
        s = str(value)
    if s and s[0] in FORMULA_PREFIXES:
        stats['sanitized'] += 1
        return "'" + s
    return s


def build_prompt_content(prompt, stats):
    """prompt is a list of {content, role}. Single message -> content string.
    Multiple -> join as '[role] content' separated by blank line."""
    if not prompt:
        return ''
    if len(prompt) == 1:
        return prompt[0].get('content') or ''
    stats['multi_prompt'] += 1
    parts = []
    for m in prompt:
        parts.append('[%s] %s' % (m.get('role') or '', m.get('content') or ''))
    return '\n\n'.join(parts)


def convert(name, is_open):
    in_path = os.path.join(DATA_DIR, name + '.parquet')
    out_path = os.path.join(DATA_DIR, name + '.csv')

    columns = BASE_COLUMNS + (OPEN_EXTRA_COLUMNS if is_open else [])
    read_cols = ['data_source', 'prompt', 'reward_model', 'extra_info']

    stats = {'sanitized': 0, 'multi_prompt': 0, 'null_question': 0, 'rows': 0}

    pf = pq.ParquetFile(in_path)
    t0 = time.time()
    row_index = 0

    with open(out_path, 'w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns)

        for g in range(pf.num_row_groups):
            table = pf.read_row_group(g, columns=read_cols)
            batch = table.to_pylist()
            for rec in batch:
                extra = rec.get('extra_info') or {}
                rm = rec.get('reward_model') or {}

                question = extra.get('question')
                if question is None:
                    stats['null_question'] += 1

                prompt_content = build_prompt_content(rec.get('prompt'), stats)

                row = [
                    row_index,
                    sanitize(extra.get('subject'), stats),
                    sanitize(extra.get('from'), stats),
                    sanitize(extra.get('difficulty'), stats),
                    sanitize(extra.get('match_rule'), stats),
                    sanitize(question, stats),
                    sanitize(rm.get('ground_truth'), stats),
                    sanitize(extra.get('reference_answer'), stats),
                    sanitize(rm.get('style'), stats),
                    sanitize(rec.get('data_source'), stats),
                    sanitize(prompt_content, stats),
                ]

                if is_open:
                    rubric = rm.get('rubric') or []
                    rubric_out = [
                        {
                            'title': (r or {}).get('title'),
                            'description': (r or {}).get('description'),
                            'weight': (r or {}).get('weight'),
                        }
                        for r in rubric
                    ]
                    rubric_json = json.dumps(rubric_out, ensure_ascii=False, separators=(',', ':'))
                    row.append(len(rubric_out))
                    row.append(sanitize(rubric_json, stats))

                writer.writerow(row)
                row_index += 1
                stats['rows'] += 1

            print('  [%s] row group %d/%d done, rows=%d, %.1fs'
                  % (name, g + 1, pf.num_row_groups, stats['rows'], time.time() - t0))

    size = os.path.getsize(out_path)
    return out_path, size, stats, columns


def report(name, out_path, size, stats, columns):
    print('\n===== %s =====' % name)
    print('output:', out_path)
    print('columns (%d):' % len(columns), columns)
    print('data rows:', stats['rows'], '(+1 header)')
    print('file size: %d bytes (%.2f MB)' % (size, size / 1024 / 1024))
    print('cells sanitized (formula guard):', stats['sanitized'])
    print('rows with multiple prompt messages:', stats['multi_prompt'])
    print('rows with null question:', stats['null_question'])

    # Verify BOM + read back header and one sample row.
    with open(out_path, 'rb') as fb:
        head = fb.read(3)
    print('BOM present (EF BB BF):', head == b'\xef\xbb\xbf')

    with open(out_path, 'r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        print('header line:', header)
        sample = next(reader)
        trunc = [c if len(c) <= 80 else c[:80] + '...' for c in sample]
        print('sample row (truncated):', trunc)

    # Count lines by re-reading (accounts for quoted newlines correctly via csv).
    with open(out_path, 'r', encoding='utf-8-sig', newline='') as fh:
        total = sum(1 for _ in csv.reader(fh))
    print('total CSV records incl header:', total)


def main():
    results = []
    for name, is_open in [('Dr_SCI_verifiable', False), ('Dr_SCI_open-ended', True)]:
        print('Converting', name, '...')
        out_path, size, stats, columns = convert(name, is_open)
        results.append((name, out_path, size, stats, columns))
    for r in results:
        report(*r)


if __name__ == '__main__':
    main()
