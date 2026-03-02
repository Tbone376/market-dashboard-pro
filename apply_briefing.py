#!/usr/bin/env python3
import json

with open('briefing_patches.json') as f:
    patches = json.load(f)
with open('index.html', 'r') as f:
    lines = f.readlines()
for p in reversed(patches):
    insert_lines = p['content'].splitlines(keepends=True)
    lines[p['after_line']:p['after_line']] = insert_lines
with open('index.html', 'w') as f:
    f.writelines(lines)
print(f'Done. {len(lines)} lines written.')
