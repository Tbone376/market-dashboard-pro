#!/usr/bin/env python3
"""Apply briefing data preservation fix to index.html."""
import sys

with open('index.html', 'r') as f:
    content = f.read()

# Verify we have the original content (not placeholder)
if len(content) < 50000:
    print(f"ERROR: index.html is too small ({len(content)} chars) - likely a placeholder!")
    sys.exit(1)

# Fix 1: In try block - preserve briefing before replacing DATA
old1 = "    if(!resp.ok) throw new Error('No data.json yet');\n    DATA = await resp.json();\n    renderAll(DATA);"
new1 = "    if(!resp.ok) throw new Error('No data.json yet');\n    const _briefing = DATA && DATA.briefing ? DATA.briefing : null;\n    DATA = await resp.json();\n    if(_briefing) DATA.briefing = _briefing;\n    renderAll(DATA);"

# Fix 2: In catch block - preserve briefing before replacing DATA with DEMO
old2 = "    DATA = DEMO;\n    DATA.crypto = CRYPTO_DEMO;\n    renderAll(DATA);"
new2 = "    const _briefing2 = DATA && DATA.briefing ? DATA.briefing : null;\n    DATA = DEMO;\n    DATA.crypto = CRYPTO_DEMO;\n    if(_briefing2) DATA.briefing = _briefing2;\n    renderAll(DATA);"

applied = 0

if old1 in content:
    content = content.replace(old1, new1)
    print("Fix 1 applied: briefing preservation in try block")
    applied += 1
else:
    if 'const _briefing = DATA' in content:
        print("Fix 1 already applied")
        applied += 1
    else:
        print("ERROR: Fix 1 pattern not found!")
        sys.exit(1)

if old2 in content:
    content = content.replace(old2, new2)
    print("Fix 2 applied: briefing preservation in catch block")
    applied += 1
else:
    if 'const _briefing2 = DATA' in content:
        print("Fix 2 already applied")
        applied += 1
    else:
        print("ERROR: Fix 2 pattern not found!")
        sys.exit(1)

with open('index.html', 'w') as f:
    f.write(content)

print(f"Done. {applied}/2 fixes applied. File: {len(content)} chars.")
