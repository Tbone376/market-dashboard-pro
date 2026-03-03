#!/usr/bin/env python3
"""Assembles index.html from base64 chunks and commits it."""
import base64
import hashlib
import sys
import os

# Import all base64 parts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b64_p1 import B64_P1
from b64_p2 import B64_P2
from b64_p3 import B64_P3
from b64_p4 import B64_P4
from b64_p5 import B64_P5
from b64_p6 import B64_P6
from b64_p7 import B64_P7
from b64_p8 import B64_P8
from b64_p9 import B64_P9
from b64_p10 import B64_P10

# Assemble full base64 string
full_b64 = B64_P1 + B64_P2 + B64_P3 + B64_P4 + B64_P5 + B64_P6 + B64_P7 + B64_P8 + B64_P9 + B64_P10

# Decode
html_bytes = base64.b64decode(full_b64)
html_content = html_bytes.decode('utf-8')

print(f"Decoded: {len(html_bytes)} bytes, {len(html_content)} chars")

# Verify MD5
md5 = hashlib.md5(html_bytes).hexdigest()
print(f"MD5: {md5}")
expected_md5 = "b85adb68e7eea7ce5430e96f81a750df"
if md5 != expected_md5:
    print(f"ERROR: MD5 mismatch! Expected {expected_md5}, got {md5}")
    sys.exit(1)
print("MD5 verified OK")

# Write index.html
with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Written index.html: {os.path.getsize('index.html')} bytes")
