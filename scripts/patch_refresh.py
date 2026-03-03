"""Patch index.html to add the REFRESH button + manualRefresh() JS."""
import re

with open('index.html', 'r') as f:
    html = f.read()

if len(html) < 50000:
    raise SystemExit('ERROR: index.html is too small, restore it first')

if 'refreshBtn' in html:
    print('Refresh button already present, skipping')
    raise SystemExit(0)

# 1. Add @keyframes spin after @keyframes fade
html = html.replace(
    '@keyframes fade{from{opacity:.4}to{opacity:1}}',
    '@keyframes fade{from{opacity:.4}to{opacity:1}}\n@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}'
)

# 2. Replace <div class="hdr-r"> with version including refresh button
BTN_HTML = '''<div class="hdr-r" style="display:flex;align-items:center;gap:8px">
    <button id="refreshBtn" onclick="manualRefresh()" title="Refresh all data" style="
      background:var(--bg3);border:1px solid var(--border);border-radius:3px;
      color:var(--text2);font-size:10px;letter-spacing:1px;padding:5px 12px;
      cursor:pointer;display:flex;align-items:center;gap:5px;font-family:inherit;
      transition:all 0.2s ease;
    " onmouseover="this.style.borderColor='var(--accent)';this.style.color='var(--accent)'" onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--text2)'">
      <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M1 1v5h5"/><path d="M15 15v-5h-5"/>
        <path d="M2.3 10a6 6 0 0 0 10.3 1.4L15 10M1 6l2.4-1.4A6 6 0 0 1 13.7 6"/>
      </svg>
      REFRESH
    </button>'''

html = html.replace('  <div class="hdr-r">', BTN_HTML, 1)

# 3. Add manualRefresh() function before the final closing </script>
REFRESH_JS = '''
// MANUAL REFRESH
async function manualRefresh() {
  const btn = document.getElementById('refreshBtn');
  const origHTML = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="animation:spin 0.8s linear infinite"><path d="M1 1v5h5"/><path d="M15 15v-5h-5"/><path d="M2.3 10a6 6 0 0 0 10.3 1.4L15 10M1 6l2.4-1.4A6 6 0 0 1 13.7 6"/></svg> REFRESHING...';
  btn.style.opacity = '0.6';
  try {
    const pat = localStorage.getItem('gh_pat');
    if (pat) {
      try {
        await fetch('https://api.github.com/repos/Tbone376/market-dashboard-pro/actions/workflows/briefing.yml/dispatches', {
          method: 'POST',
          headers: { 'Authorization': 'token ' + pat, 'Accept': 'application/vnd.github.v3+json' },
          body: JSON.stringify({ ref: 'main' })
        });
        await fetch('https://api.github.com/repos/Tbone376/market-dashboard-pro/actions/workflows/fetch-data.yml/dispatches', {
          method: 'POST',
          headers: { 'Authorization': 'token ' + pat, 'Accept': 'application/vnd.github.v3+json' },
          body: JSON.stringify({ ref: 'main' })
        });
        btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="animation:spin 0.8s linear infinite"><path d="M1 1v5h5"/><path d="M15 15v-5h-5"/><path d="M2.3 10a6 6 0 0 0 10.3 1.4L15 10M1 6l2.4-1.4A6 6 0 0 1 13.7 6"/></svg> WORKFLOWS TRIGGERED...';
        await new Promise(r => setTimeout(r, 90000));
      } catch(e) { console.warn('Workflow dispatch failed:', e); }
    }
    await loadData();
    await loadBriefing();
    calc();
    btn.innerHTML = '\u2713 UPDATED';
    btn.style.borderColor = 'var(--green)';
    btn.style.color = 'var(--green)';
    btn.style.opacity = '1';
    setTimeout(() => { btn.innerHTML = origHTML; btn.style.borderColor = 'var(--border)'; btn.style.color = 'var(--text2)'; btn.disabled = false; }, 2000);
  } catch(e) {
    console.error('Refresh failed:', e);
    btn.innerHTML = '\u2717 FAILED';
    btn.style.borderColor = 'var(--red)';
    btn.style.color = 'var(--red)';
    btn.style.opacity = '1';
    setTimeout(() => { btn.innerHTML = origHTML; btn.style.borderColor = 'var(--border)'; btn.style.color = 'var(--text2)'; btn.disabled = false; }, 2000);
  }
}
'''

# Insert before the last </script> tag
last_script_close = html.rfind('</script>')
if last_script_close == -1:
    raise SystemExit('ERROR: No closing </script> found')
html = html[:last_script_close] + REFRESH_JS + html[last_script_close:]

with open('index.html', 'w') as f:
    f.write(html)

print(f'Done. Final size: {len(html)} bytes')
