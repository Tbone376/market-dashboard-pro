/*
 * refresh.js — Injects a REFRESH button into the dashboard header
 * and handles manual data re-fetch + optional GitHub workflow dispatch.
 *
 * Loaded by index.html via <script src="scripts/refresh.js"></script>
 */
(function(){
  // Add spin keyframe if not present
  if (!document.getElementById('spin-style')) {
    const s = document.createElement('style');
    s.id = 'spin-style';
    s.textContent = '@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}';
    document.head.appendChild(s);
  }

  // Inject button into .hdr-r
  const hdrR = document.querySelector('.hdr-r');
  if (hdrR) {
    const btn = document.createElement('button');
    btn.id = 'refreshBtn';
    btn.title = 'Refresh all data';
    btn.style.cssText = 'background:var(--bg3);border:1px solid var(--border);border-radius:3px;color:var(--text2);font-size:10px;letter-spacing:1px;padding:5px 12px;cursor:pointer;display:flex;align-items:center;gap:5px;font-family:inherit;transition:all 0.2s ease;margin-right:8px;';
    btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M1 1v5h5"/><path d="M15 15v-5h-5"/><path d="M2.3 10a6 6 0 0 0 10.3 1.4L15 10M1 6l2.4-1.4A6 6 0 0 1 13.7 6"/></svg> REFRESH';
    btn.onmouseover = function(){ this.style.borderColor='var(--accent)'; this.style.color='var(--accent)'; };
    btn.onmouseout  = function(){ this.style.borderColor='var(--border)'; this.style.color='var(--text2)'; };
    btn.onclick = manualRefresh;
    hdrR.insertBefore(btn, hdrR.firstChild);
  }

  async function manualRefresh() {
    const btn = document.getElementById('refreshBtn');
    if (!btn) return;
    const origHTML = btn.innerHTML;
    btn.disabled = true;
    btn.style.opacity = '0.6';
    const spinner = '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="animation:spin 0.8s linear infinite"><path d="M1 1v5h5"/><path d="M15 15v-5h-5"/><path d="M2.3 10a6 6 0 0 0 10.3 1.4L15 10M1 6l2.4-1.4A6 6 0 0 1 13.7 6"/></svg>';
    btn.innerHTML = spinner + ' REFRESHING...';

    try {
      // If user has stored a GitHub PAT, trigger workflows first
      var pat = localStorage.getItem('gh_pat');
      if (pat) {
        try {
          var hdr = { 'Authorization': 'token ' + pat, 'Accept': 'application/vnd.github.v3+json' };
          var body = JSON.stringify({ ref: 'main' });
          await fetch('https://api.github.com/repos/Tbone376/market-dashboard-pro/actions/workflows/briefing.yml/dispatches', { method:'POST', headers:hdr, body:body });
          await fetch('https://api.github.com/repos/Tbone376/market-dashboard-pro/actions/workflows/fetch-data.yml/dispatches', { method:'POST', headers:hdr, body:body });
          btn.innerHTML = spinner + ' WORKFLOWS TRIGGERED...';
          await new Promise(function(r){ setTimeout(r, 90000); });
        } catch(e) { console.warn('Workflow dispatch failed:', e); }
      }

      // Re-fetch JSON files
      if (typeof loadData === 'function') await loadData();
      if (typeof loadBriefing === 'function') await loadBriefing();
      if (typeof calc === 'function') calc();

      btn.innerHTML = '\u2713 UPDATED';
      btn.style.borderColor = 'var(--green)';
      btn.style.color = 'var(--green)';
      btn.style.opacity = '1';
    } catch(e) {
      console.error('Refresh failed:', e);
      btn.innerHTML = '\u2717 FAILED';
      btn.style.borderColor = 'var(--red)';
      btn.style.color = 'var(--red)';
      btn.style.opacity = '1';
    }
    setTimeout(function(){
      btn.innerHTML = origHTML;
      btn.style.borderColor = 'var(--border)';
      btn.style.color = 'var(--text2)';
      btn.style.opacity = '1';
      btn.disabled = false;
    }, 2000);
  }

  // One-time setup: run localStorage.setItem('gh_pat','ghp_xxx') in console to enable workflow triggers
})();
