// Optional: split layout — dashboard served from this PC (e.g. START_LOCAL_WEB_UI.bat on :8080)
// while FastAPI runs on VantaBlack. Uncomment and set (no trailing slash):
// window.AI_ECOSYSTEM_API_BASE = 'http://vantablack:8000';

// Optional override (browser console): localStorage.setItem('ai_ecosystem_api_base', 'http://HOST:8000'); location.reload();
(function () {
    try {
        var o = localStorage.getItem('ai_ecosystem_api_base');
        if (o && o.trim()) window.AI_ECOSYSTEM_API_BASE = o.trim().replace(/\/$/, '');
    } catch (e) { /* ignore */ }
})();
