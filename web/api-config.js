// Split layout: dashboard on this PC, FastAPI on VantaBlack (no trailing slash).
// Default: Wi‑Fi LAN (same subnet as e.g. 192.168.0.12 → 192.168.0.15).
window.AI_ECOSYSTEM_API_BASE = 'http://192.168.0.15:8000';
// When your PC uses the direct Ethernet link only, set to: http://192.168.2.151:8000
// (or use localStorage key ai_ecosystem_api_base — see below)

// Optional override (browser console): localStorage.setItem('ai_ecosystem_api_base', 'http://HOST:8000'); location.reload();
(function () {
    try {
        var o = localStorage.getItem('ai_ecosystem_api_base');
        if (o && o.trim()) window.AI_ECOSYSTEM_API_BASE = o.trim().replace(/\/$/, '');
    } catch (e) { /* ignore */ }
})();
