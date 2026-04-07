// Split layout: dashboard on this PC, FastAPI on VantaBlack (no trailing slash).
// Use the Ethernet / air-gapped LAN IP so the browser hits IPv4 :8000 (not mDNS/IPv6 on Wi‑Fi).
window.AI_ECOSYSTEM_API_BASE = 'http://192.168.2.151:8000';

// Optional override (browser console): localStorage.setItem('ai_ecosystem_api_base', 'http://HOST:8000'); location.reload();
(function () {
    try {
        var o = localStorage.getItem('ai_ecosystem_api_base');
        if (o && o.trim()) window.AI_ECOSYSTEM_API_BASE = o.trim().replace(/\/$/, '');
    } catch (e) { /* ignore */ }
})();
