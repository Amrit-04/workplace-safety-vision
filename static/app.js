const ppeBtn = document.getElementById("ppeBtn");
const intrusionBtn = document.getElementById("intrusionBtn");
const downloadLogsBtn = document.getElementById("downloadLogsBtn");
const statusPill = document.getElementById("statusPill");
const fpsEl = document.getElementById("fps");
const deviceEl = document.getElementById("device");
const zonesEl = document.getElementById("zones");
const qualityEl = document.getElementById("quality");
const logsEl = document.getElementById("logs");
const streamEl = document.getElementById("stream");
const canvas = document.getElementById("zoneCanvas");
const ctx = canvas.getContext("2d");
const drawToggleBtn = document.getElementById("drawToggleBtn");
const finishZoneBtn = document.getElementById("finishZoneBtn");
const undoPointBtn = document.getElementById("undoPointBtn");
const clearZonesBtn = document.getElementById("clearZonesBtn");
const saveZonesBtn = document.getElementById("saveZonesBtn");

let zones = [];
let currentZone = [];
let drawMode = false;
let currentMode = "ppe";

function setDrawMode(enabled) {
  drawMode = enabled;
  drawToggleBtn.textContent = drawMode ? "Stop Drawing" : "Start Drawing";
  drawToggleBtn.classList.toggle("active", drawMode);
}

function toNormalized(x, y) {
  return [x / canvas.width, y / canvas.height];
}

function resizeCanvas() {
  const rect = streamEl.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width));
  canvas.height = Math.max(1, Math.floor(rect.height));
  drawOverlay();
}

function drawPolygon(pts, color) {
  if (pts.length === 0) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(pts[0][0] * canvas.width, pts[0][1] * canvas.height);
  for (let i = 1; i < pts.length; i += 1) {
    ctx.lineTo(pts[i][0] * canvas.width, pts[i][1] * canvas.height);
  }
  ctx.closePath();
  ctx.stroke();
}

function drawCurrent() {
  if (currentZone.length === 0) return;
  ctx.strokeStyle = "#f59e0b";
  ctx.fillStyle = "#f59e0b";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(currentZone[0][0] * canvas.width, currentZone[0][1] * canvas.height);
  for (let i = 1; i < currentZone.length; i += 1) {
    ctx.lineTo(currentZone[i][0] * canvas.width, currentZone[i][1] * canvas.height);
  }
  ctx.stroke();
  for (const p of currentZone) {
    ctx.beginPath();
    ctx.arc(p[0] * canvas.width, p[1] * canvas.height, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawOverlay() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const z of zones) drawPolygon(z, "#22c55e");
  drawCurrent();
}

canvas.addEventListener("click", (e) => {
  if (!drawMode) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  currentZone.push(toNormalized(x, y));
  drawOverlay();
});

drawToggleBtn.addEventListener("click", () => setDrawMode(!drawMode));

finishZoneBtn.addEventListener("click", () => {
  if (currentZone.length >= 3) zones.push([...currentZone]);
  currentZone = [];
  drawOverlay();
});

undoPointBtn.addEventListener("click", () => {
  if (currentZone.length > 0) currentZone.pop();
  drawOverlay();
});

clearZonesBtn.addEventListener("click", async () => {
  zones = [];
  currentZone = [];
  drawOverlay();
  const res = await fetch("/api/zones", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ zones: [], normalized: true }),
  });
  statusPill.textContent = res.ok ? "Zones cleared" : "Clear failed";
});

saveZonesBtn.addEventListener("click", async () => {
  const payload = { zones, normalized: true };
  const res = await fetch("/api/zones", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (res.ok) {
    const out = await res.json();
    statusPill.textContent = `Saved ${out.zones} zone(s)`;
  } else {
    statusPill.textContent = "Zone save failed";
  }
});

async function setMode(mode) {
  const res = await fetch("/api/mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  if (!res.ok) return;
  currentMode = mode;
  ppeBtn.classList.toggle("active", mode === "ppe");
  intrusionBtn.classList.toggle("active", mode === "intrusion");
  refreshLogs();
}

ppeBtn.addEventListener("click", () => setMode("ppe"));
intrusionBtn.addEventListener("click", () => setMode("intrusion"));

downloadLogsBtn.addEventListener("click", () => {
  window.location.href = `/api/logs/download?mode=${encodeURIComponent(currentMode)}`;
});


async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const s = await res.json();
    currentMode = s.mode;
    statusPill.textContent = `Mode: ${s.mode.toUpperCase()}`;
    fpsEl.textContent = `FPS: ${s.fps}`;
    deviceEl.textContent = `Device: ${s.device}`;
    zonesEl.textContent = `Zones: ${s.zones}`;
    qualityEl.textContent = `IMG/JPEG: ${s.img_size}/${s.jpeg_quality} | PPE:${s.ppe_infer_fps} Intr:${s.intrusion_infer_fps} | Active:${s.active_intrusions}`;
    ppeBtn.classList.toggle("active", s.mode === "ppe");
    intrusionBtn.classList.toggle("active", s.mode === "intrusion");
  } catch (_) {
    statusPill.textContent = "Disconnected";
  }
}

async function refreshLogs() {
  try {
    const res = await fetch(`/api/logs?limit=120&mode=${encodeURIComponent(currentMode)}`);
    const data = await res.json();
    logsEl.innerHTML = "";
    for (const row of data.items) {
      const div = document.createElement("div");
      const categoryClass = row.category === "intrusion" ? " intrusion" : row.category === "ppe" ? " ppe" : "";
      div.className = `log-line ${row.level.toLowerCase()}${categoryClass}`;
      div.textContent = `[${row.ts}] ${row.level}: ${row.message}`;
      logsEl.appendChild(div);
    }
  } catch (_) {}
}

async function loadZones() {
  try {
    const res = await fetch("/api/zones");
    const out = await res.json();
    if (res.ok && out.ok && Array.isArray(out.zones)) {
      zones = out.zones;
      drawOverlay();
    }
  } catch (_) {}
}

window.addEventListener("resize", resizeCanvas);
streamEl.addEventListener("load", resizeCanvas);

setDrawMode(false);
resizeCanvas();
refreshStatus();
refreshLogs();
loadZones();
setInterval(refreshStatus, 1000);
setInterval(refreshLogs, 1500);
