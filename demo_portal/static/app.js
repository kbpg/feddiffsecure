const MIN_ZOOM_SCALE = 0.75;
const MAX_ZOOM_SCALE = 3;
const ZOOM_STEP = 0.25;

async function refreshLiveMonitor() {
  const monitorRoot = document.querySelector("[data-live-monitor]");
  if (!monitorRoot) return;

  try {
    const response = await fetch("/api/monitor/live", { credentials: "same-origin" });
    if (!response.ok) return;
    const payload = await response.json();

    const roundNode = document.querySelector("[data-monitor-latest-round]");
    const etaNode = document.querySelector("[data-monitor-eta]");
    if (roundNode && payload.latest_round !== undefined) {
      roundNode.textContent = payload.latest_round;
    }
    if (etaNode) {
      if (payload.eta_window && payload.eta_window.point_estimate_hours) {
        etaNode.textContent = `${payload.eta_window.point_estimate_hours.toFixed(1)} 小时`;
      } else {
        etaNode.textContent = "预热中";
      }
    }

    const tableBody = document.getElementById("live-round-table");
    if (tableBody && Array.isArray(payload.recent_rounds)) {
      tableBody.innerHTML = payload.recent_rounds
        .map((row) => {
          const classifier = row.classifier_fid !== undefined ? row.classifier_fid : "n/a";
          return `<tr>
            <td>${row.round}</td>
            <td>${row.avg_client_loss}</td>
            <td>${row.proxy_fid}</td>
            <td>${classifier}</td>
            <td>${row.round_duration_sec}</td>
          </tr>`;
        })
        .join("");
    }
  } catch (err) {
    console.warn("live monitor refresh failed", err);
  }
}

function drawTrendChart(canvas, runs) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const width = canvas.width;
  const height = canvas.height;
  const padding = 30;
  ctx.clearRect(0, 0, width, height);
  const values = runs
    .map((run) => Number(run.best_metric_value))
    .filter((value) => Number.isFinite(value));
  if (!values.length) return;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-6);
  const stepX = (width - padding * 2) / Math.max(values.length - 1, 1);

  ctx.strokeStyle = "#d6deff";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, height - padding);
  ctx.lineTo(width - padding, height - padding);
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding, height - padding);
  ctx.stroke();

  ctx.strokeStyle = "#6a5af9";
  ctx.lineWidth = 3;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = padding + stepX * index;
    const normalized = (value - min) / span;
    const y = height - padding - normalized * (height - padding * 2);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#ff4d8d";
  values.forEach((value, index) => {
    const x = padding + stepX * index;
    const normalized = (value - min) / span;
    const y = height - padding - normalized * (height - padding * 2);
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawRadarChart(canvas, runs) {
  const ctx = canvas.getContext("2d");
  if (!ctx || !runs.length) return;
  const width = canvas.width;
  const height = canvas.height;
  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.min(width, height) * 0.32;
  const labels = ["质量", "稳定性", "泛化", "效率", "可解释性"];
  const run = runs[0];
  const quality = Number.isFinite(Number(run.best_metric_value))
    ? Math.max(20, Math.min(100, 110 - Number(run.best_metric_value) * 5))
    : 60;
  const stability = Number.isFinite(Number(run.best_metric_round))
    ? Math.max(20, Math.min(100, 100 - Number(run.best_metric_round) * 1.1))
    : 55;
  const generalization = run.dataset === "STL10" ? 88 : 72;
  const efficiency = Number.isFinite(Number(run.trainable_ratio))
    ? Math.max(20, Math.min(100, 100 - Number(run.trainable_ratio) * 40))
    : 58;
  const explainability = run.mode === "联邦训练" ? 90 : 75;
  const values = [quality, stability, generalization, efficiency, explainability];

  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#d6deff";
  for (let ring = 1; ring <= 4; ring += 1) {
    ctx.beginPath();
    for (let i = 0; i < labels.length; i += 1) {
      const angle = (Math.PI * 2 * i) / labels.length - Math.PI / 2;
      const x = cx + (radius * ring / 4) * Math.cos(angle);
      const y = cy + (radius * ring / 4) * Math.sin(angle);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
  }

  labels.forEach((label, i) => {
    const angle = (Math.PI * 2 * i) / labels.length - Math.PI / 2;
    const x = cx + (radius + 18) * Math.cos(angle);
    const y = cy + (radius + 18) * Math.sin(angle);
    ctx.fillStyle = "#5e6c89";
    ctx.font = "12px Segoe UI";
    ctx.fillText(label, x - 18, y);
  });

  ctx.fillStyle = "rgba(106, 90, 249, 0.22)";
  ctx.strokeStyle = "#6a5af9";
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((value, i) => {
    const angle = (Math.PI * 2 * i) / labels.length - Math.PI / 2;
    const ratio = value / 100;
    const x = cx + radius * ratio * Math.cos(angle);
    const y = cy + radius * ratio * Math.sin(angle);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function initCharts() {
  const trendCanvas = document.querySelector('[data-chart="trend"]');
  if (trendCanvas) {
    const runs = JSON.parse(trendCanvas.getAttribute("data-runs") || "[]");
    drawTrendChart(trendCanvas, runs);
  }
  const radarCanvas = document.querySelector('[data-chart="radar"]');
  if (radarCanvas) {
    const runs = JSON.parse(radarCanvas.getAttribute("data-runs") || "[]");
    drawRadarChart(radarCanvas, runs);
  }
}

function initRunFilters() {
  const datasetFilter = document.querySelector('[data-run-filter="dataset"]');
  const modeFilter = document.querySelector('[data-run-filter="mode"]');
  const cards = Array.from(document.querySelectorAll("[data-run-card]"));
  if (!datasetFilter || !modeFilter || !cards.length) return;

  const applyFilters = () => {
    const selectedDataset = datasetFilter.value;
    const selectedMode = modeFilter.value;
    cards.forEach((card) => {
      const dataset = card.dataset.dataset;
      const mode = card.dataset.mode;
      const datasetMatch = selectedDataset === "all" || selectedDataset === dataset;
      const modeMatch = selectedMode === "all" || selectedMode === mode;
      card.style.display = datasetMatch && modeMatch ? "" : "none";
    });
  };

  datasetFilter.addEventListener("change", applyFilters);
  modeFilter.addEventListener("change", applyFilters);
}

function initTabs() {
  const tabButtons = Array.from(document.querySelectorAll("[data-tab-target]"));
  const panes = Array.from(document.querySelectorAll(".tab-pane"));
  if (!tabButtons.length || !panes.length) return;

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const targetId = button.getAttribute("data-tab-target");
      tabButtons.forEach((item) => item.classList.remove("is-active"));
      panes.forEach((pane) => pane.classList.remove("is-active"));
      button.classList.add("is-active");
      const pane = document.getElementById(targetId);
      if (pane) {
        pane.classList.add("is-active");
      }
    });
  });
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function setViewerScale(viewer, nextScale) {
  const zoomImage = viewer.querySelector("[data-zoom-image]");
  if (!(zoomImage instanceof HTMLImageElement)) return;

  const safeScale = clamp(nextScale, MIN_ZOOM_SCALE, MAX_ZOOM_SCALE);
  zoomImage.style.setProperty("--zoom-scale", safeScale.toFixed(2));

  const scaleLabel = viewer.querySelector("[data-image-scale-label]");
  if (scaleLabel) {
    scaleLabel.textContent = `${Math.round(safeScale * 100)}%`;
  }
}

function syncViewerImage(viewer, details) {
  const zoomImage = viewer.querySelector("[data-zoom-image]");
  if (!(zoomImage instanceof HTMLImageElement) || !details) return;

  zoomImage.src = details.src;
  zoomImage.alt = details.alt;

  const captionNode = viewer.querySelector("[data-image-caption-output]");
  if (captionNode) {
    captionNode.textContent = details.caption;
  }

  const originalLink = viewer.querySelector("[data-image-original-link]");
  if (originalLink instanceof HTMLAnchorElement) {
    originalLink.href = details.src;
  }

  setViewerScale(viewer, 1);
}

function getFigureImageDetails(figure) {
  const image = figure.querySelector("[data-sample-image]");
  if (!(image instanceof HTMLImageElement)) return null;

  const caption = figure.getAttribute("data-image-name") || image.alt || "preview";
  return {
    src: image.currentSrc || image.src,
    alt: image.alt || caption,
    caption,
  };
}

function initImageViewers() {
  const viewers = Array.from(document.querySelectorAll("[data-image-viewer]"));
  viewers.forEach((viewer) => {
    setViewerScale(viewer, 1);

    const buttons = Array.from(viewer.querySelectorAll("[data-image-action]"));
    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        const scaleLabel = viewer.querySelector("[data-image-scale-label]");
        const currentScaleText = scaleLabel ? scaleLabel.textContent || "100%" : "100%";
        const currentScale = Number.parseFloat(currentScaleText) / 100 || 1;
        const action = button.getAttribute("data-image-action");

        if (action === "zoom-in") {
          setViewerScale(viewer, currentScale + ZOOM_STEP);
        } else if (action === "zoom-out") {
          setViewerScale(viewer, currentScale - ZOOM_STEP);
        } else {
          setViewerScale(viewer, 1);
        }
      });
    });
  });
}

function initCheckpointSwitcher() {
  const roots = Array.from(document.querySelectorAll("[data-checkpoint-switcher]"));
  if (!roots.length) return;

  roots.forEach((root) => {
    const panel = root.closest("[data-sample-panel]");
    if (!(panel instanceof HTMLElement)) return;

    const viewer = panel.querySelector("[data-image-viewer]");
    const buttons = Array.from(root.querySelectorAll("[data-checkpoint-btn]"));
    const figures = Array.from(panel.querySelectorAll("[data-sample-figure]"));
    if (!(viewer instanceof HTMLElement) || !figures.length) return;

    const selectFigure = (figure) => {
      const details = getFigureImageDetails(figure);
      if (!details) return;

      syncViewerImage(viewer, details);
      figures.forEach((item) => item.classList.toggle("is-active", item === figure));

      const imageName = (figure.getAttribute("data-image-name") || "").toLowerCase();
      buttons.forEach((button) => {
        const keyword = (button.getAttribute("data-image-keyword") || "").toLowerCase();
        button.classList.toggle("is-active", Boolean(keyword) && imageName.includes(keyword));
      });
    };

    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        const keyword = (button.getAttribute("data-image-keyword") || "").toLowerCase();
        const matchedFigure = figures.find((figure) => {
          const imageName = (figure.getAttribute("data-image-name") || "").toLowerCase();
          return imageName.includes(keyword);
        });
        if (matchedFigure) {
          selectFigure(matchedFigure);
        }
      });
    });

    figures.forEach((figure) => {
      const handleSelect = () => selectFigure(figure);
      figure.addEventListener("click", handleSelect);
      figure.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          handleSelect();
        }
      });
    });

    const initialFigure = figures.find((figure) => figure.classList.contains("is-active")) || figures[0];
    if (initialFigure) {
      selectFigure(initialFigure);
    }
  });
}

window.addEventListener("load", () => {
  initCharts();
  initRunFilters();
  initTabs();
  initImageViewers();
  initCheckpointSwitcher();
  refreshLiveMonitor();
  setInterval(refreshLiveMonitor, 30000);
});
