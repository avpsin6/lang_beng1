/* HealthLens shared front-end: theme tokens, Plotly theming, table views.
 * Chart specs come from Python with colour *tokens* ("@series-1", "@muted/0.1")
 * which are resolved here for the current light/dark theme, so a theme switch
 * re-renders every chart with the dark steps of the same palette.
 * All text from the data is inserted with textContent (never innerHTML).
 */
(function () {
  "use strict";

  var PALETTE = {
    light: {
      "surface": "#fcfcfb", "bg": "#f9f9f7", "ink": "#0b0b0b", "ink-2": "#52514e", "muted": "#898781",
      "grid": "#e1e0d9", "axis": "#c3c2b7", "border": "rgba(11,11,11,0.10)",
      "series-1": "#2a78d6", "series-2": "#eb6834", "series-3": "#1baf7a", "series-4": "#eda100",
      "series-5": "#e87ba4", "series-6": "#008300", "series-7": "#6250d6", "series-8": "#e34948",
      "deemph": "#c3c2b7", "good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
      "div-neg": "#2a78d6", "div-mid": "#f0efec", "div-pos": "#e34948",
      "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
      "ord": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
    },
    dark: {
      "surface": "#1a1a19", "bg": "#0d0d0d", "ink": "#f0efec", "ink-2": "#c3c2b7", "muted": "#898781",
      "grid": "#2c2c2a", "axis": "#383835", "border": "rgba(255,255,255,0.10)",
      "series-1": "#3987e5", "series-2": "#d95926", "series-3": "#199e70", "series-4": "#c98500",
      "series-5": "#d55181", "series-6": "#008300", "series-7": "#9085e9", "series-8": "#e66767",
      "deemph": "#52514e", "good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
      "div-neg": "#3987e5", "div-mid": "#383835", "div-pos": "#e66767",
      "seq": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
      "ord": ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]
    }
  };

  var FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';

  function storageGet(key) { try { return window.localStorage.getItem(key); } catch (e) { return null; } }
  function storageSet(key, value) { try { window.localStorage.setItem(key, value); } catch (e) { /* ignore */ } }

  function mode() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t === "dark" || t === "light") return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function hexToRgba(hex, alpha) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h.split("").map(function (c) { return c + c; }).join("");
    var n = parseInt(h, 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + alpha + ")";
  }

  function token(name, pal) {
    var alpha = null;
    var slash = name.indexOf("/");
    if (slash > 0) { alpha = parseFloat(name.slice(slash + 1)); name = name.slice(0, slash); }
    var idx = name.match(/^(seq|ord)-(\d)$/);
    var value = idx ? pal[idx[1]][parseInt(idx[2], 10)] : pal[name];
    if (value === undefined) return null;
    if (alpha !== null) return value.charAt(0) === "#" ? hexToRgba(value, alpha) : value;
    return value;
  }

  function resolve(obj, pal) {
    if (typeof obj === "string") {
      if (obj.charAt(0) === "@") { var v = token(obj.slice(1), pal); return v === null ? obj : v; }
      if (obj === "%seq") return pal.seq.map(function (c, i) { return [i / (pal.seq.length - 1), c]; });
      if (obj === "%div") return [[0, pal["div-neg"]], [0.5, pal["div-mid"]], [1, pal["div-pos"]]];
      return obj;
    }
    if (Array.isArray(obj)) return obj.map(function (x) { return resolve(x, pal); });
    if (obj && typeof obj === "object") {
      var out = {};
      Object.keys(obj).forEach(function (k) { out[k] = resolve(obj[k], pal); });
      return out;
    }
    return obj;
  }

  function merge(base, extra) {
    if (!extra) return base;
    Object.keys(extra).forEach(function (k) {
      var v = extra[k];
      if (v && typeof v === "object" && !Array.isArray(v) && base[k] && typeof base[k] === "object" &&
          !Array.isArray(base[k])) {
        base[k] = merge(base[k], v);
      } else {
        base[k] = v;
      }
    });
    return base;
  }

  function axis(pal) {
    return {
      gridcolor: pal.grid, gridwidth: 1, linecolor: pal.axis, linewidth: 1, zeroline: false,
      tickfont: { color: pal.muted, size: 11 }, title: { font: { color: pal["ink-2"], size: 12 } },
      automargin: true, showspikes: false, ticks: "", fixedrange: false
    };
  }

  function baseLayout(pal, compact) {
    var touch = window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
    return {
      font: { family: FONT, size: 12, color: pal["ink-2"] },
      paper_bgcolor: pal.surface, plot_bgcolor: pal.surface,
      margin: { l: 8, r: 12, t: 8, b: 8, pad: 4 },
      xaxis: axis(pal), yaxis: merge(axis(pal), { fixedrange: true }),
      hoverlabel: { bgcolor: pal.surface, bordercolor: pal.axis, font: { family: FONT, color: pal.ink, size: 12 },
                    align: "left" },
      legend: { orientation: "h", x: 0, xanchor: "left", y: 1.02, yanchor: "bottom", font: { color: pal["ink-2"] },
                bgcolor: "rgba(0,0,0,0)" },
      showlegend: false, bargap: 0.35, barcornerradius: 4,
      dragmode: touch ? false : "zoom", hovermode: "closest", autosize: true,
      height: compact ? 220 : undefined
    };
  }

  var CONFIG = { displayModeBar: false, responsive: true, scrollZoom: false, doubleClick: "reset",
                 showTips: false, topojsonURL: "", locale: "en" };

  var specs = {};
  var rendered = {};

  function loadSpecs() {
    var el = document.getElementById("hl-charts");
    if (!el) return;
    try { specs = JSON.parse(el.textContent); } catch (e) { specs = {}; }
  }

  function renderChart(el) {
    var id = el.getAttribute("data-chart");
    var spec = specs[id];
    if (!spec || !window.Plotly) return;
    var pal = PALETTE[mode()];
    var compact = el.classList.contains("plot-sm");
    var layout = merge(baseLayout(pal, compact), resolve(spec.layout || {}, pal));
    if (spec.height && !compact) layout.height = spec.height;
    var data = resolve(spec.data || [], pal);
    try {
      window.Plotly.react(el, data, layout, CONFIG);
      rendered[id] = el;
      el.removeAttribute("aria-busy");
    } catch (e) {
      el.textContent = "This chart could not be drawn: " + e.message;
    }
  }

  function renderAll() {
    var els = Array.prototype.slice.call(document.querySelectorAll("[data-chart]"));
    var i = 0;
    function step() {
      var started = Date.now();
      while (i < els.length && Date.now() - started < 40) { renderChart(els[i]); i += 1; }
      if (i < els.length) window.setTimeout(step, 0);
    }
    step();
  }

  function rerender() {
    Object.keys(rendered).forEach(function (id) { renderChart(rendered[id]); });
    if (window.HealthLens && typeof window.HealthLens.onTheme === "function") window.HealthLens.onTheme();
  }

  // ---- table views (the accessible twin of every chart) ---------------------------
  function buildTable(spec) {
    var wrap = document.createElement("div");
    wrap.className = "table-scroll";
    var table = document.createElement("table");
    table.className = "data";
    var thead = document.createElement("thead");
    var tr = document.createElement("tr");
    spec.table.columns.forEach(function (c) {
      var th = document.createElement("th");
      th.textContent = c;
      th.scope = "col";
      tr.appendChild(th);
    });
    thead.appendChild(tr);
    table.appendChild(thead);
    var tbody = document.createElement("tbody");
    spec.table.rows.forEach(function (row) {
      var r = document.createElement("tr");
      row.forEach(function (cell, j) {
        var td = document.createElement(j === 0 ? "th" : "td");
        if (j === 0) td.scope = "row";
        td.textContent = cell === null ? "–" : String(cell);
        r.appendChild(td);
      });
      tbody.appendChild(r);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  function wireTables() {
    document.querySelectorAll("[data-table-for]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var id = btn.getAttribute("data-table-for");
        var holder = document.getElementById("table-" + id);
        if (!holder) return;
        if (!holder.firstChild && specs[id] && specs[id].table) holder.appendChild(buildTable(specs[id]));
        var open = holder.hidden;
        holder.hidden = !open;
        btn.setAttribute("aria-expanded", open ? "true" : "false");
        btn.textContent = open ? "Hide table" : "Show table";
      });
    });
  }

  // ---- theme toggle -------------------------------------------------------------------
  var LABELS = { system: "Theme: system", light: "Theme: light", dark: "Theme: dark" };

  function applyTheme(choice) {
    if (choice === "light" || choice === "dark") document.documentElement.setAttribute("data-theme", choice);
    else document.documentElement.removeAttribute("data-theme");
    var btn = document.getElementById("theme-toggle");
    if (btn) btn.textContent = LABELS[choice] || LABELS.system;
  }

  function wireTheme() {
    var choice = storageGet("healthlens-theme") || "system";
    applyTheme(choice);
    var btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.addEventListener("click", function () {
        var order = ["system", "light", "dark"];
        choice = order[(order.indexOf(choice) + 1) % order.length];
        storageSet("healthlens-theme", choice);
        applyTheme(choice);
        rerender();
      });
    }
    if (window.matchMedia) {
      var mq = window.matchMedia("(prefers-color-scheme: dark)");
      var handler = function () { if (!document.documentElement.getAttribute("data-theme")) rerender(); };
      if (mq.addEventListener) mq.addEventListener("change", handler); else if (mq.addListener) mq.addListener(handler);
    }
  }

  window.HealthLens = {
    PALETTE: PALETTE, FONT: FONT, CONFIG: CONFIG, mode: mode, resolve: resolve, merge: merge,
    baseLayout: baseLayout, specs: function () { return specs; }, renderChart: renderChart, rerender: rerender,
    buildTable: buildTable, storageGet: storageGet, storageSet: storageSet
  };

  function start() {
    loadSpecs();
    wireTheme();
    wireTables();
    renderAll();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
