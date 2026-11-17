/* HealthLens dashboard: every view below the filter row is recomputed in the
 * browser from the embedded daily table. No network, no storage of health data.
 * Data-derived text goes into the DOM with textContent only.
 */
(function () {
  "use strict";
  var HL = window.HealthLens;
  var D = JSON.parse(document.getElementById("hl-data").textContent);
  var DAY = 86400000;
  var dates = D.dates.map(function (s) { return Date.parse(s + "T00:00:00Z"); });
  var N = dates.length;
  var byKey = {};
  D.metrics.forEach(function (m) { byKey[m.key] = m; });
  var state = { from: 0, to: N - 1, preset: "365", metric: D.defaults[0], metricB: null, group: "auto",
                lag: 0, wType: "All", wSort: "date", wDesc: true };
  state.metricB = (D.defaults[1] || D.defaults[0]);
  var plots = [];

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function iso(t) { return new Date(t).toISOString().slice(0, 10); }
  var MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  function nice(t) { var d = new Date(t); return d.getUTCDate() + " " + MON[d.getUTCMonth()] + " " + d.getUTCFullYear(); }

  // ---- formatting --------------------------------------------------------------
  function num(v, dec) {
    return v.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec });
  }
  function hours(h) {
    var t = Math.round(Math.abs(h) * 60), hh = Math.floor(t / 60), mm = t % 60;
    return hh === 0 ? mm + " min" : hh + " h " + (mm < 10 ? "0" : "") + mm + " min";
  }
  function clock(h) {
    var t = ((Math.round(h * 60) % 1440) + 1440) % 1440, hh = Math.floor(t / 60), mm = t % 60;
    var m2 = (mm < 10 ? "0" : "") + mm;
    if (D.clock24) return (hh < 10 ? "0" : "") + hh + ":" + m2;
    return (hh % 12 || 12) + ":" + m2 + (hh < 12 ? " am" : " pm");
  }
  function fmt(m, v) {
    if (v === null || v === undefined || !isFinite(v)) return "–";
    if (m.kind === "hours") return hours(v);
    if (m.kind === "clock") return clock(v);
    return num(v, m.decimals) + unitSuffix(m);
  }
  function unitSuffix(m) { return !m.unit ? "" : m.unit === "%" ? "%" : " " + m.unit; }
  function fmtDelta(m, d) {
    if (d === null || !isFinite(d)) return "–";
    var sign = d >= 0 ? "+" : "−";
    if (m.kind === "hours") return sign + hours(d);
    if (m.kind === "clock") return sign + Math.round(Math.abs(d) * 60) + " min";
    return sign + num(Math.abs(d), m.decimals) + unitSuffix(m);
  }

  // ---- statistics ----------------------------------------------------------------
  function slice(m, a, b) {
    var out = [];
    for (var i = Math.max(0, a); i <= Math.min(N - 1, b); i++) if (m.values[i] !== null) out.push(m.values[i]);
    return out;
  }
  function mean(a) { var s = 0; a.forEach(function (v) { s += v; }); return a.length ? s / a.length : null; }
  function median(a) {
    if (!a.length) return null;
    var s = a.slice().sort(function (x, y) { return x - y; }), h = s.length >> 1;
    return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
  }
  function summarise(m, vals) {
    if (!vals.length) return null;
    if (m.agg === "median") return median(vals);
    if (m.agg === "total") { var s = 0; vals.forEach(function (v) { s += v; }); return s; }
    return mean(vals);
  }
  function aggWord(m) { return m.agg === "median" ? "typical (median)" : m.agg === "total" ? "total" : "daily average"; }
  function rank(a) {
    var idx = a.map(function (v, i) { return [v, i]; }).sort(function (x, y) { return x[0] - y[0]; });
    var r = new Array(a.length), i = 0;
    while (i < idx.length) {
      var j = i;
      while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
      for (var k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2 + 1;
      i = j + 1;
    }
    return r;
  }
  function pearson(x, y) {
    var mx = mean(x), my = mean(y), sxy = 0, sxx = 0, syy = 0;
    for (var i = 0; i < x.length; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); syy += (y[i] - my) * (y[i] - my); }
    return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : null;
  }

  // ---- grouping --------------------------------------------------------------------
  function groupMode() {
    if (state.group !== "auto") return state.group;
    var span = state.to - state.from + 1;
    return span <= 120 ? "day" : span <= 800 ? "week" : "month";
  }
  function bucketStart(t, mode) {
    var d = new Date(t);
    if (mode === "week") return t - ((d.getUTCDay() + 6) % 7) * DAY;
    if (mode === "month") return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), 1);
    if (mode === "year") return Date.UTC(d.getUTCFullYear(), 0, 1);
    return t;
  }
  function grouped(m, mode) {
    var xs = [], ys = [], ns = [], cur = null, vals = [];
    function flush() { if (cur !== null && vals.length) { xs.push(iso(cur)); ys.push(summarise(m, vals)); ns.push(vals.length); } }
    for (var i = state.from; i <= state.to; i++) {
      var b = bucketStart(dates[i], mode);
      if (b !== cur) { flush(); cur = b; vals = []; }
      if (m.values[i] !== null) vals.push(m.values[i]);
    }
    flush();
    return { x: xs, y: ys, n: ns };
  }
  var GROUP_LABEL = { day: "each day", week: "weekly", month: "monthly", year: "yearly" };

  // ---- plotting ----------------------------------------------------------------------
  function plot(node, data, layout, height) {
    var entry = null;
    plots.forEach(function (p) { if (p.node === node) entry = p; });
    if (!entry) { entry = { node: node }; plots.push(entry); }
    entry.data = data; entry.layout = layout; entry.height = height;
    draw(entry);
  }
  function draw(p) {
    var pal = HL.PALETTE[HL.mode()];
    var layout = HL.merge(HL.baseLayout(pal, false), HL.resolve(p.layout, pal));
    layout.height = p.height || 300;
    window.Plotly.react(p.node, HL.resolve(p.data, pal), layout, HL.CONFIG);
  }
  HL.onTheme = function () { plots.forEach(draw); };
  function yAxisFor(m) {
    if (m.kind === "clock") return { tickformat: ".1f" };
    return { ticksuffix: m.kind === "hours" ? " h" : "" };
  }
  function hoverText(m, xs, ys, ns) {
    return xs.map(function (x, i) { return nice(Date.parse(x + "T00:00:00Z")) + ": " + fmt(m, ys[i]) + (ns && ns[i] > 1 ? " (" + ns[i] + " days)" : ""); });
  }

  // ---- filter row ------------------------------------------------------------------------
  function setRange(preset) {
    state.preset = preset;
    state.to = N - 1;
    if (preset === "all") state.from = 0;
    else if (/^y\d{4}$/.test(preset)) {
      var y = parseInt(preset.slice(1), 10);
      state.from = 0; state.to = -1;
      for (var i = 0; i < N; i++) { if (new Date(dates[i]).getUTCFullYear() === y) { if (state.to < 0) state.from = i; state.to = i; } }
    } else state.from = Math.max(0, N - parseInt(preset, 10));
    $("date-from").value = iso(dates[state.from]);
    $("date-to").value = iso(dates[state.to]);
  }
  function customRange() {
    var a = Date.parse($("date-from").value + "T00:00:00Z"), b = Date.parse($("date-to").value + "T00:00:00Z");
    if (!isFinite(a) || !isFinite(b)) return;
    if (a > b) { var t = a; a = b; b = t; }
    var from = 0, to = N - 1;
    for (var i = 0; i < N; i++) { if (dates[i] < a) from = i + 1; if (dates[i] <= b) to = i; }
    if (from > to) return;
    state.from = Math.min(from, N - 1); state.to = to; state.preset = "custom";
    $("range-preset").value = "custom";
    render();
  }
  function fillSelect(sel, value) {
    var groups = {};
    D.metrics.forEach(function (m) {
      if (!groups[m.domainLabel]) { groups[m.domainLabel] = document.createElement("optgroup"); groups[m.domainLabel].label = m.domainLabel; sel.appendChild(groups[m.domainLabel]); }
      var o = el("option", null, m.label); o.value = m.key; groups[m.domainLabel].appendChild(o);
    });
    sel.value = value;
  }

  // ---- views -------------------------------------------------------------------------------
  function prevWindow() { var len = state.to - state.from + 1; return [state.from - len, state.from - 1]; }

  function renderTiles() {
    var box = $("tiles"); clear(box);
    var pw = prevWindow();
    D.defaults.forEach(function (key) {
      var m = byKey[key], now = summarise(m, slice(m, state.from, state.to));
      var before = pw[1] >= 0 ? summarise(m, slice(m, pw[0], pw[1])) : null;
      var tile = el("div", "tile");
      tile.appendChild(el("span", "label", m.label + " · " + aggWord(m)));
      tile.appendChild(el("span", "value", fmt(m, now)));
      var d = el("span", "delta", "No earlier period to compare");
      if (now !== null && before !== null) {
        var diff = now - before;
        d.textContent = (diff >= 0 ? "▲ " : "▼ ") + fmtDelta(m, diff).slice(1) + " vs the period before";
        if (m.better === "higher" || m.better === "lower") d.className = "delta " + ((diff > 0) === (m.better === "higher") ? "good" : "bad");
      }
      tile.appendChild(d);
      box.appendChild(tile);
    });
    $("range-note").textContent = nice(dates[state.from]) + " – " + nice(dates[state.to]) + " · " + (state.to - state.from + 1).toLocaleString("en-US") + " days";
  }

  function renderExplorer() {
    var m = byKey[state.metric], mode = groupMode(), g = grouped(m, mode);
    $("explorer-sub").textContent = (m.agg === "total" ? "Total" : m.agg === "median" ? "Median" : "Average") + ", " + GROUP_LABEL[mode] + (m.unit ? " · " + m.unit : "") + (m.kind === "clock" ? " · hours after midnight" : "");
    $("explorer-desc").textContent = m.description || "";
    var vals = slice(m, state.from, state.to);
    var s = vals.slice().sort(function (a, b) { return a - b; });
    $("explorer-stats").textContent = vals.length ? "Days with data: " + vals.length.toLocaleString("en-US") + " · average " + fmt(m, mean(vals)) + " · median " + fmt(m, median(vals)) + " · lowest " + fmt(m, s[0]) + " · highest " + fmt(m, s[s.length - 1]) : "No data for this measure in the selected dates.";
    var bars = mode !== "day" && m.agg !== "median" && m.kind !== "clock";
    var trace = bars
      ? { type: "bar", x: g.x, y: g.y, marker: { color: "@series-1" } }
      : { type: "scatter", mode: g.x.length > 60 ? "lines" : "lines+markers", x: g.x, y: g.y, line: { color: "@series-1", width: 2 }, marker: { size: 7, color: "@series-1", line: { color: "@surface", width: 2 } }, connectgaps: false };
    trace.customdata = hoverText(m, g.x, g.y, g.n);
    trace.hovertemplate = "%{customdata}<extra></extra>";
    var y = yAxisFor(m);
    if (bars) y.rangemode = "tozero";
    plot($("explorer-plot"), [trace], { xaxis: { type: "date" }, yaxis: y, bargap: 0.25 }, 320);
    var rows = g.x.map(function (x, i) { return [nice(Date.parse(x + "T00:00:00Z")), fmt(m, g.y[i]), g.n[i]]; }).reverse();
    setTable("explorer-table", { columns: [mode === "day" ? "Day" : "Period starting", m.label, "Days with data"], rows: rows });
  }

  function renderCompare() {
    var a = byKey[state.metric], b = byKey[state.metricB], mode = groupMode();
    var ga = grouped(a, mode), gb = grouped(b, mode);
    function line(m, g, axis, color) {
      return { type: "scatter", mode: "lines", x: g.x, y: g.y, yaxis: axis, name: m.label, line: { color: color, width: 2 }, customdata: hoverText(m, g.x, g.y, g.n), hovertemplate: "%{customdata}<extra></extra>" };
    }
    var ax = { type: "date" };
    plot($("compare-plot"), [line(a, ga, "y", "@series-1"), line(b, gb, "y2", "@series-2")], {
      showlegend: true, xaxis: ax, hovermode: "x unified",
      yaxis: HL.merge(yAxisFor(a), { domain: [0.56, 1] }),
      yaxis2: HL.merge(yAxisFor(b), { domain: [0, 0.44], anchor: "x", gridcolor: "@grid", linecolor: "@axis", tickfont: { color: "@muted", size: 11 }, zeroline: false, fixedrange: true })
    }, 380);
    var xs = [], ys = [], txt = [];
    for (var i = state.from; i <= state.to; i++) {
      var j = i + state.lag;
      if (j < 0 || j >= N) continue;
      if (a.values[i] !== null && b.values[j] !== null) { xs.push(a.values[i]); ys.push(b.values[j]); txt.push(nice(dates[i]) + ": " + fmt(a, a.values[i]) + " → " + fmt(b, b.values[j])); }
    }
    var note = $("compare-note");
    if (xs.length < 10) {
      note.textContent = "Fewer than 10 days have both measures in this range, so there is nothing reliable to compare.";
      plot($("scatter-plot"), [], { xaxis: {}, yaxis: {} }, 300);
      return;
    }
    var rho = pearson(rank(xs), rank(ys));
    var strength = Math.abs(rho) >= 0.5 ? "strong" : Math.abs(rho) >= 0.3 ? "moderate" : Math.abs(rho) >= 0.1 ? "weak" : "no clear";
    var lagText = state.lag === 0 ? "the same day" : state.lag === 1 ? "the next day" : state.lag + " days later";
    note.textContent = xs.length.toLocaleString("en-US") + " days with both. Rank correlation ρ = " + (rho >= 0 ? "+" : "−") + Math.abs(rho).toFixed(2) + ": a " + strength + (strength === "no clear" ? " link" : (rho > 0 ? " positive" : " negative") + " link") + " between " + a.label + " and " + b.label + " " + lagText + ". This raw comparison includes seasons and long-term change, and a link is not proof of cause; the report's Connections section applies stricter checks.";
    plot($("scatter-plot"), [{ type: "scatter", mode: "markers", x: xs, y: ys, customdata: txt, hovertemplate: "%{customdata}<extra></extra>", marker: { size: 7, color: "@series-1/0.45", line: { color: "@surface", width: 1 } } }], {
      hovermode: "closest", xaxis: HL.merge(yAxisFor(a), { title: { text: a.label + (a.unit ? " (" + a.unit + ")" : "") }, fixedrange: true }),
      yaxis: HL.merge(yAxisFor(b), { title: { text: b.label + (b.unit ? " (" + b.unit + ")" : "") } })
    }, 320);
  }

  function renderCalendar() {
    var m = byKey[state.metric];
    var to = state.to, from = Math.max(state.from, to - 370);
    var start = bucketStart(dates[from], "week");
    var weeks = Math.floor((dates[to] - start) / (7 * DAY)) + 1;
    var z = [], text = [], x = [];
    for (var r = 0; r < 7; r++) { z.push(new Array(weeks).fill(null)); text.push(new Array(weeks).fill("")); }
    for (var w = 0; w < weeks; w++) x.push(iso(start + w * 7 * DAY));
    for (var i = from; i <= to; i++) {
      var wk = Math.floor((dates[i] - start) / (7 * DAY)), wd = (new Date(dates[i]).getUTCDay() + 6) % 7;
      z[wd][wk] = m.values[i];
      text[wd][wk] = nice(dates[i]) + ": " + fmt(m, m.values[i]);
    }
    $("calendar-sub").textContent = m.label + ", each day" + (from > state.from ? " · last 12 months of the selected range" : "") + " · darker is higher";
    plot($("calendar-plot"), [{ type: "heatmap", z: z, x: x, y: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], customdata: text, hovertemplate: "%{customdata}<extra></extra>", colorscale: "%seq", xgap: 2, ygap: 2, hoverongaps: false, colorbar: { thickness: 8, outlinewidth: 0, tickfont: { color: "@muted", size: 10 } } }],
      { xaxis: { type: "date", showgrid: false }, yaxis: { autorange: "reversed", type: "category", showgrid: false } }, 230);
  }

  function renderSleep() {
    var a = iso(dates[state.from]), b = iso(dates[state.to]);
    var ns = D.nights.filter(function (n) { return n.date >= a && n.date <= b && n.bed !== null && n.wake !== null; });
    var card = $("sleep-card");
    card.hidden = D.nights.length === 0;
    if (card.hidden) return;
    if (ns.length > 400) ns = ns.slice(ns.length - 400);
    $("sleep-sub").textContent = ns.length ? ns.length + " nights · bar runs from bedtime (top) to wake-up" : "No sleep recorded in the selected dates.";
    var kinds = [["stages", "Watch (stages)", "@series-1"], ["asleep", "Watch / app", "@series-3"], ["inbed", "iPhone time in bed", "@series-2"]];
    var data = [], lo = 99, hi = 0;
    kinds.forEach(function (k) {
      var sub = ns.filter(function (n) { return n.kind === k[0]; });
      if (!sub.length) return;
      sub.forEach(function (n) { lo = Math.min(lo, n.bed); hi = Math.max(hi, n.wake + 24); });
      data.push({ type: "bar", name: k[1], x: sub.map(function (n) { return n.date; }), base: sub.map(function (n) { return n.bed; }), y: sub.map(function (n) { return n.wake + 24 - n.bed; }), marker: { color: k[2] },
        customdata: sub.map(function (n) { return nice(Date.parse(n.date + "T00:00:00Z")) + ": " + clock(n.bed) + " → " + clock(n.wake) + (n.sleep !== null ? " · asleep " + hours(n.sleep) : "") + (n.inbed !== null ? " · in bed " + hours(n.inbed) : ""); }), hovertemplate: "%{customdata}<extra></extra>" });
    });
    var ticks = [];
    for (var t = Math.floor(lo); t <= Math.ceil(hi); t += 2) ticks.push(t);
    plot($("sleep-plot"), data, { showlegend: data.length > 1, barmode: "overlay", bargap: 0.2, xaxis: { type: "date" }, yaxis: { range: [Math.ceil(hi) + 0.3, Math.floor(lo) - 0.3], tickvals: ticks, ticktext: ticks.map(clock) } }, 320);
  }

  var W_COLS = [["date", "Date"], ["type", "Type"], ["min", "Duration"], ["dist", "Distance (" + D.distUnit + ")"], ["pace", "Pace"], ["hr", "Avg HR"], ["kcal", "kcal"]];
  function renderWorkouts() {
    var card = $("workouts-card");
    card.hidden = D.workouts.length === 0;
    if (card.hidden) return;
    var a = iso(dates[state.from]), b = iso(dates[state.to]);
    var rows = D.workouts.filter(function (w) { return w.date >= a && w.date <= b && (state.wType === "All" || w.type === state.wType); });
    var k = state.wSort;
    rows.sort(function (p, q) { var x = p[k], y = q[k]; if (x === null) return 1; if (y === null) return -1; return (x < y ? -1 : x > y ? 1 : 0) * (state.wDesc ? -1 : 1); });
    var total = 0, dist = 0;
    rows.forEach(function (w) { total += w.min || 0; dist += w.dist || 0; });
    $("workouts-sub").textContent = rows.length.toLocaleString("en-US") + " sessions · " + hours(total / 60) + (dist > 0 ? " · " + num(dist, 1) + " " + D.distUnit : "");
    var table = $("workouts-table"); clear(table);
    var thead = el("thead"), tr = el("tr");
    W_COLS.forEach(function (c) {
      var th = el("th"); th.scope = "col";
      var btn = el("button", "text sort", c[1] + (state.wSort === c[0] ? (state.wDesc ? " ▼" : " ▲") : ""));
      btn.type = "button";
      btn.addEventListener("click", function () { if (state.wSort === c[0]) state.wDesc = !state.wDesc; else { state.wSort = c[0]; state.wDesc = true; } renderWorkouts(); });
      th.appendChild(btn); tr.appendChild(th);
    });
    thead.appendChild(tr); table.appendChild(thead);
    var tbody = el("tbody");
    rows.slice(0, 300).forEach(function (w) {
      var r = el("tr");
      var th = el("th", null, nice(Date.parse(w.date + "T00:00:00Z"))); th.scope = "row"; r.appendChild(th);
      r.appendChild(el("td", null, w.type));
      r.appendChild(el("td", null, w.min === null ? "–" : hours(w.min / 60)));
      r.appendChild(el("td", null, w.dist === null || w.dist === 0 ? "–" : num(w.dist, 2)));
      r.appendChild(el("td", null, w.pace || "–"));
      r.appendChild(el("td", null, w.hr === null ? "–" : num(w.hr, 0)));
      r.appendChild(el("td", null, w.kcal === null ? "–" : num(w.kcal, 0)));
      tbody.appendChild(r);
    });
    table.appendChild(tbody);
    $("workouts-more").textContent = rows.length > 300 ? "Showing the first 300 of " + rows.length + ". Narrow the dates or type to see the rest." : "";
  }

  function renderPeriods() {
    var pw = prevWindow(), table = $("periods-table"); clear(table);
    var has = pw[1] >= 0;
    $("periods-sub").textContent = has ? "Selected dates against the " + (state.to - state.from + 1) + " days before (" + nice(dates[Math.max(0, pw[0])]) + " – " + nice(dates[pw[1]]) + ")" : "There is no earlier data to compare with; choose a shorter range.";
    var thead = el("thead"), tr = el("tr");
    ["Measure", "Selected dates", "Period before", "Change", "Days with data"].forEach(function (c) { var th = el("th", null, c); th.scope = "col"; tr.appendChild(th); });
    thead.appendChild(tr); table.appendChild(thead);
    var tbody = el("tbody");
    D.metrics.forEach(function (m) {
      var cur = slice(m, state.from, state.to);
      if (!cur.length) return;
      var now = summarise(m, cur), before = has ? summarise(m, slice(m, pw[0], pw[1])) : null;
      var r = el("tr");
      var th = el("th", null, m.label); th.scope = "row"; r.appendChild(th);
      r.appendChild(el("td", null, fmt(m, now)));
      r.appendChild(el("td", null, fmt(m, before)));
      r.appendChild(el("td", null, before === null ? "–" : fmtDelta(m, now - before)));
      r.appendChild(el("td", null, cur.length.toLocaleString("en-US")));
      tbody.appendChild(r);
    });
    table.appendChild(tbody);
  }

  function setTable(id, spec) {
    var holder = $(id); clear(holder);
    holder.appendChild(HL.buildTable({ table: spec }));
  }

  function render() {
    renderTiles(); renderExplorer(); renderCompare(); renderCalendar(); renderSleep(); renderWorkouts(); renderPeriods();
  }

  function init() {
    var preset = $("range-preset");
    var years = {};
    dates.forEach(function (t) { years[new Date(t).getUTCFullYear()] = true; });
    Object.keys(years).sort().reverse().forEach(function (y) { var o = el("option", null, y); o.value = "y" + y; preset.appendChild(o); });
    var custom = el("option", null, "Custom dates"); custom.value = "custom"; preset.appendChild(custom);
    $("date-from").min = $("date-to").min = iso(dates[0]);
    $("date-from").max = $("date-to").max = iso(dates[N - 1]);
    preset.value = state.preset; setRange(state.preset);
    preset.addEventListener("change", function () { if (preset.value !== "custom") { setRange(preset.value); render(); } });
    $("date-from").addEventListener("change", customRange);
    $("date-to").addEventListener("change", customRange);
    fillSelect($("metric-a"), state.metric); fillSelect($("metric-b"), state.metricB);
    $("metric-a").addEventListener("change", function (e) { state.metric = e.target.value; render(); });
    $("metric-b").addEventListener("change", function (e) { state.metricB = e.target.value; renderCompare(); });
    $("group").addEventListener("change", function (e) { state.group = e.target.value; renderExplorer(); renderCompare(); });
    $("lag").addEventListener("change", function (e) { state.lag = parseInt(e.target.value, 10); renderCompare(); });
    var types = { All: true };
    D.workouts.forEach(function (w) { types[w.type] = true; });
    Object.keys(types).forEach(function (t) { var o = el("option", null, t === "All" ? "All types" : t); o.value = t; $("workout-type").appendChild(o); });
    $("workout-type").addEventListener("change", function (e) { state.wType = e.target.value; renderWorkouts(); });
    var tbtn = $("explorer-table-btn");
    tbtn.addEventListener("click", function () { var h = $("explorer-table"); h.hidden = !h.hidden; tbtn.textContent = h.hidden ? "Show table" : "Hide table"; tbtn.setAttribute("aria-expanded", h.hidden ? "false" : "true"); });
    render();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
