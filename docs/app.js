/* ============================================================
   MINILOTO ANALYZER - Frontend Application (マルチ期間対応)
   ============================================================ */

let analysisData = null;
let rawData = null;
let currentPeriod = "all"; // 現在選択中の期間

/** 現在の期間のデータを返す */
function getPeriodData() {
  return analysisData.periods[currentPeriod];
}

// ============================================================
// Init
// ============================================================
document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  await loadData();
});

async function loadData() {
  const t = Date.now();
  try {
    const [analysisRes, rawRes] = await Promise.all([
      fetch(`data/analysis.json?t=${t}`),
      fetch(`data/miniloto_data.json?t=${t}`),
    ]);
    analysisData = await analysisRes.json();
    rawData = await rawRes.json();
    setupPeriodSlider();
    renderAll();
  } catch (e) {
    document.querySelector("main").innerHTML =
      '<div class="loading">データの読み込みに失敗しました。データが生成されていない可能性があります。</div>';
    console.error(e);
  }
}

function renderAll() {
  renderStatsBar();
  renderPrediction();
  renderFrequency();
  renderPull();
  renderZone();
  renderPair();
  renderRecent();
  renderArchive();
  renderMonteCarlo();
}

// ============================================================
// Tabs
// ============================================================
function setupTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
      updatePeriodVisibility();
      // 非表示タブ内で生成されたグラフは幅0で潰れるため、表示された時にリサイズし直す
      [freqChart, pullChart, zoneChart].forEach((c) => { if (c) c.resize(); });
    });
  });
}

/** 期間スライダーの表示制御：AI予想タブは全期間を並べて表示するのでスライダーを隠す */
function updatePeriodVisibility() {
  const section = document.getElementById("periodSection");
  if (!section || section.dataset.ready !== "1") return;
  const activeTab = document.querySelector(".tab.active");
  const isPrediction = activeTab && activeTab.dataset.tab === "prediction";
  section.style.display = isPrediction ? "none" : "block";
}

// ============================================================
// Period Slider
// ============================================================
function setupPeriodSlider() {
  const labels = analysisData.period_labels;
  if (!labels || labels.length <= 1) return;

  const slider = document.getElementById("periodSlider");
  const display = document.getElementById("periodDisplay");
  if (!slider || !display) return;

  slider.min = 0;
  slider.max = labels.length - 1;
  slider.value = labels.length - 1; // デフォルトは全期間（最後）
  slider.step = 1;

  function update() {
    const idx = parseInt(slider.value);
    const info = labels[idx];
    currentPeriod = info.key;
    display.innerHTML = `<span class="period-name">${info.label}</span><span class="period-range">${info.range}（${info.draws}回分）</span>`;
    renderAll();
  }

  slider.addEventListener("input", update);
  update();

  const section = document.getElementById("periodSection");
  section.dataset.ready = "1";
  updatePeriodVisibility();
}

// ============================================================
// Stats Bar
// ============================================================
function renderStatsBar() {
  const s = getPeriodData().summary_stats;
  document.getElementById("statsBar").innerHTML = `
    分析対象 <span>${s.total_draws}</span> 回分 ｜
    期間 <span>${s.date_range[0]}</span> 〜 <span>${s.date_range[1]}</span> ｜
    最終更新 <span>${analysisData.last_updated.split("T")[0]}</span>
  `;
}

// ============================================================
// Helpers
// ============================================================
function createBall(num, cls = "ball-gold") {
  return `<span class="ball ${cls}">${num}</span>`;
}

function createSmallBall(num, cls = "ball-gold") {
  return `<span class="ball ball-small ${cls}">${num}</span>`;
}

// ============================================================
// AI Prediction
// ============================================================
// 予想モードの表示順（データの mode_name をそのまま見出しに使う）
const MODE_ORDER = ["balanced", "frequency_heavy", "pull_heavy", "zone_balanced", "pair_heavy", "ml_heavy"];

function renderPrediction() {
  const container = document.getElementById("predictionResult");
  const periods = analysisData.period_labels; // [直近100, 200, 300, 400, 全期間]

  const cards = MODE_ORDER.map((mode) => {
    const base = analysisData.periods.all.predictions[mode];
    if (!base) return "";
    const isFeatured = mode === "balanced";
    const modeName = base.mode_name;

    const rows = periods
      .map((pInfo) => {
        const pdata = analysisData.periods[pInfo.key];
        const pred = pdata && pdata.predictions[mode];
        if (!pred) return "";

        // 予想が空（制約を満たす組が見つからなかった期間）はメッセージ表示
        if (!pred.numbers || pred.numbers.length === 0) {
          return `
            <div class="period-row">
              <div class="period-row-main">
                <span class="period-tag">${pInfo.label}</span>
                <span class="period-empty">この期間はデータ条件により予想を生成できませんでした</span>
              </div>
            </div>`;
        }

        const balls =
          pred.numbers.map((n) => createSmallBall(n)).join("") +
          `<span class="plus">+</span>` +
          (pred.bonus ? createSmallBall(pred.bonus, "ball-bonus") : "");

        const reasonsHtml =
          pred.numbers
            .map((n) => {
              const r = pred.reasons[String(n)];
              const mc = r.monte_carlo_pct != null ? `<span class="mc-badge" title="重み付き非復元抽出を1万回シミュレーションした際にこの数字が選ばれた割合">🎲 ${r.monte_carlo_pct}%</span>` : "";
              return `<div class="reason-item">
                <span class="ball ball-small ball-gold">${n}</span>
                <span class="reason-text">${r.reason_text}${mc}</span>
              </div>`;
            })
            .join("") +
          (pred.bonus
            ? `<div class="reason-item bonus-reason">
                <span class="ball ball-small ball-bonus">${pred.bonus}</span>
                <span class="reason-text">${pred.bonus_reason || "本数字に次ぐスコアで選出。"}</span>
              </div>`
            : "");

        return `
          <div class="period-row">
            <div class="period-row-main">
              <span class="period-tag">${pInfo.label}</span>
              <div class="balls-row compact">${balls}</div>
              <span class="period-metrics">奇偶 ${pred.metrics.odd_even} ｜ 合計 ${pred.metrics.sum}</span>
            </div>
            <button class="reasons-toggle sm" onclick="toggleReasons(this)">選出根拠 ▾</button>
            <div class="reasons-detail">${reasonsHtml}</div>
          </div>`;
      })
      .join("");

    return `
      <div class="prediction-card ${isFeatured ? "featured" : ""}">
        <h3>${modeName}</h3>
        <div class="period-rows">${rows}</div>
      </div>`;
  }).join("");

  container.innerHTML = `<p class="ball-legend">● 本数字 ○ ボーナス数字</p>` + cards;
}

function toggleReasons(btn) {
  const detail = btn.nextElementSibling;
  const open = detail.classList.toggle("open");
  btn.textContent = open ? "選出根拠を閉じる ▴" : "選出根拠 ▾";
}

// ============================================================
// Frequency
// ============================================================
let freqChart = null;
function renderFrequency() {
  const freq = getPeriodData().frequency;
  const labels = Array.from({ length: 31 }, (_, i) => i + 1);
  const counts = labels.map((n) => freq.counts[String(n)] || 0);

  const ctx = document.getElementById("freqChart").getContext("2d");
  if (freqChart) freqChart.destroy();
  freqChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "出現回数",
          data: counts,
          backgroundColor: labels.map((n) =>
            freq.hot.includes(n) ? "#e08a7a" : freq.cold.includes(n) ? "#57b0a5" : "#d6b24e"
          ),
          borderRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        title: { display: true, text: "数字別 出現回数", color: "#3a3843" },
      },
      scales: {
        x: { ticks: { color: "#888" }, grid: { display: false } },
        y: { ticks: { color: "#888" }, grid: { color: "rgba(0,0,0,0.06)" } },
      },
    },
  });

  document.getElementById("hotList").innerHTML = `<ul class="rank-list">${freq.hot
    .map(
      (n, i) =>
        `<li><span class="rank">${i + 1}</span> <span>${n}</span> <span>${freq.counts[String(n)]}回 (${freq.percentages[String(n)]}%)</span></li>`
    )
    .join("")}</ul>`;

  document.getElementById("coldList").innerHTML = `<ul class="rank-list">${freq.cold
    .map(
      (n, i) =>
        `<li><span class="rank">${i + 1}</span> <span>${n}</span> <span>${freq.counts[String(n)]}回 (${freq.percentages[String(n)]}%)</span></li>`
    )
    .join("")}</ul>`;

  const droughtNums = labels.map((n) => ({
    num: n,
    val: freq.drought[String(n)] || 0,
  }));
  const maxDrought = Math.max(...droughtNums.map((d) => d.val));

  document.getElementById("droughtGrid").innerHTML = droughtNums
    .map((d) => {
      const intensity = maxDrought > 0 ? d.val / maxDrought : 0;
      const cls = intensity > 0.7 ? "cold" : intensity < 0.2 ? "hot" : "";
      return `<div class="grid-cell ${cls}">
        <div class="num">${d.num}</div>
        <div class="val" style="color: ${intensity > 0.5 ? "#57b0a5" : "#e08a7a"}">${d.val}回前</div>
      </div>`;
    })
    .join("");
}

// ============================================================
// Pull
// ============================================================
let pullChart = null;
function renderPull() {
  const pull = getPeriodData().pull;

  document.getElementById("pullAvg").innerHTML = `
    <div class="number">${pull.average}</div>
    <div class="label">平均引っ張り数（前回からの重複数字数）</div>
  `;

  const labels = Object.keys(pull.distribution);
  const values = Object.values(pull.distribution);
  const ctx = document.getElementById("pullChart").getContext("2d");
  if (pullChart) pullChart.destroy();
  pullChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels.map((l) => `${l}個`),
      datasets: [
        {
          label: "回数",
          data: values,
          backgroundColor: "#d6b24e",
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        title: { display: true, text: "引っ張り数の分布", color: "#3a3843" },
      },
      scales: {
        x: { ticks: { color: "#888" }, grid: { display: false } },
        y: { ticks: { color: "#888" }, grid: { color: "rgba(0,0,0,0.06)" } },
      },
    },
  });

  const rows = pull.recent_pulls
    .slice()
    .reverse()
    .map(
      (p) => `<tr>
      <td>${p.round}</td>
      <td>${p.date}</td>
      <td>${p.numbers.map((n) => createSmallBall(n, p.pulled.includes(n) ? "ball-gold" : "ball-bonus")).join("")}</td>
      <td>${p.pull_count}個</td>
    </tr>`
    )
    .join("");

  document.getElementById("pullTable").innerHTML = `
    <table class="data-table">
      <thead><tr><th>回</th><th>日付</th><th>数字（金=引っ張り）</th><th>重複数</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

// ============================================================
// Zone
// ============================================================
let zoneChart = null;
function renderZone() {
  const zone = getPeriodData().zone;

  const ctx = document.getElementById("zoneChart").getContext("2d");
  if (zoneChart) zoneChart.destroy();
  zoneChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["低帯 (1-10)", "中帯 (11-21)", "高帯 (22-31)"],
      datasets: [
        {
          data: [zone.zone_averages.low, zone.zone_averages.mid, zone.zone_averages.high],
          backgroundColor: ["#e08a7a", "#d6b24e", "#57b0a5"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#3a3843" } },
        title: { display: true, text: "数字帯の出現比率", color: "#3a3843" },
      },
    },
  });

  const rows = zone.top_patterns
    .map(
      (p, i) => `<tr>
      <td style="color: var(--gold); font-weight: bold">${i + 1}</td>
      <td>${p.pattern}</td>
      <td>${p.count}回</td>
      <td>${p.percentage}%</td>
    </tr>`
    )
    .join("");

  document.getElementById("zonePatterns").innerHTML = `
    <table class="data-table">
      <thead><tr><th>#</th><th>パターン（低-中-高）</th><th>回数</th><th>割合</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

// ============================================================
// Pair
// ============================================================
function renderPair() {
  const pairs = getPeriodData().pairs;

  const rows = pairs.top_pairs
    .map(
      (p, i) => `<tr>
      <td style="color: var(--gold)">${i + 1}</td>
      <td>${createSmallBall(p.pair[0])} ${createSmallBall(p.pair[1])}</td>
      <td>${p.count}回</td>
    </tr>`
    )
    .join("");

  document.getElementById("pairTable").innerHTML = `
    <table class="data-table">
      <thead><tr><th>#</th><th>ペア</th><th>同時出現</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;

  const selector = document.getElementById("numberSelector");
  selector.innerHTML = Array.from({ length: 31 }, (_, i) => i + 1)
    .map((n) => `<button class="num-btn" data-num="${n}">${n}</button>`)
    .join("");

  // イベントリスナーの重複を防ぐためcloneで置換
  const newSelector = selector.cloneNode(true);
  selector.parentNode.replaceChild(newSelector, selector);

  newSelector.addEventListener("click", (e) => {
    if (!e.target.classList.contains("num-btn")) return;
    newSelector.querySelectorAll(".num-btn").forEach((b) => b.classList.remove("selected"));
    e.target.classList.add("selected");
    showAffinity(e.target.dataset.num);
  });
}

function showAffinity(num) {
  const affinity = getPeriodData().pairs.affinity[String(num)];
  if (!affinity) return;

  document.getElementById("affinityResult").innerHTML = `
    <div style="margin-top: 1rem">
      <p style="color: var(--text-muted); margin-bottom: 0.5rem">${num} と相性の良い数字 TOP5:</p>
      <div class="balls-row">
        ${affinity.map((a) => `<div style="text-align:center">${createSmallBall(a.number)}<br><small style="color:var(--text-muted)">${a.count}回</small></div>`).join("")}
      </div>
    </div>
  `;
}

// ============================================================
// Recent Results
// ============================================================
function renderRecent() {
  const recent = getPeriodData().recent_draws;
  const rows = recent
    .slice()
    .reverse()
    .map(
      (d) => `<tr>
      <td>${d.round}</td>
      <td>${d.date}</td>
      <td>${d.numbers.map((n) => createSmallBall(n)).join("")} ${createSmallBall(d.bonus, "ball-bonus")}</td>
      <td>${d.odd_even}</td>
      <td>${d.zones}</td>
      <td>${d.sum}</td>
    </tr>`
    )
    .join("");

  document.getElementById("recentTable").innerHTML = `
    <table class="data-table">
      <thead><tr><th>回</th><th>日付</th><th>数字 (+ bonus)</th><th>奇偶</th><th>帯</th><th>合計</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

// ============================================================
// Archive
// ============================================================
// アーカイブ（予想モードタブ × 期間まとめ × ページング）
let archiveMode = "balanced";
let archivePage = 0;
const ARCHIVE_PAGE_SIZE = 5;
const ARCHIVE_PERIOD_ORDER = ["all", "100", "200", "300", "400"];

function archivePeriodLabel(pk) {
  return pk === "all" ? "全期間" : `直近${pk}回`;
}

function renderArchive() {
  const container = document.getElementById("archiveList");
  // 確定（答え合わせ済み）分のみ・新しい順。結果待ちの次回予想はアーカイブには出さない
  const verified = (analysisData.archive || []).filter((e) => e.verified).reverse();

  const modeNameOf = (m) => {
    const p = analysisData.periods.all.predictions[m];
    return p ? p.mode_name : m;
  };

  // 予想モードタブ
  let html = `<div class="archive-mode-tabs">`;
  for (const m of MODE_ORDER) {
    html += `<button class="archive-mode-tab ${m === archiveMode ? "active" : ""}" onclick="setArchiveMode('${m}')">${modeNameOf(m)}</button>`;
  }
  html += `</div>`;

  if (verified.length === 0) {
    html += `<div class="card"><p style="color:var(--text-muted)">答え合わせ済みのアーカイブはまだありません。</p></div>`;
    container.innerHTML = html;
    return;
  }

  const totalPages = Math.ceil(verified.length / ARCHIVE_PAGE_SIZE);
  archivePage = Math.max(0, Math.min(archivePage, totalPages - 1));
  const start = archivePage * ARCHIVE_PAGE_SIZE;
  const pageEntries = verified.slice(start, start + ARCHIVE_PAGE_SIZE);

  for (const entry of pageEntries) {
    const actual = entry.actual;
    html += `<div class="card archive-entry">`;
    html += `<div class="archive-entry-head"><span class="archive-round">第${entry.predicted_round}回</span>`;
    if (actual) {
      html += `<span class="archive-actual">実際 ` +
        actual.numbers.map((n) => createSmallBall(n)).join("") +
        ` <span class="plus">+</span> ${createSmallBall(actual.bonus, "ball-bonus")}` +
        ` <span class="archive-date">${actual.date}</span></span>`;
    }
    html += `</div>`;

    for (const pk of ARCHIVE_PERIOD_ORDER) {
      const pdata = entry.predictions_by_period[pk];
      const pred = pdata && pdata.modes[archiveMode];
      html += `<div class="archive-period-row"><span class="archive-period-label">${archivePeriodLabel(pk)}</span>`;
      if (!pred || !pred.numbers || pred.numbers.length === 0) {
        html += `<span class="archive-empty">データ不足</span></div>`;
        continue;
      }
      const matched = new Set(pred.matched_numbers || []);
      const actualBonus = actual ? actual.bonus : null;
      const actualNums = actual ? actual.numbers : [];
      // 予想の各数字を、本数字一致・ボーナス一致・ハズレの3通りでマーキング
      const ballClass = (n) =>
        matched.has(n)
          ? "ball-matched"
          : actualBonus != null && n === actualBonus
          ? "ball-bonus-hit"
          : "ball-miss";
      html += `<span class="archive-balls-inline">`;
      for (const n of pred.numbers) {
        html += `<span class="ball ball-small ${ballClass(n)}">${n}</span>`;
      }
      if (pred.bonus != null) {
        // 予想ボーナスも同様に判定（本数字に当たれば本数字一致、実ボーナスに当たればボーナス一致）
        const bcls = actualNums.includes(pred.bonus)
          ? "ball-matched"
          : pred.bonus_matched
          ? "ball-bonus-hit"
          : "ball-miss";
        html += ` <span class="plus">+</span> <span class="ball ball-small ${bcls}">${pred.bonus}</span>`;
      }
      html += `</span>`;

      // 予想内にボーナス数字を含んでいたか（本数字予想 or 予想ボーナスが実ボーナスに一致）
      const bonusHit =
        actualBonus != null &&
        (pred.numbers.includes(actualBonus) || pred.bonus_matched);
      if (bonusHit) {
        html += `<span class="archive-bonus-flag">＋ボーナス${actualBonus}的中</span>`;
      }

      const mc = pred.match_count;
      const cls = mc >= 4 ? "match-high" : mc >= 3 ? "match-mid" : "match-low";
      let prize = "";
      if (mc === 5) prize = " 🥇1等";
      else if (mc === 4 && pred.bonus_matched) prize = " 🥈2等";
      else if (mc === 4) prize = " 🥉3等";
      else if (mc === 3) prize = " 4等";
      html += `<span class="archive-match ${cls}">${mc}個一致${prize}</span></div>`;
    }
    html += `</div>`;
  }

  html += `<div class="archive-pager">`;
  html += `<button class="pager-btn" onclick="setArchivePage(${archivePage - 1})" ${archivePage === 0 ? "disabled" : ""}>← 前へ</button>`;
  html += `<span class="pager-info">${archivePage + 1} / ${totalPages}</span>`;
  html += `<button class="pager-btn" onclick="setArchivePage(${archivePage + 1})" ${archivePage >= totalPages - 1 ? "disabled" : ""}>次へ →</button>`;
  html += `</div>`;

  container.innerHTML = html;
}

function setArchiveMode(m) {
  archiveMode = m;
  archivePage = 0;
  renderArchive();
}

function setArchivePage(p) {
  archivePage = p;
  renderArchive();
}

// ============================================================
// Monte Carlo (モンテカルロ信頼度)
// ============================================================
function renderMonteCarlo() {
  renderMcConfidenceGrid();
}

function renderMcConfidenceGrid() {
  const container = document.getElementById("mcConfidenceGrid");
  if (!container) return;

  const modeInput = document.querySelector('input[name="mode"]:checked');
  const mode = modeInput ? modeInput.value : "balanced";
  const pred = getPeriodData().predictions[mode];
  const mc = pred && pred.monte_carlo;
  if (!mc) {
    container.innerHTML = `<p style="color:var(--text-muted)">データがありません。</p>`;
    return;
  }

  const labels = Array.from({ length: 31 }, (_, i) => i + 1);
  const maxPct = Math.max(...labels.map((n) => mc[String(n)] || 0));

  container.innerHTML = labels
    .map((n) => {
      const pct = mc[String(n)] || 0;
      const isSelected = pred.numbers.includes(n);
      const intensity = maxPct > 0 ? pct / maxPct : 0;
      const color = intensity > 0.6 ? "#e08a7a" : intensity > 0.3 ? "#d6b24e" : "#57b0a5";
      return `<div class="grid-cell ${isSelected ? "hot" : ""}">
        <div class="num">${n}</div>
        <div class="val" style="color:${color}">${pct.toFixed(1)}%</div>
      </div>`;
    })
    .join("");
}
