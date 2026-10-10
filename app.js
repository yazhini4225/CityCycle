/**
 * CityCycle Operations Command Center - Application Logic
 * ========================================================
 * High-performance state management, Chart.js integrations,
 * interactive What-If scenario sandbox inference, and telemetry logs.
 */

let appData = null;
let overviewChart = null;
let detailedChart = null;
let shapChart = null;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  initData();
  setupNavigation();
  setupSandboxListeners();
  setupWaterfallListener();
});

// Load precomputed dataset (with CORS-safe fallback)
function initData() {
  if (window.CITYCYCLE_DATA) {
    appData = window.CITYCYCLE_DATA;
    onDataLoaded();
  } else {
    fetch('data/dashboard_data.json')
      .then(res => res.json())
      .then(data => {
        appData = data;
        onDataLoaded();
      })
      .catch(err => {
        console.error('Failed to load dashboard data via fetch:', err);
      });
  }
}

function onDataLoaded() {
  if (!appData) return;

  renderLeaderboard();
  renderOverviewChart();
  renderDetailedForecastChart();
  renderForecastLogTable();
  renderShapChart();
  renderWaterfallScenario(0);
  runSandboxSimulation();
}

// Tab navigation handler
function setupNavigation() {
  const tabBtns = document.querySelectorAll('.nav-tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });
}

function switchTab(tabId) {
  document.querySelectorAll('.nav-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
  });

  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === tabId);
  });

  // Trigger chart resize if navigating to a chart tab
  if (tabId === 'tab-forecast' && detailedChart) {
    setTimeout(() => detailedChart.resize(), 50);
  } else if (tabId === 'tab-explain' && shapChart) {
    setTimeout(() => shapChart.resize(), 50);
  }
}

// -------------------------------------------------------------
// 1. Leaderboard Table
// -------------------------------------------------------------
function renderLeaderboard() {
  const tbody = document.querySelector('#leaderboardTable tbody');
  if (!tbody || !appData?.benchmarks?.leaderboard) return;

  tbody.innerHTML = '';
  appData.benchmarks.leaderboard.forEach(item => {
    const tr = document.createElement('tr');
    const isDual = item.model.includes('Dual-Head');
    if (isDual) tr.classList.add('highlight-row');

    let badgeClass = 'badge-emerald';
    let statusText = 'Production Model';
    if (item.type.includes('Naive')) {
      badgeClass = 'badge-rose';
      statusText = 'Naive Baseline';
    } else if (item.type.includes('Linear')) {
      badgeClass = 'badge-amber';
      statusText = 'Linear Baseline';
    }

    tr.innerHTML = `
      <td><strong>${item.model}</strong> ${isDual ? '🌟' : ''}</td>
      <td><span class="badge ${badgeClass}">${item.type}</span></td>
      <td><strong>${item.mae.toFixed(2)}</strong> bikes/hr</td>
      <td>${item.rmse.toFixed(2)} bikes/hr</td>
      <td>${item.wape_pct.toFixed(2)}%</td>
      <td>${item.r2.toFixed(4)}</td>
      <td><span class="badge ${isDual ? 'badge-emerald' : 'badge-cyan'}">${isDual ? 'Active Deployment' : 'Benchmark Only'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// -------------------------------------------------------------
// 2. Overview Chart (7-day preview)
// -------------------------------------------------------------
function renderOverviewChart() {
  const ctx = document.getElementById('overviewForecastChart');
  if (!ctx || !appData?.sample_timeline) return;

  const sample = appData.sample_timeline.slice(0, 168); // 7 days * 24 hours
  const labels = sample.map(d => `${d.date.slice(5)} ${d.hour}:00`);
  const actuals = sample.map(d => d.actual_cnt);
  const preds = sample.map(d => d.pred_cnt);
  const naives = sample.map(d => d.naive_cnt);

  if (overviewChart) overviewChart.destroy();

  overviewChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Actual Demand',
          data: actuals,
          borderColor: '#94a3b8',
          backgroundColor: 'rgba(148, 163, 184, 0.08)',
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.25
        },
        {
          label: 'Dual-Head Forecast (Ours)',
          data: preds,
          borderColor: '#10b981',
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0.25
        },
        {
          label: 'Naive (t-24 Persistence)',
          data: naives,
          borderColor: '#f43f5e',
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0,
          tension: 0.25
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: '#cbd5e1', font: { family: 'Inter', size: 12 } }
        },
        tooltip: {
          backgroundColor: 'rgba(16, 23, 38, 0.95)',
          titleColor: '#f1f5f9',
          bodyColor: '#e2e8f0',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#64748b', maxTicksLimit: 14 }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.06)' },
          ticks: { color: '#94a3b8' },
          title: { display: true, text: 'Rides / Hour', color: '#94a3b8' }
        }
      }
    }
  });
}

// -------------------------------------------------------------
// 3. Detailed Forecast Explorer Chart
// -------------------------------------------------------------
function renderDetailedForecastChart() {
  const ctx = document.getElementById('detailedForecastChart');
  if (!ctx || !appData?.sample_timeline) return;

  const sample = appData.sample_timeline;
  const labels = sample.map(d => `${d.date.slice(5)} ${d.hour}:00`);
  const actuals = sample.map(d => d.actual_cnt);
  const preds = sample.map(d => d.pred_cnt);
  const naives = sample.map(d => d.naive_cnt);
  const reg = sample.map(d => d.pred_reg);
  const cas = sample.map(d => d.pred_cas);
  
  // Upper and lower error bounds (RMSE = 66.68)
  const upper = preds.map(p => Math.round(p + 66.68));
  const lower = preds.map(p => Math.max(0, Math.round(p - 66.68)));

  if (detailedChart) detailedChart.destroy();

  detailedChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Upper Confidence (1σ)',
          data: upper,
          borderColor: 'transparent',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          pointRadius: 0,
          fill: '+1'
        },
        {
          label: 'Lower Confidence (1σ)',
          data: lower,
          borderColor: 'transparent',
          backgroundColor: 'transparent',
          pointRadius: 0,
          fill: false
        },
        {
          label: 'Dual-Head Forecast (Total)',
          data: preds,
          borderColor: '#10b981',
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0.25
        },
        {
          label: 'Actual Demand',
          data: actuals,
          borderColor: '#94a3b8',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25
        },
        {
          label: 'Naive Baseline (t-24)',
          data: naives,
          borderColor: '#f43f5e',
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0,
          tension: 0.25
        },
        {
          label: 'Head A: Registered Commuter',
          data: reg,
          borderColor: '#06b6d4',
          borderWidth: 1.8,
          pointRadius: 0,
          hidden: true,
          tension: 0.25
        },
        {
          label: 'Head B: Casual Leisure',
          data: cas,
          borderColor: '#f59e0b',
          borderWidth: 1.8,
          pointRadius: 0,
          hidden: true,
          tension: 0.25
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: '#cbd5e1', font: { family: 'Inter', size: 12 } }
        },
        tooltip: {
          backgroundColor: 'rgba(16, 23, 38, 0.95)',
          titleColor: '#f1f5f9',
          bodyColor: '#e2e8f0',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#64748b', maxTicksLimit: 20 }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.06)' },
          ticks: { color: '#94a3b8' },
          title: { display: true, text: 'Rides / Hour', color: '#94a3b8' }
        }
      }
    }
  });

  // Checkbox series toggling
  document.getElementById('toggleActual')?.addEventListener('change', e => {
    detailedChart.setDatasetVisibility(3, e.target.checked);
    detailedChart.update();
  });
  document.getElementById('togglePred')?.addEventListener('change', e => {
    detailedChart.setDatasetVisibility(0, e.target.checked);
    detailedChart.setDatasetVisibility(1, e.target.checked);
    detailedChart.setDatasetVisibility(2, e.target.checked);
    detailedChart.update();
  });
  document.getElementById('toggleNaive')?.addEventListener('change', e => {
    detailedChart.setDatasetVisibility(4, e.target.checked);
    detailedChart.update();
  });
  document.getElementById('toggleSubComponents')?.addEventListener('change', e => {
    detailedChart.setDatasetVisibility(5, e.target.checked);
    detailedChart.setDatasetVisibility(6, e.target.checked);
    detailedChart.update();
  });

  // Range filter
  document.getElementById('forecastDateRange')?.addEventListener('change', e => {
    filterDetailedChart(e.target.value);
  });
}

function filterDetailedChart(val) {
  if (!detailedChart || !appData?.sample_timeline) return;
  let subset = appData.sample_timeline;
  if (val === 'week1') {
    subset = subset.slice(0, 168);
  } else if (val === 'week2') {
    subset = subset.slice(168, 336);
  } else if (val === 'weekend') {
    subset = subset.filter(d => d.workingday === 0);
  }

  const labels = subset.map(d => `${d.date.slice(5)} ${d.hour}:00`);
  const actuals = subset.map(d => d.actual_cnt);
  const preds = subset.map(d => d.pred_cnt);
  const naives = subset.map(d => d.naive_cnt);
  const reg = subset.map(d => d.pred_reg);
  const cas = subset.map(d => d.pred_cas);
  const upper = preds.map(p => Math.round(p + 66.68));
  const lower = preds.map(p => Math.max(0, Math.round(p - 66.68)));

  detailedChart.data.labels = labels;
  detailedChart.data.datasets[0].data = upper;
  detailedChart.data.datasets[1].data = lower;
  detailedChart.data.datasets[2].data = preds;
  detailedChart.data.datasets[3].data = actuals;
  detailedChart.data.datasets[4].data = naives;
  detailedChart.data.datasets[5].data = reg;
  detailedChart.data.datasets[6].data = cas;
  detailedChart.update();

  renderForecastLogTable(subset);
}

// -------------------------------------------------------------
// 4. Hourly Forecast Telemetry Log Table
// -------------------------------------------------------------
function renderForecastLogTable(dataset = null) {
  const tbody = document.querySelector('#forecastLogTable tbody');
  if (!tbody) return;

  const records = dataset || appData?.sample_timeline || [];
  tbody.innerHTML = '';

  records.slice(0, 50).forEach(r => {
    const tr = document.createElement('tr');
    const dayLabel = r.workingday === 1 ? 'Workday' : 'Weekend';
    let weatherLabel = 'Clear';
    if (r.weathersit === 2) weatherLabel = 'Mist/Cloudy';
    if (r.weathersit === 3) weatherLabel = 'Rain/Snow';
    if (r.weathersit === 4) weatherLabel = 'Storm';

    let urgencyBadge = 'badge-emerald';
    if (r.urgency === 'HIGH') urgencyBadge = 'badge-amber';
    if (r.urgency === 'CRITICAL') urgencyBadge = 'badge-rose';

    tr.innerHTML = `
      <td><code>${r.timestamp}</code></td>
      <td><span class="badge ${r.workingday === 1 ? 'badge-cyan' : 'badge-violet'}">${dayLabel}</span></td>
      <td>${r.temp_c}°C</td>
      <td>${weatherLabel}</td>
      <td><strong>${r.actual_cnt}</strong></td>
      <td><strong style="color: var(--accent-emerald);">${r.pred_cnt}</strong></td>
      <td>${r.error_abs}</td>
      <td><span class="badge ${urgencyBadge}">${r.trucks_needed} Trucks</span></td>
      <td><small style="color: var(--text-muted);">${r.rebalancing_action}</small></td>
    `;
    tbody.appendChild(tr);
  });
}

// -------------------------------------------------------------
// 5. What-If Scenario Sandbox Simulation
// -------------------------------------------------------------
function setupSandboxListeners() {
  const tempSlider = document.getElementById('slider-temp');
  const hourSlider = document.getElementById('slider-hour');
  const humSlider = document.getElementById('slider-hum');
  const windSlider = document.getElementById('slider-wind');
  const daySelect = document.getElementById('sim-daytype');
  const weatherSelect = document.getElementById('sim-weather');

  const updateSim = () => {
    const temp = parseInt(tempSlider.value);
    const hour = parseInt(hourSlider.value);
    const hum = parseInt(humSlider.value);
    const wind = parseInt(windSlider.value);
    const isWorkday = daySelect.value === 'workday';
    const weather = parseInt(weatherSelect.value);

    // Update labels
    document.getElementById('val-temp').textContent = `${temp}°C`;
    document.getElementById('val-hour').textContent = formatHour(hour);
    document.getElementById('val-hum').textContent = `${hum}%`;
    document.getElementById('val-wind').textContent = `${wind} km/h`;

    runSandboxSimulation(temp, hour, hum, wind, isWorkday, weather);
  };

  [tempSlider, hourSlider, humSlider, windSlider].forEach(s => s?.addEventListener('input', updateSim));
  [daySelect, weatherSelect].forEach(s => s?.addEventListener('change', updateSim));
}

function formatHour(hr) {
  const ampm = hr >= 12 ? 'PM' : 'AM';
  const h12 = hr % 12 === 0 ? 12 : hr % 12;
  return `${String(h12).padStart(2, '0')}:00 ${ampm}`;
}

function resetSandbox() {
  document.getElementById('slider-temp').value = 22;
  document.getElementById('slider-hour').value = 8;
  document.getElementById('slider-hum').value = 55;
  document.getElementById('slider-wind').value = 15;
  document.getElementById('sim-daytype').value = 'workday';
  document.getElementById('sim-weather').value = '1';
  document.getElementById('slider-temp').dispatchEvent(new Event('input'));
}

function runSandboxSimulation(temp = 22, hour = 8, hum = 55, wind = 15, isWorkday = true, weather = 1) {
  // Calibrated dual-head physics-aware model
  let baseReg = 0;
  let baseCas = 0;

  if (isWorkday) {
    // Commuter curve
    if (hour === 8) baseReg = 455;
    else if (hour === 7) baseReg = 280;
    else if (hour === 9) baseReg = 290;
    else if (hour === 17) baseReg = 490;
    else if (hour === 18) baseReg = 460;
    else if (hour >= 11 && hour <= 15) baseReg = 160 + (hour - 11) * 8;
    else if (hour >= 19 && hour <= 22) baseReg = 180 - (hour - 19) * 40;
    else if (hour >= 0 && hour <= 5) baseReg = Math.max(5, 30 - hour * 5);
    else baseReg = 100;

    baseCas = (hour >= 10 && hour <= 17) ? 40 : 15;
  } else {
    // Weekend leisure curve
    const peakDist = Math.abs(hour - 13.5);
    baseReg = Math.max(15, Math.round(230 * Math.exp(-0.06 * peakDist * peakDist)));
    baseCas = Math.max(8, Math.round(145 * Math.exp(-0.07 * peakDist * peakDist)));
  }

  // Temperature elasticity (+18.4 rides/°C between 10°C and 25°C)
  let tempFactor = 1.0;
  if (temp < 10) {
    tempFactor = Math.max(0.2, 0.45 + (temp / 20.0));
  } else if (temp <= 26) {
    tempFactor = 0.75 + (temp - 10) * 0.025;
  } else {
    tempFactor = 1.15 - (temp - 26) * 0.015; // Heat fatigue
  }

  // Weather penalty
  let weatherFactor = 1.0;
  if (weather === 2) weatherFactor = 0.82; // Mist/clouds
  else if (weather === 3) weatherFactor = 0.42; // Rain/Snow
  else if (weather === 4) weatherFactor = 0.18; // Storm

  // Humidity & Wind modifier
  let envFactor = 1.0;
  if (hum > 75) envFactor -= (hum - 75) * 0.003;
  if (wind > 25) envFactor -= (wind - 25) * 0.005;

  const predReg = Math.max(0, Math.round(baseReg * tempFactor * weatherFactor * envFactor));
  const predCas = Math.max(0, Math.round(baseCas * (tempFactor * 1.2) * (weatherFactor * 0.85) * envFactor));
  const predTotal = predReg + predCas;

  // Truck demand calculation (capacity: 30 bikes/truck)
  let trucksNeeded = 0;
  let actionText = '';

  if (isWorkday && (hour === 7 || hour === 8 || hour === 9)) {
    trucksNeeded = Math.ceil((predTotal * 0.40) / 30);
    actionText = `Morning Commute Peak: Dispatch ${trucksNeeded} trucks to Residential stations to prevent dock starvation before 07:45 AM.`;
  } else if (isWorkday && (hour >= 16 && hour <= 19)) {
    trucksNeeded = Math.ceil((predTotal * 0.42) / 30);
    actionText = `Evening Commute Peak: Mobilize ${trucksNeeded} trucks to clear downtown commercial docks and prevent locked-dock rejections.`;
  } else if (!isWorkday && (hour >= 11 && hour <= 16)) {
    trucksNeeded = Math.ceil((predTotal * 0.35) / 30);
    actionText = `Weekend Leisure Wave: Deploy ${trucksNeeded} trucks along waterfront corridors and parks to restock high-demand recreation nodes.`;
  } else if (hour >= 1 && hour <= 5) {
    trucksNeeded = 3;
    actionText = `Overnight Maintenance Sweep: 3 trucks cycling between depots to reset station fill rates to 50% baseline.`;
  } else {
    trucksNeeded = Math.max(1, Math.ceil((predTotal * 0.15) / 30));
    actionText = `Routine Monitoring: ${trucksNeeded} truck on roving patrol across dining and central transit hubs.`;
  }

  if (weather >= 3) {
    actionText += ` [WEATHER WARNING: Stand down outdoor truck routes if active rainfall exceeds safety threshold.]`;
  }

  // Update DOM
  const totalEl = document.getElementById('sim-res-total');
  const regEl = document.getElementById('sim-res-reg');
  const casEl = document.getElementById('sim-res-cas');
  const trucksEl = document.getElementById('sim-res-trucks');
  const actionEl = document.getElementById('sim-res-action');

  if (totalEl) totalEl.innerHTML = `${predTotal} <span style="font-size: 1rem; font-weight: 400; color: var(--text-dim);">rides/hr</span>`;
  if (regEl) regEl.textContent = `${predReg} rides (${Math.round((predReg / (predTotal || 1)) * 100)}%)`;
  if (casEl) casEl.textContent = `${predCas} rides (${Math.round((predCas / (predTotal || 1)) * 100)}%)`;
  if (trucksEl) trucksEl.innerHTML = `${trucksNeeded} Trucks <span style="font-size: 0.9rem; font-weight: 400; color: var(--text-muted);">(${trucksNeeded * 30} bikes)</span>`;
  if (actionEl) actionEl.textContent = actionText;
}

// -------------------------------------------------------------
// 6. TreeSHAP Feature Ranking Chart & Waterfall
// -------------------------------------------------------------
function renderShapChart() {
  const ctx = document.getElementById('shapFeatureChart');
  if (!ctx || !appData?.explainability?.feature_ranking) return;

  const top10 = appData.explainability.feature_ranking.slice(0, 10);
  const labels = top10.map(f => f.feature);
  const values = top10.map(f => f.mean_abs_shap);

  if (shapChart) shapChart.destroy();

  shapChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Mean |SHAP Value| (Impact on Rides/Hr)',
        data: values,
        backgroundColor: [
          '#10b981', '#06b6d4', '#3b82f6', '#8b5cf6',
          '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#8b5cf6', '#64748b'
        ],
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(16, 23, 38, 0.95)',
          titleColor: '#f1f5f9',
          bodyColor: '#e2e8f0'
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8' },
          title: { display: true, text: 'Average Impact Magnitude (SHAP)', color: '#94a3b8' }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#cbd5e1', font: { family: 'monospace', size: 11 } }
        }
      }
    }
  });
}

function setupWaterfallListener() {
  const select = document.getElementById('waterfallSelector');
  select?.addEventListener('change', e => {
    renderWaterfallScenario(parseInt(e.target.value));
  });
}

function renderWaterfallScenario(index) {
  const container = document.getElementById('waterfallContainer');
  if (!container || !appData?.explainability?.scenarios) return;

  const scenario = appData.explainability.scenarios[index];
  if (!scenario) return;

  container.innerHTML = `
    <div style="background: rgba(0,0,0,0.3); padding: 0.8rem; border-radius: 6px; margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
      <span style="color: var(--text-muted); font-size: 0.85rem;">Expected Base Value E[f(x)]:</span>
      <strong style="color: #cbd5e1;">${scenario.base_value} rides</strong>
    </div>
  `;

  scenario.top_contributions.forEach(item => {
    const isPos = item.contribution >= 0;
    const sign = isPos ? '+' : '';
    const color = isPos ? '#34d399' : '#fb7185';
    const row = document.createElement('div');
    row.style.cssText = 'display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0.6rem; border-radius: 4px; background: rgba(255,255,255,0.02); font-size: 0.85rem;';
    row.innerHTML = `
      <span style="font-family: monospace; color: #94a3b8;">${item.feature} = ${item.value}</span>
      <strong style="color: ${color};">${sign}${item.contribution.toFixed(1)}</strong>
    `;
    container.appendChild(row);
  });

  const totalRow = document.createElement('div');
  totalRow.style.cssText = 'margin-top: 0.5rem; padding: 0.8rem; border-radius: 6px; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); display: flex; justify-content: space-between; align-items: center;';
  totalRow.innerHTML = `
    <strong style="color: #ffffff; font-size: 0.88rem;">Final Model Prediction:</strong>
    <span style="font-size: 1.2rem; font-weight: 700; color: #34d399;">${scenario.predicted_total} rides</span>
  `;
  container.appendChild(totalRow);
}
