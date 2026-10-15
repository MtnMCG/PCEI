(function() {
    'use strict';

    // ── State ──────────────────────────────────────────────────────────────────
    const state = {
        apiStatus: 'disconnected',
        wsStatus: 'disconnected',
        isTraining: false,
        activePid: null,
        activeScript: null,
        trainHistory: [],   // {step, loss}
        evalHistory:  [],   // {step, loss}
        MAX_HISTORY: 1000,
        divergence: 0,
    };

    const API_BASE = '';

    // ── Formatters ─────────────────────────────────────────────────────────────
    const format = {
        number: (n) => new Intl.NumberFormat('en-US').format(n),
        bytes: (b) => {
            if (!b || b === 0) return '0 B';
            const k = 1024, sizes = ['B','KB','MB','GB'];
            const i = Math.floor(Math.log(b) / Math.log(k));
            return parseFloat((b / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        },
        time: () => new Date().toLocaleTimeString('en-US', { hour12: false }),
        relativeTime: (ts) => {
            if (!ts) return 'Unknown';
            const rtf = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });
            const diff = ts - Date.now() / 1000;
            if (Math.abs(diff) > 86400) return rtf.format(Math.round(diff/86400), 'day');
            if (Math.abs(diff) > 3600)  return rtf.format(Math.round(diff/3600), 'hour');
            return rtf.format(Math.round(diff/60), 'minute');
        }
    };

    // ── DOM Elements ───────────────────────────────────────────────────────────
    const els = {
        timeDisplay: document.getElementById('system-time'),
        apiStatus:   document.getElementById('api-status'),
        wsStatus:    document.getElementById('ws-status'),

        // Model
        totalParams:      document.getElementById('total-params'),
        trainableParams:  document.getElementById('trainable-params'),
        ternaryBar:       document.getElementById('ternary-dist-bar'),
        ternaryLblNeg:    document.getElementById('ternary-lbl-neg'),
        ternaryLblZero:   document.getElementById('ternary-lbl-zero'),
        ternaryLblPos:    document.getElementById('ternary-lbl-pos'),
        stackTableBody:   document.querySelector('#stack-table tbody'),
        modelDtype:       document.getElementById('model-dtype'),
        modelVocab:       document.getElementById('model-vocab'),
        modelDevice:      document.getElementById('model-device'),
        currentCkptName:  document.getElementById('current-ckpt-name'),
        currentCkptStep:  document.getElementById('current-ckpt-step'),
        currentCkptSize:  document.getElementById('current-ckpt-size'),

        // Ambient
        confHeatmap:  document.getElementById('confidence-heatmap'),
        fastNorm:     document.getElementById('fast-norm'),
        slowNorm:     document.getElementById('slow-norm'),
        fsDivergence: document.getElementById('fs-divergence'),
        lrBars:       document.getElementById('lr-bars'),

        // Entanglement
        eActive:    document.getElementById('e-active'),
        eFormed:    document.getElementById('e-formed'),
        eDissolved: document.getElementById('e-dissolved'),
        topo3d:     document.getElementById('topology-3d'),

        // Metrics Bar
        liveStep:       document.getElementById('live-step'),
        liveLoss:       document.getElementById('live-loss'),
        liveEvalLoss:   document.getElementById('live-eval-loss'),
        livePerplexity: document.getElementById('live-perplexity'),
        liveGradNorm:   document.getElementById('live-grad-norm'),
        liveThroughput: document.getElementById('live-throughput'),
        liveLr:         document.getElementById('live-lr'),
        liveBonds:      document.getElementById('live-bonds'),
        liveDivergence: document.getElementById('live-divergence'),

        // Chart
        lossCanvas: document.getElementById('loss-chart'),

        // Launcher
        launcherTabs:      document.getElementById('launcher-tabs'),
        tabPanes:          document.querySelectorAll('.tab-pane'),
        forms:             document.querySelectorAll('.launcher-form'),
        processView:       document.getElementById('active-process-view'),
        runningScriptName: document.getElementById('running-script-name'),
        processRuntime:    document.getElementById('process-runtime'),
        btnKillProcess:    document.getElementById('btn-kill-process'),
        processLogs:       document.getElementById('process-logs'),
        
        // Director Console
        directorConsole:   document.getElementById('director-console'),
        directorInput:     document.getElementById('director-input'),
        btnDirectorSend:   document.getElementById('btn-director-send'),

        // Checkpoints
        checkpointsTableBody: document.querySelector('#checkpoints-table tbody'),
        btnRefreshCkpts:      document.getElementById('refresh-checkpoints'),
        btnRefreshModel:      document.getElementById('refresh-model'),
    };

    // ── API Wrapper ────────────────────────────────────────────────────────────
    async function apiFetch(endpoint, options = {}, retries = 3) {
        for (let i = 0; i < retries; i++) {
            try {
                const res = await fetch(`${API_BASE}${endpoint}`, {
                    ...options,
                    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) }
                });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                setApiStatus('connected');
                return await res.json();
            } catch (err) {
                if (i === retries - 1) {
                    setApiStatus('disconnected');
                    console.error(`API fetch failed: ${endpoint}`, err);
                    return null;
                }
                await new Promise(r => setTimeout(r, 1000 * (i + 1)));
            }
        }
    }

    function setApiStatus(s) {
        if (state.apiStatus === s) return;
        state.apiStatus = s;
        els.apiStatus.className = `status-indicator ${s === 'connected' ? 'connected' : ''}`;
        els.apiStatus.querySelector('.status-text').textContent = s === 'connected' ? 'API Online' : 'API Offline';
    }

    function setWsStatus(s) {
        if (state.wsStatus === s) return;
        state.wsStatus = s;
        els.wsStatus.className = `status-indicator ${s === 'connected' ? 'connected pulsing' : ''}`;
        els.wsStatus.querySelector('.status-text').textContent = s === 'connected' ? 'Live Stream' : 'WS Offline';
    }

    // ── Clock ──────────────────────────────────────────────────────────────────
    setInterval(() => { els.timeDisplay.textContent = format.time(); }, 1000);

    // ── Chart.js — Dual Loss Chart ─────────────────────────────────────────────
    let lossChart = null;

    function initLossChart() {
        if (!els.lossCanvas || !window.Chart) return;
        const ctx = els.lossCanvas.getContext('2d');
        lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Train Loss',
                        data: [],
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0,212,255,0.08)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.3,
                        fill: true,
                    },
                    {
                        label: 'Eval Loss',
                        data: [],
                        borderColor: '#ff8c00',
                        backgroundColor: 'rgba(255,140,0,0.08)',
                        borderWidth: 2,
                        pointRadius: 3,
                        pointBackgroundColor: '#ff8c00',
                        tension: 0.3,
                        fill: false,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 0 },
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(0,0,0,0.8)',
                        titleColor: '#aaa',
                        bodyColor: '#fff',
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Step', color: '#666', font: { size: 11 } },
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: { color: '#555', maxTicksLimit: 8 }
                    },
                    y: {
                        title: { display: true, text: 'Loss', color: '#666', font: { size: 11 } },
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: { color: '#555' },
                        min: 0
                    }
                }
            }
        });
    }

    function updateLossChart() {
        if (!lossChart) return;
        lossChart.data.datasets[0].data = state.trainHistory.map(p => ({ x: p.step, y: p.loss }));
        lossChart.data.datasets[1].data = state.evalHistory.map(p => ({ x: p.step, y: p.loss }));
        lossChart.update('none');
    }

    // ── Model Stats ────────────────────────────────────────────────────────────
    async function loadModelStats() {
        const data = await apiFetch('/api/model/stats');
        if (!data) return;

        els.totalParams.textContent     = format.number(data.total_params || 0);
        els.trainableParams.textContent = format.number(data.trainable_params || 0);
        els.modelDtype.textContent  = data.dtype || 'N/A';
        els.modelVocab.textContent  = format.number(data.vocab_size || 0);
        els.modelDevice.textContent = data.device || 'N/A';

        if (data.checkpoint) {
            els.currentCkptName.textContent = data.checkpoint.filename;
            els.currentCkptStep.textContent = format.number(data.checkpoint.step);
            els.currentCkptSize.textContent = format.bytes(data.checkpoint.size);
        }

        if (data.weight_distribution) {
            const { neg, zero, pos } = data.weight_distribution;
            els.ternaryBar.querySelector('.neg').style.width  = `${neg}%`;
            els.ternaryBar.querySelector('.zero').style.width = `${zero}%`;
            els.ternaryBar.querySelector('.pos').style.width  = `${pos}%`;
            els.ternaryLblNeg.textContent  = `-1: ${neg.toFixed(1)}%`;
            els.ternaryLblZero.textContent = `0: ${zero.toFixed(1)}%`;
            els.ternaryLblPos.textContent  = `+1: ${pos.toFixed(1)}%`;
        }

        if (data.stacks) {
            els.stackTableBody.innerHTML = data.stacks.map(s => `
                <tr>
                    <td>${s.name}</td>
                    <td>${s.layers}</td>
                    <td>${s.d_model}</td>
                    <td>${format.number(s.params)}</td>
                </tr>
            `).join('');
        }
    }

    // ── Ambient Field ──────────────────────────────────────────────────────────
    async function loadAmbientField() {
        const data = await apiFetch('/api/model/ambient');
        if (!data) return;

        const fsDiv = data.fast_slow_divergence || data.divergence || 0;
        state.divergence = fsDiv;
        els.fastNorm.textContent     = (data.fast_state_norm || data.fast_norm || 0).toFixed(4);
        els.slowNorm.textContent     = (data.slow_state_norm || data.slow_norm || 0).toFixed(4);
        els.fsDivergence.textContent = fsDiv.toFixed(4);
        els.liveDivergence.textContent = fsDiv.toFixed(4);

        if (data.layer_confidence) {
            const confs = data.layer_confidence;
            els.confHeatmap.innerHTML = confs.map((c, i) => {
                let r, g, b;
                if (c < 0.5) {
                    const t = c * 2;
                    r = 0; g = Math.floor(17*(1-t)+212*t); b = Math.floor(51*(1-t)+255*t);
                } else {
                    const t = (c - 0.5) * 2;
                    r = Math.floor(255*t); g = Math.floor(212*(1-t)+255*t); b = 255;
                }
                return `<div class="heatmap-row">
                    <span class="heatmap-label">L${i}</span>
                    <div class="heatmap-cell" style="background:rgb(${r},${g},${b})" title="Conf: ${c.toFixed(3)}"></div>
                </div>`;
            }).join('');
        }

        if (data.effective_lr) {
            const maxLr = Math.max(...data.effective_lr, 1e-4);
            els.lrBars.innerHTML = data.effective_lr.map((lr, i) => {
                const pct = Math.max(2, (lr / maxLr) * 100);
                return `<div class="lr-bar-row">
                    <span class="lr-bar-lbl">L${i}</span>
                    <div class="lr-bar-track"><div class="lr-bar-fill" style="width:${pct}%"></div></div>
                </div>`;
            }).join('');
        }
    }

    // ── 3D Entanglement Topology (Plotly) ──────────────────────────────────────
    let topoInitialized = false;

    async function loadEntanglement() {
        const data = await apiFetch('/api/model/entanglement');
        if (!data) return;

        const activeCount = typeof data.active === 'number'
            ? data.active
            : (Array.isArray(data.active_bonds) ? data.active_bonds.length : 0);
        els.eActive.textContent    = format.number(activeCount);
        els.eFormed.textContent    = format.number(data.total_formed || data.bonds_formed_total || 0);
        els.eDissolved.textContent = format.number(data.total_dissolved || data.bonds_dissolved_total || 0);

        if (data.topology && window.Plotly) {
            draw3DTopology(data.topology);
        }
    }

    function draw3DTopology(topology) {
        const nodes = Array.isArray(topology.nodes) ? topology.nodes : [];
        const bonds = Array.isArray(topology.bonds) ? topology.bonds : [];

        if (nodes.length === 0) {
            // Generate demo nodes from confidence if no checkpoint bonds yet
            const fakeNodes = [];
            for (let i = 0; i < 24; i++) {
                const stack = i < 8 ? 'TEXT' : i < 16 ? 'IMAGE' : 'AUDIO';
                fakeNodes.push({
                    id: `layer_${i}`, stack,
                    x: 0.1 + (i * 0.13) % 0.8,
                    y: 0.15 + (i * 0.09) % 0.7,
                    z: Math.random(),
                    confidence: 0.4 + Math.random() * 0.5
                });
            }
            return draw3DTopology({ nodes: fakeNodes, bonds: [] });
        }

        // Colours per stack
        const colorMap = { TEXT: '#00d4ff', IMAGE: '#00ff88', AUDIO: '#ffb800' };

        // Nodes scatter
        const nodeTrace = {
            type: 'scatter3d',
            mode: 'markers',
            x: nodes.map(n => n.x || 0),
            y: nodes.map(n => n.y || 0),
            z: nodes.map(n => n.z || 0),
            marker: {
                size: nodes.map(n => 4 + (n.confidence || 0.5) * 8),
                color: nodes.map(n => n.confidence || 0.5),
                colorscale: [
                    [0,   'rgba(0,17,51,1)'],
                    [0.5, 'rgba(0,212,255,1)'],
                    [1,   'rgba(255,255,255,1)']
                ],
                opacity: 0.9,
                line: { width: 0 }
            },
            text: nodes.map(n => `${n.id || n.layer_id || ''}<br>Stack: ${n.stack || '?'}<br>Conf: ${(n.confidence||0).toFixed(3)}`),
            hovertemplate: '%{text}<extra></extra>',
            name: 'Layers'
        };

        // Bond lines (each bond = 3 points: n1, n2, null to break)
        const bx = [], by = [], bz = [], bOpacity = [];
        bonds.forEach(bond => {
            const n1 = nodes.find(n => (n.id||n.layer_id) === (bond.source||bond.layer_a));
            const n2 = nodes.find(n => (n.id||n.layer_id) === (bond.target||bond.layer_b));
            if (n1 && n2) {
                bx.push(n1.x||0, n2.x||0, null);
                by.push(n1.y||0, n2.y||0, null);
                bz.push(n1.z||0, n2.z||0, null);
            }
        });

        const bondTrace = {
            type: 'scatter3d',
            mode: 'lines',
            x: bx, y: by, z: bz,
            line: { color: 'rgba(0,212,255,0.35)', width: 2 },
            hoverinfo: 'none',
            name: 'Bonds'
        };

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor:  'rgba(0,0,0,0)',
            scene: {
                bgcolor: 'rgba(0,0,0,0)',
                xaxis: { showgrid: false, zeroline: false, showticklabels: false, title: '' },
                yaxis: { showgrid: false, zeroline: false, showticklabels: false, title: '' },
                zaxis: { showgrid: false, zeroline: false, showticklabels: false, title: '' },
                camera: { eye: { x: 1.5, y: 1.5, z: 0.8 } }
            },
            margin: { l: 0, r: 0, t: 0, b: 0 },
            showlegend: false,
        };

        const config = {
            responsive: true,
            displayModeBar: false,
        };

        if (!topoInitialized) {
            Plotly.newPlot(els.topo3d, [bondTrace, nodeTrace], layout, config);
            topoInitialized = true;
        } else {
            Plotly.react(els.topo3d, [bondTrace, nodeTrace], layout, config);
        }
    }

    // ── Checkpoints ────────────────────────────────────────────────────────────
    async function loadCheckpoints() {
        const data = await apiFetch('/api/checkpoints');
        if (!data || !Array.isArray(data)) return;
        els.checkpointsTableBody.innerHTML = data.map(c => `
            <tr>
                <td>${c.filename}</td>
                <td>${format.bytes(c.size || (c.size_mb ? c.size_mb*1024*1024 : 0))}</td>
                <td>${format.number(c.step || c.global_step || 0)}</td>
                <td>${format.relativeTime(c.modified || c.last_modified)}</td>
                <td><button class="btn btn-sm btn-primary" onclick="alert('Checkpoint noted.')">Load</button></td>
            </tr>
        `).join('');

        // Populate the datalist so text inputs have autocomplete
        const datalist = document.getElementById('checkpoint-list');
        if (datalist) {
            let options = data.map(c => `<option value="${c.filename}">`).join('');
            // Ensure pcei_100m.pt is always an available option to select
            if (!data.find(c => c.filename === 'pcei_100m.pt')) {
                options = `<option value="pcei_100m.pt">` + options;
            }
            datalist.innerHTML = options;
        }
    }

    els.btnRefreshCkpts.addEventListener('click', loadCheckpoints);
    els.btnRefreshModel.addEventListener('click', () => { loadModelStats(); loadAmbientField(); loadEntanglement(); });

    // ── WebSocket (Metrics) ────────────────────────────────────────────────────
    let ws;
    function setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host || 'localhost:8000';

        function connect() {
            ws = new WebSocket(`${protocol}//${host}/ws/metrics`);
            ws.onopen  = () => setWsStatus('connected');
            ws.onclose = () => { setWsStatus('disconnected'); setTimeout(connect, 3000); };
            ws.onmessage = (e) => {
                try {
                    const d = JSON.parse(e.data);
                    const step       = d.step || 0;
                    if (step > 0) {
                        els.liveStep.textContent = format.number(step);
                        state.lastStep = step;
                    }
                    const currentStep = step > 0 ? step : (state.lastStep || 0);
                    const loss       = (typeof d.loss === 'number' && !isNaN(d.loss)) ? d.loss : null;
                    const evalLoss   = (typeof d.eval_loss === 'number') ? d.eval_loss : null;
                    const perplexity = (typeof d.perplexity === 'number') ? d.perplexity : null;
                    const gradNorm   = (typeof d.grad_norm === 'number') ? d.grad_norm : null;
                    const throughput = typeof d.throughput === 'number' ? d.throughput : (d.tokens_per_sec || 0);
                    const lr         = typeof d.lr === 'number' ? d.lr : (d.learning_rate || 0.0004);
                    const bonds      = typeof d.active_bonds === 'number' ? d.active_bonds : 0;

                    els.liveStep.textContent       = format.number(currentStep);
                    els.liveLoss.textContent       = loss !== null ? loss.toFixed(4) : '--';
                    els.liveEvalLoss.textContent   = evalLoss !== null ? evalLoss.toFixed(4) : '--';
                    els.livePerplexity.textContent = perplexity !== null ? perplexity.toFixed(2) : '--';
                    els.liveGradNorm.textContent   = gradNorm !== null ? gradNorm.toFixed(3) : '--';
                    els.liveThroughput.textContent = format.number(Math.round(throughput));
                    els.liveLr.textContent         = lr.toExponential(2);
                    els.liveBonds.textContent      = format.number(bonds);

                    // Update divergence from structured telemetry or ambient poll
                    const div = typeof d.fast_slow_divergence === 'number' ? d.fast_slow_divergence : state.divergence;
                    state.divergence = div;
                    els.liveDivergence.textContent = div.toFixed(4);

                    // Mean confidence from telemetry
                    if (typeof d.mean_confidence === 'number') {
                        state.meanConfidence = d.mean_confidence;
                    }

                    if (loss !== null && loss > 0) {
                        state.trainHistory.push({ step: currentStep, loss });
                        if (state.trainHistory.length > state.MAX_HISTORY) state.trainHistory.shift();
                    }
                    if (evalLoss !== null) {
                        state.evalHistory.push({ step: currentStep, loss: evalLoss });
                        if (state.evalHistory.length > state.MAX_HISTORY) state.evalHistory.shift();
                    }

                    requestAnimationFrame(updateLossChart);

                } catch(err) { console.error('WS parse err', err); }
            };
        }
        connect();
    }

    // ── Launcher ───────────────────────────────────────────────────────────────
    els.launcherTabs.addEventListener('click', e => {
        if (e.target.tagName !== 'BUTTON') return;
        els.launcherTabs.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        const targetId = e.target.dataset.target;
        els.tabPanes.forEach(p => {
            p.classList.remove('active');
            if (p.id === targetId) p.classList.add('active');
        });
    });

    els.forms.forEach(form => {
        form.addEventListener('submit', async e => {
            e.preventDefault();
            const script = form.dataset.script;
            const formData = new FormData(form);
            const args = {};

            for (let [k, v] of formData.entries()) {
                if (k.startsWith('_')) continue;
                args[k] = v;
            }

            const modalityEl = form.querySelector('[name="_modality"]');
            if (modalityEl) {
                if (modalityEl.value === 'audio') args['audio-only'] = 'true';
                else if (modalityEl.value === 'image') args['image-only'] = 'true';
            }

            const btn = form.querySelector('.launch-btn');
            btn.disabled = true;

            // Single-slot: if something is running, show killing state
            if (state.isTraining) {
                btn.textContent = 'Killing previous job...';
            } else {
                btn.textContent = 'Launching...';
            }

            // Backend handles killing the old job automatically
            const res = await apiFetch('/api/training/launch', {
                method: 'POST',
                body: JSON.stringify({ script, args })
            });

            if (res && res.pid) {
                state.isTraining = true;
                state.activePid = res.pid;
                state.activeScript = script;
                state.launchTime = Date.now();
                updateLauncherUI();
            } else if (res && res.error) {
                alert(`Launch error: ${res.error}`);
            }

            // Re-enable all launch buttons
            els.forms.forEach(f => {
                const b = f.querySelector('.launch-btn');
                b.disabled = false;
                b.textContent = b.dataset.label || 'Launch Script';
            });
        });
    });

    // Store original button labels
    els.forms.forEach(f => {
        const b = f.querySelector('.launch-btn');
        b.dataset.label = b.textContent;
    });

    els.btnKillProcess.addEventListener('click', async () => {
        const killBtn = els.btnKillProcess;
        killBtn.textContent = 'Saving checkpoint...';
        killBtn.disabled = true;

        await apiFetch('/api/training/kill', {
            method: 'POST',
            body: JSON.stringify({})
        });

        state.isTraining = false;
        state.activePid  = null;
        state.activeScript = null;
        killBtn.textContent = 'Kill';
        killBtn.disabled = false;
        updateLauncherUI();
    });

    function updateLauncherUI() {
        if (state.isTraining) {
            els.processView.classList.remove('hidden');
            els.runningScriptName.textContent = state.activeScript || '--';
            
            if (state.activeScript === 'ollama_trainer') {
                els.directorConsole.classList.remove('hidden');
            } else {
                els.directorConsole.classList.add('hidden');
            }
            
            startLogPolling();
        } else {
            els.processView.classList.add('hidden');
            els.directorConsole.classList.add('hidden');
            stopLogPolling();
        }
    }

    // ── Director Console API ───────────────────────────────────────────────────
    async function sendDirectorPrompt() {
        if (!els.directorInput.value.trim()) return;
        const prompt = els.directorInput.value.trim();
        els.directorInput.value = '';
        els.btnDirectorSend.disabled = true;
        
        await apiFetch('/api/training/director', {
            method: 'POST',
            body: JSON.stringify({ prompt })
        });
        
        els.btnDirectorSend.disabled = false;
        els.directorInput.focus();
    }

    if (els.btnDirectorSend) els.btnDirectorSend.addEventListener('click', sendDirectorPrompt);
    if (els.directorInput) {
        els.directorInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendDirectorPrompt();
        });
    }

    // Runtime counter
    setInterval(() => {
        if (state.isTraining && state.launchTime) {
            const elapsed = Math.floor((Date.now() - state.launchTime) / 1000);
            const m = Math.floor(elapsed / 60);
            const s = elapsed % 60;
            els.processRuntime.textContent = `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
        }
    }, 1000);

    async function checkTrainingStatus() {
        const data = await apiFetch('/api/training/status');
        if (data && data.is_running) {
            state.isTraining  = true;
            state.activePid   = data.pid;
            state.activeScript = data.script;
            if (data.start_time && !state.launchTime) {
                state.launchTime = data.start_time * 1000;
            }
        } else {
            if (state.isTraining) {
                // Process just finished
                state.isTraining  = false;
                state.activePid   = null;
                state.activeScript = null;
            }
        }
        updateLauncherUI();
    }

    let logInterval;
    function startLogPolling() {
        if (logInterval) clearInterval(logInterval);
        
        const fetchLogs = async () => {
            if (!state.activePid) return;
            const data = await apiFetch(`/api/training/logs/${state.activePid}`);
            if (data && data.logs) {
                els.processLogs.textContent = data.logs.join('\n');
                els.processLogs.scrollTop = els.processLogs.scrollHeight;
            }
        };
        
        fetchLogs(); // Fetch immediately on load/refresh
        logInterval = setInterval(fetchLogs, 3000);
    }
    function stopLogPolling() {
        if (logInterval) clearInterval(logInterval);
    }

    // ── Background Pollers ────────────────────────────────────────────────────
    setInterval(loadModelStats,      30000);   // checkpoint parse is expensive — every 30s
    setInterval(checkTrainingStatus, 10000);
    setInterval(loadAmbientField,    30000);
    setInterval(loadEntanglement,    30000);

    // ── Chat Logic ─────────────────────────────────────────────────────────────
    let chatWs;
    let currentMessageEl = null;

    const chatEls = {
        config:     document.getElementById('chat-config'),
        ckpt:       document.getElementById('chat-ckpt'),
        mode:       document.getElementById('chat-mode'),
        connectBtn: document.getElementById('btn-chat-connect'),
        killBtn:    document.getElementById('btn-chat-kill'),
        history:    document.getElementById('chat-history'),
        input:      document.getElementById('chat-input'),
        sendBtn:    document.getElementById('btn-chat-send'),
    };

    function appendMessage(role, text) {
        const msg = document.createElement('div');
        msg.className = `chat-message ${role}`;
        msg.textContent = text;
        chatEls.history.appendChild(msg);
        chatEls.history.scrollTop = chatEls.history.scrollHeight;
        return msg;
    }

    if (chatEls.connectBtn) {
        chatEls.connectBtn.addEventListener('click', () => {
            const ckpt = chatEls.ckpt.value;
            const mode = chatEls.mode ? chatEls.mode.value : 'inference';
            chatEls.history.innerHTML = '';
            appendMessage('system', 'Initializing Resonance Chat...');

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host || 'localhost:8000';

            chatWs = new WebSocket(`${protocol}//${host}/ws/chat?checkpoint=${encodeURIComponent(ckpt)}&mode=${mode}`);

            chatWs.onopen = () => {
                chatEls.input.disabled = false;
                chatEls.sendBtn.disabled = false;
                if (chatEls.killBtn) chatEls.killBtn.style.display = 'inline-block';
                appendMessage('system', 'Connected to PCEI Active Learning stream.');
            };

            chatWs.onmessage = (e) => {
                if (e.data === '__START_MODEL__') {
                    currentMessageEl = appendMessage('model generating', '');
                } else if (e.data === '__END_MODEL__') {
                    if (currentMessageEl) currentMessageEl.classList.remove('generating');
                    currentMessageEl = null;
                } else {
                    if (currentMessageEl) {
                        currentMessageEl.textContent += e.data;
                        chatEls.history.scrollTop = chatEls.history.scrollHeight;
                    } else {
                        const trimmed = e.data.trim();
                        if (trimmed) appendMessage('system', trimmed);
                    }
                }
            };

            chatWs.onclose = () => {
                chatEls.input.disabled = true;
                chatEls.sendBtn.disabled = true;
                if (chatEls.killBtn) chatEls.killBtn.style.display = 'none';
                appendMessage('system', 'Connection closed.');
            };
        });

        if (chatEls.killBtn) {
            chatEls.killBtn.addEventListener('click', async () => {
                if (chatWs) chatWs.close();
                await apiFetch('/api/chat/kill', { method: 'POST' });
                chatEls.killBtn.style.display = 'none';
                appendMessage('system', 'Session killed.');
            });
        }

        const sendMessage = () => {
            const text = chatEls.input.value.trim();
            if (text && chatWs && chatWs.readyState === WebSocket.OPEN) {
                appendMessage('user', text);
                chatWs.send(text);
                chatEls.input.value = '';
            }
        };
        chatEls.sendBtn.addEventListener('click', sendMessage);
        chatEls.input.addEventListener('keypress', e => { if (e.key === 'Enter') sendMessage(); });
    }

    // ── Init ───────────────────────────────────────────────────────────────────
    async function loadDashboard() {
        initLossChart();
        await Promise.all([
            loadModelStats(),
            loadAmbientField(),
            loadEntanglement(),
            loadCheckpoints(),
            checkTrainingStatus(),
        ]);
        setupWebSocket();
    }

    loadDashboard();

})();
