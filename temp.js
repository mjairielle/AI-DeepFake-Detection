
        const API_BASE = 'http://127.0.0.1:5000';
        let lastResult = null;
        let selectedVerdict = null;

        function navigateTo(page) {
            document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
            const target = document.getElementById('page-' + page);
            if (target) target.classList.add('active');
            
            document.querySelectorAll('.sidebar-nav a').forEach(a => a.classList.remove('active'));
            const navLink = document.querySelector('.sidebar-nav a[data-page="' + page + '"]');
            if (navLink) navLink.classList.add('active');
            
            document.querySelectorAll('.mobile-nav button').forEach(b => b.classList.remove('active'));
            const mobileBtn = document.querySelector('.mobile-nav button[data-page="' + page + '"]');
            if (mobileBtn) mobileBtn.classList.add('active');
            
            if (page === 'learning') fetchLearningStatus();
            if (page === 'health') fetchHealth();
            closeSidebar();
        }

        document.querySelectorAll('.sidebar-nav a, .mobile-nav button, [data-page]').forEach(el => {
            el.addEventListener('click', (e) => {
                e.preventDefault();
                const page = el.getAttribute('data-page');
                if (page) navigateTo(page);
            });
        });

        function toggleSidebar() {
            document.getElementById('sidebar').classList.toggle('open');
        }
        
        function closeSidebar() {
            document.getElementById('sidebar').classList.remove('open');
        }

        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const analyzeBtn = document.getElementById('analyze-btn');

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        dropZone.addEventListener('click', (e) => {
            if (e.target !== fileInput) {
                fileInput.click();
            }
        });
        dropZone.addEventListener('dragover', () => { dropZone.style.borderColor = 'var(--primary)'; });
        dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = ''; });
        dropZone.addEventListener('drop', (e) => {
            dropZone.style.borderColor = '';
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                updateDropZoneLabel(e.dataTransfer.files[0].name);
            }
        });
        
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) updateDropZoneLabel(fileInput.files[0].name);
        });
        
        function updateDropZoneLabel(name) {
            dropZone.querySelector('h3').textContent = name;
            dropZone.querySelector('p').textContent = 'File selected. Click Analyze to proceed.';
        }

        analyzeBtn.addEventListener('click', async () => {
            if (!fileInput.files || fileInput.files.length === 0) {
                alert('Please select a file first.');
                return;
            }
            showLoading();
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            try {
                const res = await fetch(API_BASE + '/api/analyze', { method: 'POST', body: formData });
                const data = await res.json();
                if (res.ok) {
                    lastResult = data;
                    renderResults(data);
                    navigateTo('results');
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            } catch (err) {
                alert('Connection failed. Is the backend running at ' + API_BASE + '?');
            } finally {
                hideLoading();
            }
        });

        function renderResults(data) {
            document.getElementById('result-filename').textContent = data.original_filename || 'Unknown File';
            const vb = document.getElementById('result-verdict');
            vb.textContent = (data.verdict || '—').toUpperCase();
            vb.className = 'verdict-badge ' + (data.verdict || '');
            
            const prob = Math.round((data.deepfake_probability || 0) * 100);
            document.getElementById('gauge-value').textContent = prob + '%';
            const gaugeEl = document.getElementById('gauge');
            let gaugeColor = '#22c55e';
            if (prob > 70) gaugeColor = 'var(--error)';
            else if (prob > 30) gaugeColor = '#f59e0b';
            gaugeEl.style.background = 'radial-gradient(closest-side, #0e141a 80%, transparent 0 100%), conic-gradient(' + gaugeColor + ' ' + prob + '%, #2f353c 0)';
            document.getElementById('gauge-value').style.color = gaugeColor;
            
            const sev = document.getElementById('gauge-severity');
            if (prob > 70) { sev.textContent = 'CRITICAL'; sev.style.color = 'var(--error)'; }
            else if (prob > 30) { sev.textContent = 'MODERATE'; sev.style.color = '#f59e0b'; }
            else { sev.textContent = 'MINIMAL'; sev.style.color = '#22c55e'; }
            
            const conf = Math.round((data.confidence || 0) * 100);
            document.getElementById('confidence-value').textContent = (data.confidence || 0).toFixed(2);
            document.getElementById('confidence-bar').style.width = conf + '%';
            
            document.getElementById('manipulation-type').textContent = data.manipulation_type || 'unknown';
            document.getElementById('file-hash').textContent = (data.file_hash || '—').substring(0, 12) + '...';
            document.getElementById('processing-time').textContent = (data.processing_time_ms || 0).toLocaleString() + ' ms';
            
            const fc = document.getElementById('flags-container');
            fc.innerHTML = '';
            (data.flags || []).forEach(f => {
                const s = document.createElement('span');
                s.className = 'flag-tag';
                s.textContent = f;
                fc.appendChild(s);
            });
            
            const mc = document.getElementById('module-scores-container');
            mc.innerHTML = '';
            const scoreColors = { facial_analysis: 'var(--error)', temporal_analysis: '#f97316', gan_artifacts: '#eab308', audio_analysis: 'var(--secondary)', metadata_forensics: 'var(--tertiary-container)', noise_analysis: 'var(--primary)', spectral_analysis: 'var(--secondary)', prosody_analysis: 'var(--tertiary)', voice_clone_detect: 'var(--error)' };
            
            Object.entries(data.module_scores || {}).forEach(([key, val]) => {
                const pct = Math.round(val * 100);
                const color = scoreColors[key] || 'var(--primary)';
                let severity = 'LOW';
                if (pct > 80) severity = 'CRITICAL';
                else if (pct > 60) severity = 'HIGH';
                else if (pct > 40) severity = 'MODERATE';
                const div = document.createElement('div');
                div.className = 'module-score';
                div.innerHTML = '<div class="module-score-header"><span>' + key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) + '</span><span style="color:' + color + ';font-weight:700">' + val.toFixed(2) + ' ' + severity + '</span></div><div class="progress-bar-track"><div class="progress-bar-fill" style="width:' + pct + '%;background:' + color + '"></div></div>';
                mc.appendChild(div);
            });
            
            const fdc = document.getElementById('findings-container');
            fdc.innerHTML = '';
            (data.flags || []).slice(0, 4).forEach(f => {
                const isCritical = f.includes('deepfake') || f.includes('mismatch') || f.includes('blending');
                const isSafe = f.includes('valid') || f.includes('authentic');
                const cls = isCritical ? 'critical' : (isSafe ? 'safe' : 'info');
                const icon = isCritical ? 'warning' : (isSafe ? 'check_circle' : 'info');
                const iconColor = isCritical ? 'var(--error)' : (isSafe ? 'var(--primary)' : 'var(--secondary)');
                const div = document.createElement('div');
                div.className = 'finding-card ' + cls;
                div.innerHTML = '<span class="material-symbols-outlined" style="color:' + iconColor + '">' + icon + '</span><div><span style="display:block;font-size:12px;letter-spacing:0.05em;font-weight:700;color:' + iconColor + ';margin-bottom:4px">' + f.toUpperCase().replace(/_/g, ' ') + '</span><p style="font-size:14px;color:var(--on-surface-variant)">Forensic flag detected during analysis.</p></div>';
                fdc.appendChild(div);
            });
            
            document.getElementById('fb-hash').value = data.file_hash || '';
            document.getElementById('fb-original-verdict').value = data.verdict || '';
            document.getElementById('fb-media-type').value = data.media_type || 'image';
        }

        const batchDropZone = document.getElementById('batch-drop-zone');
        const batchFileInput = document.getElementById('batch-file-input');
        const batchAnalyzeBtn = document.getElementById('batch-analyze-btn');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            batchDropZone.addEventListener(eventName, preventDefaults, false);
        });

        batchDropZone.addEventListener('click', (e) => {
            if (e.target !== batchFileInput) {
                batchFileInput.click();
            }
        });
        batchDropZone.addEventListener('dragover', () => { batchDropZone.style.borderColor = 'var(--primary)'; });
        batchDropZone.addEventListener('dragleave', () => { batchDropZone.style.borderColor = ''; });
        batchDropZone.addEventListener('drop', (e) => {
            batchDropZone.style.borderColor = '';
            if (e.dataTransfer.files.length > 0) {
                batchFileInput.files = e.dataTransfer.files;
                batchDropZone.querySelector('h3').textContent = e.dataTransfer.files.length + ' files selected';
            }
        });
        
        batchFileInput.addEventListener('change', () => {
            if (batchFileInput.files.length > 0)
                batchDropZone.querySelector('h3').textContent = batchFileInput.files.length + ' files selected';
        });

        batchAnalyzeBtn.addEventListener('click', async () => {
            if (!batchFileInput.files || batchFileInput.files.length === 0) { alert('Please select files first.'); return; }
            showLoading();
            const formData = new FormData();
            for (const f of batchFileInput.files) formData.append('files', f);
            try {
                const res = await fetch(API_BASE + '/api/analyze/batch', { method: 'POST', body: formData });
                const data = await res.json();
                if (res.ok) renderBatchResults(data);
                else alert('Error: ' + (data.error || 'Unknown error'));
            } catch (err) {
                alert('Connection failed.');
            } finally { hideLoading(); }
        });

        function renderBatchResults(data) {
            const panel = document.getElementById('batch-results-panel');
            panel.style.display = 'block';
            const tbody = document.getElementById('batch-results-body');
            tbody.innerHTML = '';
            let counts = { authentic: 0, suspicious: 0, deepfake: 0, error: 0 };
            
            (data.results || []).forEach((r, i) => {
                const v = r.verdict || r.error || 'error';
                if (counts[v] !== undefined) counts[v]++;
                else counts.error++;
                const tr = document.createElement('tr');
                const verdictClass = v === 'authentic' ? 'authentic' : (v === 'deepfake' ? 'deepfake' : (v === 'suspicious' ? 'suspicious' : ''));
                tr.innerHTML = '<td>' + (i + 1) + '</td><td>' + (r.original_filename || '—') + '</td><td>' + (r.media_type || '—') + '</td><td><span class="verdict-badge ' + verdictClass + '">' + v.toUpperCase() + '</span></td><td>' + (r.deepfake_probability !== undefined ? r.deepfake_probability.toFixed(2) : '—') + '</td><td>' + (r.confidence !== undefined ? r.confidence.toFixed(2) : '—') + '</td><td>' + (r.processing_time_ms !== undefined ? Math.round(r.processing_time_ms) : '—') + '</td>';
                tbody.appendChild(tr);
            });
            
            const summary = document.getElementById('batch-summary');
            summary.style.display = 'block';
            summary.textContent = data.total + ' files analyzed — ' + counts.deepfake + ' Deepfake, ' + counts.suspicious + ' Suspicious, ' + counts.authentic + ' Authentic';
        }

        function selectVerdict(v) {
            selectedVerdict = v;
            document.getElementById('toggle-authentic').classList.toggle('selected', v === 'authentic');
            document.getElementById('toggle-deepfake').classList.toggle('selected', v === 'deepfake');
        }

        document.getElementById('feedback-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!selectedVerdict) { alert('Please select a corrected verdict.'); return; }
            if (!lastResult) { alert('Analyze a file first.'); return; }
            
            const payload = {
                file_hash: document.getElementById('fb-hash').value,
                media_type: document.getElementById('fb-media-type').value,
                original_verdict: document.getElementById('fb-original-verdict').value,
                corrected_verdict: selectedVerdict,
                module_scores: lastResult.module_scores || {},
                flags: lastResult.flags || [],
                manipulation_type: document.getElementById('fb-manipulation-type').value
            };
            
            try {
                const res = await fetch(API_BASE + '/api/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                const data = await res.json();
                if (res.ok) { alert('Feedback submitted successfully! Total records: ' + data.total_feedback); }
                else { alert('Error: ' + (data.error || 'Unknown error')); }
            } catch (err) { alert('Connection failed.'); }
        });

        async function fetchLearningStatus() {
            try {
                const res = await fetch(API_BASE + '/api/learning/status');
                const data = await res.json();
                document.getElementById('lr-feedback').textContent = data.feedback?.total_records ?? '—';
                document.getElementById('lr-patterns').textContent = data.pattern_memory?.stored_patterns ?? '—';
                const ready = data.classifier?.is_ready;
                document.getElementById('lr-ready').textContent = ready ? 'Yes' : 'No';
                document.getElementById('lr-ready-dot').className = 'status-dot ' + (ready ? 'online' : '');
                document.getElementById('lr-accuracy').textContent = ready ? ' (' + (data.classifier.accuracy * 100).toFixed(1) + '%)' : '';
                document.getElementById('lr-adjusted').textContent = data.adaptive_thresholds?.modules_adjusted ?? '—';
                
                const tbody = document.getElementById('threshold-table-body');
                tbody.innerHTML = '';
                const details = data.adaptive_thresholds?.details || {};
                Object.entries(details).forEach(([mod, info]) => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = '<td>' + mod + '</td><td>' + (info.authentic_samples || 0) + '</td><td>' + (info.deepfake_samples || 0) + '</td><td>' + (info.adjustment !== undefined ? (info.adjustment >= 0 ? '+' : '') + info.adjustment.toFixed(4) : '—') + '</td>';
                    tbody.appendChild(tr);
                });
            } catch (err) { console.warn('Could not fetch learning status:', err); }
        }

        document.getElementById('retrain-btn').addEventListener('click', async () => {
            if (!confirm('Force retrain the classifier?')) return;
            try {
                const res = await fetch(API_BASE + '/api/learning/retrain', { method: 'POST' });
                const data = await res.json();
                alert('Retrain complete. Accuracy: ' + ((data.accuracy || 0) * 100).toFixed(1) + '%');
                fetchLearningStatus();
            } catch (err) { alert('Connection failed.'); }
        });

        async function fetchHealth() {
            try {
                const res = await fetch(API_BASE + '/api/health');
                const data = await res.json();
                document.getElementById('engine-version').textContent = data.engine || 'unknown';
            } catch (err) { console.warn('Could not fetch health:', err); }
        }

        function showLoading() { document.getElementById('loading-overlay').classList.add('active'); }
        function hideLoading() { document.getElementById('loading-overlay').classList.remove('active'); }

        fetchHealth();
    
