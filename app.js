const BACKEND_URL = window.BACKEND_URL || "http://localhost:8000";

let occupationsData = [];
const tabLoaded = {};

async function switchTab(tabName, event) {
    // Update active nav link
    document.querySelectorAll('.nav-links li').forEach(li => li.classList.remove('active'));
    if (event && event.currentTarget) {
        event.currentTarget.classList.add('active');
    } else {
        const navIdx = tabName === 'tab1' ? 0 : 1;
        document.querySelectorAll('.nav-links li')[navIdx].classList.add('active');
    }

    // Hide all panes instantly
    document.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');

    let pane = document.getElementById(`pane-${tabName}`);

    if (!pane) {
        // First visit: fetch HTML once and create pane
        try {
            const response = await fetch(`${tabName}.html`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const html = await response.text();

            pane = document.createElement('div');
            pane.id = `pane-${tabName}`;
            pane.className = 'tab-pane';
            pane.innerHTML = html;
            document.getElementById('tab-content').appendChild(pane);

            if (tabName === 'tab1') initTab1();
        } catch (err) {
            console.error("Tab load failed:", err);
            const container = document.getElementById('tab-content');
            container.innerHTML = `<p style="color:red">Error loading ${tabName}: ${err.message}</p>`;
            return;
        }
    }

    pane.style.display = 'block';

    if (tabName === 'tab1') {
        requestAnimationFrame(() => requestAnimationFrame(scaleIframes));
    }
}

document.addEventListener("DOMContentLoaded", () => {
    switchTab('tab1');
});

function scaleIframes() {
    document.querySelectorAll('.comparison-pane .viz-container').forEach(container => {
        const iframe = container.querySelector('iframe');
        if (!iframe || !container.clientWidth) return;
        const scale = container.clientWidth / 1100;
        iframe.style.width = '1100px';
        iframe.style.height = Math.round(container.clientHeight / scale) + 'px';
        iframe.style.transform = `scale(${scale})`;
    });
}

window.addEventListener('resize', scaleIframes);

/* --- Tab 1 Logic --- */
async function initTab1() {
    if (occupationsData.length === 0) {
        try {
            const res = await fetch(`${BACKEND_URL}/api/occupations`);
            occupationsData = await res.json();
        } catch (e) {
            console.error("Failed to load occupations:", e);
            const listEl = document.getElementById('occList');
            if(listEl) listEl.innerHTML = '<p style="color:red">Failed to connect to backend.</p>';
            return;
        }
    }
    
    renderOccList(occupationsData);

    const searchInput = document.getElementById('occSearch');
    if(searchInput) {
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            const filtered = occupationsData.filter(o => o.name.toLowerCase().includes(query));
            renderOccList(filtered);
        });
    }
}

function renderOccList(data) {
    const listEl = document.getElementById('occList');
    if (!listEl) return;
    listEl.innerHTML = '';
    
    data.forEach(occ => {
        const div = document.createElement('div');
        div.className = 'topic-item';
        div.onclick = () => selectOccupation(occ.name, div);
        
        div.innerHTML = `
            <h4>${occ.name}</h4>
            <p><span class="topic-badge">Count: ${occ.count.toLocaleString()} jobs</span></p>
        `;
        listEl.appendChild(div);
    });
}
async function selectOccupation(name, element) {
    // UI Feedback: Highlight the selected sidebar item
    document.querySelectorAll('.topic-item').forEach(el => el.classList.remove('selected'));
    element.classList.add('selected');
    
    // Toggle visibility: Hide placeholder, show details panel
    const placeholder = document.getElementById('occPlaceholder');
    const details = document.getElementById('occDetails');
    if(placeholder) placeholder.style.display = 'none';
    if(details) details.style.display = 'block';
    
    // Update Header Text
    const nameEl = document.getElementById('detailOccName');
    const statsEl = document.getElementById('occStats');
    if(nameEl) nameEl.innerText = name;
    if(statsEl) statsEl.innerText = `Retrieving authentic job descriptions from dataset...`;
    
    // Show Loading Spinner
    const jobsEl = document.getElementById('occJobs');
    if(jobsEl) jobsEl.innerHTML = '<div class="loader"></div>';

    try {
        const res = await fetch(`${BACKEND_URL}/api/occupation_jobs/${encodeURIComponent(name)}`);
        const jobs = await res.json();
        
        if(jobsEl) jobsEl.innerHTML = '';
        if(statsEl) statsEl.innerText = `Showing 5 random samples from your collection for: ${name}`;

        // Render each job card
        jobs.forEach(job => {
            const div = document.createElement('div');
            div.className = 'doc-card';
            
            // Use job.title instead of Job ID for a better user experience
            div.innerHTML = `
                <div style="margin-bottom: 12px;">
                    <strong style="color: #6366f1; font-size: 1.1rem; display: block;">${job.title}</strong>
                    <div style="height: 1px; background: #e2e8f0; margin-top: 8px;"></div>
                </div>
                <div class="job-description-text" style="line-height: 1.5; color: var(--text-secondary);">
                    ${job.job_text.replace(/\n/g, '<br>')}
                </div>
            `;
            
            if(jobsEl) jobsEl.appendChild(div);
        });
    } catch (e) {
        console.error("Failed to load jobs:", e);
        if(jobsEl) jobsEl.innerHTML = '<p style="color:#ef4444; padding: 1rem;">Failed to load job descriptions. Please check your backend connection.</p>';
    }
}

function toggleNeighborJob(i) {
    const body    = document.getElementById(`neighbor-job-${i}`);
    const chevron = document.getElementById(`chevron-${i}`);
    const isOpen  = body.style.display === 'block';

    if (isOpen) {
        body.style.display = 'none';
        chevron.style.transform = '';
    } else {
        body.textContent = (window._neighborJobs[i] || {}).job_text || '';
        body.style.display = 'block';
        chevron.style.transform = 'rotate(180deg)';
    }
}

function showViz(vizType) {
    document.querySelectorAll('.viz-tab').forEach(b => b.classList.remove('active'));
    event.currentTarget.classList.add('active');

    const iframe = document.getElementById('viz-iframe');
    const img = document.getElementById('viz-img');

    if (vizType === 'scatter' || vizType === 'datamap') {
        if(iframe) iframe.style.display = 'none';
        if(img) img.style.display = 'block';
        const filename = vizType === 'scatter' ? 'cluster_scatter.png' : 'viz_datamap.png';
        if(img) img.src = `${BACKEND_URL}/outputs/${filename}`;
    } else {
        if(img) img.style.display = 'none';
        if(iframe) iframe.style.display = 'block';
        if(iframe) iframe.src = `${BACKEND_URL}/outputs/viz_${vizType}.html`;
    }
}

/* --- Tab 2 Logic --- */

// NEW: Feature 1 - Compare 2 Jobs
async function compareTwoJobs() {
    const job1 = document.getElementById('compareJob1').value;
    const job2 = document.getElementById('compareJob2').value;
    const resultDiv = document.getElementById('compareResult');
    const scoreSpan = document.getElementById('matchPercent');
    const circlePath = document.getElementById('scoreCirclePath');

    if (!job1.trim() || !job2.trim()) return;

    try {
        const response = await fetch(`${BACKEND_URL}/api/compare_jobs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job1, job2 })
        });
        const data = await response.json();
        
        const score = data.skill_fit_score;

        resultDiv.style.display = 'block';
        scoreSpan.innerText = `${score}%`;
        circlePath.setAttribute('stroke-dasharray', `${score}, 100`);
        circlePath.style.stroke = `hsl(${(score / 100) * 120}, 80%, 40%)`;

        const tiers = [
            { max: 20,  label: 'Very Low Compatibility',  desc: 'These roles operate in largely different skill domains. A transition would require substantial retraining.' },
            { max: 40,  label: 'Low Compatibility',       desc: 'Some surface-level overlap exists, but the core competencies differ significantly between these roles.' },
            { max: 60,  label: 'Moderate Compatibility',  desc: 'A meaningful share of skills transfers. With targeted upskilling, this transition is feasible.' },
            { max: 75,  label: 'Good Compatibility',      desc: 'Solid skill overlap — the core competencies of Job A align well with a good portion of Job B\'s requirements.' },
            { max: 88,  label: 'High Compatibility',      desc: 'Strong skill alignment. The two roles share most of their key competencies in the 128D skill space.' },
            { max: 101, label: 'Excellent Compatibility', desc: 'Near-identical skill profiles. These roles are highly interchangeable in terms of required competencies.' },
        ];
        const tier = tiers.find(t => score < t.max);
        document.getElementById('compatLabel').innerText = tier.label;
        document.getElementById('compatDesc').innerText  = tier.desc;
        
    } catch (e) {
        console.error("Comparison Error:", e);
    }
}

// Feature 2 - Single Job Prediction
async function runInference() {
    const text = document.getElementById('jobQuery').value;
    if (!text.trim()) return;

    // Show result container and loader
    const resultsContainer = document.getElementById('predictionResults');
    const loader = document.getElementById('inferenceLoader');
    const output = document.getElementById('inferenceOutput');

    if(resultsContainer) resultsContainer.style.display = 'block';
    if(loader) loader.style.display = 'block';
    if(output) output.style.display = 'none';

    try {
        const response = await fetch(`${BACKEND_URL}/api/predict_occupation`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ description: text })
        });

        if (!response.ok) throw new Error(`API Error: ${response.statusText}`);

        const data = await response.json();

        // Format Similar Jobs list
        // Store job texts for safe access on click (avoids embedding raw text in innerHTML)
        window._neighborJobs = data.similar_jobs || [];

        let similarJobsHtml = '';
        if (window._neighborJobs.length > 0) {
            window._neighborJobs.forEach((job, i) => {
                similarJobsHtml += `
                    <div class="neighbor-job-item" onclick="toggleNeighborJob(${i})">
                        <div class="neighbor-job-header">
                            <strong>${job.title}</strong>
                            <span class="neighbor-chevron" id="chevron-${i}">▼</span>
                        </div>
                        <div class="neighbor-job-body" id="neighbor-job-${i}"></div>
                    </div>`;
            });
        } else {
            similarJobsHtml = '<p style="color:#94a3b8; font-size:0.9rem;">No transferable roles found in training set.</p>';
        }

        if(output) {
            output.innerHTML = `
                <div class="result-data">
                    <div class="result-item">
                        <div class="label">Predicted Occupation Group</div>
                        <div class="value" style="color: #6366f1;">${data.predicted_group}</div>
                        <div style="font-size:0.85rem; color:var(--text-secondary); margin-top:0.3rem">
                            Voting Confidence: ${(data.confidence * 100).toFixed(1)}%
                        </div>
                    </div>
                    <div class="result-item">
                        <div class="label">Transferable Roles — click to view description</div>
                        <div style="margin-top:0.5rem">${similarJobsHtml}</div>
                    </div>
                </div>
            `;
        }

    } catch (e) {
        console.error("Inference Error:", e);
        if(output) output.innerHTML = `<p style="color:#ef4444; padding:1rem;">Failed to run prediction: ${e.message}</p>`;
    } finally {
        if(loader) loader.style.display = 'none';
        if(output) output.style.display = 'block';
    }
}


