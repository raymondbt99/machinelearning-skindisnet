document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const btnBrowse = document.getElementById('btn-browse');
    const btnClassify = document.getElementById('btn-classify');
    const dropzonePrompt = document.getElementById('dropzone-prompt');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const btnRemoveImage = document.getElementById('btn-remove-image');
    const previewMeta = document.getElementById('preview-meta');
    const samplesContainer = document.getElementById('samples-container');
    
    // Results DOM
    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const resultsContent = document.getElementById('results-content');
    const predClassName = document.getElementById('pred-class-name');
    const predCategory = document.getElementById('pred-category');
    const gaugeFillCircle = document.getElementById('gauge-fill-circle');
    const gaugeVal = document.getElementById('gauge-val');
    const probabilitiesList = document.getElementById('probabilities-list');
    
    // Clinical Profile DOM
    const descText = document.getElementById('desc-text');
    const sitesText = document.getElementById('sites-text');
    const hallmarksText = document.getElementById('hallmarks-text');
    const differentialText = document.getElementById('differential-text');
    
    // Modal DOM
    const btnShowMetrics = document.getElementById('btn-show-metrics');
    const btnCloseModal = document.getElementById('btn-close-modal');
    const metricsModal = document.getElementById('metrics-modal');
    const valAccuracy = document.getElementById('val-accuracy');
    const valMacroF1 = document.getElementById('val-macro-f1');
    const valBaselineAcc = document.getElementById('val-baseline-acc');
    const systemStatus = document.getElementById('system-status');

    let currentFile = null;
    let currentSamplePath = null;

    // Load System Info & Samples on startup
    fetchSystemInfo();
    fetchSamples();

    // Event: Browse Button
    btnBrowse.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener('click', () => {
        if (!currentFile && !currentSamplePath) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag & Drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files[0]) {
            handleFileSelect(dt.files[0]);
        }
    });

    // Remove Image
    btnRemoveImage.addEventListener('click', (e) => {
        e.stopPropagation();
        resetImageInput();
    });

    function handleFileSelect(file) {
        if (!file.type.match('image.*')) {
            alert('Please select an image file (JPEG, PNG, WEBP).');
            return;
        }
        currentFile = file;
        currentSamplePath = null;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            displayPreview(e.target.result, `${file.name} (${Math.round(file.size / 1024)} KB)`);
        };
        reader.readAsDataURL(file);
    }

    function displayPreview(src, metaText) {
        imagePreview.src = src;
        previewMeta.textContent = metaText || '224 × 224 RGB';
        dropzonePrompt.style.display = 'none';
        previewContainer.style.display = 'flex';
        btnClassify.disabled = false;
    }

    function resetImageInput() {
        currentFile = null;
        currentSamplePath = null;
        fileInput.value = '';
        imagePreview.src = '';
        dropzonePrompt.style.display = 'flex';
        previewContainer.style.display = 'none';
        btnClassify.disabled = true;
        showEmptyState();
    }

    // Load Quick Test Samples
    async function fetchSamples() {
        try {
            const res = await fetch('/api/samples');
            const data = await res.json();
            if (data.samples && data.samples.length > 0) {
                renderSamples(data.samples);
            }
        } catch (err) {
            console.error('Failed to load samples:', err);
        }
    }

    function renderSamples(samples) {
        samplesContainer.innerHTML = '';
        samples.forEach(sample => {
            const card = document.createElement('div');
            card.className = 'sample-card';
            card.innerHTML = `
                <img src="/api/image/${sample.rel_path}" alt="${sample.class_name}" class="sample-thumb" loading="lazy">
                <div class="sample-info">
                    <div class="sample-name" title="${sample.class_name}">${sample.class_name}</div>
                    <div class="sample-meta">${sample.patient_id} • ${sample.location}</div>
                </div>
            `;
            card.addEventListener('click', () => {
                selectSample(sample);
            });
            samplesContainer.appendChild(card);
        });
    }

    function selectSample(sample) {
        currentFile = null;
        currentSamplePath = sample.rel_path;
        displayPreview(`/api/image/${sample.rel_path}`, `Test Set Sample • ${sample.patient_id} (${sample.class_name})`);
        
        // Auto-fill clinical metadata
        if (sample.location) {
            const locEl = document.getElementById('meta-location');
            const cleanLoc = sample.location.trim().toLowerCase();
            for (let opt of locEl.options) {
                if (opt.value.toLowerCase() === cleanLoc || cleanLoc.includes(opt.value.toLowerCase())) {
                    locEl.value = opt.value;
                    break;
                }
            }
        }
        if (sample.age) {
            document.getElementById('meta-age').value = Math.round(parseFloat(sample.age));
        }
        if (sample.sex) {
            const sexEl = document.getElementById('meta-sex');
            const sexClean = sample.sex.trim().toLowerCase();
            for (let opt of sexEl.options) {
                if (opt.value.toLowerCase() === sexClean) {
                    sexEl.value = opt.value;
                    break;
                }
            }
        }
    }

    // Fetch System & Model Info
    async function fetchSystemInfo() {
        try {
            const res = await fetch('/api/info');
            const data = await res.json();
            
            if (data.status === 'ready') {
                systemStatus.className = 'status-indicator ready';
                systemStatus.querySelector('.status-label').textContent = data.model_architecture || 'EfficientNetB0 Active';
            } else {
                systemStatus.className = 'status-indicator training';
                systemStatus.querySelector('.status-label').textContent = 'Model Training in Progress...';
            }

            if (data.metrics) {
                if (data.metrics.test_accuracy) {
                    valAccuracy.textContent = (data.metrics.test_accuracy * 100).toFixed(1) + '%';
                }
                if (data.metrics.test_macro_f1) {
                    valMacroF1.textContent = (data.metrics.test_macro_f1 * 100).toFixed(1) + '%';
                }
                if (data.metrics.baseline_test_accuracy) {
                    valBaselineAcc.textContent = (data.metrics.baseline_test_accuracy * 100).toFixed(1) + '%';
                }
            }
        } catch (err) {
            console.error('Failed to load system info:', err);
        }
    }

    // Event: Classify Button
    btnClassify.addEventListener('click', async () => {
        if (!currentFile && !currentSamplePath) return;

        showLoading();

        const locVal = document.getElementById('meta-location').value;
        const ageVal = document.getElementById('meta-age').value || '30';
        const sexVal = document.getElementById('meta-sex').value;

        try {
            let res;
            if (currentFile) {
                const formData = new FormData();
                formData.append('image', currentFile);
                formData.append('location', locVal);
                formData.append('age', ageVal);
                formData.append('sex', sexVal);
                res = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
            } else if (currentSamplePath) {
                res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_path: currentSamplePath,
                        location: locVal,
                        age: ageVal,
                        sex: sexVal
                    })
                });
            }

            const data = await res.json();
            if (res.ok && data.success) {
                renderResults(data);
            } else {
                alert(data.error || 'Failed to complete classification.');
                showEmptyState();
            }
        } catch (err) {
            console.error('Classification error:', err);
            alert('Network error while analyzing image.');
            showEmptyState();
        }
    });

    function showLoading() {
        emptyState.style.display = 'none';
        resultsContent.style.display = 'none';
        loadingState.style.display = 'flex';
    }

    function showEmptyState() {
        loadingState.style.display = 'none';
        resultsContent.style.display = 'none';
        emptyState.style.display = 'flex';
    }

    function renderResults(data) {
        loadingState.style.display = 'none';
        emptyState.style.display = 'none';
        resultsContent.style.display = 'block';

        // Hero Details
        predClassName.textContent = data.predicted_class;
        predCategory.textContent = data.clinical_info.clinical_category || 'Dermatological Lesion';
        
        // Gauge
        const percent = data.confidence_percent;
        gaugeVal.textContent = `${percent}%`;
        
        // Calculate circumference: 2 * PI * 42 = 263.89
        const circumference = 2 * Math.PI * 42;
        const offset = circumference - (percent / 100) * circumference;
        gaugeFillCircle.style.strokeDashoffset = offset;

        // Probabilities Breakdown
        probabilitiesList.innerHTML = '';
        data.probabilities.forEach(item => {
            const row = document.createElement('div');
            row.className = `prob-row ${item.is_top ? 'top-class' : ''}`;
            row.innerHTML = `
                <div class="prob-header">
                    <span class="prob-class">${item.class_name}</span>
                    <span class="prob-percent">${item.percent}%</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width: 0%;"></div>
                </div>
            `;
            probabilitiesList.appendChild(row);

            // Animate progress bar fill
            setTimeout(() => {
                row.querySelector('.prob-bar-fill').style.width = `${item.percent}%`;
            }, 50);
        });

        // Clinical Info
        const info = data.clinical_info;
        descText.textContent = info.description || 'Clinical overview not available.';
        sitesText.textContent = info.common_sites || 'Various anatomical sites.';
        hallmarksText.textContent = info.visual_hallmarks || 'Erythema and epidermal changes.';
        differentialText.textContent = info.differential_diagnoses || 'Review clinical history.';
    }

    // Modal Events
    btnShowMetrics.addEventListener('click', () => {
        metricsModal.style.display = 'flex';
        fetchSystemInfo(); // Refresh metrics
    });

    btnCloseModal.addEventListener('click', () => {
        metricsModal.style.display = 'none';
    });

    metricsModal.addEventListener('click', (e) => {
        if (e.target === metricsModal) {
            metricsModal.style.display = 'none';
        }
    });
});
