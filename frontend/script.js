document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadSection = document.querySelector('.upload-section');
    const resultsSection = document.getElementById('results-section');
    const previewImg = document.getElementById('preview-img');
    const loadingOverlay = document.getElementById('loading-overlay');
    const predictionsList = document.getElementById('predictions-list');

    // API Endpoint (Make sure your FastAPI server is running on this port)
    const API_URL = 'http://localhost:8000/predict/';

    // Click to upload
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('drop-zone--over');
    }

    function unhighlight(e) {
        dropZone.classList.remove('drop-zone--over');
    }

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        let dt = e.dataTransfer;
        let files = dt.files;
        handleFiles(files);
    }

    // Handle file selection via click
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        
        // Only process image files.
        if (!file.type.match('image.*')) {
            alert('Please select an image file.');
            return;
        }

        previewFile(file);
        uploadFile(file);
    }

    function previewFile(file) {
        let reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            previewImg.src = reader.result;
            // Hide upload, show results
            uploadSection.style.display = 'none';
            resultsSection.style.display = 'grid';
            predictionsList.innerHTML = ''; // clear previous
        }
    }

    async function uploadFile(file) {
        loadingOverlay.classList.add('active');
        
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            displayResults(data.predictions);

        } catch (error) {
            console.error('Error uploading file:', error);
            predictionsList.innerHTML = `<p class="danger">Error connecting to server. Is the API running?</p>`;
        } finally {
            loadingOverlay.classList.remove('active');
        }
    }

    function displayResults(predictions) {
        predictionsList.innerHTML = '';

        if (!predictions || predictions.length === 0) {
            predictionsList.innerHTML = '<div class="no-results">No objects detected.</div>';
            return;
        }

        predictions.forEach(pred => {
            const item = document.createElement('div');
            item.className = 'prediction-item';
            
            // Format confidence as percentage
            const confPercent = (pred.confidence * 100).toFixed(1) + '%';
            
            item.innerHTML = `
                <span class="pred-class">${pred.class}</span>
                <span class="pred-conf">${confPercent}</span>
            `;
            predictionsList.appendChild(item);
        });
        
        // Add a button to try another image
        const resetButton = document.createElement('button');
        resetButton.textContent = 'Upload Another Image';
        resetButton.style.marginTop = '1.5rem';
        resetButton.style.padding = '0.8rem 1.5rem';
        resetButton.style.background = 'var(--accent-primary)';
        resetButton.style.color = 'white';
        resetButton.style.border = 'none';
        resetButton.style.borderRadius = '8px';
        resetButton.style.cursor = 'pointer';
        resetButton.style.fontWeight = '600';
        resetButton.style.fontFamily = 'inherit';
        resetButton.style.width = '100%';
        resetButton.style.transition = 'background 0.3s ease';
        
        resetButton.addEventListener('mouseover', () => {
            resetButton.style.background = '#7c3aed';
        });
        resetButton.addEventListener('mouseout', () => {
            resetButton.style.background = 'var(--accent-primary)';
        });
        
        resetButton.addEventListener('click', () => {
            resultsSection.style.display = 'none';
            uploadSection.style.display = 'flex';
            fileInput.value = '';
        });
        
        predictionsList.appendChild(resetButton);
    }
});
