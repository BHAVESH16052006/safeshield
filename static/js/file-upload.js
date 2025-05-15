document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const textArea = document.getElementById('comment');
    const resultsContainer = document.getElementById('results');
    const confidenceBar = document.querySelector('.confidence-bar');
    const confidenceScore = document.querySelector('.confidence-score');
    const resultText = document.querySelector('.result-text');
    const spinner = document.getElementById('loadingSpinner');

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const text = textArea.value.trim();
        if (text.length > 0) {
            if (spinner) spinner.style.display = 'flex';
            resultsContainer.style.display = 'none';
            confidenceBar.style.width = '0%';
            confidenceScore.textContent = '0%';
            resultText.textContent = '';
            fetchPrediction(text);
        }
    });

    function fetchPrediction(text) {
        const formData = new FormData();
        formData.append('comment', text);
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            resultsContainer.style.display = 'block';
            const confidence = parseFloat(data.confidence);
            confidenceBar.style.width = `${confidence}%`;
            confidenceScore.textContent = data.confidence;
            resultText.textContent = `This message is ${data.prediction}`;
            resultText.className = `result-text ${data.prediction.toLowerCase()}`;
            if (spinner) spinner.style.display = 'none';
        })
        .catch(error => {
            console.error('Error:', error);
            resultsContainer.style.display = 'none';
            if (spinner) spinner.style.display = 'none';
        });
    }
});

function updateAnalytics() {
    fetch('/analytics')
        .then(response => response.json())
        .then(data => {
            // Update charts with new data
            updateCharts(data);
            // Update prediction history
            updatePredictionHistory(data.prediction_history);
        });
} 