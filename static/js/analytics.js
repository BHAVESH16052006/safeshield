document.addEventListener('DOMContentLoaded', function() {
    // Initialize charts
    initializeCharts();
    
    // Load prediction history
    loadPredictionHistory();
});

function initializeCharts() {
    // Spam Distribution Chart
    const spamCtx = document.getElementById('spamDistributionChart').getContext('2d');
    new Chart(spamCtx, {
        type: 'doughnut',
        data: {
            labels: ['Spam', 'Ham'],
            datasets: [{
                data: [30, 70],
                backgroundColor: ['#ff6384', '#36a2eb']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                title: {
                    display: true,
                    text: 'Spam Distribution'
                }
            }
        }
    });

    // Accuracy Trend Chart
    const accuracyCtx = document.getElementById('accuracyTrendChart').getContext('2d');
    new Chart(accuracyCtx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [{
                label: 'Accuracy',
                data: [85, 88, 90, 92, 95, 97],
                borderColor: '#4CAF50',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                title: {
                    display: true,
                    text: 'Accuracy Trend'
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 80,
                    max: 100
                }
            }
        }
    });
}

function loadPredictionHistory() {
    fetch('/analytics')
        .then(response => response.json())
        .then(data => {
            const history = data.prediction_history || [];
            const tbody = document.getElementById('predictionHistory');
            tbody.innerHTML = history.map(item => `
                <tr>
                    <td>${item.date}</td>
                    <td>${item.message}</td>
                    <td>${item.result}</td>
                    <td>${item.confidence}</td>
                </tr>
            `).join('');
        });
} 