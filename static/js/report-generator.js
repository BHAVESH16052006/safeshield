document.addEventListener('DOMContentLoaded', function() {
    const generateReportBtn = document.getElementById('generateReport');
    
    generateReportBtn.addEventListener('click', function() {
        const message = document.getElementById('comment').value;
        const prediction = document.querySelector('.result-text').textContent.split(' ').pop();
        const confidence = document.querySelector('.confidence-score').textContent;
        
        const reportData = {
            message: message,
            prediction: prediction,
            confidence: confidence
        };
        
        fetch('/generate_report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(reportData)
        })
        .then(response => response.json())
        .then(data => {
            displayReport(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while generating the report.');
        });
    });
});

function displayReport(report) {
    // Create report modal
    const modal = document.createElement('div');
    modal.className = 'report-modal';
    modal.innerHTML = `
        <div class="report-content">
            <h2>Spam Analysis Report</h2>
            <div class="report-section">
                <h3>Basic Information</h3>
                <p><strong>Timestamp:</strong> ${report.timestamp}</p>
                <p><strong>Prediction:</strong> ${report.prediction}</p>
                <p><strong>Confidence:</strong> ${report.confidence}</p>
            </div>
            <div class="report-section">
                <h3>Message Preview</h3>
                <p>${report.message_preview}</p>
            </div>
            <div class="report-section">
                <h3>Key Indicators</h3>
                <ul>
                    ${report.analysis.key_indicators.map(indicator => `<li>${indicator}</li>`).join('')}
                </ul>
            </div>
            <div class="report-section">
                <h3>Recommendations</h3>
                <ul>
                    ${report.analysis.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
            <button class="close-report">Close</button>
        </div>
    `;
    
    // Add modal to page
    document.body.appendChild(modal);
    
    // Add styles
    const style = document.createElement('style');
    style.textContent = `
        .report-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .report-content {
            background: white;
            padding: 20px;
            border-radius: 8px;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
        }
        .report-section {
            margin: 20px 0;
        }
        .close-report {
            margin-top: 20px;
            padding: 10px 20px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .close-report:hover {
            background: #45a049;
        }
    `;
    document.head.appendChild(style);
    
    // Add close functionality
    const closeBtn = modal.querySelector('.close-report');
    closeBtn.addEventListener('click', function() {
        document.body.removeChild(modal);
        document.head.removeChild(style);
    });
} 