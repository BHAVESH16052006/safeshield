// --- Copy Result to Clipboard ---
document.getElementById('copyResult').addEventListener('click', function() {
    const result = document.querySelector('.result-text').textContent;
    const confidence = document.querySelector('.confidence-score').textContent;
    const text = `${result} (Confidence: ${confidence})`;
    navigator.clipboard.writeText(text).then(() => {
        this.textContent = 'Copied!';
        setTimeout(() => { this.innerHTML = '<i class=\'fa fa-copy\'></i> Copy Result'; }, 1200);
    });
});

// --- Download PDF ---
document.getElementById('downloadPDF').addEventListener('click', function() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    const result = document.querySelector('.result-text').textContent;
    const confidence = document.querySelector('.confidence-score').textContent;
    doc.setFontSize(18);
    doc.text('Spam Analysis Result', 10, 20);
    doc.setFontSize(12);
    doc.text(`Result: ${result}`, 10, 40);
    doc.text(`Confidence: ${confidence}`, 10, 50);
    doc.save('SpamAnalysisResult.pdf');
});

// --- Spinner and Animated Glow ---
function showSpinner(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) spinner.style.display = show ? 'flex' : 'none';
}
function animateGlow(prediction) {
    const card = document.getElementById('predictionCard');
    if (!card) return;
    card.classList.remove('glow-green', 'glow-red');
    if (prediction === 'Not Spam') {
        card.classList.add('glow-green');
    } else if (prediction === 'Spam') {
        card.classList.add('glow-red');
    }
    setTimeout(() => card.classList.remove('glow-green', 'glow-red'), 1500);
}

// --- Language Detection ---
function detectLanguage(text) {
    if (window.LanguageDetect) {
        const langDetect = new window.LanguageDetect();
        const result = langDetect.detect(text, 1);
        if (result && result.length > 0) {
            return result[0][0];
        }
    }
    return 'Unknown';
}

// --- Speech-to-Text ---
const micBtn = document.getElementById('micBtn');
const commentBox = document.getElementById('comment');
let recognition;

function showMicWarning(msg, recording=false) {
    // Do nothing: mic warning is now disabled and hidden
}

function insertAtCursor(textarea, text) {
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const before = textarea.value.substring(0, start);
    const after = textarea.value.substring(end);
    textarea.value = before + text + after;
    textarea.selectionStart = textarea.selectionEnd = start + text.length;
    textarea.focus();
}

if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    micBtn.addEventListener('mousedown', function(e) { e.preventDefault(); }); // Prevent blur
    micBtn.addEventListener('click', function(e) {
        e.preventDefault();
        if (micBtn.classList.contains('recording')) {
            recognition.stop();
            micBtn.classList.remove('recording');
            micBtn.innerHTML = '<i class="fa fa-microphone"></i>';
        } else {
            try {
                recognition.start();
                micBtn.classList.add('recording');
                micBtn.innerHTML = '<i class="fa fa-stop"></i>';
                showMicWarning('Listening... Speak now.', true);
            } catch (e) {
                showMicWarning('Could not start recognition. Try again.');
            }
        }
    });

    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        insertAtCursor(commentBox, transcript);
        commentBox.dispatchEvent(new Event('input'));
        showMicWarning('');
    };
    recognition.onend = function() {
        micBtn.classList.remove('recording');
        micBtn.innerHTML = '<i class="fa fa-microphone"></i>';
        showMicWarning('');
    };
    recognition.onerror = function(event) {
        showMicWarning('Speech recognition error. Please try again.');
        micBtn.classList.remove('recording');
        micBtn.innerHTML = '<i class="fa fa-microphone"></i>';
    };
} else {
    micBtn.disabled = true;
    micBtn.title = 'Speech recognition not supported in this browser.';
    showMicWarning('Speech recognition is not supported in your browser. Try Chrome.');
}

// --- Live Typing Feedback (with spinner, glow, language) ---
let debounceTimer;
commentBox.addEventListener('input', function() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        const text = commentBox.value.trim();
        if (text.length > 0) {
            showSpinner(true);
            const lang = detectLanguage(text);
            document.getElementById('languageDisplay').textContent = `Detected language: ${lang}`;
            const formData = new FormData();
            formData.append('comment', text);
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('results').style.display = 'block';
                const confidence = parseFloat(data.confidence);
                document.querySelector('.confidence-bar').style.width = `${confidence}%`;
                document.querySelector('.confidence-score').textContent = data.confidence;
                document.querySelector('.result-text').textContent = `This message is ${data.prediction}`;
                document.querySelector('.result-text').className = `result-text ${data.prediction.replace(' ', '').toLowerCase()}`;
                animateGlow(data.prediction);
                showSpinner(false);
            })
            .catch(() => showSpinner(false));
        } else {
            document.getElementById('results').style.display = 'none';
            document.getElementById('languageDisplay').textContent = '';
        }
    }, 600);
});

// --- Export CSV ---
document.getElementById('exportCSV').addEventListener('click', function() {
    fetch('/analytics')
        .then(response => response.json())
        .then(data => {
            const history = data.prediction_history || [];
            let csv = 'Date,Message,Result,Confidence\n';
            history.forEach(item => {
                csv += `"${item.date}","${item.message.replace(/"/g, '""')}",${item.result},${item.confidence}\n`;
            });
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'prediction_history.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
});

// --- Share Result ---
document.getElementById('shareResult').addEventListener('click', function() {
    const result = document.querySelector('.result-text').textContent;
    const confidence = document.querySelector('.confidence-score').textContent;
    const message = document.getElementById('comment').value;
    const data = {
        message: message,
        prediction: result.replace('This message is ', ''),
        confidence: confidence
    };
    fetch('/share_result', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        const shareText = data.share_text;
        if (navigator.share) {
            navigator.share({ text: shareText });
        } else {
            navigator.clipboard.writeText(shareText).then(() => {
                this.textContent = 'Copied!';
                setTimeout(() => { this.innerHTML = '<i class="fa fa-share-alt"></i> Share Result'; }, 1200);
            });
        }
    });
}); 