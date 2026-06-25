const hiddenInput = document.getElementById('hidden-input');
const typedSpan = document.getElementById('typed-text');
const predSpan = document.getElementById('predicted-text');
const caretSpan = document.getElementById('caret');

// Action Buttons
const btnStats = document.getElementById('btn-stats');
const btnSettings = document.getElementById('btn-settings');

let debounceTimer;
const DEBOUNCE_MS = 100;

// User Configuration and State
let maxTokens = 15;
let aiStats = { accepted: 0, rejected: 0 };

// -----------------------------------------
// UI Actions
// -----------------------------------------

// Crown Icon - Show accuracy/acceptance stats
btnStats.addEventListener('click', (e) => {
    e.preventDefault();
    const total = aiStats.accepted + aiStats.rejected;
    const rate = total > 0 ? Math.round((aiStats.accepted / total) * 100) : 0;
    
    alert(`👑 AI Usage Stats 👑\n\nAccepted Suggestions: ${aiStats.accepted}\nRejected Suggestions (Esc): ${aiStats.rejected}\n\nOverall Accuracy / Acceptance Rate: ${rate}%`);
    hiddenInput.focus();
});

// Gear Icon - Change model generation length
btnSettings.addEventListener('click', (e) => {
    e.preventDefault();
    const input = prompt("Settings:\nHow many characters should the AI predict at a time?", maxTokens);
    if (input && !isNaN(parseInt(input))) {
        maxTokens = parseInt(input);
    }
    hiddenInput.focus();
});

// -----------------------------------------
// Editor Logic
// -----------------------------------------

function updateDisplay() {
    const text = hiddenInput.value;
    
    if (text === '') {
        typedSpan.textContent = '';
        if (!predSpan.textContent) {
            predSpan.textContent = 'type to start auto-completing your story...';
            predSpan.style.opacity = '0.5';
        }
    } else {
        typedSpan.textContent = text;
        predSpan.style.opacity = '0.7'; 
    }
}

function fetchPrediction(promptText) {
    if (!promptText) {
        predSpan.textContent = 'type to start auto-completing your story...';
        predSpan.style.opacity = '0.5';
        return;
    }
    
    const context = promptText.slice(-32);
    
    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: context, max_new_tokens: maxTokens })
    })
    .then(r => r.json())
    .then(data => {
        if (data.completion) {
            let clean = data.completion.replace(/^\n+/, '').replace(/\n/g, ' ');
            predSpan.textContent = clean;
            predSpan.style.opacity = '0.7';
        } else {
            predSpan.textContent = '';
        }
    })
    .catch(err => {
        predSpan.textContent = '';
    });
}

// -----------------------------------------
// Event Listeners
// -----------------------------------------

hiddenInput.addEventListener('input', () => {
    updateDisplay();
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => fetchPrediction(hiddenInput.value), DEBOUNCE_MS);
});

hiddenInput.addEventListener('keydown', (e) => {
    const pred = predSpan.textContent;
    const hasPrediction = pred && pred !== 'type to start auto-completing your story...';
    
    if (e.key === 'Tab' && hasPrediction && hiddenInput.value !== '') {
        e.preventDefault();
        // Regex to extract the first word and any trailing spaces
        const wordMatch = pred.match(/^(\s*\S+\s*)/);
        const wordToAdd = wordMatch ? wordMatch[1] : pred;
        
        hiddenInput.value += wordToAdd;
        aiStats.accepted++;
        updateDisplay();
        predSpan.textContent = '';
        fetchPrediction(hiddenInput.value);
    } 
    else if (e.key === 'ArrowRight' && hasPrediction && hiddenInput.value !== '') {
        e.preventDefault();
        // Accept the entire string
        hiddenInput.value += pred;
        aiStats.accepted++;
        updateDisplay();
        predSpan.textContent = '';
        fetchPrediction(hiddenInput.value);
    }
    else if (e.key === 'Escape') {
        e.preventDefault();
        if (hasPrediction) {
            aiStats.rejected++;
            predSpan.textContent = '';
        }
    }
});

// Caret focus handling
hiddenInput.addEventListener('focus', () => caretSpan.classList.remove('unfocused'));
hiddenInput.addEventListener('blur', () => caretSpan.classList.add('unfocused'));

// Force focus when clicking background (but don't override link clicks)
document.addEventListener('click', (e) => {
    if(e.target.tagName !== 'A' && e.target.tagName !== 'I') {
        hiddenInput.focus();
    }
});

window.addEventListener('DOMContentLoaded', () => {
    hiddenInput.focus();
    updateDisplay();
});
