const hiddenInput = document.getElementById('hidden-input');
const typedSpan = document.getElementById('typed-text');
const predSpan = document.getElementById('predicted-text');
const caretSpan = document.getElementById('caret');

let debounceTimer;
const DEBOUNCE_MS = 100;

function updateDisplay() {
    const text = hiddenInput.value;
    
    if (text === '') {
        typedSpan.textContent = '';
        // Only show prediction placeholder if empty
        if (!predSpan.textContent) {
            predSpan.textContent = 'type to start auto-completing your story...';
            predSpan.style.opacity = '0.5';
        }
    } else {
        typedSpan.textContent = text;
        predSpan.style.opacity = '0.7'; // Restore normal ghost opacity
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
        body: JSON.stringify({ prompt: context, max_new_tokens: 15 })
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

hiddenInput.addEventListener('input', () => {
    updateDisplay();
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => fetchPrediction(hiddenInput.value), DEBOUNCE_MS);
});

hiddenInput.addEventListener('keydown', (e) => {
    const pred = predSpan.textContent;
    
    if ((e.key === 'Tab' || e.key === 'ArrowRight') && pred && hiddenInput.value !== '') {
        e.preventDefault();
        hiddenInput.value += pred;
        updateDisplay();
        predSpan.textContent = '';
        fetchPrediction(hiddenInput.value);
    } else if (e.key === 'Escape') {
        e.preventDefault();
        predSpan.textContent = '';
    }
});

// Caret focus handling
hiddenInput.addEventListener('focus', () => caretSpan.classList.remove('unfocused'));
hiddenInput.addEventListener('blur', () => caretSpan.classList.add('unfocused'));

// Force focus when clicking anywhere on the document
document.addEventListener('click', () => {
    hiddenInput.focus();
});

window.addEventListener('DOMContentLoaded', () => {
    hiddenInput.focus();
    updateDisplay();
});
