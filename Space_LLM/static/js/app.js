// Aether-Agent: Frontend JavaScript

class AetherAgentUI {
    constructor() {
        this.conversationHistory = [];
        this.init();
    }

    init() {
        this.initTheme();
        this.setupEventListeners();
        this.loadStatus();
        this.loadHistory();
    }

    initTheme() {
        // Check for saved theme preference or default to light mode
        const savedTheme = localStorage.getItem('aetherTheme') || 'light';
        this.setTheme(savedTheme);
    }

    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('aetherTheme', theme);
        this.updateThemeButton(theme);
    }

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
    }

    updateThemeButton(theme) {
        const toggle = document.getElementById('theme-toggle');
        if (toggle) {
            // Button text/content is handled by CSS
        }
    }

    setupEventListeners() {
        const submitBtn = document.getElementById('submit-btn');
        const queryInput = document.getElementById('query-input');
        const clearBtn = document.getElementById('clear-btn');
        const toggleThoughts = document.getElementById('toggle-thoughts');
        const themeToggle = document.getElementById('theme-toggle');

        submitBtn.addEventListener('click', () => this.handleSubmit());
        queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.handleSubmit();
            }
        });

        clearBtn.addEventListener('click', () => this.clearResults());
        toggleThoughts.addEventListener('click', () => this.toggleThoughtProcess());
        
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
        }
    }

    async loadStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.success) {
                document.getElementById('chunk-count').textContent = data.chunks.toLocaleString();
                document.getElementById('paper-count').textContent = data.papers;
                document.getElementById('status-text').textContent = 'Ready';
                document.getElementById('status-dot').style.background = '#34a853';
            }
        } catch (error) {
            console.error('Error loading status:', error);
        }
    }

    async handleSubmit() {
        const queryInput = document.getElementById('query-input');
        const question = queryInput.value.trim();

        if (!question) {
            alert('Please enter a question.');
            return;
        }

        // Disable submit button
        const submitBtn = document.getElementById('submit-btn');
        submitBtn.disabled = true;
        submitBtn.querySelector('.btn-text').textContent = 'Processing...';

        // Show loading indicator
        document.getElementById('loading-indicator').classList.remove('hidden');
        document.getElementById('answer-section').classList.add('hidden');
        document.getElementById('citations-section').classList.add('hidden');
        document.getElementById('thought-process-section').classList.add('hidden');

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question })
            });

            const data = await response.json();

            if (data.success) {
                this.displayResults(data);
                this.addToHistory(question, data.answer);
            } else {
                this.displayError(data.error || 'An error occurred');
            }
        } catch (error) {
            console.error('Error:', error);
            this.displayError('Failed to connect to the server. Please try again.');
        } finally {
            // Re-enable submit button
            submitBtn.disabled = false;
            submitBtn.querySelector('.btn-text').textContent = 'Query';
            document.getElementById('loading-indicator').classList.add('hidden');
        }
    }

    displayResults(data) {
        // Display answer
        const answerContent = document.getElementById('answer-content');
        answerContent.textContent = data.answer;
        document.getElementById('answer-section').classList.remove('hidden');

        // Display citations
        if (data.citations && data.citations.length > 0) {
            const citationsList = document.getElementById('citations-list');
            citationsList.innerHTML = '';
            data.citations.forEach(citation => {
                const citationItem = document.createElement('div');
                citationItem.className = 'citation-item';
                citationItem.textContent = citation.citation || JSON.stringify(citation);
                citationsList.appendChild(citationItem);
            });
            document.getElementById('citations-section').classList.remove('hidden');
        }

        // Display thought process
        if (data.thought_process && data.thought_process.length > 0) {
            this.displayThoughtProcess(data.thought_process);
            document.getElementById('thought-process-section').classList.remove('hidden');
        }
    }

    displayThoughtProcess(thoughtProcess) {
        const content = document.getElementById('thought-process-content');
        content.innerHTML = '';

        thoughtProcess.forEach((step, index) => {
            const stepDiv = document.createElement('div');
            stepDiv.className = 'thought-step';

            const header = document.createElement('div');
            header.className = 'thought-step-header';

            const number = document.createElement('div');
            number.className = 'thought-step-number';
            number.textContent = step.iteration || index + 1;

            const action = document.createElement('div');
            action.className = 'thought-step-action';
            action.textContent = step.action || 'Unknown Action';

            header.appendChild(number);
            header.appendChild(action);

            const stepContent = document.createElement('div');
            stepContent.className = 'thought-step-content';

            if (step.thought) {
                const thoughtP = document.createElement('p');
                thoughtP.innerHTML = `<strong>Thought:</strong> ${this.escapeHtml(step.thought)}`;
                stepContent.appendChild(thoughtP);
            }

            if (step.result) {
                const resultP = document.createElement('p');
                resultP.innerHTML = `<strong>Result:</strong> ${this.escapeHtml(step.result.substring(0, 200))}...`;
                stepContent.appendChild(resultP);
            }

            stepDiv.appendChild(header);
            stepDiv.appendChild(stepContent);
            content.appendChild(stepDiv);
        });

        // Show thought process content
        content.classList.remove('hidden');
    }

    toggleThoughtProcess() {
        const content = document.getElementById('thought-process-content');
        const button = document.getElementById('toggle-thoughts');
        
        if (content.classList.contains('hidden')) {
            content.classList.remove('hidden');
            button.textContent = 'Hide';
        } else {
            content.classList.add('hidden');
            button.textContent = 'Show';
        }
    }

    displayError(message) {
        const answerContent = document.getElementById('answer-content');
        answerContent.textContent = `Error: ${message}`;
        answerContent.style.color = '#ea4335';
        document.getElementById('answer-section').classList.remove('hidden');
    }

    clearResults() {
        document.getElementById('answer-section').classList.add('hidden');
        document.getElementById('citations-section').classList.add('hidden');
        document.getElementById('thought-process-section').classList.add('hidden');
        document.getElementById('query-input').value = '';
    }

    addToHistory(question, answer) {
        const historyItem = {
            question,
            answer: answer.substring(0, 100) + '...',
            timestamp: new Date().toLocaleTimeString()
        };

        this.conversationHistory.unshift(historyItem);
        if (this.conversationHistory.length > 10) {
            this.conversationHistory.pop();
        }

        this.saveHistory();
        this.updateHistoryDisplay();
    }

    updateHistoryDisplay() {
        const historyList = document.getElementById('history-list');
        historyList.innerHTML = '';

        if (this.conversationHistory.length === 0) {
            historyList.innerHTML = '<p style="color: var(--text-secondary); font-size: 14px;">No queries yet.</p>';
            return;
        }

        this.conversationHistory.forEach(item => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.addEventListener('click', () => {
                document.getElementById('query-input').value = item.question;
            });

            const query = document.createElement('div');
            query.className = 'history-query';
            query.textContent = item.question;

            const preview = document.createElement('div');
            preview.className = 'history-preview';
            preview.textContent = item.answer;

            historyItem.appendChild(query);
            historyItem.appendChild(preview);
            historyList.appendChild(historyItem);
        });
    }

    saveHistory() {
        localStorage.setItem('aetherHistory', JSON.stringify(this.conversationHistory));
    }

    loadHistory() {
        const saved = localStorage.getItem('aetherHistory');
        if (saved) {
            try {
                this.conversationHistory = JSON.parse(saved);
                this.updateHistoryDisplay();
            } catch (e) {
                console.error('Error loading history:', e);
            }
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new AetherAgentUI();
});
