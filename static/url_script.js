async function analyzeUrl() {
    const url = document.getElementById('input-url').value;
    if (!url) {
        alert('Please enter a URL');
        return;
    }

    const response = await fetch('http://localhost:8000/analyze_url', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: url })
    });

    const result = await response.json();
    const output = document.getElementById('url-output');

    if (response.ok) {
        output.innerHTML = `
            <p>Initial text: ${result.text}</p>
            <p>Score: ${result.prob}</p>
            <p>Paraphrases and their probabilities:</p>
            <ul>
                ${result.paraphrases.map((paraphrase, index) => `<li>${paraphrase} (Score: ${result.probabilities[index]})</li>`).join('')}
            </ul>
        `;
    } else {
        output.innerHTML = `<p>Error: ${result.detail}</p>`;
    }
}