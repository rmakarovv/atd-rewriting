async function compute() {
    const text = document.getElementById('input-text').value;
    if (!text) {
        alert('Please enter some text');
        return;
    }

    const response = await fetch('http://localhost:8000/inference_prob', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: text })
    });

    if (!response.ok) {
        alert('Error computing probability');
        return;
    }

    const result = await response.json();
    document.getElementById('output').innerText = `Probability that this text is AI-generated: ${result.prob}%`;
}

async function change() {
    const text = document.getElementById('input-text').value;
    if (!text) {
        alert('Please enter some text');
        return;
    }

    const response = await fetch('http://localhost:8000/inference_change', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: text })
    });

    if (!response.ok) {
        alert('Error fetching paraphrases');
        return;
    }

    const result = await response.json();

    document.getElementById('output').innerText = `Probability that the initial text is AI-generated: ${result.prob}%`;

    let paraphrasesHTML = '<h3>Paraphrases and their probabilities:</h3><ul>';
    result.paraphrases.forEach((paraphrase, index) => {
        paraphrasesHTML += `Probability: ${result.probabilities[index]}% <br> <li>${paraphrase}</li>`;
    });
    paraphrasesHTML += '</ul>';
    document.getElementById('paraphrases').innerHTML = paraphrasesHTML;
}