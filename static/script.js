async function compute() {
    const text = document.getElementById('input-text').value;
    if (!text) {
        alert('Please enter some text');
        return;
    }

    // Clear output fields
    document.getElementById('output').innerText = '';
    document.getElementById('paraphrases').innerHTML = '';

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
    document.getElementById('output').innerText = `Text score: ${result.prob}`;
}

async function change() {
    const text = document.getElementById('input-text').value;
    if (!text) {
        alert('Please enter some text');
        return;
    }

    // Clear output fields
    document.getElementById('output').innerText = '';
    document.getElementById('paraphrases').innerHTML = '';

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

    document.getElementById('output').innerText = `Initial text score: ${result.prob}`;

    let paraphrasesHTML = '';
    result.paraphrases.forEach((paraphrase, index) => {
        paraphrasesHTML += `<b> Score: </b> ${result.probabilities[index]} <br> <li>${paraphrase}</li> <br>`;
    });
    document.getElementById('paraphrases').innerHTML = paraphrasesHTML;
}
