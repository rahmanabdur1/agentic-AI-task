
document.addEventListener('DOMContentLoaded', () => {
    const queryForm = document.getElementById('queryForm');
    const queryInput = document.getElementById('queryInput');
    const submitButton = document.getElementById('submitButton');
    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    const answerText = document.getElementById('answerText');
    const summaryText = document.getElementById('summaryText');
    const citationsList = document.getElementById('citationsList');
    const errorDiv = document.getElementById('error');
    const errorMessage = document.getElementById('errorMessage');

    queryForm.addEventListener('submit', async (event) => {
        event.preventDefault(); 

        const query = queryInput.value.trim();
        if (!query) {
            alert('Please enter a question!');
            return;
        }

       
        resultsDiv.classList.add('hidden');
        errorDiv.classList.add('hidden');
        loadingDiv.classList.remove('hidden');
        submitButton.disabled = true;
        queryInput.disabled = true;

        try {
       
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query }),
            });

            if (!response.ok) {
                // If response is not OK (e.g., 400, 500 status)
                const errorData = await response.json();
           
                throw new Error(errorData.detail || `Server error! Status: ${response.status}`);
            }

            const data = await response.json();

            // Populate results in the UI
            answerText.textContent = data.answer || 'No answer found.';
            summaryText.textContent = data.summary || 'No summary provided.';
            
            citationsList.innerHTML = ''; 
            if (data.citations && data.citations.length > 0) {
                data.citations.forEach(citation => {
                    const li = document.createElement('li');
                    li.textContent = citation;
                    citationsList.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.textContent = 'No specific citations available.';
                citationsList.appendChild(li);
            }

            resultsDiv.classList.remove('hidden'); // Show results
        } catch (err) {
            console.error('Error fetching data:', err);
            errorMessage.textContent = err.message || 'An unknown error occurred. Please check the server logs.';
            errorDiv.classList.remove('hidden'); // Show error
        } finally {
            loadingDiv.classList.add('hidden'); // Hide loading spinner
            submitButton.disabled = false; // Re-enable button
            queryInput.disabled = false; // Re-enable input
        }
    });
});