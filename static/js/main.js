// ============================================
// Crop Price Predictor - Main JavaScript
// ============================================

document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('prediction-form');
    const resultSection = document.getElementById('result-section');
    const predictedPriceEl = document.getElementById('predicted-price');
    const resultDetailsEl = document.getElementById('result-details');
    const predictBtn = document.getElementById('predict-btn');

    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        // Collect form data
        const data = {
            state: document.getElementById('state').value,
            district: document.getElementById('district').value,
            market: document.getElementById('market').value,
            commodity: document.getElementById('commodity').value,
            variety: document.getElementById('variety').value,
            grade: document.getElementById('grade').value,
        };

        // Validate
        for (const [key, value] of Object.entries(data)) {
            if (!value) {
                alert(`Please select a ${key}`);
                return;
            }
        }

        // Loading state
        predictBtn.classList.add('loading');
        predictBtn.innerHTML = '<span class="spinner"></span> Predicting...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });

            const result = await response.json();

            if (result.success) {
                // Show result with animation
                resultSection.style.display = 'block';

                // Animate the price counter
                animatePrice(0, result.predicted_price, 1000);

                // Show input details
                resultDetailsEl.innerHTML = `
                    <span><strong>State:</strong> ${data.state}</span>
                    <span><strong>District:</strong> ${data.district}</span>
                    <span><strong>Market:</strong> ${data.market}</span>
                    <span><strong>Commodity:</strong> ${data.commodity}</span>
                    <span><strong>Variety:</strong> ${data.variety}</span>
                    <span><strong>Grade:</strong> ${data.grade}</span>
                `;

                // Scroll to result
                resultSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            } else {
                showError(result.error || 'Prediction failed. Please try again.');
            }
        } catch (error) {
            showError('Could not connect to server. Please ensure the server is running.');
        } finally {
            // Reset button
            predictBtn.classList.remove('loading');
            predictBtn.innerHTML = '<i class="fas fa-search-dollar"></i> Predict Price';
        }
    });

    // Animate price counter
    function animatePrice(start, end, duration) {
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease out cubic
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = start + (end - start) * easeOut;

            predictedPriceEl.textContent = current.toLocaleString('en-IN', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
            });

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }

        requestAnimationFrame(update);
    }

    // Show error
    function showError(message) {
        resultSection.style.display = 'block';
        resultSection.innerHTML = `
            <div class="result-content">
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle"></i> ${message}
                </div>
            </div>
        `;
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
});
