async function predict() {
    const inputData = {
        region: document.getElementById("region").value.trim(),  // No lowercase conversion
        crop_type: document.getElementById("crop").value.trim(),
        irrigation: document.getElementById("irrigation").value.trim(),
        soil_quality: document.getElementById("soil_quality").value.trim(),
        harvest_period: document.getElementById("harvest_period").value.trim(),
        numerical_features: [
            parseFloat(document.getElementById("feature1").value),
            parseFloat(document.getElementById("feature2").value),
            parseFloat(document.getElementById("feature3").value)
        ]
    };

    const response = await fetch("https://food-scarcity-backend.onrender.com/predict", {  // Use your Render API URL
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(inputData)
    });

    const result = await response.json();
    document.getElementById("output").innerText = JSON.stringify(result, null, 2);
}
