async function predict() {
    const inputData = {
        region: document.getElementById("region").value.toLowerCase(),
        crop_type: document.getElementById("crop").value.toLowerCase(),
        irrigation: document.getElementById("irrigation").value.toLowerCase(),
        soil_quality: document.getElementById("soil_quality").value.toLowerCase(),
        harvest_period: document.getElementById("harvest_period").value.toLowerCase(),
        numerical_features: [
            parseFloat(document.getElementById("feature1").value),
            parseFloat(document.getElementById("feature2").value),
            parseFloat(document.getElementById("feature3").value)
        ]
    };

    const response = await fetch("https://food-scarcity-backend.onrender.com/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"  // Ensures the request is JSON
        },
        body: JSON.stringify(inputData)  // Converts data to JSON format
    });

    const result = await response.json();
    document.getElementById("output").innerText = JSON.stringify(result, null, 2);
}
