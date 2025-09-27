let model;

// Load pre-trained model
async function loadModel() {
  model = await tf.loadLayersModel('model/model.json'); // Make sure this is hosted
  console.log("Model loaded!");
}

async function recommendCrop() {
  const recBox = document.getElementById('recommendationBox');
  recBox.classList.add('d-none');
  recBox.textContent = "Getting location and data...";

  if (!model) await loadModel();

  navigator.geolocation.getCurrentPosition(async position => {
    const lat = position.coords.latitude;
    const lon = position.coords.longitude;

    const envData = simulateEnvData(lat, lon); // Simulated weather/soil
    const input = tf.tensor2d([envData]);
    const prediction = model.predict(input);
    const cropIndex = prediction.argMax(1).dataSync()[0];
    const confidence = prediction.dataSync()[cropIndex];
    const cropMap = ["Maize", "Beans", "Sorghum", "Rice", "Potato"];

    recBox.classList.remove('d-none');
    recBox.textContent = `Recommended Crop: ${cropMap[cropIndex]} (Confidence: ${(confidence*100).toFixed(1)}%)`;
  }, error => {
    recBox.classList.remove('d-none');
    recBox.textContent = "Failed to get your location. Please allow location access.";
  });
}

// Simulated environmental data based on region
function simulateEnvData(lat, lon) {
  // Example logic for Kenya
  let temperature = 25 + Math.random() * 5;
  let humidity = 60 + Math.random() * 20;
  let precipitation = 10 + Math.random() * 10;
  let soilPh = 6 + Math.random();
  let nitrogen = 30 + Math.random() * 10;
  let phosphorus = 20 + Math.random() * 5;
  let potassium = 25 + Math.random() * 5;
  let moisture = 30 + Math.random() * 10;
  let month = new Date().getMonth() + 1;

  return [temperature, humidity, precipitation, soilPh, nitrogen, phosphorus, potassium, moisture, month];
}

window.onload = loadModel;
