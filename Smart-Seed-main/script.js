document.getElementById("predictForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const formData = new FormData(form);
    const body = {};
    formData.forEach((v, k) => body[k] = isNaN(v) ? v : parseFloat(v));
  
    const res = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const result = await res.json();
    document.getElementById("result").innerText = 
      result.recommended_crop ? 
      `Recommended Crop (${result.method}): ${result.recommended_crop}` : 
      `Error: ${result.error}`;
  });
  