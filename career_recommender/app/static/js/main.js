// Main JS
// app/static/js/app.js
document.addEventListener("DOMContentLoaded", () => {
  const btnPredict = document.getElementById("btnPredict");
  if (btnPredict) {
    btnPredict.addEventListener("click", async () => {
      const txt = document.getElementById("profileText").value.trim();
      if (!txt) return alert("Please paste your skills / profile first.");
      const payload = { profile: { profile_text: txt } };

      btnPredict.disabled = true;
      btnPredict.innerText = "Predicting...";

      try {
        const r = await fetch("/api/predict", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify(payload)
        });
        const j = await r.json();
        btnPredict.disabled = false; btnPredict.innerText = "Get Recommendations";
        showPrediction(j.top3);
      } catch (err) {
        btnPredict.disabled = false; btnPredict.innerText = "Get Recommendations";
        alert("Prediction failed: " + err.message);
      }
    });
  }

  const uploadBtn = document.getElementById("btnUpload");
  if (uploadBtn) {
    uploadBtn.addEventListener("click", async () => {
      const fileInput = document.getElementById("fileInput");
      if (!fileInput.files.length) return alert("Select a file first.");
      const fd = new FormData();
      fd.append("file", fileInput.files[0]);
      fd.append("user_id", 1);

      uploadBtn.disabled = true;
      uploadBtn.innerText = "Uploading...";

      try {
        const r = await fetch("/api/upload_resume", { method: "POST", body: fd });
        const j = await r.json();
        uploadBtn.disabled = false;
        uploadBtn.innerText = "Upload & Predict";
        if (j.parsed) {
          document.getElementById("parsedCard").classList.remove("d-none");
          document.getElementById("parsedJson").innerText = JSON.stringify(j.parsed, null, 2);
          const top = j.top3 || [];
          const ol = document.getElementById("parsedTop3");
          ol.innerHTML = "";
          top.forEach(t => {
            const li = document.createElement("li");
            li.innerText = `${t.label} (${(t.score*100).toFixed(1)}%)`;
            ol.appendChild(li);
          });
        } else {
          alert("No parse result returned.");
        }
      } catch (err) {
        uploadBtn.disabled = false;
        uploadBtn.innerText = "Upload & Predict";
        alert("Upload failed: " + err.message);
      }
    });
  }
});

function showPrediction(top3) {
  const card = document.getElementById("resultCard");
  const ol = document.getElementById("topRoles");
  if (!top3 || !top3.length) {
    ol.innerHTML = "<li>No recommendation</li>";
  } else {
    ol.innerHTML = "";
    top3.forEach(t => {
      const li = document.createElement("li");
      li.innerText = `${t.label} — ${(t.score*100).toFixed(1)}%`;
      ol.appendChild(li);
    });
  }
  if (card) card.classList.remove("d-none");
}
