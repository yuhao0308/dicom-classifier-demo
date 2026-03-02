(function () {
  "use strict";

  var POLL_INTERVAL_MS = 2000;
  var MAX_POLLS = 150;

  var form = document.getElementById("upload-form");
  var fileInput = document.getElementById("dicom-file");
  var uploadButton = document.getElementById("upload-button");
  var statusText = document.getElementById("status-text");
  var progressBar = document.getElementById("job-progress");

  if (!form || !fileInput || !uploadButton || !statusText || !progressBar) {
    return;
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    if (!fileInput.files || fileInput.files.length === 0) {
      setStatus("Choose a .zip archive before uploading.");
      return;
    }

    var file = fileInput.files[0];
    if (!file.name.toLowerCase().endsWith(".zip")) {
      setStatus("Only .zip uploads are supported.");
      return;
    }

    void startUpload(file);
  });

  async function startUpload(file) {
    setProcessingState(true);
    setProgress(0);
    setStatus("Uploading archive...");

    var formData = new FormData();
    formData.append("file", file);

    try {
      var uploadResponse = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      var uploadPayload = await uploadResponse.json().catch(function () {
        return {};
      });

      if (!uploadResponse.ok) {
        var detail = uploadPayload.detail || "Upload failed.";
        setStatus(detail);
        setProcessingState(false);
        return;
      }

      var jobId = uploadPayload.job_id;
      if (!jobId) {
        setStatus("Upload succeeded but no job ID was returned.");
        setProcessingState(false);
        return;
      }

      setStatus("Upload accepted. Waiting for processing...");
      await pollJobUntilDone(jobId);
    } catch (error) {
      setStatus("Network error while uploading.");
      setProcessingState(false);
    }
  }

  async function pollJobUntilDone(jobId) {
    for (var attempt = 1; attempt <= MAX_POLLS; attempt += 1) {
      var pollResponse;
      try {
        pollResponse = await fetch("/jobs/" + encodeURIComponent(jobId), {
          method: "GET",
        });
      } catch (error) {
        setStatus("Network error while checking job status.");
        setProcessingState(false);
        return;
      }

      if (!pollResponse.ok) {
        setStatus("Unable to fetch job status.");
        setProcessingState(false);
        return;
      }

      var statusPayload = await pollResponse.json();
      var progress = Number(statusPayload.progress || 0);
      var statusValue = String(statusPayload.status || "processing");
      setProgress(progress);

      if (statusValue === "completed") {
        setStatus("Processing completed. Redirecting to results...");
        window.location.assign("/results/" + encodeURIComponent(jobId) + "/view");
        return;
      }

      if (statusValue === "failed") {
        setStatus("Processing failed. Please try again with a different file.");
        setProcessingState(false);
        return;
      }

      setStatus("Processing... " + progress + "%");
      await sleep(POLL_INTERVAL_MS);
    }

    setStatus("Processing is taking longer than expected.");
    setProcessingState(false);
  }

  function setStatus(message) {
    statusText.textContent = message;
  }

  function setProgress(value) {
    var safeValue = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;
    progressBar.value = safeValue;
  }

  function setProcessingState(isProcessing) {
    uploadButton.disabled = isProcessing;
    fileInput.disabled = isProcessing;
    uploadButton.textContent = isProcessing ? "Processing..." : "Upload and Analyze";
  }

  function sleep(ms) {
    return new Promise(function (resolve) {
      window.setTimeout(resolve, ms);
    });
  }
})();
