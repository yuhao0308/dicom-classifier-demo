(function () {
  "use strict";

  var POLL_INTERVAL_MS = 2000;
  var MAX_POLLS = 150;

  var STAGE_LABELS = {
    0: "Uploading...",
    25: "Parsing DICOM...",
    40: "Loading annotations...",
    55: "Running inference...",
    75: "Evaluating...",
    90: "Finalizing...",
    100: "Complete!",
  };

  var form = document.getElementById("upload-form");
  var fileInput = document.getElementById("dicom-file");
  var annotationInput = document.getElementById("annotation-file");
  var uploadButton = document.getElementById("upload-button");
  var dropZone = document.getElementById("drop-zone");
  var fileNameDisplay = document.getElementById("file-name");
  var progressSection = document.getElementById("progress-section");
  var progressFill = document.getElementById("progress-fill");
  var progressLabel = document.getElementById("progress-label");
  var advancedToggle = document.getElementById("advanced-toggle");
  var advancedSection = document.getElementById("advanced-section");
  var toggleArrow = document.getElementById("toggle-arrow");

  if (!form || !fileInput || !uploadButton) {
    return;
  }

  // ── Collapsible toggle ──
  if (advancedToggle && advancedSection) {
    advancedToggle.addEventListener("click", function () {
      var isOpen = advancedSection.classList.toggle("collapsible-content--open");
      if (toggleArrow) {
        toggleArrow.classList.toggle("collapsible-toggle__arrow--open", isOpen);
      }
    });
  }

  // ── Drag and drop ──
  if (dropZone) {
    dropZone.addEventListener("click", function () {
      fileInput.click();
    });

    dropZone.addEventListener("dragover", function (e) {
      e.preventDefault();
      dropZone.classList.add("drop-zone--active");
    });

    dropZone.addEventListener("dragleave", function () {
      dropZone.classList.remove("drop-zone--active");
    });

    dropZone.addEventListener("drop", function (e) {
      e.preventDefault();
      dropZone.classList.remove("drop-zone--active");
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        showFileName(e.dataTransfer.files[0].name);
      }
    });
  }

  fileInput.addEventListener("change", function () {
    if (fileInput.files && fileInput.files.length > 0) {
      showFileName(fileInput.files[0].name);
    }
  });

  function showFileName(name) {
    if (fileNameDisplay) {
      fileNameDisplay.textContent = name;
    }
  }

  // ── Form submit ──
  form.addEventListener("submit", function (event) {
    event.preventDefault();
    if (!fileInput.files || fileInput.files.length === 0) {
      setProgressLabel("Choose a .zip archive before uploading.");
      return;
    }

    var file = fileInput.files[0];
    if (!file.name.toLowerCase().endsWith(".zip")) {
      setProgressLabel("Only .zip uploads are supported.");
      return;
    }

    void startUpload(file);
  });

  async function startUpload(file) {
    setProcessingState(true);
    showProgress(true);
    setProgress(0);
    setProgressLabel("Uploading archive...");

    var formData = new FormData();
    formData.append("file", file);

    // Append annotation file if provided
    if (annotationInput && annotationInput.files && annotationInput.files.length > 0) {
      formData.append("annotation", annotationInput.files[0]);
    }

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
        setProgressLabel(detail);
        setProcessingState(false);
        return;
      }

      var jobId = uploadPayload.job_id;
      if (!jobId) {
        setProgressLabel("Upload succeeded but no job ID was returned.");
        setProcessingState(false);
        return;
      }

      setProgressLabel("Upload accepted. Processing started...");
      setProgress(10);
      await pollJobUntilDone(jobId);
    } catch (error) {
      setProgressLabel("Network error while uploading.");
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
        setProgressLabel("Network error while checking job status.");
        setProcessingState(false);
        return;
      }

      if (!pollResponse.ok) {
        setProgressLabel("Unable to fetch job status.");
        setProcessingState(false);
        return;
      }

      var statusPayload = await pollResponse.json();
      var progress = Number(statusPayload.progress || 0);
      var statusValue = String(statusPayload.status || "processing");
      setProgress(progress);
      setProgressLabel(getStageLabel(progress));

      if (statusValue === "completed") {
        setProgress(100);
        setProgressLabel("Processing completed. Redirecting to results...");
        progressFill.classList.add("progress-bar-fill--complete");
        window.location.assign("/results/" + encodeURIComponent(jobId) + "/view");
        return;
      }

      if (statusValue === "failed") {
        setProgressLabel("Processing failed. Please try again with a different file.");
        setProcessingState(false);
        return;
      }

      await sleep(POLL_INTERVAL_MS);
    }

    setProgressLabel("Processing is taking longer than expected.");
    setProcessingState(false);
  }

  function getStageLabel(progress) {
    var label = "Processing...";
    var thresholds = Object.keys(STAGE_LABELS)
      .map(Number)
      .sort(function (a, b) {
        return a - b;
      });
    for (var i = thresholds.length - 1; i >= 0; i--) {
      if (progress >= thresholds[i]) {
        label = STAGE_LABELS[thresholds[i]];
        break;
      }
    }
    return label + " " + progress + "%";
  }

  function showProgress(visible) {
    if (progressSection) {
      progressSection.classList.toggle("progress-section--visible", visible);
    }
  }

  function setProgressLabel(message) {
    if (progressLabel) {
      progressLabel.textContent = message;
    }
    showProgress(true);
  }

  function setProgress(value) {
    var safeValue = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;
    if (progressFill) {
      progressFill.style.width = safeValue + "%";
    }
  }

  function setProcessingState(isProcessing) {
    uploadButton.disabled = isProcessing;
    fileInput.disabled = isProcessing;
    if (annotationInput) {
      annotationInput.disabled = isProcessing;
    }
    uploadButton.textContent = isProcessing ? "Processing..." : "Upload and Analyze";
  }

  function sleep(ms) {
    return new Promise(function (resolve) {
      window.setTimeout(resolve, ms);
    });
  }
})();
