export function initPopups(app) {
    const { elements, state } = app;

    // IMAGE-QUALITY WARNING POPUP
    function closeQualityPopup() {
        elements.qualityPopupOverlay.classList.remove("active");
        elements.qualityPopupOverlay.setAttribute("aria-hidden", "true");
    }

    function showQualityPopup(quality) {
        const messages = quality.warnings.map(
            (warning) => app.text()[warning]
        );

        elements.qualityWarningList.innerHTML = messages.map((message) =>
        `<li>${app.actions.escapeHtml(message)}</li>`
        ).join("");

        elements.qualityMetrics.textContent =
            `${app.text().brightnessMetric}: ` +
            `${quality.brightness.toFixed(1)} · ` +
            `${app.text().blurMetric}: ` +
            `${quality.blurVariance.toFixed(1)}`;

        elements.qualityPopupOverlay.classList.add("active");
        elements.qualityPopupOverlay.setAttribute("aria-hidden", "false");
    }

    // CAMERA POPUP
    async function openCamera() {
        app.actions.setDateToToday();

        state.capturedBlob = null;

        elements.cameraError.textContent = "";
        elements.cameraPreview.hidden = true;
        elements.cameraVideo.hidden = false;
        elements.liveCameraActions.hidden = false;
        elements.capturedCameraActions.hidden = true;
        elements.cameraPopupOverlay.classList.add("active");
        elements.cameraPopupOverlay.setAttribute("aria-hidden", "false");

        try {
            state.cameraStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: {ideal: "environment"}
                },
                audio: false
            });

            elements.cameraVideo.srcObject = state.cameraStream;

            await elements.cameraVideo.play();
        } catch (error) {
            console.error("Camera access failed:", error);

            elements.cameraError.textContent = app.text().cameraUnavailable;
        }
    }

    function stopCameraStream() {
        if (state.cameraStream) {
            state.cameraStream.getTracks().forEach((track) => track.stop());
            state.cameraStream = null;
        }

        elements.cameraVideo.srcObject = null;
    }

    function closeCamera() {
        stopCameraStream();

        elements.cameraPopupOverlay.classList.remove("active");
        elements.cameraPopupOverlay.setAttribute("aria-hidden", "true");

        if (state.capturedPreviewUrl) {
            URL.revokeObjectURL(state.capturedPreviewUrl);
            state.capturedPreviewUrl = null;
        }
    }

    function capturePhoto() {
        const width = elements.cameraVideo.videoWidth;
        const height = elements.cameraVideo.videoHeight;

        if (!width || !height) {
            elements.cameraError.textContent = app.text().cameraUnavailable;
            return;
        }

        const context = elements.cameraCanvas.getContext("2d");

        elements.cameraCanvas.width = width;
        elements.cameraCanvas.height = height;

        context.drawImage(
            elements.cameraVideo,
            0,
            0,
            width,
            height
        );

        elements.cameraCanvas.toBlob(
            (blob) => {
                if (!blob) {
                    elements.cameraError.textContent = app.text().cameraUnavailable;
                    return;
                }

                state.capturedBlob = blob;

                if (state.capturedPreviewUrl) {
                    URL.revokeObjectURL(state.capturedPreviewUrl);
                }

                state.capturedPreviewUrl = URL.createObjectURL(blob);

                elements.cameraPreview.src = state.capturedPreviewUrl;
                elements.cameraPreview.hidden = false;
                elements.cameraVideo.hidden = true;
                elements.liveCameraActions.hidden = true;
                elements.capturedCameraActions.hidden = false;

                stopCameraStream();
            },
            "image/jpeg", 0.95
        );
    }

    async function retakePhoto() {
        state.capturedBlob = null;

        elements.cameraPreview.hidden = true;
        elements.cameraVideo.hidden = false;
        elements.liveCameraActions.hidden = false;
        elements.capturedCameraActions.hidden = true;

        await openCamera();
    }

    async function useCapturedPhoto() {
        if (!state.capturedBlob) {
            return;
        }

        const file = new File(
            [state.capturedBlob],
            `rice-leaf-${Date.now()}.jpg`,
            {
                type: "image/jpeg"
            }
        );

        closeCamera();

        await app.actions.handleCandidateImage(file, "camera"
        );
    }

    // Camera popup buttons
    elements.closeCameraPopup.addEventListener("click", closeCamera);
    elements.cancelCameraBtn.addEventListener("click", closeCamera);
    elements.capturePhotoBtn.addEventListener("click", capturePhoto);
    elements.retakePhotoBtn.addEventListener("click", retakePhoto);
    elements.usePhotoBtn.addEventListener("click", useCapturedPhoto);

    // Closes the camera popup when the user clicks the dark area outside the popup
    elements.cameraPopupOverlay.addEventListener("click", (event) => {
            if (event.target === elements.cameraPopupOverlay) {
                closeCamera();
            }
        }
    );

    // IMAGE-QUALITY POPUP BUTTONS
    elements.replaceImageBtn.addEventListener("click", () => {
            closeQualityPopup();

            state.pendingFile = null;

            if (state.pendingSource === "camera") {
                openCamera();
            } else {
                elements.fileInput.value = "";
                elements.fileInput.click();
            }
        }
    );

    elements.continueImageBtn.addEventListener("click", async () => {
            const file = state.pendingFile;

            closeQualityPopup();

            if (file) {
                await app.actions.submitImage(file);
            }
        }
    );

    // CONTACT US POPUP
    elements.contactUsBtn.addEventListener("click", () => {
            elements.contactPopupOverlay.classList.add("active");
        }
    );

    elements.closeContactPopup.addEventListener("click", () => {
            elements.contactPopupOverlay.classList.remove("active");
        }
    );

    // Closes the Contact Us popup when the user clicks outside the popup content
    elements.contactPopupOverlay.addEventListener("click", (event) => {
            if (event.target === elements.contactPopupOverlay) {
                elements.contactPopupOverlay.classList.remove("active");
            }
        }
    );

    // Escape key closes all popups controlled here
    document.addEventListener("keydown", (event) => {
        if (event.key !== "Escape") {
            return;
        }

        elements.contactPopupOverlay.classList.remove("active");

        closeQualityPopup();
        closeCamera();
    });

    // Makes popup and camera functions available to the other modules
    app.actions.openCamera = openCamera;
    app.actions.stopCameraStream = stopCameraStream;
    app.actions.closeCamera = closeCamera;
    app.actions.capturePhoto = capturePhoto;
    app.actions.retakePhoto = retakePhoto;
    app.actions.useCapturedPhoto = useCapturedPhoto;
    app.actions.showQualityPopup = showQualityPopup;
    app.actions.closeQualityPopup = closeQualityPopup;
}