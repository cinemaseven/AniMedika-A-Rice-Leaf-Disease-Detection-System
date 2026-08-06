export function initImageUpload(app) {
    const { elements, state, constants } = app;

    const { MAX_FILE_BYTES, QUALITY_THRESHOLDS } = constants;

    // Displays the selected image in the results section
    function setPreview(file) {
        if (state.currentPreviewUrl) {
            URL.revokeObjectURL(state.currentPreviewUrl);
        }

        state.currentPreviewUrl = URL.createObjectURL(file);
        elements.previewImage.src = state.currentPreviewUrl;
    }

    // Checks the image type and file size
    function validateFile(file) {
        const allowedTypes = new Set([
            "image/jpeg",
            "image/png",
            "image/webp"
        ]);

        if (!file || !allowedTypes.has(file.type)) {
            alert(app.text().invalidImage);
            return false;
        }

        if (file.size > MAX_FILE_BYTES) {
            alert(app.text().imageTooLarge);
            return false;
        }

        return true;
    }

    // Loads an uploaded image so it can be checked
    async function loadImageForCanvas(file) {
        if ("createImageBitmap" in window) {
            return createImageBitmap(file, {imageOrientation: "from-image"});
        }

        return new Promise((resolve, reject) => {
            const image = new Image();
            const url = URL.createObjectURL(file);

            image.onload = () => {
                URL.revokeObjectURL(url);
                resolve(image);
            };

            image.onerror = () => {
                URL.revokeObjectURL(url);

                reject(
                    new Error("Image could not be decoded.")
                );
            };

            image.src = url;
        });
    }

    // Checks the brightness and blur level of the image
    async function inspectImageQuality(file) {
        const image = await loadImageForCanvas(file);
        const scale = Math.min(1, QUALITY_THRESHOLDS.analysisMaxSide / Math.max(image.width, image.height));
        const width = Math.max(3, Math.round(image.width * scale));
        const height = Math.max(3, Math.round(image.height * scale));
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d", {willReadFrequently: true});

        canvas.width = width;
        canvas.height = height;

        context.drawImage(
            image,
            0,
            0,
            width,
            height
        );

        if (typeof image.close === "function") {
            image.close();
        }

        const pixels = context.getImageData(
            0,
            0,
            width,
            height
        ).data;

        const gray = new Float32Array(width * height);

        let brightnessSum = 0;

        // Converts every pixel to grayscale and calculates the image's average brightness
        for (
            let pixelIndex = 0, grayIndex = 0;
            pixelIndex < pixels.length;
            pixelIndex += 4, grayIndex += 1
        ) {
            const value =
                0.299 * pixels[pixelIndex] +
                0.587 * pixels[pixelIndex + 1] +
                0.114 * pixels[pixelIndex + 2];

            gray[grayIndex] = value;
            brightnessSum += value;
        }

        let laplacianSum = 0;
        let laplacianSquareSum = 0;
        let count = 0;

        // Uses Laplacian variance to estimate whether the image is blurry.
        for (let y = 1; y < height - 1; y += 1) {
            for (let x = 1; x < width - 1; x += 1) {
                const index = y * width + x;
                const laplacian =
                    4 * gray[index] -
                    gray[index - 1] -
                    gray[index + 1] -
                    gray[index - width] -
                    gray[index + width];

                laplacianSum += laplacian;
                laplacianSquareSum +=
                    laplacian * laplacian;

                count += 1;
            }
        }

        const brightness = brightnessSum / gray.length;
        const laplacianMean = count ? laplacianSum / count : 0;
        const blurVariance = count ? laplacianSquareSum / count - laplacianMean ** 2 : 0;
        const warnings = [];

        if (brightness < QUALITY_THRESHOLDS.minBrightness) {
            warnings.push("tooDark");
        }

        if (brightness > QUALITY_THRESHOLDS.maxBrightness) {
            warnings.push("tooBright");
        }

        if (blurVariance < QUALITY_THRESHOLDS.minBlurVariance) {
            warnings.push("tooBlurry");
        }

        return {brightness, blurVariance, warnings};
    }

    // Processes an image selected from the gallery, drag-and-drop, or camera
    async function handleCandidateImage(file, source) {
        if (!validateFile(file)) {
            return;
        }

        state.pendingFile = file;
        state.pendingSource = source;

        setPreview(file);

        try {
            const quality = await inspectImageQuality(file);

            if (quality.warnings.length) {
                app.actions.showQualityPopup(quality);
                return;
            }
        } catch (error) {
            console.warn("Image quality check was skipped:", error);
        }

        await app.actions.submitImage(file);
    }

    // Opens the device's file picker
    elements.galleryBtn.addEventListener("click", () => {
        elements.fileInput.value = "";
        elements.fileInput.click();
    });

    // Processes the image selected from the file picker
    elements.fileInput.addEventListener("change", () => {
        const selectedFile = elements.fileInput.files[0];

        if (selectedFile) {
            handleCandidateImage(selectedFile, "gallery");
        }
    });

    // Allows an image to be dragged over the upload area
    elements.dropZone.addEventListener("dragover",
        (event) => {event.preventDefault();}
    );

    // Processes an image dropped into the upload area
    elements.dropZone.addEventListener("drop",
        (event) => {event.preventDefault();
            const droppedFile = event.dataTransfer.files[0];

            if (droppedFile) {
                handleCandidateImage(droppedFile, "gallery");
            }
        }
    );

    // Opens the camera popup when the camera button is clicked
    elements.cameraBtn.addEventListener("click", () => {
        app.actions.openCamera();
    });

    // Makes these functions available to the other modules
    app.actions.setPreview = setPreview;
    app.actions.validateFile = validateFile;
    app.actions.inspectImageQuality = inspectImageQuality;
    app.actions.handleCandidateImage = handleCandidateImage;
}