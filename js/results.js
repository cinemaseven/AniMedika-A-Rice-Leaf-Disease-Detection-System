export function initResults(app) {
    const { elements, state } = app;

    // Prevents text returned by the backend from being interpreted as HTML
    function escapeHtml(value) {
        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    // Displays the trusted source titles returned by the backend as clickable links
    function renderMoreInformation(sources = []) {
        elements.moreInformation.replaceChildren();

        if (!Array.isArray(sources) || sources.length === 0) {
            elements.moreInformation.hidden = true;
            return;
        }

        const sourceList = document.createElement("ol");
        sourceList.className = "more-information-list";

        sources.forEach((source) => {
            if (!source?.title || !source?.url) {
                return;
            }

            let parsedUrl;

            try {
                parsedUrl = new URL(source.url);
            } catch (_error) {
                return;
            }

            if (!["http:", "https:"].includes(parsedUrl.protocol)) {
                return;
            }

            const listItem = document.createElement("li");
            const link = document.createElement("a");
            link.href = parsedUrl.href;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = source.title;

            listItem.append(link);
            sourceList.append(listItem);
        });

        if (sourceList.children.length === 0) {
            elements.moreInformation.hidden = true;
            return;
        }

        const label = document.createElement("span");
        label.className = "more-information-label";
        label.textContent = "More information:";

        elements.moreInformation.append(label, sourceList);
        elements.moreInformation.hidden = false;
    }

    // Displays the loading state while the image is being processed
    function showResultsLoading() {
        state.displayedError = null;

        elements.resultsSection.style.display = "block";
        elements.resultStatus.className = "result-status loading";
        elements.resultStatus.textContent = app.text().processing;
        elements.diseaseName.textContent = app.text().evaluating;
        elements.diseaseDescription.textContent = app.text().awaiting;
        elements.recommendationList.innerHTML = `<li>${escapeHtml(app.text().awaitingRecommendations)}</li>`;
        renderMoreInformation();
        elements.confidenceValue.textContent = "--%";
        elements.confidenceText.textContent = "";
        elements.resultSeason.textContent = "";
        elements.gaugeContainer.style.setProperty("--fill-deg", "0deg");
    }

    // Displays prediction or server errors.
    function showError(message, modelNotReady = false) {
        state.displayedError = {modelNotReady, message};

        elements.resultsSection.style.display = "block";
        elements.resultStatus.className = "result-status error";
        elements.resultStatus.textContent = message;
        elements.confidenceValue.textContent = "--%";
        elements.confidenceText.textContent = "";
        renderMoreInformation();
        elements.gaugeContainer.style.setProperty("--fill-deg", "0deg");

        const season = app.actions.updateSeasonDisplay();

        const seasonLabel = season === "wet" ? app.text().wetSeason : app.text().drySeason;

        elements.resultSeason.textContent = `${app.text().selectedSeason}: ` + `${seasonLabel}`;

        // Special error shown when the backend is running but the AI model has not been loaded
        if (modelNotReady) {
            elements.diseaseName.textContent = app.text().modelNotReadyTitle;
            elements.diseaseDescription.textContent = app.text().modelNotReadyDescription;
            elements.recommendationList.innerHTML = `<li>${escapeHtml(app.text().modelNotReadyRecommendation)}</li>`;

            return;
        }

        elements.diseaseName.textContent = app.text().requestFailed;
        elements.diseaseDescription.textContent = message;
        elements.recommendationList.innerHTML = "";
    }

    // Displays a successful prediction returned by the backend
    function renderResult(result) {
        state.displayedError = null;

        const localized = result.localized_result?.[state.currentLanguage];

        if (!localized) {
            showError(app.text().requestFailed);
            return;
        }

        const confidencePercent = Number(result.prediction.confidence) * 100;
        const safeConfidence = Number.isFinite(confidencePercent) ? confidencePercent : 0;
        const seasonLabel = result.context.season === "wet" ? app.text().wetSeason : app.text().drySeason;

        elements.resultStatus.className = "result-status success";
        elements.resultStatus.textContent = "";
        elements.diseaseName.textContent = localized.name;
        // elements.diseaseDescription.textContent = localized.description;
        renderDescription(localized.description);
        elements.confidenceValue.textContent = `${safeConfidence.toFixed(1)}%`;
        elements.confidenceText.textContent = `${app.text().confidence}: ` + `${safeConfidence.toFixed(1)}%`;
        elements.resultSeason.textContent = `${app.text().selectedSeason}: ` + `${seasonLabel}`;

        // Updates the confidence gauge. 100% confidence corresponds to 180 degrees
        elements.gaugeContainer.style.setProperty(
            "--fill-deg",
            `${Math.max(0, Math.min(180, safeConfidence * 1.8))}deg`
        );

        if (elements.seasonNote) {
            elements.seasonNote.textContent = localized.season_note || "";
            elements.seasonNote.hidden = !localized.season_note;
        }

        const recommendations = [...localized.general_recommendations];

        elements.recommendationList.innerHTML = recommendations
            .map((item) => `<li>${escapeHtml(item)}</li>`)
            .join("");

        renderMoreInformation(localized.sources);
    }

    function renderDescription(description) {
        const paragraphs = Array.isArray(description)
            ? description
            : [description];

        elements.diseaseDescription.replaceChildren();

        paragraphs
            .filter(
                (paragraph) =>
                    paragraph !== null &&
                    paragraph !== undefined &&
                    String(paragraph).trim()
            )
            .forEach((paragraph, index) => {
                if (index > 0) {
                    elements.diseaseDescription.append(
                        document.createElement("br"),
                        document.createElement("br")
                    );
                }

                elements.diseaseDescription.append(
                    document.createTextNode(String(paragraph))
                );
            });
    }

    // Sends the uploaded or captured image to the Flask backend
    async function submitImage(file, { scroll = true } = {}) {
        state.lastSubmittedFile = file;
        state.lastResult = null;

        showResultsLoading();

        if (scroll) {
            elements.resultsSection.scrollIntoView({behavior: "smooth",block: "start"});
        }

        const formData = new FormData();

        formData.append(
            "image", file,
            file.name || `rice-leaf-${Date.now()}.jpg`
        );

        formData.append("selected_date", elements.datePicker.value);

        formData.append("language", state.currentLanguage);

        // Helps ignore an older request if the user submits another image before it finishes
        const requestId = ++state.requestSequence;

        try {
            const response = await fetch(
                "/api/predict",
                {
                    method: "POST",
                    body: formData
                }
            );

            const payload =
                await response
                    .json()
                    .catch(() => ({}));

            if (requestId !== state.requestSequence) {
                return;
            }

            if (!response.ok) {
                if (response.status === 503 && payload.code === "MODEL_NOT_READY") {
                    showError(
                        payload.error || app.text().modelNotReadyTitle,
                        true
                    );

                    return;
                }

                throw new Error(payload.error || app.text().requestFailed);
            }

            state.lastResult = payload;

            renderResult(payload);
        } catch (error) {
            console.error("Prediction request failed:", error
            );

            showError(error.message || app.text().requestFailed);
        }
    }

    // Makes the result functions available to the other modules
    app.actions.escapeHtml = escapeHtml;
    app.actions.showResultsLoading = showResultsLoading;
    app.actions.showError = showError;
    app.actions.renderResult = renderResult;
    app.actions.submitImage = submitImage;
}