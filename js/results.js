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

    // Displays NLP-extracted key phrases
    function renderKeyPhrases(keyPhrases = []) {
        elements.keyPhraseList.replaceChildren();

        if (!Array.isArray(keyPhrases) || keyPhrases.length === 0) {
            elements.keyPhrases.hidden = true;
            return;
        }

        elements.keyPhrasesLabel.textContent = app.text().keySigns;

        keyPhrases.forEach((phrase) => {
            const listItem = document.createElement("li");
            listItem.textContent = phrase;
            elements.keyPhraseList.append(listItem);
        });

        elements.keyPhrases.hidden = false;
    }

     // Displays recommended actions and makes only configured rice variety names clickable
    function renderRecommendations(recommendations = []) {
        elements.recommendationList.replaceChildren();

        recommendations.forEach((item) => {
            const listItem = document.createElement("li");

            if (typeof item === "string") {
                listItem.textContent = item;
                elements.recommendationList.append(listItem);
                return;
            }

            if (!item?.text) {
                return;
            }

            const text = String(item.text);
            const links = Array.isArray(item.links) ? item.links : [];
            const matches = [];

            links.forEach((source) => {
                if (!source?.text || !source?.url) {
                    return;
                }

                const position = text.indexOf(source.text);

                if (position === -1) {
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

                matches.push({
                    start: position,
                    end: position + source.text.length,
                    text: source.text,
                    url: parsedUrl.href
                });
            });

            matches.sort((a, b) => a.start - b.start);

            if (matches.length === 0) {
                listItem.textContent = text;
                elements.recommendationList.append(listItem);
                return;
            }

            let cursor = 0;

            matches.forEach((match) => {
                if (match.start < cursor) {
                    return;
                }

                listItem.append(
                    document.createTextNode(text.slice(cursor, match.start))
                );

                const link = document.createElement("a");
                link.href = match.url;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                link.className = "recommendation-variety-link";
                link.textContent = match.text;

                listItem.append(link);
                cursor = match.end;
            });

            listItem.append(document.createTextNode(text.slice(cursor)));
            elements.recommendationList.append(listItem);
        });
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
        elements.diseaseName.classList.remove("analysis-error");
        elements.diseaseName.textContent = app.text().evaluating;
        elements.diseaseDescription.textContent = app.text().awaiting;
        renderKeyPhrases();

        const recommendationCard = elements.recommendationList.closest(".recommendation-card");

        if (recommendationCard) {
            recommendationCard.hidden = false;
        }

        elements.recommendationList.innerHTML = `<li>${escapeHtml(app.text().awaitingRecommendations)}</li>`;
        renderMoreInformation();
        elements.resultSeason.textContent = "";
    }

    // Displays prediction or server errors.
    function showError(message, modelNotReady = false) {
        state.displayedError = { modelNotReady, message };

        elements.resultsSection.style.display = "block";
        elements.resultStatus.className = "result-status error";
        elements.resultStatus.textContent = "";
        elements.resultSeason.textContent = "";

        renderKeyPhrases();
        renderMoreInformation();

        if (elements.seasonNote) {
            elements.seasonNote.textContent = "";
            elements.seasonNote.hidden = true;
        }

        const recommendationCard =
            elements.recommendationList.closest(".recommendation-card");

        // Special error shown when the backend is running but the AI model has not been loaded
        if (modelNotReady) {
            elements.diseaseName.classList.add("analysis-error");
            elements.diseaseName.textContent = app.text().modelNotReadyTitle;
            elements.diseaseDescription.textContent =
                app.text().modelNotReadyDescription;

            if (recommendationCard) {
                recommendationCard.hidden = false;
            }

            elements.recommendationList.innerHTML =
                `<li>${escapeHtml(app.text().modelNotReadyRecommendation)}</li>`;

            return;
        }

        // Generic analysis error
        elements.diseaseName.classList.add("analysis-error");
        elements.diseaseName.textContent = app.text().requestFailed;
        elements.diseaseDescription.textContent = "";

        if (recommendationCard) {
            recommendationCard.hidden = true;
        }

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

        const seasonLabel = result.context.season === "wet" ? app.text().wetSeason : app.text().drySeason;

        elements.resultStatus.className = "result-status success";
        elements.resultStatus.textContent = "";
        elements.diseaseName.classList.remove("analysis-error");
        elements.diseaseName.textContent = localized.name;

        const recommendationCard = elements.recommendationList.closest(".recommendation-card");

        if (recommendationCard) {
            recommendationCard.hidden = false;
        }
        renderDescription(localized.description);
        renderKeyPhrases(localized.key_phrases);

        elements.resultSeason.textContent = `${app.text().selectedSeason}: ` + `${seasonLabel}`;

        if (elements.seasonNote) {
            elements.seasonNote.textContent = localized.season_note || "";
            elements.seasonNote.hidden = !localized.season_note;
        }

        const recommendations = [...localized.general_recommendations];

        renderRecommendations(recommendations);

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