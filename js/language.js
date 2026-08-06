export function initLanguage(app, WEBSITE_TEXT) {
    const { elements, state } = app;

    // Applies the selected language to the website
    function applyLanguage(language) {
        const text = WEBSITE_TEXT[language];

        if (!text) {
            return;
        }

        state.currentLanguage = language;

        // Language selector
        elements.currentLanguageFlag.textContent = text.flag;
        elements.currentLanguageText.textContent = text.label;

        // Website title
        document.querySelector(".site-title").textContent = text.siteTitle;

        // About section
        document.querySelector(".about-header h2").textContent = text.aboutTitle;
        document.querySelector(".about-text p").textContent = text.aboutShort;
        document.querySelectorAll("#extendedAboutText p").forEach((paragraph, index) => {
                if (text.aboutLong[index]) {
                    paragraph.textContent = text.aboutLong[index];
                }
            });

        // Instructions section
        document.querySelector(".instructions-card h2").textContent = text.howToUse;
        document.querySelectorAll(".instructions-card li").forEach((item, index) => {
                if (text.steps[index]) {
                    item.innerHTML = text.steps[index];
                }
            });

        // Upload section
        document.querySelector(".upload-card h2").textContent = text.selectImageSource;

        elements.galleryBtn.textContent = text.gallery;
        elements.cameraBtn.textContent = text.camera;

        // Results section 
        document.querySelector(".results-header span").textContent = text.results;
        document.querySelector(".recommendation-card h2").textContent = text.recommendedActions;
        document.querySelector(".temporary-storage-note").textContent = text.storageNote;
        document.querySelector(".disclaimer").innerHTML = text.disclaimer;

        // Footer
        elements.contactUsBtn.textContent = text.contactUs;

        document.querySelector(".footer-left p").textContent = text.backToTop;
        document.querySelector(".copyright").textContent = text.copyright;

        // Initial language popup
        // document.querySelector(".language-popup h2").textContent = text.popupTitle;
        // document.querySelector(".language-popup p").textContent = text.popupText;
        document.querySelector(".language-popup-primary-title").textContent = WEBSITE_TEXT.fil.popupTitle;
        document.querySelector(".language-popup-primary-text").textContent = WEBSITE_TEXT.fil.popupText;
        document.querySelector(".language-popup-title-translation").textContent = WEBSITE_TEXT.en.popupTitle;
        document.querySelector(".language-popup-text-translation").textContent = WEBSITE_TEXT.en.popupText;

        // Camera popup
        document.getElementById("cameraPopupTitle").textContent = text.cameraTitle;
        document.getElementById("cameraGuidance").textContent = text.cameraGuidance;

        elements.cancelCameraBtn.textContent = text.cancel;
        elements.capturePhotoBtn.textContent = text.capture;
        elements.retakePhotoBtn.textContent = text.retake;
        elements.usePhotoBtn.textContent = text.usePhoto;

        // Image-quality warning popup
        document.getElementById("qualityPopupTitle").textContent = text.qualityTitle;
        document.getElementById("qualityPopupText").textContent = text.qualityText;
        elements.replaceImageBtn.textContent = text.replaceImage;
        elements.continueImageBtn.textContent = text.continueAnyway;

        // About section button
        const expandedAbout = document.getElementById("extendedAboutText");

        document.querySelector("#toggleAboutBtn .btn-text").textContent = expandedAbout.style.display === "flex" ? text.showLess : text.showMore;

        // Updates the season text to the newly selected language
        app.actions.updateSeasonDisplay();

        // Re-renders the previous prediction result when the language changes
        if (state.lastResult) {
            app.actions.renderResult(state.lastResult);
        } else if (
            state.displayedError?.modelNotReady
        ) {
            app.actions.showError(text.modelNotReadyTitle, true);
        } else if (state.displayedError) {
            app.actions.showError(text.requestFailed);
        }
    }

    // Applies the language and closes both language selection popups
    function chooseLanguage(language) {
        applyLanguage(language);

        elements.languagePopupOverlay.classList.remove("active");
        elements.languageMenu.classList.remove("active");
    }

    // Opens or closes the language dropdown
    elements.languageToggleBtn.addEventListener("click", (event) => {
            event.stopPropagation();

            elements.languageMenu.classList.toggle("active");
        }
    );

    // Language buttons inside the initial popup
    elements.languageOptionBtns.forEach(
        (button) => {
            button.addEventListener("click", () => {
                    chooseLanguage(button.dataset.lang);
                }
            );
        }
    );

    // Language buttons inside the dropdown menu
    elements.languageMenuBtns.forEach(
        (button) => {
            button.addEventListener("click", () => {
                    chooseLanguage(button.dataset.lang);
                }
            );
        }
    );

    // Closes the language menu when the user clicks outside it
    document.addEventListener("click", (event) => {
            const clickedOutsideMenu = !elements.languageMenu.contains(event.target);
            const clickedOutsideButton = !elements.languageToggleBtn.contains(event.target);

            if (clickedOutsideMenu &&clickedOutsideButton) {
                elements.languageMenu.classList.remove("active");
            }
        }
    );

    // Closes the language dropdown using Escape
    document.addEventListener("keydown", (event) => {
            if (event.key === "Escape") {
                elements.languageMenu.classList.remove("active");
            }
        }
    );

    // Makes language functions available to the other JavaScript files
    app.actions.applyLanguage = applyLanguage;
    app.actions.chooseLanguage = chooseLanguage;
}