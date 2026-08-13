import { WEBSITE_TEXT } from "./translations.js";
import { initDate } from "./date.js";
import { initImageUpload } from "./image-upload.js";
import { initLanguage } from "./language.js";
import { initPopups } from "./popups.js";
import { initResults } from "./results.js";

document.addEventListener("DOMContentLoaded", () => {

    const elements = {
        // Image upload
        dropZone: document.getElementById("dropZone"),
        fileInput: document.getElementById("leafFileInput"),
        galleryBtn: document.getElementById("customUploadBtn"),
        cameraBtn: document.getElementById("cameraUploadBtn"),

        // Date and season
        dateInput: document.getElementById("date-display"),
        datePicker: document.getElementById("date-picker"),
        calendarBox: document.querySelector(".calendar"),
        seasonDisplay: document.getElementById("seasonDisplay"),

        // Results
        resultsSection: document.getElementById("resultsSection"),
        previewImage: document.getElementById("previewImage"),
        resultStatus: document.getElementById("resultStatus"),
        resultSeason: document.getElementById("resultSeason"),
        diseaseName: document.getElementById("diseaseName"),
        diseaseDescription: document.getElementById("diseaseDescription"),
        keyPhrases: document.getElementById("keyPhrases"),
        keyPhrasesLabel: document.getElementById("keyPhrasesLabel"),
        keyPhraseList: document.getElementById("keyPhraseList"),
        seasonNote: document.getElementById("seasonNote"),
        recommendationList: document.getElementById("recommendationList"),
        moreInformation: document.getElementById("moreInformation"),

        // Language selector
        languagePopupOverlay: document.getElementById("languagePopupOverlay"),
        languageOptionBtns: document.querySelectorAll(".language-option-btn"),
        languageToggleBtn: document.getElementById("languageToggleBtn"),
        languageMenu: document.getElementById("languageMenu"),
        languageMenuBtns: document.querySelectorAll(".language-menu-btn"),
        currentLanguageFlag: document.getElementById("currentLanguageFlag"),
        currentLanguageText: document.getElementById("currentLanguageText"),

        // Contact Us popup
        contactUsBtn: document.getElementById("contactUsBtn"),
        contactPopupOverlay: document.getElementById("contactPopupOverlay"),
        closeContactPopup: document.getElementById("closeContactPopup"),

        // Camera popup
        cameraPopupOverlay: document.getElementById("cameraPopupOverlay"),
        cameraVideo: document.getElementById("cameraVideo"),
        cameraPreview: document.getElementById("cameraPreview"),
        cameraCanvas: document.getElementById("cameraCanvas"),
        cameraError: document.getElementById("cameraError"),
        closeCameraPopup: document.getElementById("closeCameraPopup"),
        cancelCameraBtn: document.getElementById("cancelCameraBtn"),
        capturePhotoBtn: document.getElementById("capturePhotoBtn"),
        retakePhotoBtn: document.getElementById("retakePhotoBtn"),
        usePhotoBtn: document.getElementById("usePhotoBtn"),
        liveCameraActions: document.getElementById("liveCameraActions"),
        capturedCameraActions: document.getElementById("capturedCameraActions"),

        // Image-quality popup
        qualityPopupOverlay: document.getElementById("qualityPopupOverlay"),
        qualityWarningList: document.getElementById("qualityWarningList"),
        qualityMetrics: document.getElementById("qualityMetrics"),
        replaceImageBtn: document.getElementById("replaceImageBtn"),
        continueImageBtn: document.getElementById("continueImageBtn")
    };

    // Shared website state. These values can be accessed and updated by all modules
    const state = {
        currentLanguage: "fil",
        currentPreviewUrl: null,
        cameraStream: null,
        capturedBlob: null,
        capturedPreviewUrl: null,
        pendingFile: null,
        pendingSource: "gallery",
        lastSubmittedFile: null,
        lastResult: null,
        displayedError: null,
        requestSequence: 0
    };

    // Shared application object
    const app = {elements, state,
        constants: {MAX_FILE_BYTES: 10 * 1024 * 1024,
            QUALITY_THRESHOLDS: {
                minBrightness: 45,
                maxBrightness: 225,
                minBlurVariance: 60,
                analysisMaxSide: 360
            }
        },

        text() {
            return WEBSITE_TEXT[state.currentLanguage];
        },

        actions: {}
    };

    // Initialize the modules
    initResults(app);
    initPopups(app);
    initImageUpload(app);
    initDate(app);
    initLanguage(app, WEBSITE_TEXT);
    initAboutSection(app);

    // Initial website settings
    app.actions.setDateToToday();
    app.actions.applyLanguage("fil");
    elements.languagePopupOverlay.classList.add("active");

    // Clean up temporary browser resources before leaving the page
    window.addEventListener("beforeunload", () => {
        app.actions.stopCameraStream();

        if (state.currentPreviewUrl) {
            URL.revokeObjectURL(state.currentPreviewUrl);
        }

        if (state.capturedPreviewUrl) {
            URL.revokeObjectURL(state.capturedPreviewUrl);
        }
    });
});

// About section Show More / Show Less button
function initAboutSection(app) {
    const toggleButton = document.getElementById("toggleAboutBtn");
    const content = document.getElementById("extendedAboutText");
    const buttonText = document.querySelector("#toggleAboutBtn .btn-text");
    const icon = document.querySelector("#toggleAboutBtn .material-icons");

    toggleButton.addEventListener("click", () => {
        const shouldShow = !content.classList.contains("active");

        content.classList.toggle("active", shouldShow);
        toggleButton.classList.toggle("expanded", shouldShow);

        buttonText.textContent = shouldShow
            ? app.text().showLess
            : app.text().showMore;

        icon.textContent = shouldShow
            ? "arrow_circle_up"
            : "arrow_circle_down";
    });
}