import { getDiseaseResult } from "./results.js";
import { WEBSITE_TEXT } from "./translations.js";

document.addEventListener("DOMContentLoaded", async () => {

    const dropZone = document.getElementById("dropZone");

    const languagePopupOverlay = document.getElementById("languagePopupOverlay");
    const languageOptionBtns = document.querySelectorAll(".language-option-btn");
    const languageToggleBtn = document.getElementById("languageToggleBtn");
    const languageMenu = document.getElementById("languageMenu");
    const languageMenuBtns = document.querySelectorAll(".language-menu-btn");
    const currentLanguageFlag = document.getElementById("currentLanguageFlag");
    const currentLanguageText = document.getElementById("currentLanguageText");


    const dateInput = document.getElementById("date-display");
    const datePicker = document.getElementById("date-picker");
    const calendarBox = document.querySelector(".calendar");

    const cameraBtn = document.getElementById("cameraUploadBtn");
    const fileInput = document.getElementById("leafFileInput");
    const customBtn = document.getElementById("customUploadBtn");

    const resultsSection = document.getElementById("resultsSection");

    const previewImage = document.getElementById("previewImage");
    const confidenceValue = document.getElementById("confidenceValue");
    const confidenceText = document.getElementById("confidenceText");
    const diseaseName = document.getElementById("diseaseName");
    const diseaseDescription = document.getElementById("diseaseDescription");
    const recommendationList = document.getElementById("recommendationList");
    const gaugeContainer = document.querySelector('.confidence-overlay');

    const contactUsBtn = document.getElementById("contactUsBtn");
    const contactPopupOverlay = document.getElementById("contactPopupOverlay");
    const closeContactPopup = document.getElementById("closeContactPopup");

    // const CLASS_MAP = {
    //     0: { name: "bacterial_leaf_blight", desc: "Disease affecting rice leaves", rec: ["Remove infected leaves"] },
    //     1: { name: "brown_spot", desc: "Brown lesions on leaves", rec: ["Use fungicide"] },
    //     2: { name: "healthy_rice_plant", desc: "Plant is healthy", rec: ["Continue normal care"] },
    //     3: { name: "rice_blast", desc: "Severe fungal infection", rec: ["Apply treatment early"] },
    //     4: { name: "sheath_blight", desc: "Sheath infection", rec: ["Improve field drainage"] },
    //     5: { name: "tungro_virus", desc: "Viral disease", rec: ["Remove infected plants"] }
    // };

    let model = null;
    let selectedImageSource = "gallery";
    let currentLanguage = "fil";
    let lastPrediction = null;

    function getTodayDateObject() {
        return new Date();
    }

    function toDatePickerValue(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, "0");
        const day = String(date.getDate()).padStart(2, "0");

        return `${year}-${month}-${day}`;
    }

    function toDisplayDate(value) {
        const [year, month, day] = value.split("-");
        return `${month}/${day}/${year}`;
    }

    function setCalendarToToday() {
        const today = getTodayDateObject();
        const pickerValue = toDatePickerValue(today);

        if (datePicker) {
            datePicker.value = pickerValue;
        }

        if (dateInput) {
            dateInput.value = toDisplayDate(pickerValue);
        }
    }

    setCalendarToToday();

    if (datePicker && dateInput) {
        datePicker.addEventListener("change", () => {
            dateInput.value = toDisplayDate(datePicker.value);
        });
    }

    if (calendarBox && datePicker) {
        calendarBox.addEventListener("click", () => {
            if (datePicker.showPicker) {
                datePicker.showPicker();
            } else {
                datePicker.click();
            }
        });
    }

    function applyLanguage(lang) {
        const t = WEBSITE_TEXT[lang];

        if (!t) return;

        currentLanguage = lang;

        if (currentLanguageFlag) currentLanguageFlag.textContent = t.flag;
        if (currentLanguageText) currentLanguageText.textContent = t.label;

        const siteTitle = document.querySelector(".site-title");
        if (siteTitle) siteTitle.textContent = t.siteTitle;

        const aboutTitle = document.querySelector(".about-header h2");
        if (aboutTitle) aboutTitle.textContent = t.aboutTitle;

        const aboutShort = document.querySelector(".about-text p");
        if (aboutShort) aboutShort.textContent = t.aboutShort;

        const aboutLongParagraphs = document.querySelectorAll("#extendedAboutText p");
        aboutLongParagraphs.forEach((p, index) => {
            if (t.aboutLong[index]) p.textContent = t.aboutLong[index];
        });

        const textSpan = document.querySelector("#toggleAboutBtn .btn-text");
        const extendedAboutText = document.getElementById("extendedAboutText");

        if (textSpan && extendedAboutText) {
            textSpan.textContent = extendedAboutText.style.display === "flex" ? t.showLess : t.showMore;
        }

        const instructionsTitle = document.querySelector(".instructions-card h2");
        if (instructionsTitle) instructionsTitle.textContent = t.howToUse;

        const instructionItems = document.querySelectorAll(".instructions-card li");
        instructionItems.forEach((li, index) => {
            if (t.steps[index]) li.innerHTML = t.steps[index];
        });

        const uploadTitle = document.querySelector(".upload-card h2");
        if (uploadTitle) uploadTitle.textContent = t.selectImageSource;

        if (customBtn) customBtn.textContent = t.gallery;
        if (cameraBtn) cameraBtn.textContent = t.camera;

        const resultsHeader = document.querySelector(".results-header span");
        if (resultsHeader) resultsHeader.textContent = t.results;

        const recommendationTitle = document.querySelector(".recommendation-card h2");
        if (recommendationTitle) recommendationTitle.textContent = t.recommendedActions;

        const disclaimer = document.querySelector(".disclaimer");
        if (disclaimer) disclaimer.innerHTML = t.disclaimer;

        const contactTitle = document.getElementById("contactUsBtn");
        if (contactTitle) contactTitle.textContent = t.contactUs;

        const backToTop = document.querySelector(".footer-left p");
        if (backToTop) backToTop.textContent = t.backToTop;

        const copyright = document.querySelector(".copyright");
        if (copyright) copyright.textContent = t.copyright;

        const footerLinks = document.querySelectorAll(".footer-links a");
        if (footerLinks[0]) footerLinks[0].textContent = t.advertisement;
        if (footerLinks[1]) footerLinks[1].textContent = t.privacy;
        if (footerLinks[2]) footerLinks[2].textContent = t.visitor;

        const languagePopupTitle = document.querySelector(".language-popup h2");
        if (languagePopupTitle) languagePopupTitle.textContent = t.popupTitle;

        const languagePopupText = document.querySelector(".language-popup p");
        if (languagePopupText) languagePopupText.textContent = t.popupText;

        const contactPopupTitle = document.querySelector(".contact-popup h2");
        if (contactPopupTitle) contactPopupTitle.textContent = t.contactUs;

        if (lastPrediction) {
            updateUI(lastPrediction.index, lastPrediction.score);
        }
    }

    function chooseLanguage(lang) {
        applyLanguage(lang);

        if (languagePopupOverlay) {
            languagePopupOverlay.classList.remove("active");
        }

        if (languageMenu) {
            languageMenu.classList.remove("active");
        }
    }

    applyLanguage("fil");

    if (languagePopupOverlay) {
        languagePopupOverlay.classList.add("active");
    }

    if (languageToggleBtn && languageMenu) {
        languageToggleBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            languageMenu.classList.toggle("active");
        });
    }

    languageOptionBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            chooseLanguage(btn.dataset.lang);
        });
    });

    languageMenuBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            chooseLanguage(btn.dataset.lang);
        });
    });

    document.addEventListener("click", (e) => {
        if (
            languageMenu &&
            languageToggleBtn &&
            !languageMenu.contains(e.target) &&
            !languageToggleBtn.contains(e.target)
        ) {
            languageMenu.classList.remove("active");
        }
    });

    async function loadModel() {
        try {
            console.log("Loading model...");

            if (!window.tf) {
                throw new Error("TensorFlow.js not found. Check CDN script.");
            }

            // IMPORTANT: use absolute path to avoid 404 confusion
            model = await tf.loadLayersModel('/model_web/model.json');

            console.log("Model loaded successfully");
        } catch (err) {
            console.error("MODEL LOAD FAILED:", err);
            alert("Model failed to load. Check model_web folder and model.json path.");
        }
    }

    await loadModel();

    async function processImage(file) {

        if (!file.type.startsWith("image/")) {
            alert("Please upload an image file.");
            return;
        }

        previewImage.src = URL.createObjectURL(file);
        resultsSection.style.display = "block";

        // diseaseName.textContent = "Analyzing...";
        // diseaseDescription.textContent = "Processing image...";
        diseaseName.innerHTML = `${WEBSITE_TEXT[currentLanguage].evaluating} <span class="material-icons">info</span>`;
        diseaseDescription.textContent = WEBSITE_TEXT[currentLanguage].processing;

        previewImage.onload = async () => {

            if (!model) {
                alert("Model not loaded.");
                return;
            }

            const tensor = tf.browser.fromPixels(previewImage)
                .resizeNearestNeighbor([224, 224])
                .toFloat()
                .div(255.0)
                .expandDims();

            const prediction = await model.predict(tensor).data();

            const maxIndex = prediction.indexOf(Math.max(...prediction));
            const confidence = (prediction[maxIndex] * 100).toFixed(1);

            updateUI(maxIndex, confidence);
        };
    }

    // function updateUI(index, score) {

    //     const data = CLASS_MAP[index] || {
    //         name: "Unknown",
    //         desc: "No data available",
    //         rec: []
    //     };

    //     if (confidenceValue) confidenceValue.textContent = score + "%";
    //     if (confidenceText) confidenceText.textContent = score + "%";

    //     diseaseName.textContent = data.name;
    //     diseaseDescription.textContent = data.desc;

    //     recommendationList.innerHTML = data.rec.map(r => `<li>${r}</li>`).join("");

    //     if (gaugeContainer) {
    //         gaugeContainer.style.setProperty('--fill-deg', (score / 100) * 180 + 'deg');
    //     }
    // }

    function updateUI(index, score) {

        lastPrediction = { index, score };

        const data = getDiseaseResult(index, currentLanguage, dateInput.value);

        if (confidenceValue) confidenceValue.textContent = score + "%";
        if (confidenceText) confidenceText.textContent = score + "%";

        diseaseName.textContent = data.name;
        diseaseDescription.textContent = data.desc;

        recommendationList.innerHTML = data.rec.map(r => `<li>${r}</li>`).join("");

        if (gaugeContainer) {
            gaugeContainer.style.setProperty('--fill-deg', (score / 100) * 180 + 'deg');
        }
    }

    // === ABOUT CARD TOGGLE FUNCTIONALITY ===
    const toggleAboutBtn = document.getElementById("toggleAboutBtn");
    const extendedAboutText = document.getElementById("extendedAboutText");

    if (toggleAboutBtn && extendedAboutText) {
        toggleAboutBtn.addEventListener("click", () => {
            const isHidden = extendedAboutText.style.display === "none";
            
            const textSpan = toggleAboutBtn.querySelector(".btn-text");
            const iconNode = toggleAboutBtn.querySelector(".material-icons");

            if (isHidden) {
                extendedAboutText.style.display = "flex";
                // if (textSpan) textSpan.textContent = "Show less";
                if (textSpan) textSpan.textContent = WEBSITE_TEXT[currentLanguage].showLess;
                if (iconNode) iconNode.textContent = "arrow_circle_up";
            } else {
                extendedAboutText.style.display = "none";
                // if (textSpan) textSpan.textContent = "Show more";
                if (textSpan) textSpan.textContent = WEBSITE_TEXT[currentLanguage].showMore;
                if (iconNode) iconNode.textContent = "arrow_circle_down";
            }
        });
    }

    // === PHOTO UPLOAD HANDLERS ===
    if (customBtn) {
        customBtn.addEventListener("click", () => {
            selectedImageSource = "gallery";
            fileInput.removeAttribute("capture");
            fileInput.value = "";
            fileInput.click();
        });
    }

    if (cameraBtn) {
        cameraBtn.addEventListener("click", () => {
            selectedImageSource = "camera";
            fileInput.setAttribute("capture", "environment");
            fileInput.value = "";
            fileInput.click();
        });
    }

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            if (selectedImageSource === "camera") {
                setCalendarToToday();
            }

            processImage(e.target.files[0]);
        }
    });

    dropZone.addEventListener("dragover", (e) => e.preventDefault());

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();

        if (e.dataTransfer.files.length > 0) {
            selectedImageSource = "gallery";
            processImage(e.dataTransfer.files[0]);
        }
    });
    // if (customBtn) {
    //     customBtn.addEventListener("click", () => fileInput.click());
    // }

    // fileInput.addEventListener("change", (e) => {
    //     if (e.target.files.length > 0) {
    //         processImage(e.target.files[0]);
    //     }
    // });

    // dropZone.addEventListener("dragover", (e) => e.preventDefault());

    // dropZone.addEventListener("drop", (e) => {
    //     e.preventDefault();
    //     if (e.dataTransfer.files.length > 0) {
    //         processImage(e.dataTransfer.files[0]);
    //     }
    // });

    // === CONTACT POPUP FUNCTIONALITY ===
    if (contactUsBtn && contactPopupOverlay && closeContactPopup) {
        contactUsBtn.addEventListener("click", () => {
            contactPopupOverlay.classList.add("active");
        });

        closeContactPopup.addEventListener("click", () => {
            contactPopupOverlay.classList.remove("active");
        });

        contactPopupOverlay.addEventListener("click", (e) => {
            if (e.target === contactPopupOverlay) {
                contactPopupOverlay.classList.remove("active");
            }
        });

        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape") {
                contactPopupOverlay.classList.remove("active");
            }
        });
    }

});