export function initDate(app) {
    const { elements, state } = app;

    function toPickerValue(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, "0");
        const day = String(date.getDate()).padStart(2, "0");

        return `${year}-${month}-${day}`;
    }

    // Converts YYYY-MM-DD into MM/DD/YYYY for the visible date display
    function toDisplayDate(value) {
        if (!value) {
            return "";
        }
        const [year, month, day] = value.split("-");

        return `${month}/${day}/${year}`;
    }

    // Determines the season using the selected month
    function determineSeason(dateValue) {
        const month = Number(dateValue?.split("-")[1]);

        return month >= 6 && month <= 11 ? "wet" : "dry";
    }

    function updateSeasonDisplay() {
        const season = determineSeason(elements.datePicker.value);
        const seasonLabel = season === "wet" ? app.text().wetSeason : app.text().drySeason;

        elements.seasonDisplay.textContent = seasonLabel;
        elements.seasonDisplay.dataset.season = season;
        elements.resultSeason.textContent = `${app.text().selectedSeason}: ${seasonLabel}`;

        return season;
    }

    // Sets the date picker and date display to today's date
    function setDateToToday() {
        const pickerValue = toPickerValue(new Date());

        elements.datePicker.value = pickerValue;
        elements.dateInput.value = toDisplayDate(pickerValue);

        updateSeasonDisplay();
    }

    // Runs whenever the user selects a different date
    elements.datePicker.addEventListener("change", async () => {
        if (!elements.datePicker.value) {
            setDateToToday();

            setTimeout(() => {
                elements.datePicker.blur();
            }, 0);
            
        } else {
            elements.dateInput.value =
                toDisplayDate(elements.datePicker.value);

            updateSeasonDisplay();
        }

        if (state.lastSubmittedFile) {
            await app.actions.submitImage(
                state.lastSubmittedFile,
                { scroll: false }
            );
        }
    });

    // Opens the hidden date picker when the visible calendar box is clicked
    elements.calendarBox.addEventListener("click", () => {
            if (typeof elements.datePicker.showPicker === "function") {
                elements.datePicker.showPicker();
            } else {
                elements.datePicker.click();
            }
        }
    );

    // Makes the date functions available to other files
    app.actions.toPickerValue = toPickerValue;
    app.actions.toDisplayDate = toDisplayDate;
    app.actions.determineSeason = determineSeason;
    app.actions.updateSeasonDisplay = updateSeasonDisplay;
    app.actions.setDateToToday = setDateToToday;
}