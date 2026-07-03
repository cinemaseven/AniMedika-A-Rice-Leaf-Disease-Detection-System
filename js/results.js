import { ENGLISH_RESULTS } from "./translations.js";

const FILIPINO_RESULTS = {
    0: {
        name: "Bacterial Leaf Blight",
        desc: "Isang bacterial disease na nagdudulot ng parang basang guhit at paninilaw sa dahon ng palay.",
        rec: {
            default: ["Alisin ang mga apektadong dahon", "Iwasang gumamit ng binhi mula sa mga apektadong halaman"],
            dry: ["Alisin ang mga apektadong dahon", "Panatilihin ang tamang patubig upang mabawasan ang stress ng halaman"],
            wet: ["Alisin ang mga apektadong dahon", "Iwasan ang sobrang ipong tubig sa bukid"]
        }
    },
    1: {
        name: "Brown Spot",
        desc: "Isang fungal disease na makikita bilang kayumangging bilog o pahabang batik sa dahon ng palay.",
        rec: {
            default: ["Gumamit ng angkop na fungicide", "Panatilihin ang kalinisan ng bukid"],
            dry: ["Gumamit ng angkop na fungicide", "Panatilihin ang wastong sustansya ng lupa"],
            wet: ["Gumamit ng angkop na fungicide", "Panatilihin ang kalinisan ng bukid pagkatapos ng malakas na ulan"]
        }
    },
    2: {
        name: "Malusog na Palay",
        desc: "Ang dahon ng palay ay mukhang malusog at walang nakikitang palatandaan ng sakit.",
        rec: {
            default: ["Ipagpatuloy ang normal na pangangalaga sa pananim", "Regular na bantayan ang halaman"],
            dry: ["Ipagpatuloy ang normal na pangangalaga sa pananim", "Panatilihin ang sapat na suplay ng tubig"],
            wet: ["Ipagpatuloy ang normal na pangangalaga sa pananim", "Bantayan ang maagang sintomas ng sakit"]
        }
    },
    3: {
        name: "Rice Blast",
        desc: "Isang malubhang fungal disease na maaaring magdulot ng pahabang sugat sa dahon ng palay.",
        rec: {
            default: ["Maglagay agad ng angkop na paggamot", "Alisin ang malubhang apektadong bahagi ng halaman"],
            dry: ["Maglagay agad ng angkop na paggamot", "Iwasan ang sobrang paggamit ng nitrogen fertilizer"],
            wet: ["Maglagay agad ng angkop na paggamot", "Mas bantayan pagkatapos ng maalinsangan o maulang araw"]
        }
    },
    4: {
        name: "Sheath Blight",
        desc: "Isang fungal disease na nakaaapekto sa sheath o bahagi ng palay at maaaring kumalat kapag mahalumigmig ang kondisyon.",
        rec: {
            default: ["Ayusin ang drainage sa bukid", "Iwasan ang sobrang dikit-dikit na pagtatanim"],
            dry: ["Ayusin ang drainage sa bukid", "Iwasan ang sobrang dikit-dikit na pagtatanim"],
            wet: ["Ayusin ang drainage sa bukid", "Bawasan ang sobrang nakatayong tubig"]
        }
    },
    5: {
        name: "Tungro Virus",
        desc: "Isang viral disease sa palay na kadalasang nagdudulot ng paninilaw, pagliit, at mabagal na paglaki ng halaman.",
        rec: {
            default: ["Alisin ang mga apektadong halaman", "Bantayan ang mga insektong maaaring magkalat ng sakit"],
            dry: ["Alisin ang mga apektadong halaman", "Bantayan ang mga insektong maaaring magkalat ng sakit"],
            wet: ["Alisin ang mga apektadong halaman", "Kontrolin ang pagkalat ng insekto pagkatapos ng ulan"]
        }
    }
};

function getSeasonFromDate(dateValue) {
    const month = Number(dateValue.split("/")[0]);

    if (!month) return "default";

    return month >= 6 && month <= 11 ? "wet" : "dry";
}

export function getDiseaseResult(index, language, dateValue) {
    const resultSet = language === "en" ? ENGLISH_RESULTS : FILIPINO_RESULTS;
    const result = resultSet[index];

    if (!result) {
        return {
            name: language === "en" ? "Unknown" : "Hindi Matukoy",
            desc: language === "en" ? "No data available" : "Walang available na datos",
            rec: []
        };
    }

    const season = getSeasonFromDate(dateValue);
    const recommendations = result.rec[season] || result.rec.default;

    return {
        name: result.name,
        desc: result.desc,
        rec: recommendations
    };
}