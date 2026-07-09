export const WEBSITE_TEXT = {
    fil: {
        flag: "🇵🇭",
        label: "Filipino",

        siteTitle: "AniMedika",

        aboutTitle: "Tungkol sa Website",
        aboutShort: "Ang website na ito ay nagsisilbing opisyal na digital platform at deployment interface para sa software project na nagbibigay ng madaling gamitin at mataas ang katumpakang kasangkapan para sa mabilis na pagsusuri ng kalusugan ng pananim.",
        aboutLong: [
            'Ang website na ito ay nagsisilbing opisyal na digital platform at deployment interface para sa software project na pinamagatang "Rice Leaf Disease Detection Using EfficientNet-Based Convolutional Neural Network with Multi-Metric Performance Evaluation." Binuo ito ng mga computer science na estudyante mula sa Holy Angel University upang pagdugtungin ang deep learning models at praktikal na pamamahala sa agrikultura.',
            "Nagbibigay ang platform ng madaling gamitin at mataas ang katumpakang kasangkapan para sa mabilis na pagsusuri ng kalusugan ng pananim gamit ang fine-tuned EfficientNet Convolutional Neural Network o CNN. Maaaring mag-upload ang mga user ng larawan ng dahon ng palay upang makatanggap ng real-time prediction at mga rekomendadong aksyon.",
            "Layunin ng sistema na maagang matukoy ang mahahalagang sakit sa dahon ng palay upang matulungan ang mga komunidad sa agrikultura na mabawasan ang banta bago pa ito makaapekto sa ani. Sa kabuuan, ang platform na ito ay isang teknolohikal na solusyon para sa mas makabagong proteksyon ng pananim, mas matatag na pagsasaka, at seguridad sa pagkain."
        ],

        showMore: "Ipakita pa",
        showLess: "Ipakita nang mas kaunti",

        howToUse: "Paano Gamitin",
        steps: [
            'Maghanda ng isang malinaw na close-up na larawan ng dahon ng palay. <i id="img-specs">Siguraduhing malinaw ito at hindi masyadong madilim.</i>',
            "Piliin o i-update ang petsa gamit ang calendar icon sa itaas upang tumugma sa petsa kung kailan kinuha ang larawan.",
            "I-upload ang larawan gamit ang camera o <i>select from gallery</i> na opsyon.",
            "Awtomatikong ipapakita ang disease prediction at mga rekomendadong aksyon.",
            "Upang magsuri muli ng panibagong larawan, ulitin lamang ang mga hakbang sa itaas."
        ],

        selectImageSource: "Pumili ng Pinagmulan ng Larawan",
        gallery: "Pumili mula sa gallery",
        camera: "Gamitin ang camera",

        results: "Mga Resulta",
        evaluating: "Sinusuri...",
        processing: "Pinoproseso ang larawan...",
        awaiting: "Naghihintay ng pagsusuri ng larawan...",
        recommendedActions: "Mga Rekomendadong Aksyon",
        awaitingRecommendations: "Naghihintay ng lokal na pagproseso...",

        disclaimer: "<strong>Paalala:</strong> Ang sistemang ito ay ginawa upang tumulong, hindi upang palitan, ang mga magsasakang Pilipino at mga propesyonal sa agrikultura. Ang mga resultang ibinibigay ay suporta lamang sa pagtukoy ng sakit at paggawa ng desisyon, at dapat pa ring gamitin kasama ng obserbasyon sa bukid at payo ng eksperto kung kinakailangan.",

        contactUs: "Contact Us",
        backToTop: "Back to Top",
        copyright: "All rights reserved 2026",

        popupTitle: "Pumili ng Wika",
        popupText: "Piliin ang nais mong gamitin na wika."
    },

    en: {
        flag: "🇺🇸",
        label: "English",

        siteTitle: "AniMedika",

        aboutTitle: "About this Website",
        aboutShort: "This website serves as the official digital platform and deployment interface for a software project that provides an accessible, high-precision tool for rapid crop health assessment.",
        aboutLong: [
            'This website serves as the official digital platform and deployment interface for the software project titled "Rice Leaf Disease Detection Using EfficientNet-Based Convolutional Neural Network with Multi-Metric Performance Evaluation." Developed by computer science students at Holy Angel University, this web application directly bridges the gap between complex deep learning models and practical agricultural management.',
            "The platform provides an accessible, high-precision tool for rapid crop health assessment by leveraging a fine-tuned EfficientNet Convolutional Neural Network (CNN) architecture. Users can seamlessly upload images of symptomatic rice leaves directly through the web interface to receive real-time predictions with recommended actions.",
            "The underlying system is engineered to detect critical rice leaf diseases early, empowering agricultural communities to mitigate threats before they compromise seasonal crop yields. Ultimately, this user-centric platform stands as a dedicated technological solution aimed at modernizing crop protection, supporting sustainable farming practices, and contributing to food security."
        ],

        showMore: "Show more",
        showLess: "Show less",

        howToUse: "How to Use",
        steps: [
            'Prepare one close-up image of a rice leaf. <i id="img-specs">Please make sure that it is clear and not too dark.</i>',
            "Select or update the date using the calendar icon above to match when the image was taken.",
            "Upload the image using the camera or <i>select from gallery</i> option.",
            "The disease prediction and recommended actions will be displayed automatically.",
            "To analyze another image, simply repeat the steps above."
        ],

        selectImageSource: "Select Image Source",
        gallery: "Select from gallery",
        camera: "Use camera",

        results: "Results",
        evaluating: "Evaluating...",
        processing: "Processing image...",
        awaiting: "Awaiting image analysis...",
        recommendedActions: "Recommended Actions",
        awaitingRecommendations: "Awaiting local performance processing...",

        disclaimer: "<strong>Disclaimer:</strong> This system is intended to assist, not replace, Filipino farmers and agricultural professionals. The results provided are meant to support disease identification and decision-making and should be used together with field observations and expert advice when necessary.",

        contactUs: "Contact Us",
        backToTop: "Back to Top",
        copyright: "All rights reserved 2026",

        popupTitle: "Choose a Language",
        popupText: "Please select your preferred language."
    }
};

export const ENGLISH_RESULTS = {
    0: {
        name: "Bacterial Leaf Blight",
        desc: "A bacterial disease that causes water-soaked streaks and yellowing on rice leaves.",
        rec: {
            default: ["Remove infected leaves", "Avoid using seeds from infected plants"],
            dry: ["Remove infected leaves", "Maintain proper irrigation to reduce plant stress"],
            wet: ["Remove infected leaves", "Avoid excessive water buildup in the field"]
        }
    },
    1: {
        name: "Brown Spot",
        desc: "A fungal disease that appears as brown circular or oval spots on rice leaves.",
        rec: {
            default: ["Use appropriate fungicide", "Improve field sanitation"],
            dry: ["Use appropriate fungicide", "Maintain proper soil nutrients"],
            wet: ["Use appropriate fungicide", "Improve field sanitation after heavy rain"]
        }
    },
    2: {
        name: "Healthy Rice Plant",
        desc: "The rice leaf appears healthy with no visible signs of disease.",
        rec: {
            default: ["Continue normal crop care", "Monitor the plant regularly"],
            dry: ["Continue normal crop care", "Maintain enough water supply"],
            wet: ["Continue normal crop care", "Monitor for early disease symptoms"]
        }
    },
    3: {
        name: "Rice Blast",
        desc: "A serious fungal disease that can cause spindle-shaped lesions on rice leaves.",
        rec: {
            default: ["Apply treatment early", "Remove heavily infected plant parts"],
            dry: ["Apply treatment early", "Avoid over-fertilizing with nitrogen"],
            wet: ["Apply treatment early", "Monitor closely after humid or rainy days"]
        }
    },
    4: {
        name: "Sheath Blight",
        desc: "A fungal disease that affects the sheath area and may spread upward under humid conditions.",
        rec: {
            default: ["Improve field drainage", "Avoid dense planting"],
            dry: ["Improve field drainage", "Avoid dense planting"],
            wet: ["Improve field drainage", "Reduce excess standing water"]
        }
    },
    5: {
        name: "Tungro Virus",
        desc: "A viral rice disease often associated with yellowing, stunting, and reduced plant growth.",
        rec: {
            default: ["Remove infected plants", "Monitor for insect vectors"],
            dry: ["Remove infected plants", "Monitor for insect vectors"],
            wet: ["Remove infected plants", "Control insect spread after rainfall"]
        }
    }
};