import requests
import json

BASE_URL = "http://localhost:8000"

def analyze_batch_reports(reports):
    url = f"{BASE_URL}/analyze/batch"
    payload = {"reports": reports}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        batch_results = response.json()
        print("Batch Hazard Analysis Results:")
        print(json.dumps(batch_results, indent=4))
        return batch_results
    else:
        print(f"Error ({response.status_code}): {response.text}")
        return None

if __name__ == "__main__":
    reports_to_analyze = [
        {
        "text": "मुंबई में भीषण बाढ़, हजारों लोग फंसे हुए हैं। तत्काल बचाव कार्य की जरूरत है।",
        "source": "social_media"
    },
    {
        "text": "चेन्नई तट पर सुनामी की चेतावनी जारी, लोगों को तुरंत सुरक्षित स्थान पर जाने की सलाह दी गई है।",
        "source": "official_alert"
    },
    {
        "text": "दिल्ली-एनसीआर में 6.2 तीव्रता का भूकंप, इमारतें हिल गईं। लोगों में दहशत फैली हुई है।",
        "source": "news_report"
    },
    {
        "text": "गुजरात में चक्रवाती तूफान की चेतावनी, मछुआरों को समुद्र में न जाने की सलाह।",
        "source": "weather_service"
    },
    {
        "text": "उत्तराखंड में भूस्खलन से 15 लोग दबे, बचाव टीम मौके पर पहुंची।",
        "source": "emergency_services"
    },
    {
        "text": "राजस्थान में जंगलों में भीषण आग, कई गाँव खाली कराए गए।",
        "source": "local_news"
    },
    
    {
        "text": "কলকাতায় ভয়াবহ ঝড়ের আশঙ্কা। আবহাওয়া দপ্তর জরুরি সতর্কতা জারি করেছে।",
        "source": "weather_service"
    },
    {
        "text": "সুন্দরবনে সাইক্লোন আম্ফানের মতো ঝড়ের সম্ভাবনা, জেলেদের সতর্ক করা হয়েছে।",
        "source": "official_alert"
    },
    {
        "text": "সিলেটে প্রবল বন্যায় হাজারো পরিবার বিচ্ছিন্ন। ত্রাণ সামগ্রীর জরুরি প্রয়োজন।",
        "source": "social_media"
    },
    {
        "text": "ঢাকায় ভূমিকম্প অনুভূত হয়েছে, মানুষ ভবন থেকে বেরিয়ে এসেছে।",
        "source": "seismic_monitor"
    },
    
    {
        "text": "சென்னையில் நீர் வெள்ளம் பரவி உள்ளது। பல பகுதிகளில் மீட்புப் பணிகள் நடந்து வருகின்றன।",
        "source": "local_news"
    },
    {
        "text": "கோவையில் வன தீ விபத்து, 500 ஏக்கர் காடுகள் எரிந்து சாம்பலாகின।",
        "source": "fire_department"
    },
    {
        "text": "நீலகிரி மாவட்டத்தில் மண் சரிவு, சாலை போக்குவரத்து முற்றிலும் பாதிக்கப்பட்டது।",
        "source": "traffic_alert"
    },
    {
        "text": "கன்யாகுமரியில் கடல் அலைகளின் உயரம் அதிகரித்துள்ளது, கடலோர பகுதி மக்களை வெளியேற்றம்।",
        "source": "coastal_guard"
    },
    
    {
        "text": "కొండ చరియు హైదరాబాదులో, అనేక ఇళ్ళు దెబ్బతిన్నాయి। రెస్క్యూ టీమ్స్ వచ్చాయి।",
        "source": "emergency_services"
    },
    {
        "text": "విశాఖపట్నంలో తుఫాను హెచ్చరిక, మత్స్యకారులను సముద్రంలోకి వెళ్లవద్దని సలహా।",
        "source": "fisheries_dept"
    },
    {
        "text": "విజయవాడలో కృష్ణా నది పొంగి పొర్లుతోంది, వరద హెచ్చరిక జారీ చేశారు।",
        "source": "irrigation_dept"
    },
    {
        "text": "Severe flooding reported in multiple areas of Chennai. Emergency services are overwhelmed. Immediate assistance required.",
        "source": "news_report"
    },
    {
        "text": "Earthquake tremors felt in Delhi NCR region. Magnitude estimated at 4.5. No major damage reported so far.",
        "source": "seismic_monitor"
    },
    {
        "text": "Cyclone Biparjoy approaching Gujarat coast. Wind speeds reaching 120 kmph. Coastal evacuation underway.",
        "source": "meteorology"
    },
    {
        "text": "Flash floods in Kedarnath region. Pilgrimage suspended. Helicopter rescue operations initiated.",
        "source": "disaster_management"
    },
    {
        "text": "Forest fires spreading rapidly in ramesh nagar. Several villages evacuated as precautionary measure.",
        "source": "forest_department"
    },
    {
        "text": "Landslide blocks national highway in Uttarakhand. Traffic diverted through alternate routes.",
        "source": "highway_authority"
    },
    {
        "text": "અમદાવાદમાં ભારે વરસાદથી પાણી ભરાયું છે। ઘણા લોકો મકાનોમાં ફસાયેલા છે।",
        "source": "local_media"
    },
    {
        "text": "સૌરાષ્ટ્રમાં વાવાઝોડાની ચેતવણી, માછીમારોને સમુદ્રમાં ન જવાની સલાહ.",
        "source": "port_authority"
    },
    {
        "text": "കേരളത്തിൽ വെള്ളപ്പൊക്കം, പല ജില്ലകളിലും റെഡ് അലർട്ട് പ്രഖ്യാപിച്ചു।",
        "source": "state_disaster"
    },
    {
        "text": "കോട്ടയത്ത് മണ്ണിടിച്ചിൽ, അഞ്ച് വീടുകൾ നശിച്ചു. രക്ഷാപ്രവർത്തനം പുരോഗമിക്കുന്നു।",
        "source": "fire_rescue"
    },
    {
        "text": "ಬೆಂಗಳೂರಲ್ಲಿ ಭೀಕರ ಮಳೆ, ಅನೇಕ ಪ್ರದೇಶಗಳಲ್ಲಿ ನೀರು ನಿಂತಿದೆ। ಸಂಚಾರ ಸ್ಥಗಿತ.",
        "source": "traffic_police"
    },
    {
        "text": "ಕೊಡಗಿನಲ್ಲಿ ಭೂಕುಸಿತದಿಂದ ಮೂವರು ಸಾವು, ಹತ್ತು ಮಂದಿ ಗಾಯ.",
        "source": "district_collector"
    },
    {
        "text": "पुण्यात मुसळधार पाऊस, अनेक भागांमध्ये पाणी साचलं आहे। वाहतूक कोलमडली.",
        "source": "municipal_corp"
    },
    {
        "text": "कोल्हापूरमध्ये कृष्णा नदी पूर्ण क्षमतेने वाहत आहे, पूरग्रस्त भागातील लोकांना स्थलांतर करण्याचे आदेश.",
        "source": "collector_office"
    },
    {
        "text": "ਲੁਧਿਆਣੇ ਵਿੱਚ ਜ਼ੋਰਦਾਰ ਬਾਰਿਸ਼, ਸ਼ਹਿਰ ਵਿੱਚ ਪਾਣੀ ਭਰ ਗਿਆ। ਲੋਕ ਮੁਸ਼ਕਿਲਾਂ ਵਿੱਚ।",
        "source": "civic_body"
    },
    {
        "text": "ଭୁବନେଶ୍ୱରରେ ପ୍ରବଳ ଘୂର୍ଣ୍ଣିବାତ୍ୟାର ଆଶଙ୍କା, ଉପକୂଳବର୍ତୀ ଅଞ୍ଚଳରେ ସତର୍କତା।",
        "source": "cyclone_center"
    },
    {
        "text": "London में भारतीय community को flood warning दी गई है। Thames नदी का जल स्तर बढ़ रहा है।",
        "source": "diaspora_news"
    },
    {
        "text": "Dubai में रहने वाले Indians को sandstorm की चेतावनी। तेज़ हवाओं के साथ धूल भरी आंधी का अनुमान।",
        "source": "expat_alert"
    },
    {
        "text": "Help needed! बाढ़ का पानी घर में घुस गया है Bandra में। Kids scared. Anyone nearby please help! #MumbaiFloods",
        "source": "twitter"
    },
    {
        "text": "URGENT: Earthquake felt in Gurgaon। Building shaking बहुत ज्यादा। Everyone evacuating। Stay safe everyone! 🙏",
        "source": "facebook"
    }

    ]
    
    analyze_batch_reports(reports_to_analyze)
