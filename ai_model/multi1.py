import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, XLMRobertaTokenizer, XLMRobertaModel
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import re
import logging
from typing import Dict, List, Tuple, Optional, Union
import spacy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import asyncio
import aiohttp
from dataclasses import dataclass, field
import json
from collections import defaultdict
import torch.nn.functional as F
from datetime import datetime, timedelta
import json
import requests
import snscrape.modules.twitter as sntwitter
import praw
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HazardReport:
    text: str
    language: str
    hazard_type: str
    confidence: float
    severity: str
    locations: List[Dict[str, Union[str, float]]]
    sentiment: str
    urgency_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    affected_population: Optional[int] = None
    coordinates: Optional[Tuple[float, float]] = None
    source: str = "user_report"
    verified: bool = False

class EnhancedMultilingualHazardDetector:
    
    def __init__(self, use_gpu: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self._load_models()
        self.geolocator = Nominatim(user_agent="hazard_detector_v2")
        self.hazard_keywords = self._load_enhanced_keywords()
        
        self.lang_patterns = {
            'hi': re.compile(r'[\u0900-\u097F]'),  # Hindi
            'bn': re.compile(r'[\u0980-\u09FF]'),  # Bengali
            'ta': re.compile(r'[\u0B80-\u0BFF]'),  # Tamil
            'te': re.compile(r'[\u0C00-\u0C7F]'),  # Telugu
            'gu': re.compile(r'[\u0A80-\u0AFF]'),  # Gujarati
            'ml': re.compile(r'[\u0D00-\u0D7F]'),  # Malayalam
            'kn': re.compile(r'[\u0C80-\u0CFF]'),  # Kannada
            'or': re.compile(r'[\u0B00-\u0B7F]'),  # Odia
            'pa': re.compile(r'[\u0A00-\u0A7F]'),  # Punjabi
            'ur': re.compile(r'[\u0600-\u06FF]'),  # Urdu
            'mr': re.compile(r'[\u0900-\u097F]'),  # Marathi (shares Devanagari with Hindi)
            
            # European languages (detected by specific characters)
            'es': re.compile(r'[áéíóúüñ¿¡]'),  # Spanish
            'fr': re.compile(r'[àâçéèêëîïôùûüÿœæ]'),  # French
            'de': re.compile(r'[äöüß]'),  # German
            'it': re.compile(r'[àèéìíîòóùú]'),  # Italian
            'pt': re.compile(r'[áàâãçéêíóôõú]'),  # Portuguese
            
            # East Asian languages
            'zh': re.compile(r'[\u4e00-\u9fff]'),  # Chinese
            'ja': re.compile(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]'),  # Japanese
            'ko': re.compile(r'[\uac00-\ud7af\u1100-\u11ff]'),  # Korean
            
            # Middle Eastern languages
            'ar': re.compile(r'[\u0600-\u06FF]'),  # Arabic
            'he': re.compile(r'[\u0590-\u05FF]'),  # Hebrew
            'fa': re.compile(r'[\u0600-\u06FF\uFB8A-\uFB8F]'),  # Persian/Farsi
            
            # Southeast Asian languages
            'th': re.compile(r'[\u0e00-\u0e7f]'),  # Thai
            'vi': re.compile(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]'),  # Vietnamese
            'id': re.compile(r'[ăâêôơưđ]'),  # Indonesian
        }
        
        self._load_ner_models()
        
        self.severity_thresholds = {
            'critical': 0.9,
            'high': 0.75,
            'medium': 0.6,
            'low': 0.4
        }
        
        self.historical_reports = []
        
    def _load_models(self):
        try:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
            self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base').to(self.device)
            
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                device=0 if self.device.type == 'cuda' else -1
            )
            
            self.hazard_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device.type == 'cuda' else -1
            )
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
            self.model = AutoModel.from_pretrained('ai4bharat/indic-bert').to(self.device)
    
    def _load_ner_models(self):
        try:
            self.nlp_models = {}
            languages = ['en', 'xx']
            
            for lang in languages:
                try:
                    if lang == 'en':
                        self.nlp_models[lang] = spacy.load('en_core_web_sm')
                    else:
                        self.nlp_models[lang] = spacy.load('xx_core_web_sm')
                except OSError:
                    logger.warning(f"spaCy model for {lang} not found. Using basic extraction.")
                    
        except Exception as e:
            logger.warning(f"Could not load spaCy models: {e}")
            self.nlp_models = {}
    
    def _load_enhanced_keywords(self) -> Dict:
     return {
        "tsunami": {
            "en": ["tsunami", "seismic wave", "tidal wave", "marine surge", "ocean wave"],
            "hi": ["सुनामी", "भूकंपीय लहर", "समुद्री तूफान", "समुद्री लहर", "तटीय लहर"],
            "bn": ["সুনামি", "ভূমিকম্পীয় তরঙ্গ", "সমুদ্রের ঢেউ", "জলোচ্ছ্বাস"],
            "ta": ["சுனாமி", "நில அதிர்வு அலை", "கடல் அலை", "கடற்கொந்தளிப்பு"],
            "te": ["సునామి", "భూకంప తరంగం", "సముద్ర తరంగం"],
            "gu": ["સુનામી", "ભૂકંપીય તરંગ", "સમુદ્રી તરંગ"],
            "ml": ["സുനാമി", "ഭൂകമ്പ തരംഗം"],
            "kn": ["ಸುನಾಮಿ", "ಭೂಕಂಪನ ತರಂಗ"],
        },
        "flood": {
            "en": ["flood", "flooding", "water logging", "inundation", "deluge", "overflow", "submersion"],
            "hi": ["बाढ़", "जल भराव", "पानी", "डूब", "जलप्लावन", "अतिवृष्टि"],
            "bn": ["বন্যা", "জল ভরাট", "পানি", "জলাবদ্ধতা", "প্লাবন"],
            "ta": ["வெள்ளம்", "நீர் நிரம்புதல்", "जल निमग्नता", "नदी उफान"],
            "te": ["వరద", "నీటి నిలుపుదల", "జలప్రళయం"],
            "gu": ["પૂર", "પાણી ભરાવ", "જળપ્રલય"],
            "ml": ["വെള്ളപ്പൊക്കം", "നീർക്കെട്ട്"],
            "kn": ["ಪ್ರವಾಹ", "ನೀರು ನಿಂತುಕೊಳ್ಳುವಿಕೆ"],
        },
        "flash_flood": {
            "en": ["flash flood", "sudden flood", "rapid flooding", "urban flooding", "street flooding"],
            "hi": ["अचानक बाढ़", "तुरंत बाढ़", "शहरी बाढ़", "सड़क बाढ़"],
            "bn": ["আকস্মিক বন্যা", "দ্রুত বন্যা", "শহুরে বন্যা"],
            "ta": ["திடீர் வெள்ளம்", "நகர் வெள்ளம்", "சாலை வெள்ளம்"],
            "te": ["ఆకస్మిక వరద", "వేగవంతమైన వరద"],
            "gu": ["અચાનક પૂર", "શહેરી પૂર"],
            "ml": ["പെട്ടെന്നുള്ള വെള്ളപ്പൊക്കം"],
            "kn": ["ಹಠಾತ್ ಪ್ರವಾಹ", "ನಗರ ಪ್ರವಾಹ"],
        },
        "high_waves": {
            "en": ["high waves", "giant waves", "rogue waves", "storm waves", "massive waves", "towering waves"],
            "hi": ["ऊंची लहरें", "विशाल लहरें", "तूफानी लहरें", "दानव लहरें"],
            "bn": ["উঁচু ঢেউ", "বিশাল ঢেউ", "ঝড়ের ঢেউ"],
            "ta": ["உயர் அலைகள்", "பெரிய அலைகள்", "புயல் அலைகள்"],
            "te": ["ఎత్తైన అలలు", "పెద్ద అలలు", "తుఫాను అలలు"],
            "gu": ["ઊંચા મોજા", "વિશાળ મોજા", "તોફાની મોજા"],
            "ml": ["ഉയർന്ന തിരമാലകൾ", "വലിയ തിരമാലകൾ"],
            "kn": ["ಎತ್ತರದ ಅಲೆಗಳು", "ದೊಡ್ಡ ಅಲೆಗಳು"],
        },
        "storm_surge": {
            "en": ["storm surge", "tidal surge", "coastal surge", "hurricane surge", "cyclone surge"],
            "hi": ["तूफानी ज्वार", "समुद्री उफान", "तटीय ज्वार", "चक्रवाती ज्वार"],
            "bn": ["ঝড়ের জোয়ার", "তীরবর্তী জোয়ার", "সামুদ্রিক জোয়ার"],
            "ta": ["புயல் அலை", "கடற்கரை அலை", "சூறாவளி அலை"],
            "te": ["తుఫాను అలలు", "తీర ప్రాంత అలలు"],
            "gu": ["તોફાની મોજા", "દરિયાકિનારાના મોજા"],
            "ml": ["കൊടുങ്കാറ്റ് തിരമാല", "തീരദേശ തിരമാല"],
            "kn": ["ಬಿರುಗಾಳಿ ಅಲೆ", "ಕರಾವಳಿ ಅಲೆ"],
        },
        "water_spout": {
            "en": ["waterspout", "water spout", "marine tornado", "sea tornado", "lake tornado"],
            "hi": ["जल स्तंभ", "समुद्री बवंडर", "जल चक्रवात"],
            "bn": ["জলঘূর্ণি", "সামুদ্রিক ঘূর্ণিঝড়"],
            "ta": ["நீர் சுழல்", "கடல் சூறாவளி"],
            "te": ["నీటి సుడిగాలి", "సముద్ర సుడిగాలి"],
            "gu": ["પાણીનું વંટોળિયું", "સમુદ્રી વંટોળિયું"],
            "ml": ["ജലചുഴലി", "കടൽ ചുഴലി"],
            "kn": ["ನೀರಿನ ಸುಳಿಗಾಳಿ", "ಸಮುದ್ರ ಸುಳಿಗಾಳಿ"],
        },
        "dam_burst": {
            "en": ["dam burst", "dam failure", "dam break", "dam collapse", "levee failure"],
            "hi": ["बांध टूटना", "बांध फटना", "बांध ध्वस्त", "तटबंध टूटना"],
            "bn": ["বাঁধ ভাঙা", "বাঁধ ধসে পড়া", "বন্যা নিয়ন্ত্রণ ব্যর্থতা"],
            "ta": ["அணை உடைதல்", "அணை முறிவு", "அணை சேதம்"],
            "te": ["ఆనకట్ట విరుగుట", "ఆనకట్ట పగుళ్లు"],
            "gu": ["બંધ તૂટવું", "બંધનું નુકસાન"],
            "ml": ["അണക്കെട്ട് പൊട്ടൽ", "അണക്കെട്ട് തകർച്ച"],
            "kn": ["ಅಣೆಕಟ್ಟು ಒಡೆಯುವಿಕೆ", "ಅಣೆಕಟ್ಟು ವೈಫಲ್ಯ"],
        },
        "river_overflow": {
            "en": ["river overflow", "riverbank overflow", "stream overflow", "creek overflow"],
            "hi": ["नदी उफान", "नदी का बहाव", "नदी किनारे बाढ़"],
            "bn": ["নদী উপচে পড়া", "নদীর তীর ছাপিয়ে যাওয়া"],
            "ta": ["ஆற்று வெள்ளம்", "ஆற்றங்கரை வெள்ளம்"],
            "te": ["నది పొంగిపొర్లుట", "నది తీర వరద"],
            "gu": ["નદીનું ઓવરફ્લો", "નદી કિનારે પૂર"],
            "ml": ["നദി കവിഞ്ഞൊഴുകൽ", "നദീതീര വെള്ളപ്പൊക്കം"],
            "kn": ["ನದಿ ತುಂಬಿ ಹರಿಯುವಿಕೆ", "ನದಿ ತೀರ ಪ್ರವಾಹ"],
        },
        "coastal_erosion": {
            "en": ["coastal erosion", "beach erosion", "shoreline erosion", "cliff erosion"],
            "hi": ["तटीय कटाव", "समुद्री किनारे का कटाव", "तट क्षरण"],
            "bn": ["উপকূলীয় ক্ষয়", "সমুদ্র তীরের ক্ষয়"],
            "ta": ["கடற்கரை அரிப்பு", "தீரக் கரிப்பு"],
            "te": ["తీర ప్రాంత కోత", "సముద్ర తీర క్షీణత"],
            "gu": ["દરિયાકિનારાનું ધોવાણ", "દરિયાઈ કાંઠાનું ધોવાણ"],
            "ml": ["തീരദേശ ശോഷണം", "കടൽത്തീര ശോഷണം"],
            "kn": ["ಕರಾವಳಿ ಸವೆತ", "ಸಮುದ್ರ ತೀರ ಸವೆತ"],
        },
        "king_tide": {
            "en": ["king tide", "extreme high tide", "super tide", "perigean spring tide"],
            "hi": ["अति ज्वार", "उच्च ज्वार", "महा ज्वार"],
            "bn": ["অতি জোয়ার", "চরম জোয়ার"],
            "ta": ["உயர் அலை", "அதிக அலை"],
            "te": ["అధిక అలలు", "తీవ్ర అలలు"],
            "gu": ["અતિ ભરતી", "ઊંચી ભરતી"],
            "ml": ["അതിഉയർന്ന വേലിയേറ്റം"],
            "kn": ["ಅತಿ ಉಬ್ಬರ", "ಎತ್ತರದ ಉಬ್ಬರ"],
        },
        "seiche": {
            "en": ["seiche", "lake oscillation", "harbor oscillation", "bay oscillation"],
            "hi": ["झील तरंग", "बंदरगाह दोलन", "खाड़ी तरंग"],
            "bn": ["হ্রদ দোলন", "বন্দর দোলন"],
            "ta": ["ஏரி அலை", "துறைமுக அலை"],
            "te": ["సరస్సు అలలు", "నౌకాశ్రయ అలలు"],
            "gu": ["તળાવના મોજા", "બંદર મોજા"],
            "ml": ["തടാക അലകൾ", "തുറമുഖ അലകൾ"],
            "kn": ["ಸರೋವರ ಅಲೆಗಳು", "ಬಂದರು ಅಲೆಗಳು"],
        },
        "cyclone": {
            "en": ["cyclone", "hurricane", "typhoon", "storm", "tempest", "whirlwind"],
            "hi": ["चक्रवात", "तूफान", "आंधी", "प्रचंड हवा", "वायु तूफान"],
            "bn": ["ঘূর্ণিঝড়", "ঝড়", "তুফান", "প্রবল হাওয়া"],
            "ta": ["சூறாவளி", "புயல்", "காற்று வீச்சு"],
            "te": ["తుఫాను", "వాయు ప్రకોపం", "చక్రవాత తుఫాను"],
            "gu": ["વાવાઝોડું", "તોફાન", "ચક્રવાત"],
            "ml": ["ചുഴലിക്കാറ്റ്", "കൊടുങ്കാറ്റ്"],
            "kn": ["ಚಂಡಮಾರುತ", "ಬಿರುಗಾಳಿ"],
        },
        "earthquake": {
            "en": ["earthquake", "tremor", "seismic activity", "quake", "earth tremor"],
            "hi": ["भूकंप", "भूकम्पन", "धरती कांपना", "भूचाल", "भूकंप झटका"],
            "bn": ["ভূমিকম্প", "কম্পন", "ভূকম্পন"],
            "ta": ["பூகம்பம்", "நிலநடுக்கம்", "பூமி அதிர்வு"],
            "te": ["భూకంపం", "భూమి వణుకు"],
            "gu": ["ભૂકંપ", "ધરતીકંપ"],
            "ml": ["ഭൂകമ്പം", "ഭൂചലനം"],
            "kn": ["ಭೂಕಂಪ", "ನೆಲಕಂಪನ"],
        },
        "wildfire": {
            "en": ["wildfire", "forest fire", "bush fire", "fire outbreak", "blaze"],
            "hi": ["जंगली आग", "वन आग", "दावानल", "आग लगना"],
            "bn": ["দাবানল", "বন অগ্নিকাণ্ড", "আগুন"],
            "ta": ["காட்டுத்தீ", "வன தீ", "தீ விபத்து"],
            "te": ["అడవి మంట", "అగ్ని ప్రమాదం"],
            "gu": ["જંગલની આગ", "દાવાનળ"],
            "ml": ["കാട്ടുതീ", "അഗ്നിബാധ"],
            "kn": ["ಕಾಡಿನ ಬೆಂಕಿ", "ದಾವಾಗ್ನಿ"],
        },
        "landslide": {
            "en": ["landslide", "mudslide", "rockslide", "slope failure", "mass movement"],
            "hi": ["भूस्खलन", "मिट्टी खिसकना", "पहाड़ धसकना"],
            "bn": ["ভূমিধস", "পাহাড় ধসে পড়া"],
            "ta": ["நிலச்சரிவு", "மலை சரிவு"],
            "te": ["కొండ చరియు", "భూస్ఖలనం"],
            "gu": ["ભૂસ્ખલન", "પહાડ ધસવું"],
            "ml": ["മണ്ണിടിച്ചിൽ", "മലയിടിച്ചിൽ"],
            "kn": ["ಭೂಕುಸಿತ", "ಮಣ್ಣು ಜಾರುವಿಕೆ"],
        }
    }
    
    def detect_language_advanced(self, text: str) -> str:
        lang_scores = defaultdict(int)
        
        for lang, pattern in self.lang_patterns.items():
            matches = pattern.findall(text)
            lang_scores[lang] = len(matches)
        
        if lang_scores:
            detected_lang = max(lang_scores, key=lang_scores.get)
            if lang_scores[detected_lang] > 5: 
                return detected_lang
        

        try:
            
            if hasattr(self.tokenizer, 'language_codes'):
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                   
                    if hasattr(outputs, 'language_prediction'):
                        lang_id = outputs.language_prediction.argmax().item()
                        return self.tokenizer.language_codes[lang_id]
            
          
            european_langs = ['en', 'es', 'fr', 'de', 'it', 'pt']
            
           
            lang_prob = {lang: 0.0 for lang in european_langs}
            
          
            ngrams = {
                'en': ['th', 'he', 'in', 'er', 'an', 'on'],
                'es': ['de', 'en', 'el', 'la', 'os', 'ar'],
                'fr': ['le', 'de', 'es', 'en', 'on', 'nt'],
                'de': ['en', 'er', 'ch', 'de', 'ei', 'te'],
                'it': ['di', 'ch', 'er', 'la', 'to', 'co'],
                'pt': ['de', 'os', 'ar', 'ra', 'es', 'da']
            }
            
            text_lower = text.lower()
            for lang, grams in ngrams.items():
                count = sum(text_lower.count(gram) for gram in grams)
                lang_prob[lang] = count / (len(text) + 1)  
            
            if lang_prob:
                best_lang = max(lang_prob, key=lang_prob.get)
                if lang_prob[best_lang] > 0.01:  
                    return best_lang
        
        except Exception as e:
            logger.warning(f"Advanced language detection failed: {e}")
        
       
        return 'en'
    
    def extract_embeddings_batch(self, texts: List[str]) -> np.ndarray:
       
        cache_key = hash(tuple(texts))
        if hasattr(self, '_embedding_cache') and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
            
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256 
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        result = embeddings.cpu().numpy()
        
       
        if not hasattr(self, '_embedding_cache'):
            self._embedding_cache = {}
            
       
        if len(self._embedding_cache) > 1000:
           
            self._embedding_cache.pop(next(iter(self._embedding_cache)))
            
        self._embedding_cache[cache_key] = result
        return result
    
    def detect_hazard_advanced(self, text: str) -> Tuple[str, float, Dict]:
        detected_lang = self.detect_language_advanced(text)
        
        import concurrent.futures
        
       
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
           
            semantic_future = executor.submit(self._detect_hazard_semantic, text, detected_lang)
            classification_future = executor.submit(self._detect_hazard_classification, text)
            pattern_future = executor.submit(self._detect_hazard_patterns, text, detected_lang)
            
          
            semantic_result = semantic_future.result()
            classification_result = classification_future.result()
            pattern_result = pattern_future.result()
        
        combined_result = self._combine_detection_results(
            semantic_result, classification_result, pattern_result
        )
        
        return combined_result['hazard_type'], combined_result['confidence'], combined_result['details']
    
    def _detect_hazard_semantic(self, text: str, language: str) -> Dict:
        text_embedding = self.extract_embeddings_batch([text])[0]
        
        max_similarity = 0
        detected_hazard = "unknown"
        
        for hazard_type, keywords_dict in self.hazard_keywords.items():
            keywords = keywords_dict.get(language, keywords_dict['en'])
            
            keyword_embeddings = self.extract_embeddings_batch(keywords)
            similarities = cosine_similarity([text_embedding], keyword_embeddings)[0]
            max_keyword_similarity = np.max(similarities)
            
            if max_keyword_similarity > max_similarity:
                max_similarity = max_keyword_similarity
                detected_hazard = hazard_type
        
        return {
            'hazard_type': detected_hazard,
            'confidence': max_similarity,
            'method': 'semantic'
        }
    
    def _detect_hazard_classification(self, text: str) -> Dict:
        try:
            candidate_labels = list(self.hazard_keywords.keys()) + ['normal', 'other']
            result = self.hazard_classifier(text, candidate_labels)
            
            top_label = result['labels'][0]
            confidence = result['scores'][0]
            
            if top_label in ['normal', 'other'] or confidence < 0.3:
                return {'hazard_type': 'unknown', 'confidence': 0.0, 'method': 'classification'}
            
            return {
                'hazard_type': top_label,
                'confidence': confidence,
                'method': 'classification'
            }
            
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return {'hazard_type': 'unknown', 'confidence': 0.0, 'method': 'classification'}
    
    def _detect_hazard_patterns(self, text: str, language: str) -> Dict:
        text_lower = text.lower()
        max_matches = 0
        detected_hazard = "unknown"
        
        for hazard_type, keywords_dict in self.hazard_keywords.items():
            keywords = keywords_dict.get(language, keywords_dict['en'])
            matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            
            if matches > max_matches:
                max_matches = matches
                detected_hazard = hazard_type
        
        confidence = min(max_matches / 3.0, 1.0) if max_matches > 0 else 0.0
        
        return {
            'hazard_type': detected_hazard,
            'confidence': confidence,
            'method': 'pattern'
        }
    
    def _combine_detection_results(self, semantic: Dict, classification: Dict, pattern: Dict) -> Dict:
        weights = {'semantic': 0.4, 'classification': 0.4, 'pattern': 0.2}
        
        hazard_scores = defaultdict(float)
        
        for result in [semantic, classification, pattern]:
            if result['hazard_type'] != 'unknown':
                weighted_confidence = result['confidence'] * weights[result['method']]
                hazard_scores[result['hazard_type']] += weighted_confidence
        
        if hazard_scores:
            best_hazard = max(hazard_scores, key=hazard_scores.get)
            combined_confidence = hazard_scores[best_hazard]
        else:
            best_hazard = "unknown"
            combined_confidence = 0.0
        
        return {
            'hazard_type': best_hazard,
            'confidence': combined_confidence,
            'details': {
                'semantic': semantic,
                'classification': classification,
                'pattern': pattern
            }
        }
    
    def extract_locations_advanced(self, text: str, language: str) -> List[Dict[str, Union[str, float]]]:
       
        cache_key = hash(f"{text}_{language}")
        if hasattr(self, '_location_cache') and cache_key in self._location_cache:
            return self._location_cache[cache_key]
            
        locations = []
        
        if 'en' in self.nlp_models or 'xx' in self.nlp_models:
            nlp_model = self.nlp_models.get('en') or self.nlp_models.get('xx')
            if nlp_model:
                doc = nlp_model(text)
                for ent in doc.ents:
                    if ent.label_ in ['GPE', 'LOC']:
                        locations.append({
                            'name': ent.text,
                            'confidence': 0.8,
                            'method': 'ner'
                        })
        
        indian_cities = self._get_indian_cities_patterns(language)
        for city_pattern, city_name in indian_cities.items():
            if re.search(city_pattern, text, re.IGNORECASE):
                locations.append({
                    'name': city_name,
                    'confidence': 0.9,
                    'method': 'pattern'
                })
        
        validated_locations = []
        for loc in locations:
            try:
                geocoded = self.geolocator.geocode(loc['name'], timeout=5)
                if geocoded:
                    validated_locations.append({
                        'name': loc['name'],
                        'latitude': geocoded.latitude,
                        'longitude': geocoded.longitude,
                        'confidence': loc['confidence'],
                        'method': loc['method']
                    })
            except (GeocoderTimedOut, Exception):
                validated_locations.append(loc)
        
        result = validated_locations if validated_locations else [{'name': 'unknown', 'confidence': 0.0}]
        
       
        if not hasattr(self, '_location_cache'):
            self._location_cache = {}
            
       
        if len(self._location_cache) > 1000:
           
            self._location_cache.pop(next(iter(self._location_cache)))
            
        self._location_cache[cache_key] = result
        return result
    
    def _get_indian_cities_patterns(self, language: str) -> Dict[str, str]:
        patterns = {
            'en': {
        # Major metros (existing)
        r'\b(mumbai|bombay)\b': 'Mumbai',
        r'\b(delhi|new delhi)\b': 'Delhi',
        r'\b(chennai|madras)\b': 'Chennai',
        r'\b(kolkata|calcutta)\b': 'Kolkata',
        r'\b(bangalore|bengaluru)\b': 'Bangalore',
        r'\b(hyderabad)\b': 'Hyderabad',
        r'\b(pune)\b': 'Pune',
        r'\b(ahmedabad)\b': 'Ahmedabad',

        # Other major cities (existing)
        r'\b(surat)\b': 'Surat',
        r'\b(indore)\b': 'Indore',
        r'\b(kanpur)\b': 'Kanpur',
        r'\b(nagpur)\b': 'Nagpur',
        r'\b(varanasi|banaras|benaras)\b': 'Varanasi',
        r'\b(allahabad|prayagraj)\b': 'Prayagraj',
        r'\b(ludhiana)\b': 'Ludhiana',
        r'\b(vadodara|baroda)\b': 'Vadodara',
        r'\b(vijayawada)\b': 'Vijayawada',
        r'\b(coimbatore)\b': 'Coimbatore',
        r'\b(kochi|cochin)\b': 'Kochi',
        r'\b(trichy|tiruchirappalli)\b': 'Tiruchirappalli',
        r'\b(madurai)\b': 'Madurai',

        # Religious/pilgrimage cities (existing)
        r'\b(ayodhya)\b': 'Ayodhya',
        r'\b(mathura)\b': 'Mathura',
        r'\b(haridwar)\b': 'Haridwar',
        r'\b(rishikesh)\b': 'Rishikesh',
        r'\b(ujjain)\b': 'Ujjain',
        r'\b(bodh gaya|bodhgaya)\b': 'Bodh Gaya',
        r'\b(gaya)\b': 'Gaya',
        r'\b(tirupati)\b': 'Tirupati',
        r'\b(shirdi)\b': 'Shirdi',
        r'\b(dwarka|dwaraka)\b': 'Dwarka',
        r'\b(somnath)\b': 'Somnath',
        r'\b(ajmer)\b': 'Ajmer',
        r'\b(pushkar)\b': 'Pushkar',
        r'\b(rameswaram|rameshwaram)\b': 'Rameswaram',
        r'\b(kanyakumari|cape comorin)\b': 'Kanyakumari',
        r'\b(amaravati)\b': 'Amaravati',
        r'\b(sanchi)\b': 'Sanchi',
        r'\b(sarnath)\b': 'Sarnath',
        r'\b(puri)\b': 'Puri',
        r'\b(jagannath puri)\b': 'Puri',
        r'\b(hampi)\b': 'Hampi',
        r'\b(mahabalipuram|mamallapuram)\b': 'Mahabalipuram',
        r'\b(chidambaram)\b': 'Chidambaram',
        r'\b(kedarnath)\b': 'Kedarnath',
        r'\b(badrinath)\b': 'Badrinath',
        r'\b(gangotri)\b': 'Gangotri',
        r'\b(yamunotri)\b': 'Yamunotri',

        # State capitals and major cities (existing)
        r'\b(jaipur)\b': 'Jaipur',
        r'\b(lucknow)\b': 'Lucknow',
        r'\b(bhopal)\b': 'Bhopal',
        r'\b(patna)\b': 'Patna',
        r'\b(ranchi)\b': 'Ranchi',
        r'\b(bhubaneswar)\b': 'Bhubaneswar',
        r'\b(dehradun)\b': 'Dehradun',
        r'\b(shimla)\b': 'Shimla',
        r'\b(chandigarh)\b': 'Chandigarh',
        r'\b(shillong)\b': 'Shillong',
        r'\b(aizawl)\b': 'Aizawl',
        r'\b(itanagar)\b': 'Itanagar',
        r'\b(agartala)\b': 'Agartala',
        r'\b(gangtok)\b': 'Gangtok',
        r'\b(kohima)\b': 'Kohima',
        r'\b(imphal)\b': 'Imphal',
        r'\b(dispur)\b': 'Dispur',
        r'\b(panaji|panjim)\b': 'Panaji',
        r'\b(thiruvananthapuram|trivandrum)\b': 'Thiruvananthapuram',
        r'\b(kavaratti)\b': 'Kavaratti',
        r'\b(port blair)\b': 'Port Blair',
        r'\b(daman)\b': 'Daman',
        r'\b(silvassa)\b': 'Silvassa',
        r'\b(srinagar|jammu)\b': 'Srinagar/Jammu',
        r'\b(leh)\b': 'Leh',
        r'\b(kargil)\b': 'Kargil',
        r'\b(puducherry|pondicherry)\b': 'Puducherry',
        r'\b(raipur)\b': 'Raipur',
        r'\b(naya raipur|atal nagar)\b': 'Naya Raipur',

        # COASTAL CITIES - WEST COAST
        # Gujarat Coast
        r'\b(kandla)\b': 'Kandla',
        r'\b(jamnagar)\b': 'Jamnagar',
        r'\b(porbandar)\b': 'Porbandar',
        r'\b(veraval)\b': 'Veraval',
        r'\b(bharuch|broach)\b': 'Bharuch',
        r'\b(navsari)\b': 'Navsari',
        r'\b(valsad)\b': 'Valsad',
        r'\b(bhavnagar)\b': 'Bhavnagar',
        r'\b(junagadh)\b': 'Junagadh',
        r'\b(okha)\b': 'Okha',
        r'\b(mandvi)\b': 'Mandvi',

        # Maharashtra Coast
        r'\b(thane)\b': 'Thane',
        r'\b(navi mumbai|new mumbai)\b': 'Navi Mumbai',
        r'\b(alibaug|alibag)\b': 'Alibaug',
        r'\b(ratnagiri)\b': 'Ratnagiri',
        r'\b(sindhudurg)\b': 'Sindhudurg',
        r'\b(malvan)\b': 'Malvan',
        r'\b(murud)\b': 'Murud',
        r'\b(dahanu)\b': 'Dahanu',
        r'\b(vasai)\b': 'Vasai',
        r'\b(virar)\b': 'Virar',
        r'\b(ganpatipule)\b': 'Ganpatipule',
        r'\b(tarkarli)\b': 'Tarkarli',
        r'\b(vengurla)\b': 'Vengurla',

        # Goa Coast
        r'\b(margao|madgaon)\b': 'Margao',
        r'\b(mapusa)\b': 'Mapusa',
        r'\b(vasco da gama|vasco)\b': 'Vasco da Gama',
        r'\b(calangute)\b': 'Calangute',
        r'\b(baga)\b': 'Baga',
        r'\b(anjuna)\b': 'Anjuna',
        r'\b(arambol)\b': 'Arambol',
        r'\b(candolim)\b': 'Candolim',
        r'\b(colva)\b': 'Colva',
        r'\b(benaulim)\b': 'Benaulim',
        r'\b(palolem)\b': 'Palolem',
        r'\b(agonda)\b': 'Agonda',

        # Karnataka Coast
        r'\b(mangalore|mangaluru)\b': 'Mangalore',
        r'\b(udupi)\b': 'Udupi',
        r'\b(karwar)\b': 'Karwar',
        r'\b(kundapura)\b': 'Kundapura',
        r'\b(bhatkal)\b': 'Bhatkal',
        r'\b(kumta)\b': 'Kumta',
        r'\b(gokarna)\b': 'Gokarna',
        r'\b(murdeshwar)\b': 'Murdeshwar',
        r'\b(malpe)\b': 'Malpe',
        r'\b(manipal)\b': 'Manipal',
        r'\b(surathkal)\b': 'Surathkal',

        # Kerala Coast
        r'\b(kozhikode|calicut)\b': 'Kozhikode',
        r'\b(kannur|cannanore)\b': 'Kannur',
        r'\b(kasaragod)\b': 'Kasaragod',
        r'\b(kollam|quilon)\b': 'Kollam',
        r'\b(alappuzha|alleppey)\b': 'Alappuzha',
        r'\b(thrissur|trichur)\b': 'Thrissur',
        r'\b(malappuram)\b': 'Malappuram',
        r'\b(palakkad|palghat)\b': 'Palakkad',
        r'\b(varkala)\b': 'Varkala',
        r'\b(kovalam)\b': 'Kovalam',
        r'\b(bekal)\b': 'Bekal',
        r'\b(kumarakom)\b': 'Kumarakom',
        r'\b(munnar)\b': 'Munnar',
        r'\b(wayanad)\b': 'Wayanad',
        r'\b(thekkady)\b': 'Thekkady',

        # COASTAL CITIES - EAST COAST
        # Tamil Nadu Coast
        r'\b(tuticorin|thoothukudi)\b': 'Tuticorin',
        r'\b(nagapattinam)\b': 'Nagapattinam',
        r'\b(thanjavur|tanjore)\b': 'Thanjavur',
        r'\b(cuddalore)\b': 'Cuddalore',
        r'\b(pondicherry|puducherry)\b': 'Puducherry',
        r'\b(karaikal)\b': 'Karaikal',
        r'\b(villupuram)\b': 'Villupuram',
        r'\b(chidambaram)\b': 'Chidambaram',
        r'\b(kumbakonam)\b': 'Kumbakonam',
        r'\b(mayiladuthurai)\b': 'Mayiladuthurai',
        r'\b(sirkazhi)\b': 'Sirkazhi',
        r'\b(velankanni)\b': 'Velankanni',
        r'\b(marina beach)\b': 'Marina Beach',
        r'\b(ennore)\b': 'Ennore',

        # Andhra Pradesh Coast
        r'\b(visakhapatnam|vizag)\b': 'Visakhapatnam',
        r'\b(guntur)\b': 'Guntur',
        r'\b(nellore)\b': 'Nellore',
        r'\b(machilipatnam|masulipatnam)\b': 'Machilipatnam',
        r'\b(kakinada)\b': 'Kakinada',
        r'\b(rajahmundry|rajamahendravaram)\b': 'Rajahmundry',
        r'\b(eluru)\b': 'Eluru',
        r'\b(bhimavaram)\b': 'Bhimavaram',
        r'\b(ongole)\b': 'Ongole',
        r'\b(chirala)\b': 'Chirala',
        r'\b(bapatla)\b': 'Bapatla',
        r'\b(tenali)\b': 'Tenali',
        r'\b(narasaraopet)\b': 'Narasaraopet',
        r'\b(amalapuram)\b': 'Amalapuram',
        r'\b(tanuku)\b': 'Tanuku',

        # Odisha Coast
        r'\b(cuttack)\b': 'Cuttack',
        r'\b(berhampur|brahmapur)\b': 'Berhampur',
        r'\b(balasore)\b': 'Balasore',
        r'\b(puri)\b': 'Puri',
        r'\b(konark)\b': 'Konark',
        r'\b(gopalpur)\b': 'Gopalpur',
        r'\b(chandrabhaga)\b': 'Chandrabhaga',
        r'\b(chilika)\b': 'Chilika',
        r'\b(paradip|paradeep)\b': 'Paradip',
        r'\b(jagatsinghpur)\b': 'Jagatsinghpur',
        r'\b(kendrapara)\b': 'Kendrapara',
        r'\b(bhadrak)\b': 'Bhadrak',

        # West Bengal Coast
        r'\b(digha)\b': 'Digha',
        r'\b(mandarmani)\b': 'Mandarmani',
        r'\b(bakkhali)\b': 'Bakkhali',
        r'\b(frazerganj)\b': 'Frazerganj',
        r'\b(shankarpur)\b': 'Shankarpur',
        r'\b(tajpur)\b': 'Tajpur',
        r'\b(haldia)\b': 'Haldia',
        r'\b(diamond harbour)\b': 'Diamond Harbour',
        r'\b(kakdwip)\b': 'Kakdwip',
        r'\b(namkhana)\b': 'Namkhana',
        r'\b(sundarbans)\b': 'Sundarbans',

        # ISLAND TERRITORIES
        # Andaman & Nicobar Islands
        r'\b(havelock|swaraj dweep)\b': 'Havelock Island',
        r'\b(neil island|shaheed dweep)\b': 'Neil Island',
        r'\b(ross island)\b': 'Ross Island',
        r'\b(baratang)\b': 'Baratang',
        r'\b(mayabunder)\b': 'Mayabunder',
        r'\b(rangat)\b': 'Rangat',
        r'\b(car nicobar)\b': 'Car Nicobar',
        r'\b(campbell bay)\b': 'Campbell Bay',

        # Lakshadweep Islands
        r'\b(minicoy)\b': 'Minicoy',
        r'\b(agatti)\b': 'Agatti',
        r'\b(bangaram)\b': 'Bangaram',
        r'\b(kadmat)\b': 'Kadmat',
        r'\b(kalpeni)\b': 'Kalpeni',
        r'\b(amini)\b': 'Amini',

        # Diu & Daman (Union Territory coastal areas)
        r'\b(diu)\b': 'Diu',

        # ADDITIONAL COASTAL TOWNS AND BEACHES
        r'\b(besant nagar)\b': 'Besant Nagar',
        r'\b(ecr|east coast road)\b': 'ECR',
        r'\b(mahabalipuram|mamallapuram)\b': 'Mahabalipuram',
        r'\b(yelagiri)\b': 'Yelagiri',
        r'\b(rameshwaram|rameswaram)\b': 'Rameswaram',
        r'\b(dhanushkodi)\b': 'Dhanushkodi',
        r'\b(kodaikanal)\b': 'Kodaikanal',
        r'\b(ooty|udhagamandalam)\b': 'Ooty',
    },
    'hi': {
        # Existing Hindi patterns
        r'मुंबई|मुम्बई': 'Mumbai',
        r'दिल्ली|नई दिल्ली': 'Delhi',
        r'चेन्नई': 'Chennai',
        r'कोलकाता': 'Kolkata',
        r'बेंगलुरु|बेंगलूर': 'Bangalore',
        r'हैदराबाद': 'Hyderabad',
        r'पुणे': 'Pune',
        r'अहमदाबाद': 'Ahmedabad',

        r'सूरत': 'Surat',
        r'इंदौर': 'Indore',
        r'कानपुर': 'Kanpur',
        r'नागपुर': 'Nagpur',
        r'वाराणसी|बनारस|बेन्नारस': 'Varanasi',
        r'इलाहाबाद|प्रयागराज': 'Prayagraj',
        r'लुधियाना': 'Ludhiana',
        r'वडोदरा|बड़ौदा': 'Vadodara',
        r'विजयवाड़ा': 'Vijayawada',
        r'कोयंबटूर': 'Coimbatore',
        r'कोच्चि|कोचीन': 'Kochi',
        r'तिरुचिरापल्ली|त्रिची': 'Tiruchirappalli',
        r'मदुरै': 'Madurai',

        r'अयोध्या': 'Ayodhya',
        r'मथुरा': 'Mathura',
        r'हरिद्वार': 'Haridwar',
        r'ऋषिकेश': 'Rishikesh',
        r'उज्जैन': 'Ujjain',
        r'बोधगया': 'Bodh Gaya',
        r'गया': 'Gaya',
        r'तिरुपति': 'Tirupati',
        r'शिरडी': 'Shirdi',
        r'द्वारका': 'Dwarka',
        r'सोमनाथ': 'Somnath',
        r'अजमेर': 'Ajmer',
        r'पुष्कर': 'Pushkar',
        r'रामेश्वरम': 'Rameswaram',
        r'कन्याकुमारी': 'Kanyakumari',
        r'सांची': 'Sanchi',
        r'सारनाथ': 'Sarnath',
        r'पुरी|जगन्नाथ पुरी': 'Puri',
        r'हम्पी': 'Hampi',
        r'महाबलीपुरम|ममल्लापुरम': 'Mahabalipuram',
        r'चिदंबरम': 'Chidambaram',
        r'केदारनाथ': 'Kedarnath',
        r'बद्रीनाथ': 'Badrinath',
        r'गंगोत्री': 'Gangotri',
        r'यमुनोत्री': 'Yamunotri',

        r'जयपुर': 'Jaipur',
        r'लखनऊ': 'Lucknow',
        r'भोपाल': 'Bhopal',
        r'पटना': 'Patna',
        r'रांची': 'Ranchi',
        r'भुवनेश्वर': 'Bhubaneswar',
        r'देहरादून': 'Dehradun',
        r'शिमला': 'Shimla',
        r'चंडीगढ़': 'Chandigarh',
        r'शिलांग': 'Shillong',
        r'आइजोल': 'Aizawl',
        r'ईटानगर': 'Itanagar',
        r'अगरतला': 'Agartala',
        r'गंगटोक': 'Gangtok',
        r'कोहिमा': 'Kohima',
        r'इंफाल': 'Imphal',
        r'दिसपुर': 'Dispur',
        r'पणजी|पणजीम': 'Panaji',
        r'तिरुवनंतपुरम|त्रिवेंद्रम': 'Thiruvananthapuram',
        r'कवरत्ती': 'Kavaratti',
        r'पोर्ट ब्लेयर': 'Port Blair',
        r'दमण': 'Daman',
        r'सिलवासा': 'Silvassa',
        r'श्रीनगर|जम्मू': 'Srinagar/Jammu',
        r'लेह': 'Leh',
        r'कारगिल': 'Kargil',
        r'पुडुचेरी|पांडिचेरी': 'Puducherry',
        r'रायपुर': 'Raipur',
        r'नया रायपुर|अटल नगर': 'Naya Raipur',

        # COASTAL CITIES IN HINDI
        # Gujarat Coast
        r'कांडला': 'Kandla',
        r'जामनगर': 'Jamnagar',
        r'पोरबंदर': 'Porbandar',
        r'वेरावल': 'Veraval',
        r'भरूच': 'Bharuch',
        r'नवसारी': 'Navsari',
        r'वलसाड': 'Valsad',
        r'भावनगर': 'Bhavnagar',
        r'जूनागढ़': 'Junagadh',
        r'ओखा': 'Okha',

        # Maharashtra Coast
        r'ठाणे': 'Thane',
        r'नवी मुंबई': 'Navi Mumbai',
        r'अलीबाग': 'Alibaug',
        r'रत्नागिरी': 'Ratnagiri',
        r'सिंधुदुर्ग': 'Sindhudurg',
        r'मालवन': 'Malvan',
        r'मुरुड': 'Murud',
        r'दहानू': 'Dahanu',
        r'वसई': 'Vasai',
        r'विरार': 'Virar',

        # Goa Coast
        r'मारगांव|मडगांव': 'Margao',
        r'मापुसा': 'Mapusa',
        r'वास्को डा गामा|वास्को': 'Vasco da Gama',
        r'कलंगुट': 'Calangute',
        r'बागा': 'Baga',
        r'अंजुना': 'Anjuna',
        r'अरामबोल': 'Arambol',
        r'पालोलेम': 'Palolem',

        # Karnataka Coast
        r'मंगलौर|मंगलूरु': 'Mangalore',
        r'उडुपी': 'Udupi',
        r'कारवार': 'Karwar',
        r'कुंडापुर': 'Kundapura',
        r'भटकल': 'Bhatkal',
        r'गोकर्ण': 'Gokarna',
        r'मुरुदेश्वर': 'Murdeshwar',
        r'मणिपाल': 'Manipal',

        # Kerala Coast
        r'कोझिकोड|कालीकट': 'Kozhikode',
        r'कन्नूर': 'Kannur',
        r'कासरगोड': 'Kasaragod',
        r'कोल्लम|क्विलोन': 'Kollam',
        r'अलाप्पुझा|अल्लेप्पी': 'Alappuzha',
        r'त्रिशूर': 'Thrissur',
        r'मलप्पुरम': 'Malappuram',
        r'वरकला': 'Varkala',
        r'कोवलम': 'Kovalam',
        r'कुमारकोम': 'Kumarakom',

        # Tamil Nadu Coast
        r'तूतिकोरिन|तूतुकुडी': 'Tuticorin',
        r'नागपट्टिनम': 'Nagapattinam',
        r'तंजावुर|तंजौर': 'Thanjavur',
        r'कडलूर': 'Cuddalore',
        r'करैकल': 'Karaikal',
        r'वेलांकन्नी': 'Velankanni',

        # Andhra Pradesh Coast
        r'विशाखापत्तनम|विजाग': 'Visakhapatnam',
        r'गुंटूर': 'Guntur',
        r'नेल्लोर': 'Nellore',
        r'मछलीपत्तनम': 'Machilipatnam',
        r'काकीनाडा': 'Kakinada',
        r'राजहमुंद्री': 'Rajahmundry',

        # Odisha Coast
        r'कटक': 'Cuttack',
        r'बेरहामपुर|ब्रह्मपुर': 'Berhampur',
        r'बालासोर': 'Balasore',
        r'कोणार्क': 'Konark',
        r'गोपालपुर': 'Gopalpur',
        r'पारादीप': 'Paradip',

        # West Bengal Coast
        r'दीघा': 'Digha',
        r'मंदारमणि': 'Mandarmani',
        r'बक्खाली': 'Bakkhali',
        r'हलदिया': 'Haldia',
        r'सुंदरबन': 'Sundarbans',

        # Island Territories
        r'हैवलॉक द्वीप|स्वराज द्वीप': 'Havelock Island',
        r'नील द्वीप|शहीद द्वीप': 'Neil Island',
        r'रॉस द्वीप': 'Ross Island',
        r'मिनिकॉय': 'Minicoy',
        r'अगत्ति': 'Agatti',
        r'दीव': 'Diu',
    }}
        return patterns.get(language, patterns['en'])

    def calculate_urgency_score(self, text: str, hazard_type: str, confidence: float, sentiment: str) -> float:
        urgency_score = 0.0
        
        urgency_score += confidence * 0.3
        
        hazard_urgency = {
            'tsunami': 1.0,
            'earthquake': 0.95,
            'flash_flood': 0.9,
            'cyclone': 0.85,
            'flood': 0.7,
            'wildfire': 0.90,
            'landslide': 0.8
        }
        urgency_score += hazard_urgency.get(hazard_type, 0.5) * 0.3
        
        sentiment_impact = {
            'negative/urgent': 0.3,
            'neutral/concerning': 0.2,
            'positive/informative': 0.1
        }
        urgency_score += sentiment_impact.get(sentiment, 0.1)
        
        urgent_keywords = [
            'emergency', 'urgent', 'help', 'rescue', 'trapped', 'immediate','severe',
            'आपातकाल', 'मदद', 'बचाव', 'फंसे', 'तुरंत',
            'জরুরি', 'সাহায্য', 'উদ্ধার',
            'அவசர', 'உதவி', 'மீட்பு'
        ]
        
        urgent_count = sum(1 for keyword in urgent_keywords if keyword.lower() in text.lower())
        urgency_score += min(urgent_count * 0.1, 0.1)
        
        return min(urgency_score, 1.0)
    
    def analyze_sentiment_advanced(self, text: str, language: str) -> str:
       
        cache_key = hash(f"{text}_{language}")
        if hasattr(self, '_sentiment_cache') and cache_key in self._sentiment_cache:
            return self._sentiment_cache[cache_key]
            
        try:
           
            if len(text) > 500:
                text = text[:500]
                
            sentiment_result = self.sentiment_analyzer(text)
            label = sentiment_result[0]['label']
            score = sentiment_result[0]['score']
            
            if label in ['NEGATIVE'] and score > 0.7:
                sentiment = "negative/urgent"
            elif label in ['NEGATIVE']:
                sentiment = "neutral/concerning"
            else:
                sentiment = "positive/informative"
                
        except Exception:
            sentiment = self._analyze_sentiment_keywords(text, language)
        
       
        if not hasattr(self, '_sentiment_cache'):
            self._sentiment_cache = {}
            
      
        if len(self._sentiment_cache) > 1000:
           
            self._sentiment_cache.pop(next(iter(self._sentiment_cache)))
            
        self._sentiment_cache[cache_key] = sentiment
        return sentiment
    
    def _analyze_sentiment_keywords(self, text: str, language: str) -> str:
        negative_words = {
            'en': ['terrible', 'awful', 'disaster', 'emergency', 'help', 'critical', 'severe'],
            'hi': ['भयानक', 'आपातकाल', 'मदद', 'समस्या', 'गंभीर', 'खतरनाक'],
            'bn': ['ভয়ানক', 'জরুরি', 'সাহায্য', 'সমস্যা', 'গুরুতর'],
            'ta': ['பயங்கரமான', 'அவசர', 'உதவி', 'கடுமையான'],
            'te': ['భయంకరమైన', 'అత్యవసర', 'సహాయం', 'తీవ్రమైన'],
        }
        
        words = negative_words.get(language, negative_words['en'])
        negative_count = sum(1 for word in words if word.lower() in text.lower())
        
        if negative_count > 3:
            return "negative/urgent"
        elif negative_count > 1:
            return "neutral/concerning"
        else:
            return "positive/informative"
    
    def determine_severity(self, confidence: float, urgency_score: float, hazard_type: str) -> str:
        hazard_severity_multiplier = {
            'tsunami': 1.4,
            'earthquake': 1.2,
            'cyclone': 0.9,
            'flood': 1.0,
            'wildfire': 1.3,
            'landslide': 0.9
        }
        
        severity_score = (confidence * 0.7 + urgency_score * 0.3) * hazard_severity_multiplier.get(hazard_type, 1.0)
        
        if severity_score >= self.severity_thresholds['critical']:
            return 'critical'
        elif severity_score >= self.severity_thresholds['high']:
            return 'high'
        elif severity_score >= self.severity_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def process_multilingual_report_enhanced(self, text: str, source: str = "user_report") -> HazardReport:
        try:
            import concurrent.futures
            
           
            language = self.detect_language_advanced(text)
            
           
            def detect_hazard():
                return self.detect_hazard_advanced(text)
                
            def extract_locations():
                return self.extract_locations_advanced(text, language)
                
            def analyze_sentiment():
                return self.analyze_sentiment_advanced(text, language)
            
          
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
               
                hazard_future = executor.submit(detect_hazard)
                locations_future = executor.submit(extract_locations)
                sentiment_future = executor.submit(analyze_sentiment)
                
              
                hazard_type, confidence, detection_details = hazard_future.result()
                locations = locations_future.result()
                sentiment = sentiment_future.result()
            
            urgency_score = self.calculate_urgency_score(text, hazard_type, confidence, sentiment)
            severity = self.determine_severity(confidence, urgency_score, hazard_type)
            
            report = HazardReport(
                text=text,
                timestamp=datetime.now(),
                language=language,
                hazard_type=hazard_type,
                confidence=float(confidence),
                severity=severity,
                locations=locations,
                sentiment=sentiment,
                urgency_score=float(urgency_score),
                source=source
            )
            
            self.historical_reports.append(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error processing report: {e}")
            return HazardReport(
                text=text,
                timestamp = datetime.now(),
                language='en',
                hazard_type='unknown',
                 confidence=float(confidence),
                severity='low',
                locations=[{'name': 'unknown'}],
                sentiment=sentiment,
                urgency_score=0.0,
                source=source
            )
    
    def batch_process_reports(self, reports: List[str], sources: List[str] = None) -> List[HazardReport]:
        if sources is None:
            sources = ['user_report'] * len(reports)
            
        import concurrent.futures
        import os
        
       
        max_workers = min(32, (os.cpu_count() or 4) * 2)
        
        processed_reports = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
           
            future_to_report = {executor.submit(self.process_multilingual_report_enhanced, text, source): (text, source) 
                               for text, source in zip(reports, sources)}
            
           
            for future in concurrent.futures.as_completed(future_to_report):
                try:
                    report = future.result()
                    processed_reports.append(report)
                except Exception as e:
                    
                    text, source = future_to_report[future]
                    logger.error(f"Error processing report: {str(e)}")
                   
                    processed_reports.append(HazardReport(
                        text=text,
                        timestamp=datetime.now(),
                        language='en',
                        hazard_type='unknown',
                        confidence=0.0,
                        severity='low',
                        locations=[{'name': 'unknown', 'confidence': 0.0}],
                        sentiment='neutral/concerning',
                        urgency_score=0.0,
                        source=source
                    ))
        
        return processed_reports
    
    def get_trend_analysis(self, days: int = 7) -> Dict:
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_reports = [r for r in self.historical_reports if r.timestamp >= cutoff_date]
        
        if not recent_reports:
            return {"message": "No recent reports available for trend analysis"}
        
        hazard_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        language_counts = defaultdict(int)
        
        for report in recent_reports:
            hazard_counts[report.hazard_type] += 1
            severity_counts[report.severity] += 1
            language_counts[report.language] += 1
        
        return {
            "total_reports": len(recent_reports),
            "hazard_distribution": dict(hazard_counts),
            "severity_distribution": dict(severity_counts),
            "language_distribution": dict(language_counts),
            "average_confidence": float(np.mean([r.confidence for r in recent_reports])),
            "average_urgency": float(np.mean([r.urgency_score for r in recent_reports]))
        }
    
    def export_reports(self, format: str = "json", filename: str = None) -> str:
        if not self.historical_reports:
            return "No reports available for export"
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hazard_reports_{timestamp}"
        
        if format.lower() == "json":
            data = []
            for report in self.historical_reports:
                location_data = []
                for loc in report.locations:
                    loc_dict = {}
                    for key, value in loc.items():
                        if isinstance(value, (np.floating, np.integer)):
                            loc_dict[key] = float(value)
                        else:
                            loc_dict[key] = value
                    location_data.append(loc_dict)
                
                data.append({
                    "text": report.text,
                    "timestamp": report.timestamp.isoformat(),
                    "language": report.language,
                    "hazard_type": report.hazard_type,
                    "confidence": float(report.confidence),
                    "severity": report.severity,
                    "locations": location_data,
                    "sentiment": report.sentiment,
                    "urgency_score": float(report.urgency_score),
                    "source": report.source,
                    "verified": report.verified
                })
            
            with open(f"{filename}.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return f"Reports exported to {filename}.json"
        
        elif format.lower() == "csv":
            import csv
            with open(f"{filename}.csv", 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'text', 'language', 'hazard_type', 'confidence',
                    'severity', 'sentiment', 'urgency_score', 'locations', 'source', 'verified'
                ])
                
                for report in self.historical_reports:
                    locations_str = '; '.join([loc['name'] for loc in report.locations])
                    writer.writerow([
                        report.timestamp.isoformat(),
                        report.text,
                        report.language,
                        report.hazard_type,
                        float(report.confidence),
                        report.severity,
                        report.sentiment,
                        float(report.urgency_score),
                        locations_str,
                        report.source,
                        report.verified
                    ])
            
            return f"Reports exported to {filename}.csv"
        
        else:
            return f"Unsupported format: {format}. Use 'json' or 'csv'"
    
    def cluster_similar_reports(self, threshold: float = 0.8) -> Dict:
        if len(self.historical_reports) < 2:
            return {"message": "Not enough reports for clustering"}
        
        texts = [report.text for report in self.historical_reports]
        embeddings = self.extract_embeddings_batch(texts)
        
        clustering = DBSCAN(eps=1-threshold, metric='cosine', min_samples=2)
        clusters = clustering.fit_predict(embeddings)
        
        clustered_reports = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            clustered_reports[cluster_id].append({
                'report_index': i,
                'report': self.historical_reports[i],
                'similarity_score': 1.0
            })
        
        return {
            "total_clusters": len([c for c in clustered_reports.keys() if c != -1]),
            "noise_reports": len(clustered_reports.get(-1, [])),
            "clusters": dict(clustered_reports)
        }
    
    async def real_time_monitoring(self, data_sources: List[str], interval: int = 300):
        logger.info(f"Starting real-time monitoring with {interval} second intervals")
        
       
        processed_cache = set()
        
        async def fetch_from_source(session, source_url):
            try:
                async with session.get(source_url) as response:
                    if response.status == 200:
                        data = await response.text()
                        return self._parse_source_data(data, source_url)
                    return []
            except Exception as e:
                logger.error(f"Error fetching from {source_url}: {e}")
                return []
        
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    tasks = [fetch_from_source(session, source) for source in data_sources]
                    results = await asyncio.gather(*tasks)
                    
                    for source_reports in results:
                        if source_reports:
                            
                            new_reports = []
                            for r in source_reports:
                                content_hash = hash(r['text'])
                                if content_hash not in processed_cache:
                                    processed_cache.add(content_hash)
                                    new_reports.append(r)
                            
                            if not new_reports:
                                continue
                                
                            logger.info(f"Processing {len(new_reports)} new reports")
                            
                           
                            batch_size = 10 
                            for i in range(0, len(new_reports), batch_size):
                                batch = new_reports[i:i+batch_size]
                                batch_reports = self.batch_process_reports(
                                    [r['text'] for r in batch],
                                    [r['source'] for r in batch]
                                )
                                
                                for report in batch_reports:
                                    if report.severity in ['critical', 'high'] and report.confidence > 0.7:
                                        logger.warning(f"HIGH PRIORITY ALERT: {report.hazard_type} detected in {report.locations[0]['name']}")
                    
                   
                    if len(processed_cache) > 10000:
                        processed_cache = set(list(processed_cache)[-5000:])
                        
                    await asyncio.sleep(interval)
                    
                except KeyboardInterrupt:
                    logger.info("Real-time monitoring stopped")
                    break
                except Exception as e:
                    logger.error(f"Error in real-time monitoring: {e}")
                    
                    await asyncio.sleep(min(interval, 30))
    
    def set_twitter_credentials(self, api_key: str, api_secret: str, access_token: str, access_token_secret: str):
        """Set up Twitter API credentials for real-time monitoring"""
        try:
            import tweepy
            auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
            self.twitter_api = tweepy.API(auth)
            
            self.twitter_api.verify_credentials()
            logger.info("Twitter API credentials verified successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting up Twitter API: {e}")
            return False
    
    def set_reddit_credentials(self, client_id: str, client_secret: str, user_agent: str):
        """Set up Reddit API credentials for real-time monitoring"""
        try:
            import praw
            self.reddit_api = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
           
            username = self.reddit_api.user.me()
            logger.info(f"Reddit API credentials verified successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting up Reddit API: {e}")
            return False
    
    def _parse_source_data(self, data: str, source_url: str) -> List[Dict]:
        parsed_reports = []
        
        try:
            if 'json' in source_url.lower():
                json_data = json.loads(data)
                if isinstance(json_data, list):
                    for item in json_data:
                        if 'text' in item or 'content' in item:
                            parsed_reports.append({
                                'text': item.get('text', item.get('content', '')),
                                'source': f"feed_{source_url.split('/')[-1]}"
                            })
            
            elif 'rss' in source_url.lower() or 'xml' in source_url.lower():
                import feedparser
                feed = feedparser.parse(data)
                
                for entry in feed.entries:
                    text = f"{entry.title}. {entry.get('description', '')}"
                    parsed_reports.append({
                        'text': text,
                        'source': entry.get('link', source_url),
                        'timestamp': datetime.now()
                    })
            
           
            elif 'twitter' in source_url.lower() or 'x.com' in source_url.lower():
                if hasattr(self, 'twitter_api'):
                    import tweepy
                  
                    query = source_url.split('/')[-1] if '/' in source_url else source_url
                    
                    
                    tweets = self.twitter_api.search_tweets(q=query, count=100, tweet_mode='extended')
                    
                    for tweet in tweets:
                        parsed_reports.append({
                            'text': tweet.full_text,
                            'source': f"https://twitter.com/user/status/{tweet.id}",
                            'timestamp': tweet.created_at,
                            'location': tweet.user.location if hasattr(tweet.user, 'location') else None,
                            'user': tweet.user.screen_name
                        })
                else:
                    logger.warning("Twitter API credentials not set. Use set_twitter_credentials() method first.")
            
           
            elif 'reddit' in source_url.lower():
                if hasattr(self, 'reddit_api'):
                    import praw
                    
                   
                    if 'r/' in source_url:
                      
                        subreddit_name = source_url.split('r/')[-1].split('/')[0]
                        submissions = self.reddit_api.subreddit(subreddit_name).hot(limit=50)
                    elif 'search' in source_url:
                       
                        query = source_url.split('search/')[-1]
                        submissions = self.reddit_api.subreddit('all').search(query, limit=50)
                    else:
                       
                        submission = self.reddit_api.submission(url=source_url)
                        submissions = [submission]
                    
                    for submission in submissions:
                        
                        text = f"{submission.title}. {submission.selftext}"
                        
                        parsed_reports.append({
                            'text': text,
                            'source': f"https://reddit.com{submission.permalink}",
                            'timestamp': datetime.fromtimestamp(submission.created_utc),
                            'subreddit': submission.subreddit.display_name
                        })
                        
                       
                        submission.comments.replace_more(limit=0)  
                        for comment in submission.comments[:20]:  
                            parsed_reports.append({
                                'text': comment.body,
                                'source': f"https://reddit.com{comment.permalink}",
                                'timestamp': datetime.fromtimestamp(comment.created_utc),
                                'subreddit': submission.subreddit.display_name
                            })
                else:
                    logger.warning("Reddit API credentials not set. Use set_reddit_credentials() method first.")
            
            else:
                lines = data.split('\n')
                for line in lines:
                    if line.strip():
                        parsed_reports.append({
                            'text': line.strip(),
                            'source': f"text_{source_url.split('/')[-1]}"
                        })
        
        except Exception as e:
            logger.error(f"Error parsing data from {source_url}: {e}")
        
        return parsed_reports
    
    def generate_alert_summary(self, severity_threshold: str = "medium") -> Dict:
        severity_levels = ['low', 'medium', 'high', 'critical']
        threshold_index = severity_levels.index(severity_threshold)
        
        alert_reports = [
            r for r in self.historical_reports 
            if severity_levels.index(r.severity) >= threshold_index
        ]
        
        if not alert_reports:
            return {"message": f"No alerts above {severity_threshold} severity level"}
        
        alerts_by_type = defaultdict(list)
        alerts_by_location = defaultdict(list)
        
        for report in alert_reports:
            alerts_by_type[report.hazard_type].append(report)
            for location in report.locations:
                if location['name'] != 'unknown':
                    alerts_by_location[location['name']].append(report)
        
        summary = {
            "total_alerts": len(alert_reports),
            "severity_threshold": severity_threshold,
            "time_range": {
                "earliest": min(r.timestamp for r in alert_reports).isoformat(),
                "latest": max(r.timestamp for r in alert_reports).isoformat()
            },
            "hazard_breakdown": {
                hazard_type: {
                    "count": len(reports),
                    "avg_confidence": float(np.mean([r.confidence for r in reports])),
                    "avg_urgency": float(np.mean([r.urgency_score for r in reports]))
                }
                for hazard_type, reports in alerts_by_type.items()
            },
            "location_breakdown": {
                location: {
                    "count": len(reports),
                    "hazard_types": list(set(r.hazard_type for r in reports))
                }
                for location, reports in alerts_by_location.items()
            },
            "most_urgent": max(alert_reports, key=lambda r: r.urgency_score).__dict__
        }
        
        return summary





# === Example Integration with Hazard Detector ===
if __name__ == "__main__":
    print("=== Enhanced Multilingual Hazard Detection System ===\n")

    detector = EnhancedMultilingualHazardDetector(use_gpu=False)

    # Fetch daily news articles
    daily_news_reports = fetch_daily_news()

    processed_reports = []
    for i, report_data in enumerate(daily_news_reports, 1):
        print(f"\nProcessing News Report {i}:")
        print(f"Source: {report_data['source']}")
        print(f"Text: {report_data['text']}")

        # Pass each article into your hazard detection pipeline
        # (replace with your actual detector object & method)
        # Example:
        report = detector.process_multilingual_report_enhanced(report_data['text'], report_data['source'])
        processed_reports.append(report)

        # For now, just print out placeholder
        print(f"Processed => Hazard Type: flood/earthquake/... (example)")
        print("-" * 50)

    print("\n=== News Processing Done ===")
    
if __name__ == "__main__":
    detector = EnhancedMultilingualHazardDetector(use_gpu=False)
    enhanced_test_reports = [
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
    print("=== Enhanced Multilingual Hazard Detection System ===\n")
    
    processed_reports = []
    for i, report_data in enumerate(enhanced_test_reports, 1):
        print(f"Processing Report {i}:")
        print(f"Source: {report_data['source']}")
        print(f"Text: {report_data['text']}")
        
        report = detector.process_multilingual_report_enhanced(
            report_data['text'], 
            report_data['source']
        )
        processed_reports.append(report)
        
        print(f"Results:")
        print(f"  Language: {report.language}")
        print(f"  Hazard Type: {report.hazard_type}")
        print(f"  Confidence: {report.confidence}")
        print(f"  Severity: {report.severity}")
        print(f"  Urgency Score: {report.urgency_score}")
        print(f"  Sentiment: {report.sentiment}")
        print(f"  Locations: {[loc['name'] for loc in report.locations]}")
        print(f"  Timestamp: {report.timestamp}")
        print("-" * 50)
    
    print("\n=== Trend Analysis ===")
    trends = detector.get_trend_analysis(days=1)
    for key, value in trends.items():
        print(f"{key}: {value}")
    
    print("\n=== Alert Summary (Medium+ Severity) ===")
    alerts = detector.generate_alert_summary("medium")
    if "message" not in alerts:
        print(f"Total Alerts: {alerts['total_alerts']}")
        print("Hazard Breakdown:")
        for hazard, stats in alerts['hazard_breakdown'].items():
            print(f"  {hazard}: {stats['count']} reports (avg confidence: {stats['avg_confidence']:.3f})")
        
        print("Location Breakdown:")
        for location, stats in alerts['location_breakdown'].items():
            print(f"  {location}: {stats['count']} reports - {', '.join(stats['hazard_types'])}")
    else:
        print(alerts['message'])
    
    print("\n=== Similarity Clustering ===")
    clusters = detector.cluster_similar_reports(threshold=0.7)
    if "message" not in clusters:
        print(f"Found {clusters['total_clusters']} clusters")
        print(f"Noise reports: {clusters['noise_reports']}")
    else:
        print(clusters['message'])
    
    print("\n=== Export Reports ===")
    export_result = detector.export_reports("json", "sample_hazard_reports")
    print(export_result)
    
    print("\n=== System Performance Summary ===")
    print(f"Total reports processed: {len(processed_reports)}")
    print(f"Languages detected: {len(set(r.language for r in processed_reports))}")
    print(f"Hazard types identified: {len(set(r.hazard_type for r in processed_reports if r.hazard_type != 'unknown'))}")
    print(f"Average processing confidence: {np.mean([r.confidence for r in processed_reports]):.3f}")
    print(f"High-priority alerts: {len([r for r in processed_reports if r.severity in ['high', 'critical']])}")
    
    print("\n=== Starting Real-time Monitoring (Example) ===")
    data_sources = [
    "https://news.google.com/rss/search?q=flood+OR+earthquake+OR+tsunami&hl=en-IN&gl=IN&ceid=IN:en",
]
