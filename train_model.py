#!/usr/bin/env python3
"""
SMS Spam Detection Model Training Script
Trains a machine learning model on the spam collection dataset
"""

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

class SMSFeatureExtractor:
    """Extract features from SMS text for spam detection"""
    
    def __init__(self):
        self.spam_keywords = [
            'free', 'win', 'winner', 'congratulations', 'urgent', 'limited time',
            'click here', 'act now', 'cash prize', 'guaranteed', 'no obligation',
            'risk free', 'special promotion', 'limited offer', 'exclusive deal',
            'call now', 'text stop', 'unsubscribe', 'opt out', 'sms', 'txt',
            'mobile', 'phone', 'call', 'text', 'reply', 'stop', 'end',
            'prize', 'award', 'claim', 'winner', 'selected', 'chosen',
            'congratulations', 'urgent', 'immediate', 'expires', 'limited',
            'offer', 'deal', 'discount', 'save', 'money', 'cash', 'dollar',
            'pound', 'euro', '£', '$', '€', 'million', 'thousand',
            'subscription', 'service', 'charge', 'billing', 'payment',
            'credit', 'card', 'account', 'bank', 'security', 'verify',
            'confirm', 'update', 'suspended', 'blocked', 'expired',
            'click', 'link', 'website', 'www', 'http', 'https', 'com',
            'download', 'install', 'software', 'virus', 'malware',
            'lottery', 'raffle', 'contest', 'competition', 'draw',
            'ticket', 'entry', 'participate', 'join', 'register',
            'subscribe', 'newsletter', 'alert', 'notification',
            'reminder', 'appointment', 'meeting', 'schedule',
            'delivery', 'shipping', 'order', 'purchase', 'buy',
            'sell', 'sale', 'auction', 'bid', 'offer', 'price',
            'cost', 'fee', 'charge', 'payment', 'billing', 'invoice',
            'receipt', 'refund', 'return', 'exchange', 'warranty',
            'insurance', 'coverage', 'policy', 'claim', 'settlement',
            'loan', 'credit', 'debt', 'mortgage', 'finance', 'investment',
            'stock', 'share', 'trading', 'broker', 'advisor',
            'consultant', 'expert', 'professional', 'specialist',
            'doctor', 'lawyer', 'attorney', 'counsel', 'legal',
            'medical', 'health', 'treatment', 'therapy', 'cure',
            'medicine', 'drug', 'pharmacy', 'prescription', 'dosage',
            'side effect', 'adverse', 'warning', 'caution', 'danger',
            'risk', 'hazard', 'safety', 'security', 'protection',
            'privacy', 'confidential', 'secret', 'private', 'personal',
            'identity', 'theft', 'fraud', 'scam', 'fake', 'phishing',
            'spoofing', 'hacking', 'breach', 'leak', 'exposure',
            'data', 'information', 'record', 'file', 'document',
            'report', 'statement', 'summary', 'analysis', 'review',
            'rating', 'score', 'grade', 'rank', 'position', 'status',
            'level', 'tier', 'category', 'class', 'type', 'kind',
            'sort', 'variety', 'selection', 'choice', 'option',
            'alternative', 'substitute', 'replacement', 'backup',
            'copy', 'duplicate', 'original', 'genuine', 'authentic',
            'real', 'true', 'false', 'fake', 'counterfeit', 'imitation',
            'replica', 'model', 'sample', 'example', 'instance',
            'case', 'scenario', 'situation', 'circumstance', 'condition',
            'state', 'status', 'position', 'location', 'place',
            'address', 'contact', 'phone', 'mobile', 'cell', 'number',
            'email', 'mail', 'message', 'text', 'sms', 'notification',
            'alert', 'reminder', 'announcement', 'notice', 'update',
            'news', 'information', 'data', 'details', 'facts',
            'statistics', 'numbers', 'figures', 'results', 'outcomes',
            'consequences', 'effects', 'impacts', 'influences', 'changes',
            'modifications', 'adjustments', 'improvements', 'enhancements',
            'upgrades', 'updates', 'revisions', 'corrections', 'fixes',
            'solutions', 'answers', 'responses', 'replies', 'feedback',
            'comments', 'suggestions', 'recommendations', 'advice',
            'tips', 'hints', 'clues', 'signs', 'indicators', 'markers',
            'flags', 'warnings', 'alerts', 'notifications', 'messages',
            'communications', 'correspondence', 'letters', 'emails',
            'calls', 'contacts', 'connections', 'relationships',
            'associations', 'partnerships', 'collaborations', 'alliances',
            'agreements', 'contracts', 'deals', 'arrangements', 'plans',
            'strategies', 'approaches', 'methods', 'techniques', 'procedures',
            'processes', 'systems', 'frameworks', 'structures', 'models',
            'patterns', 'templates', 'formats', 'standards', 'guidelines',
            'rules', 'regulations', 'policies', 'procedures', 'protocols',
            'requirements', 'specifications', 'criteria', 'conditions',
            'terms', 'clauses', 'sections', 'parts', 'components',
            'elements', 'factors', 'variables', 'parameters', 'settings',
            'configurations', 'customizations', 'personalizations',
            'adaptations', 'modifications', 'adjustments', 'changes',
            'updates', 'upgrades', 'improvements', 'enhancements',
            'optimizations', 'refinements', 'revisions', 'corrections',
            'fixes', 'repairs', 'maintenance', 'servicing', 'support',
            'assistance', 'help', 'aid', 'guidance', 'direction',
            'instruction', 'education', 'training', 'learning', 'teaching',
            'coaching', 'mentoring', 'advising', 'consulting', 'counseling',
            'therapy', 'treatment', 'healing', 'recovery', 'rehabilitation',
            'restoration', 'renewal', 'revival', 'refreshment', 'rejuvenation',
            'regeneration', 'reconstruction', 'rebuilding', 'redevelopment',
            'renovation', 'remodeling', 'refurbishment', 'restoration',
            'conservation', 'preservation', 'protection', 'maintenance',
            'care', 'attention', 'focus', 'concentration', 'emphasis',
            'priority', 'importance', 'significance', 'relevance',
            'applicability', 'usefulness', 'value', 'worth', 'benefit',
            'advantage', 'merit', 'quality', 'excellence', 'superiority',
            'premium', 'deluxe', 'luxury', 'exclusive', 'special',
            'unique', 'rare', 'uncommon', 'unusual', 'extraordinary',
            'exceptional', 'outstanding', 'remarkable', 'notable',
            'significant', 'important', 'crucial', 'critical', 'essential',
            'vital', 'necessary', 'required', 'mandatory', 'compulsory',
            'obligatory', 'binding', 'enforceable', 'valid', 'legal',
            'legitimate', 'authorized', 'approved', 'certified', 'verified',
            'confirmed', 'authenticated', 'validated', 'tested', 'proven',
            'reliable', 'trustworthy', 'credible', 'believable', 'convincing',
            'persuasive', 'compelling', 'attractive', 'appealing', 'desirable',
            'wanted', 'needed', 'required', 'demanded', 'requested',
            'ordered', 'purchased', 'bought', 'acquired', 'obtained',
            'received', 'delivered', 'shipped', 'sent', 'dispatched',
            'transmitted', 'transferred', 'moved', 'relocated', 'transported',
            'carried', 'conveyed', 'delivered', 'handed', 'passed',
            'given', 'provided', 'supplied', 'furnished', 'equipped',
            'outfitted', 'prepared', 'ready', 'available', 'accessible',
            'obtainable', 'achievable', 'attainable', 'reachable',
            'approachable', 'contactable', 'reachable', 'available',
            'present', 'here', 'there', 'everywhere', 'anywhere',
            'somewhere', 'nowhere', 'always', 'never', 'sometimes',
            'often', 'rarely', 'occasionally', 'frequently', 'regularly',
            'constantly', 'continuously', 'permanently', 'temporarily',
            'briefly', 'momentarily', 'instantly', 'immediately',
            'quickly', 'rapidly', 'swiftly', 'speedily', 'fast',
            'slow', 'gradual', 'steady', 'consistent', 'stable',
            'reliable', 'dependable', 'trustworthy', 'faithful',
            'loyal', 'devoted', 'committed', 'dedicated', 'focused',
            'determined', 'persistent', 'tenacious', 'resilient',
            'strong', 'powerful', 'mighty', 'forceful', 'intense',
            'severe', 'serious', 'grave', 'critical', 'urgent',
            'immediate', 'instant', 'quick', 'fast', 'rapid',
            'swift', 'speedy', 'hasty', 'hurried', 'rushed',
            'pressed', 'stressed', 'strained', 'tension', 'pressure',
            'burden', 'load', 'weight', 'responsibility', 'duty',
            'obligation', 'commitment', 'promise', 'pledge', 'vow',
            'oath', 'swear', 'declare', 'announce', 'proclaim',
            'state', 'say', 'tell', 'speak', 'talk', 'discuss',
            'converse', 'chat', 'communicate', 'correspond', 'contact',
            'reach', 'connect', 'link', 'join', 'unite', 'combine',
            'merge', 'blend', 'mix', 'integrate', 'incorporate',
            'include', 'involve', 'participate', 'engage', 'take part',
            'join in', 'contribute', 'add', 'supply', 'provide',
            'offer', 'give', 'present', 'deliver', 'hand over',
            'transfer', 'pass', 'move', 'shift', 'relocate',
            'transport', 'carry', 'convey', 'bring', 'take',
            'fetch', 'get', 'obtain', 'acquire', 'gain', 'win',
            'earn', 'achieve', 'accomplish', 'complete', 'finish',
            'end', 'stop', 'halt', 'pause', 'break', 'rest',
            'relax', 'unwind', 'calm', 'peaceful', 'quiet',
            'silent', 'still', 'motionless', 'static', 'stable',
            'steady', 'firm', 'solid', 'strong', 'tough', 'hard',
            'difficult', 'challenging', 'complex', 'complicated',
            'intricate', 'sophisticated', 'advanced', 'modern',
            'contemporary', 'current', 'latest', 'newest', 'recent',
            'fresh', 'new', 'novel', 'original', 'unique', 'special',
            'particular', 'specific', 'individual', 'personal',
            'private', 'confidential', 'secret', 'hidden', 'concealed',
            'covered', 'protected', 'secured', 'safe', 'secure',
            'reliable', 'dependable', 'trustworthy', 'faithful',
            'loyal', 'devoted', 'committed', 'dedicated', 'focused',
            'determined', 'persistent', 'tenacious', 'resilient',
            'strong', 'powerful', 'mighty', 'forceful', 'intense',
            'severe', 'serious', 'grave', 'critical', 'urgent',
            'immediate', 'instant', 'quick', 'fast', 'rapid',
            'swift', 'speedy', 'hasty', 'hurried', 'rushed',
            'pressed', 'stressed', 'strained', 'tension', 'pressure',
            'burden', 'load', 'weight', 'responsibility', 'duty',
            'obligation', 'commitment', 'promise', 'pledge', 'vow',
            'oath', 'swear', 'declare', 'announce', 'proclaim',
            'state', 'say', 'tell', 'speak', 'talk', 'discuss',
            'converse', 'chat', 'communicate', 'correspond', 'contact',
            'reach', 'connect', 'link', 'join', 'unite', 'combine',
            'merge', 'blend', 'mix', 'integrate', 'incorporate',
            'include', 'involve', 'participate', 'engage', 'take part',
            'join in', 'contribute', 'add', 'supply', 'provide',
            'offer', 'give', 'present', 'deliver', 'hand over',
            'transfer', 'pass', 'move', 'shift', 'relocate',
            'transport', 'carry', 'convey', 'bring', 'take',
            'fetch', 'get', 'obtain', 'acquire', 'gain', 'win',
            'earn', 'achieve', 'accomplish', 'complete', 'finish',
            'end', 'stop', 'halt', 'pause', 'break', 'rest',
            'relax', 'unwind', 'calm', 'peaceful', 'quiet',
            'silent', 'still', 'motionless', 'static', 'stable',
            'steady', 'firm', 'solid', 'strong', 'tough', 'hard',
            'difficult', 'challenging', 'complex', 'complicated',
            'intricate', 'sophisticated', 'advanced', 'modern',
            'contemporary', 'current', 'latest', 'newest', 'recent',
            'fresh', 'new', 'novel', 'original', 'unique', 'special',
            'particular', 'specific', 'individual', 'personal',
            'private', 'confidential', 'secret', 'hidden', 'concealed',
            'covered', 'protected', 'secured', 'safe', 'secure'
        ]
        
        self.spam_patterns = [
            r'\b\d{4,}\b',  # Long numbers
            r'[A-Z]{3,}',   # Multiple caps
            r'[!]{2,}',     # Multiple exclamation marks
            r'\$[0-9]+',    # Dollar amounts
            r'%[0-9]+',     # Percentages
            r'http[s]?://',  # URLs
            r'www\.',       # www links
            r'\b[A-Z]{2,}\b',  # All caps words
            r'\b\d{3,}\b',     # Numbers with 3+ digits
            r'[!?]{2,}',       # Multiple punctuation
            r'\b(?:call|text|sms|txt)\b',  # Action words
            r'\b(?:now|today|immediately|urgent)\b',  # Urgency words
        ]
    
    def extract_features(self, text):
        """Extract comprehensive features from SMS text"""
        features = []
        
        # Basic text features
        features.append(len(text))  # Length
        features.append(len(text.split()))  # Word count
        features.append(len(text.split('\n')))  # Line count
        
        # Character features
        features.append(text.count('!'))  # Exclamation marks
        features.append(text.count('?'))   # Question marks
        features.append(text.count('$'))   # Dollar signs
        features.append(text.count('%'))   # Percent signs
        features.append(text.count('@'))   # At symbols
        features.append(text.count('#'))   # Hash symbols
        features.append(text.count('*'))   # Asterisks
        
        # Case features
        uppercase_count = sum(1 for c in text if c.isupper())
        features.append(uppercase_count)
        features.append(uppercase_count / len(text) if len(text) > 0 else 0)  # Uppercase ratio
        
        # Number features
        number_count = sum(1 for c in text if c.isdigit())
        features.append(number_count)
        features.append(number_count / len(text) if len(text) > 0 else 0)  # Number ratio
        
        # Spam keyword features
        text_lower = text.lower()
        spam_keyword_count = sum(1 for keyword in self.spam_keywords if keyword in text_lower)
        features.append(spam_keyword_count)
        features.append(spam_keyword_count / len(text.split()) if len(text.split()) > 0 else 0)
        
        # Pattern matching features
        pattern_matches = sum(1 for pattern in self.spam_patterns if re.search(pattern, text))
        features.append(pattern_matches)
        
        # Special character features
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        features.append(special_chars)
        features.append(special_chars / len(text) if len(text) > 0 else 0)
        
        # Word length features
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            features.append(avg_word_length)
            features.append(max(len(word) for word in words))  # Max word length
        else:
            features.extend([0, 0])
        
        # Repetition features
        features.append(len(set(words)) / len(words) if words else 0)  # Unique word ratio
        
        return features

def load_and_preprocess_data(file_path):
    """Load and preprocess the SMS dataset"""
    print("Loading SMS dataset...")
    
    # Read the dataset
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    label, text = parts
                    data.append({'label': label, 'text': text})
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} messages")
    print(f"Spam: {len(df[df['label'] == 'spam'])}")
    print(f"Ham: {len(df[df['label'] == 'ham'])}")
    
    return df

def train_models(df):
    """Train multiple models and return the best one"""
    print("\nTraining models...")
    
    # Prepare data
    X_text = df['text'].values
    y = df['label'].map({'spam': 1, 'ham': 0}).values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature extractor
    feature_extractor = SMSFeatureExtractor()
    
    # Extract features
    print("Extracting features...")
    X_train_features = np.array([feature_extractor.extract_features(text) for text in X_train])
    X_test_features = np.array([feature_extractor.extract_features(text) for text in X_test])
    
    # TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents='unicode'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Combine features
    from scipy.sparse import hstack
    X_train_combined = hstack([X_train_tfidf, X_train_features])
    X_test_combined = hstack([X_test_tfidf, X_test_features])
    
    # Train models
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_combined, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_combined)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} with accuracy: {best_score:.4f}")
    
    return best_model, vectorizer, feature_extractor, best_name

def save_model(model, vectorizer, feature_extractor, model_name):
    """Save the trained model and components"""
    print(f"\nSaving {model_name} model...")
    
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'feature_extractor': feature_extractor,
        'model_name': model_name
    }
    
    with open('sms_spam_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved successfully!")

def main():
    """Main training function"""
    print("🚀 SMS Spam Detection Model Training")
    print("=" * 50)
    
    # Load data
    df = load_and_preprocess_data('spam_collection.txt')
    
    # Train models
    model, vectorizer, feature_extractor, model_name = train_models(df)
    
    # Save model
    save_model(model, vectorizer, feature_extractor, model_name)
    
    print("\n✅ Training completed successfully!")
    print("Model saved as 'sms_spam_model.pkl'")

if __name__ == "__main__":
    main()
