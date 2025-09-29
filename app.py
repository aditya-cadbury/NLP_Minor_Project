"""
SMS Spam Detection Flask Backend
A simple web service for detecting spam SMS messages
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import re
import pickle
import numpy as np
from datetime import datetime
import os
from scipy.sparse import hstack

app = Flask(__name__)
CORS(app)

# Initialize database
def init_db():
    """Initialize SQLite database for storing SMS messages"""
    conn = sqlite3.connect('sms_detection.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sms_text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# SMS Feature Extractor Class (needed for model loading)
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

# Load trained machine learning model
class MLSpamDetector:
    """Machine Learning based spam detector using trained model"""
    
    def __init__(self, model_path='sms_spam_model.pkl'):
        """Load the trained model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.feature_extractor = model_data['feature_extractor']
            self.model_name = model_data['model_name']
            
            print(f"✅ Loaded {self.model_name} model successfully!")
            
        except FileNotFoundError:
            print("❌ Model file not found. Please run train_model.py first.")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict(self, text):
        """Predict if SMS is spam or not using trained model with threshold"""
        try:
            # Extract features using the same feature extractor used in training
            features = self.feature_extractor.extract_features(text)
            features_array = np.array(features).reshape(1, -1)
            
            # Get TF-IDF features
            tfidf_features = self.vectorizer.transform([text])
            
            # Combine features
            combined_features = hstack([tfidf_features, features_array])
            
            # Get probabilities for both classes
            probability = self.model.predict_proba(combined_features)[0]
            spam_probability = probability[1]  # Probability of spam
            not_spam_probability = probability[0]  # Probability of not spam
            
            # Apply threshold: if not_spam confidence < 80%, classify as spam
            threshold = 0.8  # 80% threshold for not spam
            if not_spam_probability >= threshold:
                label = 'not spam'
                confidence = not_spam_probability
                score = not_spam_probability
            else:
                label = 'spam'
                confidence = spam_probability
                score = spam_probability
            
            return {
                'label': label,
                'confidence': confidence,
                'score': score,
                'spam_probability': spam_probability,
                'not_spam_probability': not_spam_probability,
                'threshold': threshold,
                'model_name': self.model_name
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            # Fallback to simple rule-based detection
            return self._fallback_prediction(text)
    
    def _fallback_prediction(self, text):
        """Fallback prediction method"""
        spam_keywords = ['free', 'win', 'winner', 'urgent', 'click', 'call now']
        spam_count = sum(1 for keyword in spam_keywords if keyword.lower() in text.lower())
        
        is_spam = spam_count > 0 or len(text) > 100
        confidence = 0.7 if is_spam else 0.6
        
        return {
            'label': 'spam' if is_spam else 'not spam',
            'confidence': confidence,
            'score': 0.8 if is_spam else 0.2,
            'model_name': 'fallback'
        }

# Initialize detector
try:
    detector = MLSpamDetector()
except:
    print("⚠️  Using fallback detector")
    detector = None

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sms():
    """Predict if SMS is spam"""
    try:
        data = request.get_json()
        sms_text = data.get('sms_text', '').strip()
        
        if not sms_text:
            return jsonify({'error': 'SMS text is required'}), 400
        
        # Get prediction
        if detector is None:
            return jsonify({'error': 'Model not loaded. Please run train_model.py first.'}), 500
        
        prediction = detector.predict(sms_text)
        
        # Store in database
        conn = sqlite3.connect('sms_detection.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages (sms_text, prediction, confidence)
            VALUES (?, ?, ?)
        ''', (sms_text, prediction['label'], prediction['confidence']))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'label': prediction['label'],
            'confidence': prediction['confidence'],
            'score': prediction['score']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/messages', methods=['GET'])
def get_messages():
    """Get all stored messages"""
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        search = request.args.get('search', '')
        
        offset = (page - 1) * limit
        
        conn = sqlite3.connect('sms_detection.db')
        cursor = conn.cursor()
        
        # Build query with search
        query = 'SELECT id, sms_text, prediction, confidence, timestamp FROM messages'
        params = []
        
        if search:
            query += ' WHERE sms_text LIKE ? OR prediction LIKE ?'
            params.extend([f'%{search}%', f'%{search}%'])
        
        query += ' ORDER BY timestamp DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        messages = cursor.fetchall()
        
        # Get total count
        count_query = 'SELECT COUNT(*) FROM messages'
        count_params = []
        
        if search:
            count_query += ' WHERE sms_text LIKE ? OR prediction LIKE ?'
            count_params.extend([f'%{search}%', f'%{search}%'])
        
        cursor.execute(count_query, count_params)
        total_count = cursor.fetchone()[0]
        
        conn.close()
        
        # Format response
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                'id': msg[0],
                'sms_text': msg[1],
                'prediction': msg[2],
                'confidence': msg[3],
                'timestamp': msg[4]
            })
        
        return jsonify({
            'messages': formatted_messages,
            'total_count': total_count,
            'page': page,
            'limit': limit,
            'total_pages': (total_count + limit - 1) // limit
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics about stored messages"""
    try:
        conn = sqlite3.connect('sms_detection.db')
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute('SELECT COUNT(*) FROM messages')
        total_count = cursor.fetchone()[0]
        
        # Get spam count
        cursor.execute('SELECT COUNT(*) FROM messages WHERE prediction = "spam"')
        spam_count = cursor.fetchone()[0]
        
        # Get not spam count
        cursor.execute('SELECT COUNT(*) FROM messages WHERE prediction = "not spam"')
        not_spam_count = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'total_messages': total_count,
            'spam_count': spam_count,
            'not_spam_count': not_spam_count,
            'spam_percentage': (spam_count / total_count * 100) if total_count > 0 else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5001)
