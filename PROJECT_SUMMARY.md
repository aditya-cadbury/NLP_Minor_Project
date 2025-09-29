# SMS Spam Detection - NLP Minor Project

A comprehensive web-based SMS Spam Detection application using machine learning techniques, built with Flask backend and modern responsive frontend.

## 🎯 Project Overview

This project implements a sophisticated SMS spam detection system using a trained Random Forest machine learning model with 98.92% accuracy. The application features a modern web interface, comprehensive testing suites, and an aggressive 80% threshold for spam detection.

## 🚀 Key Features

### Backend (Flask)
- **Trained ML Model**: Random Forest with 98.92% accuracy
- **Advanced Feature Extraction**: TF-IDF + 20+ custom features
- **Real Dataset Training**: 5,574 SMS messages (747 spam, 4,827 ham)
- **Threshold-based Classification**: 80% confidence threshold for "not spam"
- **RESTful API**: Clean endpoints for prediction, messages, and statistics
- **SQLite Database**: Persistent storage for message history

### Frontend (HTML/CSS/JavaScript)
- **Modern UI**: Professional design with responsive layout
- **Real-time Analysis**: Instant spam detection with confidence scores
- **Dashboard**: Message history with search, pagination, and statistics
- **Visual Feedback**: Color-coded results, loading states, animations
- **Mobile Responsive**: Works seamlessly on all devices

## 📊 Model Performance

- **Training Accuracy**: 98.92% on test set
- **Real-world Testing**: 86.7% accuracy on diverse test cases
- **Spam Detection**: Excellent at identifying obvious spam patterns
- **Threshold-based Classification**: 80% confidence threshold for "not spam"
- **Aggressive Spam Detection**: Anything below 80% confidence classified as spam

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aditya-cadbury/NLP_Minor_Project.git
   cd NLP_Minor_Project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   # Linux/Mac
   ./start.sh
   
   # Windows
   start.bat
   
   # Manual
   python app.py
   ```

4. **Access the application**:
   Open your browser and navigate to: `http://localhost:5001`

## 📁 Project Structure

```
NLP_Minor_Project/
├── app.py                    # Flask backend application
├── train_model.py            # Machine learning model training script
├── test_threshold.py         # Threshold testing script
├── test_enhanced_model.py    # Comprehensive model testing script
├── requirements.txt          # Python dependencies
├── sms_spam_model.pkl        # Trained Random Forest model
├── spam_collection.txt       # Training dataset (5,574 SMS messages)
├── templates/
│   └── index.html           # Main HTML template
├── start.sh                 # Linux/Mac startup script
├── start.bat                # Windows startup script
├── .gitignore               # Git ignore file
└── README.md                # Complete documentation
```

## 🧪 Testing

### Run Comprehensive Tests
```bash
# Test the enhanced model
python test_enhanced_model.py

# Test threshold effects
python test_threshold.py

# Test API endpoints
python test_api.py
```

### Test Individual API Endpoints
```bash
# Predict spam
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"sms_text": "Your SMS message here"}'

# Get message history
curl http://localhost:5001/messages

# Get statistics
curl http://localhost:5001/stats
```

## 🔧 Technical Details

### Spam Detection Algorithm
The application uses a **trained machine learning model** (Random Forest) that analyzes:

1. **TF-IDF Features**: Term frequency-inverse document frequency analysis
2. **Text Length**: Message length, word count, line count
3. **Character Analysis**: Uppercase ratio, number ratio, special characters
4. **Spam Keywords**: 500+ common spam terms and phrases
5. **Pattern Matching**: URLs, excessive punctuation, phone numbers
6. **Word Analysis**: Average word length, unique word ratio
7. **Machine Learning**: Trained on 5,574 real SMS messages

### API Endpoints

- `POST /predict` - Analyze SMS for spam detection
- `GET /messages` - Retrieve paginated message history
- `GET /stats` - Get statistics about analyzed messages

## 🎯 Usage Examples

### Web Interface
1. Navigate to the "Detection" tab
2. Enter or paste an SMS message
3. Click "Check SMS" to analyze
4. View results with confidence scores
5. Check "Dashboard" for message history

### API Usage
```python
import requests

# Predict spam
response = requests.post('http://localhost:5001/predict', 
                        json={'sms_text': 'Your message here'})
result = response.json()
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']}")
```

## 📈 Results & Performance

The model demonstrates excellent performance:
- **Clear Spam Detection**: 100% accuracy on obvious spam patterns
- **Legitimate Messages**: High confidence (80%+) for genuine messages
- **Edge Cases**: Conservative approach with threshold-based classification
- **Real-time Processing**: Fast predictions with detailed confidence scores

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Aditya Kurup**
- GitHub: [@aditya-cadbury](https://github.com/aditya-cadbury)
- Project: NLP Minor Project - SMS Spam Detection

---

**🎉 Ready to detect spam with 98.92% accuracy!**
