# SMS Spam Detection Web Application

A modern, responsive web application for detecting spam SMS messages using machine learning techniques.

## Features

### Frontend
- **Modern UI**: Clean, professional design with responsive layout
- **SMS Detection**: Real-time spam detection with confidence scores
- **Dashboard**: View all analyzed messages with search and pagination
- **Statistics**: Overview of spam detection metrics
- **Loading States**: Visual feedback during processing
- **Mobile Responsive**: Works seamlessly on desktop and mobile devices

### Backend
- **Flask API**: RESTful endpoints for prediction and data management
- **Rule-based Detection**: Intelligent spam detection using multiple features
- **SQLite Database**: Persistent storage for message history
- **CORS Support**: Cross-origin requests enabled
- **Error Handling**: Comprehensive error handling and validation

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone or download the project**
   ```bash
   cd /path/to/your/project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   Open your browser and navigate to: `http://localhost:5001`

## API Endpoints

### POST /predict
Analyze an SMS message for spam detection.

**Request:**
```json
{
    "sms_text": "Your SMS message here"
}
```

**Response:**
```json
{
    "label": "spam" | "not spam",
    "confidence": 0.85,
    "score": 0.75
}
```

### GET /messages
Retrieve paginated list of analyzed messages.

**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 10)
- `search`: Search term for filtering

**Response:**
```json
{
    "messages": [...],
    "total_count": 100,
    "page": 1,
    "limit": 10,
    "total_pages": 10
}
```

### GET /stats
Get statistics about analyzed messages.

**Response:**
```json
{
    "total_messages": 100,
    "spam_count": 25,
    "not_spam_count": 75,
    "spam_percentage": 25.0
}
```

## How It Works

### Spam Detection Algorithm

The application uses a **trained machine learning model** (Random Forest) with 98.92% accuracy that analyzes multiple features:

1. **TF-IDF Features**: Term frequency-inverse document frequency analysis
2. **Text Length**: Message length, word count, line count
3. **Character Analysis**: Uppercase ratio, number ratio, special characters
4. **Spam Keywords**: 500+ common spam terms and phrases
5. **Pattern Matching**: URLs, excessive punctuation, phone numbers
6. **Word Analysis**: Average word length, unique word ratio
7. **Machine Learning**: Trained on 5,574 real SMS messages (747 spam, 4,827 ham)

### Model Performance

- **Training Accuracy**: 98.92% on test set
- **Real-world Testing**: 86.7% accuracy on diverse test cases
- **Spam Detection**: Excellent at identifying obvious spam patterns
- **Threshold-based Classification**: 80% confidence threshold for "not spam"
- **Aggressive Spam Detection**: Anything below 80% confidence classified as spam
- **Confidence Scores**: High confidence for clear cases, moderate for edge cases

## Project Structure

```
NLP_Project/
├── app.py                    # Flask backend application
├── train_model.py            # Machine learning model training script
├── test_threshold.py         # Threshold testing script
├── requirements.txt          # Python dependencies
├── sms_spam_model.pkl        # Trained Random Forest model
├── spam_collection.txt       # Training dataset (5,574 SMS messages)
├── templates/
│   └── index.html           # Main HTML template
├── sms_detection.db          # SQLite database (created automatically)
├── start.sh                 # Linux/Mac startup script
├── start.bat                # Windows startup script
└── README.md                # This file
```

## Usage

### SMS Detection
1. Navigate to the "Detection" tab
2. Enter or paste an SMS message in the text area
3. Click "Check SMS" to analyze
4. View the result with confidence score

### Dashboard
1. Navigate to the "Dashboard" tab
2. View statistics and message history
3. Use search to filter messages
4. Navigate through pages using pagination

## Customization

### Adding New Spam Keywords
Edit the `spam_keywords` list in `app.py`:

```python
self.spam_keywords = [
    'free', 'win', 'winner', 'congratulations', 'urgent',
    # Add your keywords here
]
```

### Modifying Detection Rules
Adjust the scoring system in the `predict` method:

```python
# Example: Increase penalty for spam keywords
score += features[4] * 0.4  # Instead of 0.3
```

### Styling Changes
Modify the CSS in the `<style>` section of `templates/index.html` to customize colors, fonts, and layout.

## Technical Details

### Database Schema
```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sms_text TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Dependencies
- **Flask**: Web framework
- **Flask-CORS**: Cross-origin resource sharing
- **SQLite3**: Database (built-in with Python)
- **NumPy**: Numerical operations (optional, for future ML models)

## Future Enhancements

- **Machine Learning Model**: Replace rule-based detection with trained ML model
- **User Authentication**: Add user accounts and personal dashboards
- **Bulk Upload**: Support for analyzing multiple messages at once
- **Export Features**: Download results as CSV/JSON
- **Real-time Updates**: WebSocket support for live updates
- **Advanced Analytics**: More detailed statistics and visualizations

## Troubleshooting

### Common Issues

1. **Port already in use**
   - Change the port in `app.py`: `app.run(port=5001)`

2. **Database errors**
   - Delete `sms_detection.db` to reset the database

3. **CORS issues**
   - Ensure Flask-CORS is installed: `pip install Flask-CORS`

4. **Missing dependencies**
   - Reinstall requirements: `pip install -r requirements.txt`

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
