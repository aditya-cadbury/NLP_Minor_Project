#!/usr/bin/env python3
"""
Enhanced SMS Spam Detection Test Script
Tests the trained machine learning model with various SMS examples
"""

import requests
import json
import time

# Test SMS messages with expected labels
test_cases = [
    # Obvious spam messages
    {
        "text": "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
        "expected": "spam",
        "description": "Classic lottery scam"
    },
    {
        "text": "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
        "expected": "spam",
        "description": "Competition scam"
    },
    {
        "text": "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info",
        "expected": "spam",
        "description": "Cash prize scam"
    },
    {
        "text": "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
        "expected": "spam",
        "description": "Prize jackpot scam"
    },
    {
        "text": "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL",
        "expected": "spam",
        "description": "Mobile service scam"
    },
    
    # Legitimate messages
    {
        "text": "Hey, how are you doing? Want to grab coffee later?",
        "expected": "not spam",
        "description": "Casual conversation"
    },
    {
        "text": "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.",
        "expected": "not spam",
        "description": "Personal message"
    },
    {
        "text": "I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times.",
        "expected": "not spam",
        "description": "Thank you message"
    },
    {
        "text": "I HAVE A DATE ON SUNDAY WITH WILL!!",
        "expected": "not spam",
        "description": "Excited personal message"
    },
    {
        "text": "Can you pick up some milk on your way home?",
        "expected": "not spam",
        "description": "Simple request"
    },
    
    # Edge cases
    {
        "text": "Congratulations! You've won $1000! Click here to claim now!",
        "expected": "spam",
        "description": "Generic spam template"
    },
    {
        "text": "URGENT: Your account will be closed in 24 hours. Call now to prevent this!",
        "expected": "spam",
        "description": "Urgency-based spam"
    },
    {
        "text": "Thanks for the meeting today. Let me know if you need anything else.",
        "expected": "not spam",
        "description": "Professional message"
    },
    {
        "text": "FREE! Win a new iPhone! Text STOP to opt out. Limited time offer!",
        "expected": "spam",
        "description": "Free offer spam"
    },
    {
        "text": "See you at the gym at 6 PM?",
        "expected": "not spam",
        "description": "Simple question"
    }
]

def test_prediction_api():
    """Test the prediction API endpoint with comprehensive test cases"""
    print("🧪 Testing Enhanced SMS Spam Detection API")
    print("=" * 60)
    
    base_url = "http://localhost:5001"
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        text = test_case['text']
        expected = test_case['expected']
        description = test_case['description']
        
        print(f"\n📱 Test {i}: {description}")
        print(f"   Text: {text[:60]}...")
        
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"sms_text": text},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                predicted = data['label']
                confidence = data['confidence']
                
                # Check if prediction is correct
                is_correct = predicted == expected
                if is_correct:
                    correct_predictions += 1
                
                # Color coding for output
                status_icon = "✅" if is_correct else "❌"
                result_color = "🟢" if predicted == 'not spam' else "🔴"
                
                print(f"   {status_icon} Expected: {expected.upper()}")
                print(f"   {result_color} Predicted: {predicted.upper()} (Confidence: {confidence:.2f})")
                
                if not is_correct:
                    print(f"   ⚠️  MISCLASSIFICATION!")
                    
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("   ❌ Error: Could not connect to server. Make sure the Flask app is running.")
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ Correct Predictions: {correct_predictions}/{total_predictions}")
    print(f"📈 Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 90:
        print("🎉 Excellent performance!")
    elif accuracy >= 80:
        print("👍 Good performance!")
    elif accuracy >= 70:
        print("⚠️  Moderate performance - consider retraining")
    else:
        print("❌ Poor performance - model needs improvement")
    
    return True

def test_model_performance():
    """Test model performance with detailed analysis"""
    print("\n\n🔍 DETAILED MODEL ANALYSIS")
    print("=" * 60)
    
    base_url = "http://localhost:5001"
    
    # Test different types of messages
    spam_examples = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts",
        "WINNER!! You have been selected to receive £900 prize",
        "URGENT! Your account will be closed in 24 hours",
        "SIX chances to win CASH! From 100 to 20,000 pounds",
        "XXXMobileMovieClub: To use your credit, click the WAP link"
    ]
    
    ham_examples = [
        "Hey, how are you doing? Want to grab coffee later?",
        "I'm gonna be home soon and i don't want to talk about this",
        "Can you pick up some milk on your way home?",
        "Thanks for the meeting today. Let me know if you need anything",
        "See you at the gym at 6 PM?"
    ]
    
    print("🔴 SPAM DETECTION TEST:")
    spam_correct = 0
    for i, text in enumerate(spam_examples, 1):
        try:
            response = requests.post(f"{base_url}/predict", 
                                    json={"sms_text": text},
                                    headers={"Content-Type": "application/json"})
            if response.status_code == 200:
                data = response.json()
                predicted = data['label']
                confidence = data['confidence']
                
                is_correct = predicted == 'spam'
                if is_correct:
                    spam_correct += 1
                
                status = "✅" if is_correct else "❌"
                print(f"   {status} Test {i}: {predicted.upper()} (Confidence: {confidence:.2f})")
        except:
            print(f"   ❌ Test {i}: Error")
    
    print(f"\n🟢 HAM DETECTION TEST:")
    ham_correct = 0
    for i, text in enumerate(ham_examples, 1):
        try:
            response = requests.post(f"{base_url}/predict", 
                                    json={"sms_text": text},
                                    headers={"Content-Type": "application/json"})
            if response.status_code == 200:
                data = response.json()
                predicted = data['label']
                confidence = data['confidence']
                
                is_correct = predicted == 'not spam'
                if is_correct:
                    ham_correct += 1
                
                status = "✅" if is_correct else "❌"
                print(f"   {status} Test {i}: {predicted.upper()} (Confidence: {confidence:.2f})")
        except:
            print(f"   ❌ Test {i}: Error")
    
    spam_accuracy = (spam_correct / len(spam_examples)) * 100
    ham_accuracy = (ham_correct / len(ham_examples)) * 100
    overall_accuracy = ((spam_correct + ham_correct) / (len(spam_examples) + len(ham_examples))) * 100
    
    print(f"\n📊 DETAILED ACCURACY:")
    print(f"   🔴 Spam Detection: {spam_correct}/{len(spam_examples)} ({spam_accuracy:.1f}%)")
    print(f"   🟢 Ham Detection: {ham_correct}/{len(ham_examples)} ({ham_accuracy:.1f}%)")
    print(f"   📈 Overall Accuracy: {overall_accuracy:.1f}%")

def test_api_endpoints():
    """Test all API endpoints"""
    print("\n\n🌐 API ENDPOINTS TEST")
    print("=" * 60)
    
    base_url = "http://localhost:5001"
    
    # Test messages endpoint
    try:
        response = requests.get(f"{base_url}/messages")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Messages API: {data['total_count']} messages loaded")
        else:
            print(f"❌ Messages API: Error {response.status_code}")
    except Exception as e:
        print(f"❌ Messages API: {e}")
    
    # Test stats endpoint
    try:
        response = requests.get(f"{base_url}/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stats API: {data['spam_percentage']:.1f}% spam rate")
        else:
            print(f"❌ Stats API: Error {response.status_code}")
    except Exception as e:
        print(f"❌ Stats API: {e}")

def main():
    """Main test function"""
    print("🚀 Enhanced SMS Spam Detection - Comprehensive Test Suite")
    print("=" * 70)
    print("Testing the trained Random Forest model with 98.92% accuracy")
    print("Make sure the Flask application is running on http://localhost:5001")
    print("=" * 70)
    
    # Wait a moment for user to start the server
    time.sleep(2)
    
    # Run comprehensive tests
    if test_prediction_api():
        test_model_performance()
        test_api_endpoints()
        
        print("\n\n🎉 All tests completed!")
        print("Visit http://localhost:5001 to see the web interface")
        print("\n💡 The model has been trained on 5,574 real SMS messages")
        print("   with 747 spam and 4,827 ham examples for high accuracy!")
    else:
        print("\n❌ Tests failed. Please start the Flask application first.")

if __name__ == "__main__":
    main()
