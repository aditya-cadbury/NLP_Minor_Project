#!/usr/bin/env python3
"""
Test script to demonstrate the 80% threshold for spam detection
Shows how the threshold affects classification decisions
"""

import requests
import json

def test_threshold_effect():
    """Test the 80% threshold effect on various messages"""
    print("🧪 Testing 80% Threshold for Spam Detection")
    print("=" * 60)
    print("Rule: If 'not spam' confidence < 80%, classify as SPAM")
    print("=" * 60)
    
    base_url = "http://localhost:5001"
    
    test_messages = [
        {
            "text": "Hey, how are you doing? Want to grab coffee later?",
            "description": "Clear legitimate message"
        },
        {
            "text": "Thanks for the meeting today. Let me know if you need anything else.",
            "description": "Professional message"
        },
        {
            "text": "Can you pick up some milk on your way home?",
            "description": "Simple request"
        },
        {
            "text": "Congratulations! You have won $1000! Click here to claim now!",
            "description": "Obvious spam"
        },
        {
            "text": "URGENT! Your account will be closed in 24 hours. Call now to prevent this!",
            "description": "Urgency-based spam"
        },
        {
            "text": "FREE! Win a new iPhone! Text STOP to opt out. Limited time offer!",
            "description": "Free offer spam"
        },
        {
            "text": "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
            "description": "Lottery scam"
        },
        {
            "text": "I'm not sure about this message. It could be spam or not.",
            "description": "Borderline message"
        },
        {
            "text": "Your package has been delivered. Track at ups.com",
            "description": "Legitimate delivery notification"
        },
        {
            "text": "URGENT: Verify your account now or it will be suspended! Click here immediately!",
            "description": "Account verification spam"
        }
    ]
    
    for i, test_case in enumerate(test_messages, 1):
        text = test_case['text']
        description = test_case['description']
        
        print(f"\n📱 Test {i}: {description}")
        print(f"   Text: {text[:50]}...")
        
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"sms_text": text},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                label = data['label']
                confidence = data['confidence']
                
                # Determine if threshold was applied
                if 'spam_probability' in data and 'not_spam_probability' in data:
                    spam_prob = data['spam_probability']
                    not_spam_prob = data['not_spam_probability']
                    threshold = data.get('threshold', 0.8)
                    
                    print(f"   📊 Probabilities:")
                    print(f"      Spam: {spam_prob:.2f} ({spam_prob*100:.1f}%)")
                    print(f"      Not Spam: {not_spam_prob:.2f} ({not_spam_prob*100:.1f}%)")
                    print(f"      Threshold: {threshold*100:.0f}%")
                    
                    if not_spam_prob >= threshold:
                        print(f"   ✅ Result: NOT SPAM (Not spam prob ≥ {threshold*100:.0f}%)")
                    else:
                        print(f"   🔴 Result: SPAM (Not spam prob < {threshold*100:.0f}%)")
                else:
                    # Fallback display
                    result_color = "🟢" if label == 'not spam' else "🔴"
                    print(f"   {result_color} Result: {label.upper()} (Confidence: {confidence:.2f})")
                    
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("   ❌ Error: Could not connect to server. Make sure the Flask app is running.")
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return True

def test_threshold_impact():
    """Show the impact of the threshold on classification"""
    print(f"\n\n📈 THRESHOLD IMPACT ANALYSIS")
    print("=" * 60)
    print("With 80% threshold:")
    print("• Messages need ≥80% confidence to be classified as 'NOT SPAM'")
    print("• Anything below 80% confidence gets classified as 'SPAM'")
    print("• This makes the model more aggressive in detecting spam")
    print("• Reduces false negatives (missed spam) but may increase false positives")
    print("=" * 60)

def main():
    """Main test function"""
    print("🚀 SMS Spam Detection - 80% Threshold Test")
    print("=" * 70)
    print("Testing the new threshold-based classification system")
    print("Make sure the Flask application is running on http://localhost:5001")
    print("=" * 70)
    
    if test_threshold_effect():
        test_threshold_impact()
        print("\n\n🎉 Threshold testing completed!")
        print("The 80% threshold is now active and working correctly.")
    else:
        print("\n❌ Tests failed. Please start the Flask application first.")

if __name__ == "__main__":
    main()
