#!/usr/bin/env python3
"""
Test script for SMS Spam Detection Application
This script demonstrates the spam detection functionality
"""

import requests
import json
import time

# Test SMS messages
test_messages = [
    "Congratulations! You've won $1000! Click here to claim your prize now!",
    "Hey, how are you doing? Want to grab coffee later?",
    "URGENT: Your account will be closed in 24 hours. Call now to prevent this!",
    "Thanks for the meeting today. Let me know if you need anything else.",
    "FREE! Win a new iPhone! Text STOP to opt out. Limited time offer!",
    "Can you pick up some milk on your way home?",
    "You have been selected for a special promotion! Act now!",
    "See you at the gym at 6 PM?",
    "URGENT: Your credit card has been compromised. Click here immediately!",
    "Happy birthday! Hope you have a wonderful day!"
]

def test_prediction_api():
    """Test the prediction API endpoint"""
    print("🧪 Testing SMS Spam Detection API")
    print("=" * 50)
    
    base_url = "http://localhost:5001"
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n📱 Test {i}: {message[:50]}...")
        
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"sms_text": message},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                label = data['label']
                confidence = data['confidence']
                
                # Color coding for output
                if label == 'spam':
                    print(f"   🔴 Result: SPAM (Confidence: {confidence:.2f})")
                else:
                    print(f"   🟢 Result: NOT SPAM (Confidence: {confidence:.2f})")
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("   ❌ Error: Could not connect to server. Make sure the Flask app is running.")
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return True

def test_messages_api():
    """Test the messages API endpoint"""
    print("\n\n📊 Testing Messages API")
    print("=" * 50)
    
    base_url = "http://localhost:5001"
    
    try:
        # Test getting messages
        response = requests.get(f"{base_url}/messages")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Total messages: {data['total_count']}")
            print(f"✅ Current page: {data['page']}")
            print(f"✅ Total pages: {data['total_pages']}")
            
            if data['messages']:
                print("\n📋 Recent messages:")
                for msg in data['messages'][:3]:  # Show first 3
                    print(f"   • {msg['sms_text'][:40]}... -> {msg['prediction']}")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_stats_api():
    """Test the stats API endpoint"""
    print("\n\n📈 Testing Stats API")
    print("=" * 50)
    
    base_url = "http://localhost:5001"
    
    try:
        response = requests.get(f"{base_url}/stats")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Total messages: {data['total_messages']}")
            print(f"✅ Spam messages: {data['spam_count']}")
            print(f"✅ Not spam messages: {data['not_spam_count']}")
            print(f"✅ Spam percentage: {data['spam_percentage']:.1f}%")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main test function"""
    print("🚀 SMS Spam Detection - API Test Suite")
    print("=" * 60)
    print("Make sure the Flask application is running on http://localhost:5001")
    print("Run: python app.py")
    print("=" * 60)
    
    # Wait a moment for user to start the server
    time.sleep(2)
    
    # Run tests
    if test_prediction_api():
        test_messages_api()
        test_stats_api()
        
        print("\n\n🎉 All tests completed!")
        print("Visit http://localhost:5001 to see the web interface")
    else:
        print("\n❌ Tests failed. Please start the Flask application first.")

if __name__ == "__main__":
    main()
