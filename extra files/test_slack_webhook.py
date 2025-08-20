#!/usr/bin/env python3
"""
Test script to verify Slack webhook URL configuration
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_slack_webhook():
    """Test if the Slack webhook URL is working"""
    
    # Get webhook URL from environment
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    
    print("🧪 Testing Slack Webhook URL Configuration")
    print("=" * 50)
    
    if not webhook_url:
        print("❌ SLACK_WEBHOOK_URL not found in .env file")
        print("\n💡 Please add your Slack webhook URL to .env file:")
        print("   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...")
        return False
    
    if not webhook_url.startswith('https://hooks.slack.com/services/'):
        print("❌ Invalid Slack webhook URL format")
        print("   Expected: https://hooks.slack.com/services/...")
        print(f"   Found: {webhook_url[:30]}...")
        return False
    
    print(f"✅ Webhook URL found in .env file")
    print(f"🔗 URL: {webhook_url[:50]}...")
    
    # Test sending a message
    payload = {
        "text": "🧪 Test message from Credit Intelligence Platform - Webhook integration is working!",
        "username": "Credit Alert Bot",
        "icon_emoji": ":chart_with_upwards_trend:"
    }
    
    try:
        response = requests.post(webhook_url, json=payload)
        
        if response.status_code == 200:
            print("✅ Webhook test message sent successfully!")
            print("   Check your Slack channel for the test message")
            return True
        else:
            print(f"❌ Failed to send webhook message: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing webhook: {e}")
        return False

def send_custom_message():
    """Send a custom message via webhook"""
    
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    if not webhook_url:
        print("❌ SLACK_WEBHOOK_URL not configured")
        return False
    
    message = input("Enter your custom message: ").strip()
    if not message:
        print("❌ Message cannot be empty")
        return False
    
    payload = {
        "text": message,
        "username": "Credit Alert Bot",
        "icon_emoji": ":credit_card:"
    }
    
    try:
        response = requests.post(webhook_url, json=payload)
        
        if response.status_code == 200:
            print("✅ Custom message sent successfully!")
            return True
        else:
            print(f"❌ Failed to send message: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error sending message: {e}")
        return False

if __name__ == "__main__":
    # Test webhook
    if test_slack_webhook():
        # Ask if user wants to send a custom message
        custom_message = input("\nDo you want to send a custom message? (y/N): ").strip().lower()
        if custom_message == 'y':
            send_custom_message()
    else:
        print("\nPlease check your SLACK_WEBHOOK_URL configuration.")
