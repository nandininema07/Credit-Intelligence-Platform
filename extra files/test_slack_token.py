#!/usr/bin/env python3
"""
Test script to verify Slack bot token configuration
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_slack_token():
    """Test if the Slack bot token is working"""
    
    # Get token from environment
    token = os.getenv('SLACK_BOT_TOKEN')
    
    if not token:
        print("❌ SLACK_BOT_TOKEN not found in .env file")
        return False
    
    if not token.startswith('xoxb-'):
        print("❌ Invalid Slack bot token format. Should start with 'xoxb-'")
        return False
    
    # Test API call
    url = "https://slack.com/api/auth.test"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if data.get('ok'):
            print("✅ Slack bot token is valid!")
            print(f"   Bot name: {data.get('user')}")
            print(f"   Team: {data.get('team')}")
            print(f"   User ID: {data.get('user_id')}")
            return True
        else:
            print(f"❌ Slack API error: {data.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Slack token: {e}")
        return False

def test_send_message():
    """Test sending a message to a channel"""
    
    token = os.getenv('SLACK_BOT_TOKEN')
    channel = input("Enter channel name to test (e.g., #alerts): ").strip()
    
    if not channel.startswith('#'):
        channel = f"#{channel}"
    
    url = "https://slack.com/api/chat.postMessage"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "channel": channel,
        "text": "🧪 Test message from Credit Intelligence Platform - Slack integration is working!",
        "username": "Credit Alert Bot"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        
        if data.get('ok'):
            print(f"✅ Test message sent successfully to {channel}!")
            return True
        else:
            print(f"❌ Failed to send message: {data.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error sending test message: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Slack Bot Token Configuration")
    print("=" * 50)
    
    # Test token validity
    if test_slack_token():
        # Ask if user wants to test sending a message
        test_message = input("\nDo you want to test sending a message? (y/N): ").strip().lower()
        if test_message == 'y':
            test_send_message()
    else:
        print("\nPlease check your SLACK_BOT_TOKEN configuration.")
