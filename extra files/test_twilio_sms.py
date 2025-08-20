#!/usr/bin/env python3
"""
Test script to verify Twilio SMS configuration
"""

import os
import aiohttp
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_twilio_credentials():
    """Test if Twilio credentials are configured"""
    
    print("🧪 Testing Twilio SMS Configuration")
    print("=" * 50)
    
    # Get Twilio configuration from environment
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    from_number = os.getenv('TWILIO_FROM_NUMBER')
    
    if not account_sid:
        print("❌ TWILIO_ACCOUNT_SID not found in .env file")
        print("   Add: TWILIO_ACCOUNT_SID=your-account-sid-here")
        return False
    
    if not auth_token:
        print("❌ TWILIO_AUTH_TOKEN not found in .env file")
        print("   Add: TWILIO_AUTH_TOKEN=your-auth-token-here")
        return False
    
    if not from_number:
        print("❌ TWILIO_FROM_NUMBER not found in .env file")
        print("   Add: TWILIO_FROM_NUMBER=+1234567890")
        return False
    
    print(f"✅ Account SID: {account_sid[:10]}...")
    print(f"✅ Auth Token: {auth_token[:10]}...")
    print(f"✅ From Number: {from_number}")
    
    return True

async def test_twilio_api_connection():
    """Test Twilio API connection"""
    
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    
    if not account_sid or not auth_token:
        return False
    
    # Test API connection by getting account info
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}.json"
    
    try:
        auth = aiohttp.BasicAuth(account_sid, auth_token)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, auth=auth) as response:
                
                if response.status == 200:
                    data = await response.json()
                    print("✅ Twilio API connection successful!")
                    print(f"   Account Name: {data.get('friendly_name', 'Unknown')}")
                    print(f"   Account Status: {data.get('status', 'Unknown')}")
                    print(f"   Account Type: {data.get('type', 'Unknown')}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Twilio API error: {response.status}")
                    print(f"   Response: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Error connecting to Twilio API: {e}")
        return False

async def test_send_sms():
    """Test sending an SMS message"""
    
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    from_number = os.getenv('TWILIO_FROM_NUMBER')
    
    if not all([account_sid, auth_token, from_number]):
        return False
    
    # Ask for test phone number
    test_number = input("\nEnter phone number to test SMS (e.g., +1234567890): ").strip()
    
    if not test_number:
        print("❌ No phone number provided")
        return False
    
    if not test_number.startswith('+'):
        print("❌ Phone number should start with + (international format)")
        return False
    
    # Test message
    test_message = f"🧪 Test message from Credit Intelligence Platform - {asyncio.get_event_loop().time():.0f}"
    
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    auth = aiohttp.BasicAuth(account_sid, auth_token)
    
    data = {
        'From': from_number,
        'To': test_number,
        'Body': test_message
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, auth=auth, data=data) as response:
                
                if response.status == 201:
                    response_data = await response.json()
                    message_sid = response_data.get('sid')
                    print(f"✅ Test SMS sent successfully!")
                    print(f"   Message SID: {message_sid}")
                    print(f"   To: {test_number}")
                    print(f"   From: {from_number}")
                    print(f"   Message: {test_message}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to send SMS: {response.status}")
                    print(f"   Response: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Error sending SMS: {e}")
        return False

async def test_phone_number_validation():
    """Test phone number validation"""
    
    print("\n📱 Testing Phone Number Validation")
    print("=" * 50)
    
    test_numbers = [
        "+1234567890",      # Valid US number
        "+44123456789",     # Valid UK number
        "+911234567890",    # Valid India number
        "1234567890",       # Invalid (no +)
        "+123",             # Invalid (too short)
        "+1234567890123456" # Invalid (too long)
    ]
    
    for number in test_numbers:
        is_valid = await validate_phone_number(number)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"   {number}: {status}")

async def validate_phone_number(phone_number: str) -> bool:
    """Validate phone number format"""
    
    try:
        # Basic validation - should start with + and contain only digits
        if not phone_number.startswith('+'):
            return False
        
        # Remove + and check if remaining characters are digits
        digits_only = phone_number[1:]
        if not digits_only.isdigit():
            return False
        
        # Check length (international numbers are typically 10-15 digits)
        if len(digits_only) < 10 or len(digits_only) > 15:
            return False
        
        return True
        
    except Exception:
        return False

async def get_twilio_account_info():
    """Get detailed Twilio account information"""
    
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    
    if not account_sid or not auth_token:
        return False
    
    try:
        # Get account info
        account_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}.json"
        auth = aiohttp.BasicAuth(account_sid, auth_token)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(account_url, auth=auth) as response:
                if response.status == 200:
                    account_data = await response.json()
                    
                    print("\n📊 Twilio Account Information")
                    print("=" * 50)
                    print(f"   Account Name: {account_data.get('friendly_name', 'Unknown')}")
                    print(f"   Account SID: {account_data.get('sid', 'Unknown')}")
                    print(f"   Account Status: {account_data.get('status', 'Unknown')}")
                    print(f"   Account Type: {account_data.get('type', 'Unknown')}")
                    print(f"   Date Created: {account_data.get('date_created', 'Unknown')}")
                    
                    # Get phone numbers
                    numbers_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/IncomingPhoneNumbers.json"
                    async with session.get(numbers_url, auth=auth) as numbers_response:
                        if numbers_response.status == 200:
                            numbers_data = await numbers_response.json()
                            phone_numbers = numbers_data.get('incoming_phone_numbers', [])
                            
                            if phone_numbers:
                                print(f"\n📞 Phone Numbers ({len(phone_numbers)}):")
                                for number in phone_numbers:
                                    print(f"   - {number.get('phone_number', 'Unknown')} ({number.get('friendly_name', 'No name')})")
                            else:
                                print("\n📞 No phone numbers found in account")
                    
                    return True
                else:
                    print("❌ Failed to get account information")
                    return False
                    
    except Exception as e:
        print(f"❌ Error getting account information: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Twilio SMS Test for Credit Intelligence Platform")
    print("=" * 60)
    
    # Test credentials
    if not await test_twilio_credentials():
        print("\n❌ Twilio credentials not configured properly")
        print("Please update your .env file with correct Twilio credentials")
        return
    
    # Test API connection
    if not await test_twilio_api_connection():
        print("\n❌ Cannot connect to Twilio API")
        print("Please check your credentials and internet connection")
        return
    
    # Get account information
    await get_twilio_account_info()
    
    # Test phone number validation
    await test_phone_number_validation()
    
    # Ask if user wants to send a test SMS
    send_test = input("\nDo you want to send a test SMS? (y/N): ").strip().lower()
    if send_test == 'y':
        await test_send_sms()
    
    print("\n🎉 Twilio SMS configuration test completed!")
    print("\n📝 Your platform is ready to send SMS notifications via Twilio.")

if __name__ == "__main__":
    asyncio.run(main())
