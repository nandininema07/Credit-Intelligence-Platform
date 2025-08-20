#!/usr/bin/env python3
"""
Test script to verify Jira API token configuration
"""

import os
import requests
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_jira_connection():
    """Test if the Jira API token is working"""
    
    # Get Jira configuration from environment
    jira_url = os.getenv('JIRA_URL')
    jira_username = os.getenv('JIRA_USERNAME')
    jira_api_token = os.getenv('JIRA_API_TOKEN')
    
    print("🧪 Testing Jira API Token Configuration")
    print("=" * 50)
    
    # Check if all required variables are set
    if not jira_url:
        print("❌ JIRA_URL not found in .env file")
        print("   Add: JIRA_URL=https://your-workspace.atlassian.net")
        return False
    
    if not jira_username:
        print("❌ JIRA_USERNAME not found in .env file")
        print("   Add: JIRA_USERNAME=your-email@example.com")
        return False
    
    if not jira_api_token:
        print("❌ JIRA_API_TOKEN not found in .env file")
        print("   Add: JIRA_API_TOKEN=ATATT3xFfGF0...")
        return False
    
    print(f"✅ Jira URL: {jira_url}")
    print(f"✅ Username: {jira_username}")
    print(f"✅ API Token: {jira_api_token[:10]}...")
    
    # Create auth header
    credentials = f"{jira_username}:{jira_api_token}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    auth_header = f"Basic {encoded_credentials}"
    
    # Test API call
    url = f"{jira_url}/rest/api/3/myself"
    headers = {
        "Authorization": auth_header,
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Jira API connection successful!")
            print(f"   User: {data.get('displayName', 'Unknown')}")
            print(f"   Email: {data.get('emailAddress', 'Unknown')}")
            print(f"   Account ID: {data.get('accountId', 'Unknown')}")
            return True
        else:
            print(f"❌ Jira API error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to Jira: {e}")
        return False

def test_project_access():
    """Test access to a specific project"""
    
    jira_url = os.getenv('JIRA_URL')
    jira_username = os.getenv('JIRA_USERNAME')
    jira_api_token = os.getenv('JIRA_API_TOKEN')
    
    if not all([jira_url, jira_username, jira_api_token]):
        return False
    
    credentials = f"{jira_username}:{jira_api_token}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    auth_header = f"Basic {encoded_credentials}"
    
    # Get projects
    url = f"{jira_url}/rest/api/3/project"
    headers = {
        "Authorization": auth_header,
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            projects = response.json()
            print(f"\n📋 Available Projects ({len(projects)}):")
            for project in projects[:5]:  # Show first 5 projects
                print(f"   • {project.get('key', 'N/A')}: {project.get('name', 'N/A')}")
            
            if len(projects) > 5:
                print(f"   ... and {len(projects) - 5} more projects")
            
            return True
        else:
            print(f"❌ Failed to get projects: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error getting projects: {e}")
        return False

def create_test_issue():
    """Create a test issue (optional)"""
    
    jira_url = os.getenv('JIRA_URL')
    jira_username = os.getenv('JIRA_USERNAME')
    jira_api_token = os.getenv('JIRA_API_TOKEN')
    
    if not all([jira_url, jira_username, jira_api_token]):
        return False
    
    credentials = f"{jira_username}:{jira_api_token}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    auth_header = f"Basic {encoded_credentials}"
    
    # Ask for project key
    project_key = input("\nEnter project key to create test issue (e.g., CREDIT): ").strip().upper()
    if not project_key:
        print("❌ No project key provided")
        return False
    
    # Create test issue
    url = f"{jira_url}/rest/api/3/issue"
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json"
    }
    
    payload = {
        "fields": {
            "project": {
                "key": project_key
            },
            "summary": "Test Issue from Credit Intelligence Platform",
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": "This is a test issue created by the Credit Intelligence Platform to verify Jira integration."
                            }
                        ]
                    }
                ]
            },
            "issuetype": {
                "name": "Task"
            }
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 201:
            data = response.json()
            issue_key = data.get('key')
            print(f"✅ Test issue created successfully!")
            print(f"   Issue Key: {issue_key}")
            print(f"   Issue URL: {jira_url}/browse/{issue_key}")
            return True
        else:
            print(f"❌ Failed to create test issue: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating test issue: {e}")
        return False

if __name__ == "__main__":
    # Test connection
    if test_jira_connection():
        # Test project access
        test_project_access()
        
        # Ask if user wants to create a test issue
        create_issue = input("\nDo you want to create a test issue? (y/N): ").strip().lower()
        if create_issue == 'y':
            create_test_issue()
    else:
        print("\nPlease check your Jira configuration in .env file.")
