#!/usr/bin/env python3
"""
Gmail OAuth 2.0 Token Generator
Script gets access token for Gmail API access
"""

import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import pickle

# Gmail API scopes - IMPORTANT: Need full read access for code extraction
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',  # Read full message content
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.modify'
]
# Note: gmail.readonly should allow format=full, but if you get 403 errors,
# you may need to regenerate the token with these scopes

def get_gmail_token():
    """Get Gmail access token using OAuth 2.0"""
    creds = None
    
    # Check if token file exists
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Check if client_secret.json exists
            if not os.path.exists('client_secret.json'):
                print("client_secret.json not found!")
                print("\n To get this file:")
                print("1. Go to Google Cloud Console > APIs & Services > Credentials")
                print("2. Create OAuth 2.0 Client ID (Desktop application)")
                print("3. Download the JSON file and rename it to 'client_secret.json'")
                print("4. Place it in the same directory as this script")
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

def main():
    print("Gmail OAuth 2.0 Token Generator")
    print("=" * 50)
    
    try:
        creds = get_gmail_token()
        if creds:
            print("Successfully obtained Gmail access token!")
            print(f"Token expires at: {creds.expiry}")
            print(f"Access token: {creds.token[:20]}...")
            print("\n Add this to your .env file:")
            print(f"GOOGLE_ACCESS_TOKEN={creds.token}")
            print("\n Keep this token secure and don't share it!")
        else:
            print("Failed to get access token")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\n Make sure you have:")
        print("1. Enabled Gmail API in Google Cloud Console")
        print("2. Created OAuth 2.0 credentials")
        print("3. Downloaded client_secret.json")

if __name__ == "__main__":
    main()