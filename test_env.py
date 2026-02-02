from dotenv import load_dotenv
import os

# Explicitly specify .env file location
load_dotenv('.env')

key = os.getenv('OPENAI_API_KEY')
if key and key.startswith('sk-'):
    print('✅ API key works!')
    print(f'Key starts with: {key[:15]}...')
else:
    print('❌ API key not found!')
    print('Current directory:', os.getcwd())
    print('.env exists?', os.path.exists('.env'))