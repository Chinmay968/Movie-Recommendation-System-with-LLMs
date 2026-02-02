from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv('.env')

api_key = os.getenv('OPENAI_API_KEY')
print(f'API Key found: {api_key[:20]}...' if api_key else 'No API key!')

try:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role':'user','content':'Say hi'}],
        max_tokens=10
    )
    print('✅ OpenAI works!')
    print('Response:', response.choices[0].message.content)
except Exception as e:
    print('❌ Error:', str(e))