#!/bin/bash

echo "🚀 Setting up Movie Recommender with LLM Integration..."

# Create virtual environment (optional but recommended)
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
    echo "⚠️  Please edit .env and add your OpenAI API key"
    echo "   Get it from: https://platform.openai.com/api-keys"
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OpenAI API key"
echo "2. Run: streamlit run movie_recommender.py"
echo ""
echo "💰 Cost estimate: ~$0.01 per search with GPT-4o-mini"
