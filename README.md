# Pinecone Chatbot

A Streamlit-based chatbot interface for Pinecone vector search.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Update the `.env` file with your Pinecone API key
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Features

- Chat interface for querying Pinecone vector database
- Secure API key handling
- Chat history persistence
- Real-time vector search
- Error handling and feedback

## Configuration

Update the `.env` file with your Pinecone API key:
```env
PINECONE_API_KEY=your_pinecone_api_key_here
```
```

After creating all these files, you should:

1. Create and activate a virtual environment:
```bash
cd pinecone_app
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Update the `.env` file with your actual Pinecone API key

4. Run the application:
```bash
streamlit run app.py
```

Would you like me to help you with any of these steps or explain any part of the code in more detail?