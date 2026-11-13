conda create -n langagent python=3.11 -y

conda activate langagent

pip install -r requirements.txt

Create a .env file in the project root with the keys needed for the notebook:

GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
GROQ_API_KEY=your_groq_api_key
WEATHERSTACK_API_KEY=your_weatherstack_api_key

Start the Streamlit app:

```bash
streamlit run app.py
```

The app uses the same Groq, Tavily, and Weatherstack tools as `research/agent_demo.ipynb`.
