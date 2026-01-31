# LLM Analytics Assistant

A lightweight data analysis tool designed for managers, product leaders, and non-technical decision-makers to explore datasets and get insights using plain English—without writing code or waiting on analysts.

[![Streamlit App](https://img.shields.io/badge/Live_App-Streamlit-brightgreen?logo=streamlit)](YOUR_DEPLOYED_URL_HERE)

##  Features

- **Natural Language Queries**: Ask questions in plain English
- **Automatic Analysis**: Get instant calculations, aggregations, and insights
- **Smart Visualizations**: Auto-generated charts based on your questions
- **No Code Required**: Perfect for non-technical users

##  Example Questions

- "What are the total sales?"
- "Show me average sales by category"
- "Which region performs best?"
- "Plot sales trend over time"
- "What's the correlation between sales and units sold?"

##  Tech Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-3.5
- **Agent Framework**: LangChain
- **Data**: Pandas
- **Visualization**: Plotly

##  Installation
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/llm-analytics-assistant
cd llm-analytics-assistant

# Install dependencies
pip install -r requirements.txt

# Set up environment
echo "OPENAI_API_KEY=your-key-here" > .env

# Run app
streamlit run app.py
```

##  How It Works

1. Upload your CSV file or try sample data
2. Ask questions in natural language
3. LLM agent analyzes your data using pandas
4. Get instant answers with automatic visualizations

##  Sample Data

Includes sample sales data with:
- 730 days of sales records
- Multiple categories and regions
- Sales and units sold metrics

##  Privacy

- All data processing happens locally
- Only queries are sent to OpenAI API
- No data is stored or shared

##  Deployment

Deployed on Streamlit Cloud. [Try it live](https://analytics-llm.streamlit.app/?embed_options=dark_theme)

##  License

MIT License
