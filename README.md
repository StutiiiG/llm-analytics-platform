# LLM Analytics Assistant

A lightweight data analysis tool designed for managers, product leaders, and non-technical decision-makers to explore datasets and get insights using plain English—without writing code or waiting on analysts.


## The Problem

Organizations often have data, but accessing insights is slow and fragmented.

Common challenges:
- Business teams cannot query raw data themselves
- Even simple questions require:
  - SQL knowledge
  - Pre-built dashboards
  - Back-and-forth with data teams
- Exploration is limited to predefined views
- Analysts spend time answering repetitive questions instead of higher-value work

Traditional BI tools are powerful, but they:
- Assume technical expertise
- Require upfront modeling
- Do not support open-ended, ad-hoc exploration

--- 

## The Solution

**LLM Analytics Platform** enables leaders to:

- Upload a CSV or Excel file
- Ask questions in natural language
- Instantly receive:
  - Clear analytical answers
  - Summary statistics
  - Context-aware explanations
  - Visual insights where appropriate

No dashboards to pre-build.  
No queries to write.  
No technical setup required.

---

## Who This Is For

- Product managers
- Business and operations leaders
- Strategy and growth teams
- Consultants
- Anyone who needs answers from data without technical tooling

---

## Key Capabilities

### Natural Language Data Analysis
Ask questions such as:
- “What are the main drivers of performance?”
- “Which category contributes the most?”
- “Are there trends over time?”
- “Where are the largest variations?”

The system answers using the **actual dataset schema and values**.

---

### Safe, Grounded Insights
- The model is provided with:
  - Column names and data types
  - Null counts
  - Sample rows
  - Summary statistics
- The system does **not hallucinate columns**
- Answers are explained in clear, business-oriented language

---

### Visual Explanations
- Automatic charts when relevant:
  - Numeric distributions
  - Category breakdowns
  - Time-based trends
- Visuals are generated only when supported by the data

## Screenshots

### Main Analysis View
_Add screenshot of the main analysis interface_

### Data Upload & Sidebar
_Add screenshot of the data upload and quick queries section_

### Charts View
_Add screenshot of the visualization tab_

---

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
