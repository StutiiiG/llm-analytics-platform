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

## Architecture Overview

_Add a simple diagram illustrating the flow below_

1. User uploads CSV / Excel file  
2. Data is loaded locally into Pandas  
3. A compact dataset summary is generated:
   - Schema
   - Sample rows
   - Summary statistics  
4. User question + dataset context is sent to the OpenAI API  
5. The model returns a structured analytical response  
6. Charts are generated locally using Plotly  

**Important:**  
Only summarized text context is sent to the model. Raw data remains local.

## Tech Stack

- Frontend: Streamlit  
- Data Processing: Pandas  
- Visualization: Plotly  
- LLM: OpenAI API (direct HTTP calls)  

### Why No LangChain

LangChain does not add value for this use case:
- No multi-agent orchestration required
- No retrieval pipelines
- No complex chains
- Adds unnecessary dependency risk and deployment friction

Direct API calls are:
- Faster
- More stable
- Easier to audit
- Easier to maintain

This is an intentional design decision.

---

## Privacy & Data Handling

- All data processing happens locally within the app session
- Only text summaries are sent to the LLM
- No raw data is uploaded externally
- No data is stored or logged

Suitable for internal exploratory analysis.

---

## Installation (Local)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/llm-analytics-platform
cd llm-analytics-platform

# Install dependencies
pip install -r requirements.txt

# Create .env file
OPENAI_API_KEY="your_openai_key_here"

# Run the app
streamlit run app.py
```

## Deployment 

- Deployed on streamlit cloud
- Python 3.13 compatbile 

Live app: https://analytics-llm.streamlit.app 
