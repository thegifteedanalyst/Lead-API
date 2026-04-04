## REVENUE & lEAD PRIORITY INTELLIGENT SYSTEM

## Project Overview
This project demonstrates the creation of an intelligent lead scoring system using machine learning, 
deployed as a FastAPI application, and integrated with HubSpot CRM through Zapier automation. 
The system helps businesses prioritize leads based on predicted conversion probability and deal value, 
improving sales efficiency and revenue outcomes.

## GOALS
Automatically:
Capture new leads
Score them (High / Medium / Low)
Prioritize them
Notify sales instantly
Push hot leads to CRM
Track revenue pipeline

No manual sorting. No delays. Faster revenue.

## Data Features

The system uses lead behavioral and transactional data including:

sessions_count – Number of website sessions by the lead
pages_viewed – Total pages visited
pricing_page_views – Number of times the pricing page was viewed
time_on_site_sec – Total time spent on site in seconds
recency_days – Days since last visit
deal_value – Estimated deal value in currency

Target Variable:
converted – Whether the lead converted (0 = No, 1 = Yes)

## System Architecture
1.Machine Learning Model
Trained a logistic regression model to predict lead conversion probability.
Saved the trained model and scaler using joblib.

2.FastAPI Endpoint
Endpoint: POST /score
Input: JSON payload with lead behavioral data
Output: JSON response with:

3.Cloud Deployment
Deployed API to Render for production availability.

4.Zapier Automation
Trigger: New HubSpot contact
Actions: Format numeric fields → POST to FastAPI → Update HubSpot with AI score

## Workflow Diagram
HubSpot Contact Created
          │
          ▼
   Zapier Trigger
          │
          ▼
 Formatter (Convert Fields to Numbers)
          │
          ▼
 Webhook POST → FastAPI Model → AI Scoring
          │
          ▼
HubSpot Updated with:
- Conversion Probability
- Priority Score
- Priority Level

## Results
1.High-priority leads identified automatically
2.Reduced manual lead evaluation by sales team
3.Improved conversion efficiency by focusing on high-value prospects
4.Fully automated, end-to-end AI-driven CRM workflow


## API URL

👉Click or copy the url to the recommendation APIhttps://lead-api-4sqr.onrender.com/docs
