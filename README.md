# Travel AI Assistant

Travel AI Assistant is a simple web app built with **Gradio** and **Hugging Face Transformers**.  
It helps users analyze travel-related requests, recommend a trip style, and generate a basic personalized itinerary.

## Features

- **Travel Request Analyzer**: classifies a travel request into categories like flights, hotels, budget, itinerary, transportation, packing, or travel documents.
- **Trip Style Recommender**: suggests a suitable travel style based on user preferences.
- **Itinerary Generator**: creates a simple day-by-day travel plan based on destination, trip style, and budget.

## Technologies Used

- **Python**
- **Gradio**
- **Hugging Face Transformers**
- **PyTorch**
- **facebook/bart-large-mnli** zero-shot classification model

## Installation

```bash
git clone https://github.com/jorgersantiago/travel-ai-assistant.git
cd travel-ai-assistant
pip install -r requirements.txt
python app.py
```

## Demo Inputs

- **Travel Request Analyzer**:  
  `I need a cheap hotel near the airport for two nights.`

- **Trip Style Recommender**:  
  `I want a calm vacation with good food, nice places to walk, and not too many activities.`

- **Itinerary Generator**:  
  Destination: `Rome`  
  Days: `3`  
  Style: `Relaxed`  
  Budget: `Medium`

## Deployment

Live app: https://huggingface.co/spaces/jorgersantiago/travel-ai-assistant
