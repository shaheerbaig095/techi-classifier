TECHi: Tech Headline Category ClassifierTask ID: TPT-2026-03075 
Author: Shaheer

techi-classifierTECHi is a lightweight Machine Learning web app that automatically categorizes tech news headlines into five domains: AI, Cybersecurity, Startups, Gadgets, and Software.


Technical Architecture
Feature Extraction: TfidfVectorizer 
Classifier: LogisticRegression for multi-class probability estimation
Frontend: Streamlit
Project Structure:

├── data/headlines.csv           # 150 labeled headlines dataset
├── model/tech_classifier.joblib  # Trained model binary
├── app.py                      # Streamlit application entry point
├── train.py                    # Model training pipeline
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation


DatasetTrained on 150 balanced headlines (30 samples per category in data/headlines.csv):AI: LLMs, Neural Networks, Computer Vision, Robotics.Cybersecurity: Breaches, Ransomware, Zero-day exploits, Vulnerabilities.Startups: Venture capital, Funding rounds, Valuations, Accelerators.Gadgets: Smartphones, Specs, Wearables, Consoles.Software: Operating systems, Developer tools, Frameworks, Databases.Quick StartBash# 1. Clone repository

git clone https://github.com/YOUR_USERNAME/techi-classifier.git

cd techi-classifier

# 2. Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies & train model
pip install -r requirements.txt
python train.py

# 4. Run application
streamlit run app.py
