# setup.py
import subprocess
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import json

def install_requirements():
    """Install all required packages"""
    requirements = [
        'pandas==1.5.3',
        'numpy==1.24.3',
        'scikit-learn==1.2.2',
        'plotly==5.13.1',
        'joblib==1.2.0'
    ]
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {str(e)}")
            sys.exit(1)
    
    print("\nAll packages installed successfully!")

class RiskAnalyticsProject:
    def __init__(self, base_path):
        self.base_path = base_path
        self.data_path = os.path.join(base_path, 'data')
        self.models_path = os.path.join(base_path, 'models')
        self.viz_path = os.path.join(base_path, 'visualizations')
        self.config_path = os.path.join(base_path, 'config')
        self.src_path = os.path.join(base_path, 'src')
        self.images_path = os.path.join(base_path, 'images')
        
    def create_project_structure(self):
        """Create project directory structure"""
        directories = [
            self.data_path,
            self.models_path,
            self.viz_path,
            self.config_path,
            self.src_path,
            self.images_path
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        print("Project structure created successfully!")
    
    def generate_sample_data(self):
        """Generate synthetic financial data"""
        np.random.seed(42)
        n_samples = 10000
        
        # Generate dates for the last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
        
        data = {
            'transaction_id': range(n_samples),
            'date': dates,
            'amount': np.random.lognormal(mean=4, sigma=1, size=n_samples),
            'transaction_count': np.random.poisson(lam=5, size=n_samples),
            'average_transaction_value': np.random.lognormal(mean=3.5, sigma=0.8, size=n_samples),
            'customer_tenure_days': np.random.randint(1, 3650, size=n_samples),
            'credit_score': np.random.normal(700, 50, size=n_samples).clip(300, 850),
            'payment_frequency': np.random.choice(['weekly', 'monthly', 'quarterly'], size=n_samples),
            'industry_sector': np.random.choice(['tech', 'retail', 'manufacturing', 'services'], size=n_samples),
            'geographical_region': np.random.choice(['North', 'South', 'East', 'West'], size=n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Generate risk indicators
        df['amount_volatility'] = np.random.normal(0, 1, size=n_samples)
        df['payment_delay_days'] = np.random.exponential(5, size=n_samples)
        df['compliance_score'] = np.random.normal(85, 10, size=n_samples).clip(0, 100)
        
        # Create risk label
        risk_conditions = (
            (df['amount_volatility'] > df['amount_volatility'].quantile(0.9)) |
            (df['payment_delay_days'] > df['payment_delay_days'].quantile(0.9)) |
            (df['compliance_score'] < df['compliance_score'].quantile(0.1))
        )
        df['risk_label'] = risk_conditions.astype(int)
        
        # Add derived features
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Save data
        df.to_csv(os.path.join(self.data_path, 'risk_data.csv'), index=False)
        print("Sample data generated successfully!")
        return df
    
    def create_visualizations(self, df):
        """Create all visualizations"""
        # Risk Distribution Pie Chart
        risk_dist = df['risk_label'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Low Risk', 'High Risk'],
            values=[risk_dist[0], risk_dist[1]],
            marker=dict(colors=['green', 'red'])
        )])
        fig_pie.update_layout(title='Risk Distribution')
        fig_pie.write_html(os.path.join(self.viz_path, 'risk_distribution.html'))
        
        # Industry Risk Analysis
        fig_industry = make_subplots(rows=2, cols=1,
                                   subplot_titles=('Risk by Industry', 'Average Amount by Industry'))
        
        industry_risk = df.groupby('industry_sector')['risk_label'].mean()
        fig_industry.add_trace(
            go.Bar(x=industry_risk.index, y=industry_risk.values, 
                  marker_color='orange', name='Risk Level'),
            row=1, col=1
        )
        
        industry_amount = df.groupby('industry_sector')['amount'].mean()
        fig_industry.add_trace(
            go.Bar(x=industry_amount.index, y=industry_amount.values,
                  marker_color='blue', name='Avg Amount'),
            row=2, col=1
        )
        
        fig_industry.update_layout(height=800, title_text="Industry Analysis")
        fig_industry.write_html(os.path.join(self.viz_path, 'industry_analysis.html'))

        # Additional visualizations...
        # Compliance Dashboard, Temporal Analysis, Transaction Analysis (similar to above format)
        
        print("Visualizations created successfully!")
    
    def train_risk_model(self, df):
        """Train and save risk prediction model"""
        feature_cols = ['amount', 'transaction_count', 'average_transaction_value',
                       'customer_tenure_days', 'credit_score', 'amount_volatility',
                       'payment_delay_days', 'compliance_score']
        
        X = df[feature_cols]
        y = df['risk_label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        joblib.dump(model, os.path.join(self.models_path, 'risk_classifier.joblib'))
        joblib.dump(scaler, os.path.join(self.models_path, 'scaler.joblib'))
        
        print("Model trained and saved successfully!")
    
    def setup_project(self):
        """Run complete project setup"""
        print("\nStarting project setup...")
        self.create_project_structure()
        df = self.generate_sample_data()
        self.create_visualizations(df)
        self.train_risk_model(df)
        print("\nProject setup completed successfully!")
    
def main():
    project = RiskAnalyticsProject('/home/milad/Desktop/risk')

    project.setup_project()

if __name__ == "__main__":
    install_requirements()
    main()
