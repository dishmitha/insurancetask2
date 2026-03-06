# Insurance Charges Prediction - Task Plan

## Information Gathered
- Dataset: insurance.csv with 1338 rows
- Features: age, sex, bmi, children, smoker, region
- Target: charges (continuous variable)
- Task: Predict insurance charges with interaction features using Streamlit

## Plan

### 1. Create Streamlit App Structure (app.py)
- Import required libraries (streamlit, pandas, numpy, scikit-learn, etc.)
- Create sidebar for navigation
- Create main sections for each requirement

### 2. Feature Engineering Functions
- BMI Category: Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (>=30)
- Age Group: Young (18-25), Adult (26-40), Middle-aged (41-55), Senior (56+)
- Interaction Feature: smoker × BMI (numeric interaction)

### 3. Outlier Detection & Handling
- Use IQR method on charges and bmi columns
- Cap outliers at IQR boundaries (or remove)

### 4. Model Training
- Linear Regression model
- Train/test split (80/20)
- Feature preprocessing (encoding categorical variables)

### 5. Model Evaluation
- Calculate MAE, RMSE, R² metrics
- Display results in Streamlit

### 6. Model Comparison
- Compare Linear Regression vs Random Forest (non-linear)
- Display comparison metrics

### 7. Final Model Deployment
- Allow user to input features
- Make predictions
- Display results

## Dependent Files
- insurance.csv (input data)
- app.py (main Streamlit application)

## Followup Steps
1. ✅ Created app.py with all functionality
2. ✅ Installed required packages: streamlit, pandas, numpy, scikit-learn
3. ✅ Running the app with: streamlit run app.py

## Completion Status
✅ Task completed successfully! The Streamlit app is running at http://localhost:8501

