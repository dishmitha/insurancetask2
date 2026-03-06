"""
Insurance Charges Prediction with Interaction Features
Predicts insurance charges using Linear Regression with engineered features
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Insurance Charges Predictor", page_icon="🏥", layout="wide")

# Title and description
st.title("🏥 Insurance Charges Prediction")
st.markdown("""
This app predicts insurance charges using **Linear Regression** with interaction features.
It includes feature engineering, outlier handling, and model comparison.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('insurance.csv')
    return df

df = load_data()

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Feature Engineering", "Outlier Handling", 
                                   "Model Training", "Model Comparison", "Prediction"])

# ============== DATA OVERVIEW ==============
if page == "Data Overview":
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1])
    with col3:
        st.metric("Avg Charges", f"${df['charges'].mean():,.2f}")
    with col4:
        st.metric("Smokers %", f"{(df['smoker'] == 'yes').mean()*100:.1f}%")
    
    st.subheader("Dataset Sample")
    st.dataframe(df.head(10))
    
    st.subheader("Data Types")
    st.write(df.dtypes)
    
    st.subheader("Statistical Summary")
    st.write(df.describe())

# ============== FEATURE ENGINEERING ==============
elif page == "Feature Engineering":
    st.header("🔧 Feature Engineering")
    
    # Create a copy for feature engineering
    df_features = df.copy()
    
    # 1. BMI Category
    def get_bmi_category(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    df_features['bmi_category'] = df_features['bmi'].apply(get_bmi_category)
    
    # 2. Age Group
    def get_age_group(age):
        if age <= 25:
            return 'Young'
        elif age <= 40:
            return 'Adult'
        elif age <= 55:
            return 'Middle-aged'
        else:
            return 'Senior'
    
    df_features['age_group'] = df_features['age'].apply(get_age_group)
    
    # 3. Interaction: smoker × BMI
    df_features['smoker_numeric'] = (df_features['smoker'] == 'yes').astype(int)
    df_features['smoker_bmi_interaction'] = df_features['smoker_numeric'] * df_features['bmi']
    
    st.subheader("Created Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### BMI Category Distribution")
        bmi_dist = df_features['bmi_category'].value_counts()
        st.bar_chart(bmi_dist)
        st.write(bmi_dist)
    
    with col2:
        st.write("### Age Group Distribution")
        age_dist = df_features['age_group'].value_counts()
        st.bar_chart(age_dist)
        st.write(age_dist)
    
    st.subheader("Interaction Feature: Smoker × BMI")
    st.write("This feature combines smoking status with BMI to capture the interaction effect.")
    st.write(f"Example: A smoker with BMI 30 has interaction value: 1 × 30 = 30")
    st.write(f"Non-smoker with BMI 30 has interaction value: 0 × 30 = 0")
    
    st.subheader("Dataset with New Features")
    st.dataframe(df_features.head(10))
    
    # Store for later use
    st.session_state['df_features'] = df_features

# ============== OUTLIER HANDLING ==============
elif page == "Outlier Handling":
    st.header("📈 Outlier Detection & Handling")
    
    df_outliers = df.copy()
    
    st.subheader("Outlier Detection using IQR Method")
    
    # Function to detect outliers
    def detect_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    # Detect outliers in charges
    charges_outliers, charges_lower, charges_upper = detect_outliers_iqr(df_outliers, 'charges')
    
    # Detect outliers in BMI
    bmi_outliers, bmi_lower, bmi_upper = detect_outliers_iqr(df_outliers, 'bmi')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Charges Outliers")
        st.write(f"Lower Bound: ${charges_lower:,.2f}")
        st.write(f"Upper Bound: ${charges_upper:,.2f}")
        st.write(f"Number of Outliers: {len(charges_outliers)}")
        st.write(f"Percentage: {len(charges_outliers)/len(df_outliers)*100:.2f}%")
    
    with col2:
        st.write("### BMI Outliers")
        st.write(f"Lower Bound: {bmi_lower:.2f}")
        st.write(f"Upper Bound: {bmi_upper:.2f}")
        st.write(f"Number of Outliers: {len(bmi_outliers)}")
        st.write(f"Percentage: {len(bmi_outliers)/len(df_outliers)*100:.2f}%")
    
    # Option to handle outliers
    st.subheader("Handle Outliers")
    handle_method = st.radio("Select method:", ["Cap Outliers", "Remove Outliers", "Keep Original"])
    
    if handle_method == "Cap Outliers":
        # Cap charges outliers
        df_outliers['charges'] = df_outliers['charges'].clip(lower=charges_lower, upper=charges_upper)
        # Cap BMI outliers
        df_outliers['bmi'] = df_outliers['bmi'].clip(lower=bmi_lower, upper=bmi_upper)
        st.success("Outliers have been capped!")
    elif handle_method == "Remove Outliers":
        # Remove charges outliers
        df_outliers = df_outliers[(df_outliers['charges'] >= charges_lower) & 
                                   (df_outliers['charges'] <= charges_upper)]
        # Remove BMI outliers
        df_outliers = df_outliers[(df_outliers['bmi'] >= bmi_lower) & 
                                   (df_outliers['bmi'] <= bmi_upper)]
        st.success(f"Outliers removed! Remaining records: {len(df_outliers)}")
    else:
        st.info("Keeping original data without outlier handling.")
    
    st.session_state['df_processed'] = df_outliers
    
    st.subheader("Processed Data Statistics")
    st.write(df_outliers.describe())

# ============== MODEL TRAINING ==============
elif page == "Model Training":
    st.header("🤖 Model Training - Linear Regression")
    
    # Prepare data
    df_model = df.copy()
    
    # Apply feature engineering
    df_model['bmi_category'] = df_model['bmi'].apply(
        lambda x: 'Underweight' if x < 18.5 else ('Normal' if x < 25 else ('Overweight' if x < 30 else 'Obese'))
    )
    df_model['age_group'] = df_model['age'].apply(
        lambda x: 'Young' if x <= 25 else ('Adult' if x <= 40 else ('Middle-aged' if x <= 55 else 'Senior'))
    )
    df_model['smoker_numeric'] = (df_model['smoker'] == 'yes').astype(int)
    df_model['smoker_bmi_interaction'] = df_model['smoker_numeric'] * df_model['bmi']
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    le_bmi_cat = LabelEncoder()
    le_age_group = LabelEncoder()
    
    df_model['sex_encoded'] = le_sex.fit_transform(df_model['sex'])
    df_model['smoker_encoded'] = le_smoker.fit_transform(df_model['smoker'])
    df_model['region_encoded'] = le_region.fit_transform(df_model['region'])
    df_model['bmi_category_encoded'] = le_bmi_cat.fit_transform(df_model['bmi_category'])
    df_model['age_group_encoded'] = le_age_group.fit_transform(df_model['age_group'])
    
    # Select features
    feature_columns = ['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 
                      'region_encoded', 'bmi_category_encoded', 'age_group_encoded',
                      'smoker_bmi_interaction']
    
    X = df_model[feature_columns]
    y = df_model['charges']
    
    # Train-test split
    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    st.subheader("Training Data")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Training samples: {len(X_train)}")
    with col2:
        st.write(f"Testing samples: {len(X_test)}")
    
    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = lr_model.predict(X_train)
    y_pred_test = lr_model.predict(X_test)
    
    # Evaluation metrics
    st.subheader("Model Evaluation Metrics")
    
    # Training metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    
    # Test metrics
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("MAE (Mean Absolute Error)")
        st.write(f"Train: ${train_mae:,.2f}")
        st.write(f"Test: ${test_mae:,.2f}")
    
    with col2:
        st.subheader("RMSE (Root Mean Squared Error)")
        st.write(f"Train: ${train_rmse:,.2f}")
        st.write(f"Test: ${test_rmse:,.2f}")
    
    with col3:
        st.subheader("R² Score")
        st.write(f"Train: {train_r2:.4f}")
        st.write(f"Test: {test_r2:.4f}")
    
    # Feature importance
    st.subheader("Feature Coefficients")
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': lr_model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    st.write(feature_importance)
    st.write(f"Intercept: {lr_model.intercept_:,.2f}")
    
    # Store model for comparison
    st.session_state['lr_model'] = lr_model
    st.session_state['feature_columns'] = feature_columns
    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test
    st.session_state['y_pred_test'] = y_pred_test
    st.session_state['le_sex'] = le_sex
    st.session_state['le_smoker'] = le_smoker
    st.session_state['le_region'] = le_region

# ============== MODEL COMPARISON ==============
elif page == "Model Comparison":
    st.header("📊 Model Comparison: Linear vs Non-Linear")
    
    # Prepare data (same as before)
    df_model = df.copy()
    
    df_model['bmi_category'] = df_model['bmi'].apply(
        lambda x: 'Underweight' if x < 18.5 else ('Normal' if x < 25 else ('Overweight' if x < 30 else 'Obese'))
    )
    df_model['age_group'] = df_model['age'].apply(
        lambda x: 'Young' if x <= 25 else ('Adult' if x <= 40 else ('Middle-aged' if x <= 55 else 'Senior'))
    )
    df_model['smoker_numeric'] = (df_model['smoker'] == 'yes').astype(int)
    df_model['smoker_bmi_interaction'] = df_model['smoker_numeric'] * df_model['bmi']
    
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    le_bmi_cat = LabelEncoder()
    le_age_group = LabelEncoder()
    
    df_model['sex_encoded'] = le_sex.fit_transform(df_model['sex'])
    df_model['smoker_encoded'] = le_smoker.fit_transform(df_model['smoker'])
    df_model['region_encoded'] = le_region.fit_transform(df_model['region'])
    df_model['bmi_category_encoded'] = le_bmi_cat.fit_transform(df_model['bmi_category'])
    df_model['age_group_encoded'] = le_age_group.fit_transform(df_model['age_group'])
    
    feature_columns = ['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 
                      'region_encoded', 'bmi_category_encoded', 'age_group_encoded',
                      'smoker_bmi_interaction']
    
    X = df_model[feature_columns]
    y = df_model['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    
    # Train Random Forest (Non-linear model)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    # Calculate metrics for both models
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    lr_r2 = r2_score(y_test, y_pred_lr)
    
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_r2 = r2_score(y_test, y_pred_rf)
    
    # Display comparison
    st.subheader("Performance Comparison")
    
    comparison_df = pd.DataFrame({
        'Metric': ['MAE ($)', 'RMSE ($)', 'R² Score'],
        'Linear Regression': [f"${lr_mae:,.2f}", f"${lr_rmse:,.2f}", f"{lr_r2:.4f}"],
        'Random Forest': [f"${rf_mae:,.2f}", f"${rf_rmse:,.2f}", f"{rf_r2:.4f}"]
    })
    
    st.table(comparison_df)
    
    # Visual comparison
    st.subheader("Visual Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Linear Regression")
        st.write(f"MAE: ${lr_mae:,.2f}")
        st.write(f"RMSE: ${lr_rmse:,.2f}")
        st.write(f"R²: {lr_r2:.4f}")
    
    with col2:
        st.write("### Random Forest (Non-linear)")
        st.write(f"MAE: ${rf_mae:,.2f}")
        st.write(f"RMSE: ${rf_rmse:,.2f}")
        st.write(f"R²: {rf_r2:.4f}")
    
    # Winner announcement
    st.subheader("🏆 Model Comparison Result")
    
    if lr_r2 > rf_r2:
        st.success(f"Linear Regression performs BETTER! R²: {lr_r2:.4f} vs {rf_r2:.4f}")
    elif rf_r2 > lr_r2:
        st.success(f"Random Forest (Non-linear) performs BETTER! R²: {rf_r2:.4f} vs {lr_r2:.4f}")
    else:
        st.info("Both models perform equally!")
    
    # Actual vs Predicted plot data
    st.subheader("Actual vs Predicted (Test Set)")
    
    comparison_plot_df = pd.DataFrame({
        'Actual': y_test.values,
        'Linear Regression': y_pred_lr,
        'Random Forest': y_pred_rf
    }).head(50)
    
    st.line_chart(comparison_plot_df)
    
    # Store models for prediction
    st.session_state['lr_model'] = lr_model
    st.session_state['rf_model'] = rf_model
    st.session_state['feature_columns'] = feature_columns

# ============== PREDICTION ==============
elif page == "Prediction":
    st.header("🎯 Make a Prediction")
    
    # Use the trained Linear Regression model
    if 'lr_model' not in st.session_state:
        # Train model if not available
        df_model = df.copy()
        
        df_model['bmi_category'] = df_model['bmi'].apply(
            lambda x: 'Underweight' if x < 18.5 else ('Normal' if x < 25 else ('Overweight' if x < 30 else 'Obese'))
        )
        df_model['age_group'] = df_model['age'].apply(
            lambda x: 'Young' if x <= 25 else ('Adult' if x <= 40 else ('Middle-aged' if x <= 55 else 'Senior'))
        )
        df_model['smoker_numeric'] = (df_model['smoker'] == 'yes').astype(int)
        df_model['smoker_bmi_interaction'] = df_model['smoker_numeric'] * df_model['bmi']
        
        le_sex = LabelEncoder()
        le_smoker = LabelEncoder()
        le_region = LabelEncoder()
        le_bmi_cat = LabelEncoder()
        le_age_group = LabelEncoder()
        
        df_model['sex_encoded'] = le_sex.fit_transform(df_model['sex'])
        df_model['smoker_encoded'] = le_smoker.fit_transform(df_model['smoker'])
        df_model['region_encoded'] = le_region.fit_transform(df_model['region'])
        df_model['bmi_category_encoded'] = le_bmi_cat.fit_transform(df_model['bmi_category'])
        df_model['age_group_encoded'] = le_age_group.fit_transform(df_model['age_group'])
        
        feature_columns = ['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 
                          'region_encoded', 'bmi_category_encoded', 'age_group_encoded',
                          'smoker_bmi_interaction']
        
        X = df_model[feature_columns]
        y = df_model['charges']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        st.session_state['lr_model'] = lr_model
        st.session_state['feature_columns'] = feature_columns
        st.session_state['le_sex'] = le_sex
        st.session_state['le_smoker'] = le_smoker
        st.session_state['le_region'] = le_region
        st.session_state['le_bmi_cat'] = le_bmi_cat
        st.session_state['le_age_group'] = le_age_group
    
    # Input form
    st.subheader("Enter Your Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    
    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    
    # Calculate derived features
    if bmi < 18.5:
        bmi_category = 'Underweight'
    elif bmi < 25:
        bmi_category = 'Normal'
    elif bmi < 30:
        bmi_category = 'Overweight'
    else:
        bmi_category = 'Obese'
    
    if age <= 25:
        age_group = 'Young'
    elif age <= 40:
        age_group = 'Adult'
    elif age <= 55:
        age_group = 'Middle-aged'
    else:
        age_group = 'Senior'
    
    smoker_numeric = 1 if smoker == 'yes' else 0
    smoker_bmi_interaction = smoker_numeric * bmi
    
    # Encode features
    sex_encoded = 1 if sex == 'male' else 0
    smoker_encoded = 1 if smoker == 'yes' else 0
    
    region_mapping = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
    region_encoded = region_mapping[region]
    
    bmi_cat_mapping = {'Normal': 0, 'Obese': 1, 'Overweight': 2, 'Underweight': 3}
    bmi_category_encoded = bmi_cat_mapping[bmi_category]
    
    age_group_mapping = {'Adult': 0, 'Middle-aged': 1, 'Senior': 2, 'Young': 3}
    age_group_encoded = age_group_mapping[age_group]
    
    # Create feature vector
    features = np.array([[age, sex_encoded, bmi, children, smoker_encoded, 
                          region_encoded, bmi_category_encoded, age_group_encoded,
                          smoker_bmi_interaction]])
    
    # Make prediction
    if st.button("Predict Insurance Charges"):
        model = st.session_state['lr_model']
        prediction = model.predict(features)[0]
        
        st.subheader("Prediction Result")
        
        st.success(f"Estimated Insurance Charges: **${prediction:,.2f}**")
        
        # Show breakdown
        st.write("### Feature Summary")
        st.write(f"- Age: {age} ({age_group})")
        st.write(f"- Sex: {sex}")
        st.write(f"- BMI: {bmi} ({bmi_category})")
        st.write(f"- Children: {children}")
        st.write(f"- Smoker: {smoker}")
        st.write(f"- Region: {region}")
        st.write(f"- Smoker × BMI Interaction: {smoker_bmi_interaction}")
        
        # Compare with average
        avg_charges = df['charges'].mean()
        diff = prediction - avg_charges
        if diff > 0:
            st.warning(f"This is ${diff:,.2f} ABOVE the average insurance charges (${avg_charges:,.2f})")
        else:
            st.info(f"This is ${abs(diff):,.2f} BELOW the average insurance charges (${avg_charges:,.2f})")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("### About")
st.sidebar.info("""
This app demonstrates:
1. Feature Engineering (BMI Category, Age Group, Interaction)
2. Outlier Detection (IQR Method)
3. Linear Regression Model
4. Model Evaluation (MAE, RMSE, R²)
5. Model Comparison (Linear vs Non-Linear)
6. Interactive Predictions
""")

