# Logistics Data Analyst Intern - Week 2: Data Collection, Cleaning, and Preprocessing

**Author:** Shivam Pandey  
**Program:** BCA (Artificial Intelligence and Data Analytics), 3rd Year  
**Institution:** LNCT University, Bhopal  

## Project Overview
This project simulates an end-to-end data preprocessing and cleaning pipeline for a logistics and supply chain dataset. It addresses common data quality challenges such as missing sensor entries, duplicate transaction records, GPS outlier errors, and feature scaling for machine learning models.

## Key Steps Implemented
1. **Data Simulation:** Generated a raw dataset comprising shipping distances, package weights, traffic congestion indices, and delivery times.
2. **Duplicate Removal:** Scanned transaction logs and eliminated redundant entries to ensure data uniqueness.
3. **Missing Value Treatment:** Implemented median imputation to handle missing delivery duration values without introducing distribution bias.
4. **Outlier Detection & Capping:** Applied the Interquartile Range (IQR) method to detect and cap anomalous distance values.
5. **Feature Standardization:** Utilized Scikit-Learn's `StandardScaler` to normalize numerical features for downstream modeling.

## Repository Structure
* `preprocessing_pipeline.py`: Python scripts for data cleaning, missing value imputation, IQR outlier treatment, and feature scaling.
* `requirements.txt`: Python package dependencies.
