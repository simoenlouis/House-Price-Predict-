# House-Price-Predict-

Housing Price Prediction Project
This project focuses on building an end-to-end machine learning pipeline to predict housing prices. Through rigorous exploratory data analysis (EDA), advanced feature engineering, and model optimization, we successfully handled non-linear relationships and high skewness to deliver an accurate predictive model.

🚀 Key Project Steps & Methodology
1. Missing Data Imputation
During the initial data exploration, a significant number of missing values were detected in the total_bedrooms feature. To address this without losing valuable data patterns, an advanced multivariate imputation technique was employed:
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
An IterativeImputer was utilized to estimate the missing values based on the other features in the dataset.

2. Feature Engineering & Multi-collinearity Resolution
A correlation heatmap revealed highly correlated features, indicating strong multi-collinearity among structural and population columns (total_rooms, total_bedrooms, population, households).

To resolve this and extract deeper insights, the following feature engineering steps were performed:
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_household'] = df['total_bedrooms'] / df['households']
df['population_per_household'] = df['population'] / df['households']
df = df.drop(['total_rooms', 'total_bedrooms', 'population', 'households'], axis=1)
Furthermore, geospatial coordinates (Latitude & Longitude) were utilized to construct a powerful new feature representing geographical pricing context: nearby_houses_average_price.

3. Handling Skewness
Several features, alongside the target variable, exhibited high right-skewness. To normalize the distributions and improve model convergence:

PowerTransformer was applied to the skewed input features.

TransformedTargetRegressor was utilized to dynamically apply a log-like transformation to the target variable during training and automatically back-transform predictions during evaluation.

📊 Model Evaluation & Results
We evaluated several regression algorithms, ranging from traditional linear models to advanced gradient boosting trees and ensemble techniques.
Key Model Insights & Residual AnalysisLinear vs. Tree Models: Residual plots revealed distinct behaviors. Linear models (Linear Regression, Ridge, Lasso, ElasticNet) suffered from high bias and displayed noticeable patterns in their residuals, indicating they failed to capture complex underlying trends.Tree-Based Superiority: Tree-based models (Random Forest, XGBoost, LightGBM) drastically outperformed linear models. They showed minimal bias and no distinct residual patterns, demonstrating a powerful capability to capture non-linear relationships.The Winner: Random Forest achieved the best individual performance. By ensembling the top estimators into a Voting Regressor, the test metric improved even further ($R^2 = 0.8642$), while maintaining a controlled overfitting gap.🔍 Model Interpretability (SHAP)To understand what drives the model's decisions, we performed a SHAP (SHapley Additive exPlanations) analysis.The summary plot (image_230e40.png) and feature importance bar chart (image_230e46.png) reveal that our engineered geospatial feature, nearby_houses_average_price, is by far the most powerful predictor in the dataset, followed closely by median_income. This proves that leveraging spatial context was crucial for the model's performance.

<img width="785" height="619" alt="image" src="https://github.com/user-attachments/assets/57bd1086-c2f6-41e6-ab34-abdea43af404" />
<img width="804" height="640" alt="image" src="https://github.com/user-attachments/assets/4c31e13f-dc8a-4b08-91d5-d50605aa2e59" />
<img width="733" height="470" alt="image" src="https://github.com/user-attachments/assets/7ef0f650-098f-429f-a0eb-9f7daf06f67a" />
<img width="733" height="470" alt="image" src="https://github.com/user-attachments/assets/b5ff1b76-81e8-4771-9abf-683d10eb2714" />




