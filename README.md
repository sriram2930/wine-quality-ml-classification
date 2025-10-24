# wine-quality-ml-classification
Classification of Portuguese red wine quality based on 11 physicochemical properties. Implements EDA, feature engineering (Yeo–Johnson transformation and chemical ratios), and model comparison between Logistic Regression and Random Forest. A showcase of end-to-end applied machine learning with interpretable insights
Wine Quality Classification Report
1.	Introduction
This project aims to classify red wine quality based on physicochemical properties using machine learning models. The dataset, sourced from the UCI Machine Learning Repository, contains laboratory analyses of 1,599 Portuguese “Vinho Verde” red wines. Each sample includes 11 numerical attributes describing its chemical composition and a quality score (an integer between 3 and 8) assigned by professional tasters. The task is a multiclass classification problem where we predict the quality rating using measurable chemical features.
2.	Dataset Description

<img width="1050" height="201" alt="image" src="https://github.com/user-attachments/assets/a5537269-8b4b-4253-a626-71558cadff28" />

 

The dataset’s 11 input variables are objective physicochemical measurements:
•	Fixed acidity, volatile acidity, and citric acid (g/dm³): Represent total and volatile acid levels, influencing tartness and spoilage.
•	Residual sugar (g/dm³): Sugar remaining after fermentation. Red wines in this dataset are mostly dry.
•	Chlorides (g/dm³): Reflect salt concentration; excessive amounts reduce quality.
•	Free and total sulphur dioxide (mg/dm³): Antioxidants and preservatives preventing oxidation and spoilage.
•	Density (g/cm³): Indicates sugar and alcohol content; alcohol decreases density while sugar increases it.
•	pH: Measures acidity strength. Lower pH means higher acidity.
•	Sulphates (g/dm³): Strengthen antioxidant effects; often correlate positively with quality.
•	Alcohol (% by volume): A key determinant of flavour balance and body, usually positively associated with quality.
The target variable quality is an integer score rated by experts, typically ranging from 3 (poor) to 8 (excellent). 

3.	Exploratory Data Analysis (EDA)
EDA showed that alcohol has the strongest positive correlation with quality, while volatile acidity correlates negatively, confirming that high acidity often reduces quality.

<img width="688" height="512" alt="image" src="https://github.com/user-attachments/assets/1073904f-c59f-40fe-8605-51b0f5a8daf5" />

 
Sulphates and citric acid show moderate positive correlations, and density shows a negative one. 



<img width="461" height="310" alt="image" src="https://github.com/user-attachments/assets/86d4ec2a-8db7-4820-8889-a811d37c66b8" />
<img width="465" height="307" alt="image" src="https://github.com/user-attachments/assets/12ff0f48-5185-4e0e-a84a-6e3cdd11c5ff" />


<img width="477" height="318" alt="image" src="https://github.com/user-attachments/assets/05b15690-04b4-4337-b0ac-bddbfd3bae3a" />

<img width="474" height="313" alt="image" src="https://github.com/user-attachments/assets/af59bdb1-2a27-4b37-be00-c10747ede2ad" />








Several features—such as chlorides, residual sugar, and sulphur dioxide—are right-skewed, warranting power transformations for variance stabilization.



<img width="487" height="324" alt="image" src="https://github.com/user-attachments/assets/763b57f1-7100-4317-a37f-0921bf200656" />

<img width="492" height="324" alt="image" src="https://github.com/user-attachments/assets/27b2480b-32da-467f-802c-1e08e3cb2730" />

<img width="491" height="327" alt="image" src="https://github.com/user-attachments/assets/2b3eeea5-bb0f-4c55-9751-26389127eaea" />

<img width="505" height="332" alt="image" src="https://github.com/user-attachments/assets/786e27d4-ac32-4353-b5f0-cb0314eaf906" />















Boxplots were used to compare feature distributions across wine quality levels, highlighting trends and differences between low- and high-quality wines. They also expose outliers and reveal how median values shift with quality, providing visual evidence of potential predictors. Class imbalance was evident, necessitating stratified train-test splits and macro-F1 evaluation.
4. Baseline Models
Two baseline models were trained:
1.	Multinomial Logistic Regression (scaled) – Accuracy: 0.59, Macro-F1: 0.28
2.	Random Forest (400 trees) – Accuracy: 0.68, Macro-F1: 0.41

<img width="652" height="161" alt="image" src="https://github.com/user-attachments/assets/ac5eae11-05ad-4455-acc4-6df7804d4c10" />

<img width="502" height="413" alt="image" src="https://github.com/user-attachments/assets/25c6f8ad-b2f7-4711-b30b-fd0e831dc1f7" />

<img width="499" height="410" alt="image" src="https://github.com/user-attachments/assets/5f2e4d7f-f7b5-42a8-9299-8721f430e032" />




 
The Random Forest model outperformed Logistic Regression, effectively capturing nonlinear feature relationships and handling skewed inputs. Feature importance analysis highlighted alcohol, sulphates, volatile acidity, and density as the most influential predictors.

5. Feature Engineering
Two feature-engineering strategies were applied:
(A) Variance stabilization using the Yeo–Johnson transform on skewed variables (chlorides, residual sugar, SO₂), to normalize distributions and help linear models.
(B) Domain-inspired features created from chemical intuition:
•	sulfur_ratio = free SO₂ / total SO₂ (oxidation balance),
•	Total_acidity = fixed + volatile + citric acid,
•	acid_sugar_balance = total_acidity / (residual sugar + 1),
•	alcohol_density = alcohol / density.
These new ratios capture wine chemistry relationships known to affect taste and stability.
6. Model Re-evaluation
After engineering, both models were retrained:
•	Logistic Regression: Accuracy 0.59, Macro-F1 0.28 → 0.283 (slight F1 improvement).
•	Random Forest: Accuracy 0.68 → 0.69, Macro-F1 0.41 (stable).
The improvements were modest but consistent, especially for underrepresented classes. The Random Forest remained the best-performing model overall
 
<img width="705" height="166" alt="image" src="https://github.com/user-attachments/assets/f0ea8b48-862e-4ce8-8b4c-6601d6645933" />

<img width="500" height="415" alt="image" src="https://github.com/user-attachments/assets/9d7e3b15-2126-4f5d-b4c2-ae2fb6d833c9" />

<img width="494" height="411" alt="image" src="https://github.com/user-attachments/assets/a520ab03-46a8-446a-87cb-036c1e4f505d" />



7. Conclusion
This project demonstrated how physicochemical variables can reasonably predict wine quality. Alcohol, volatile acidity, sulphates, and density emerged as dominant quality indicators. Feature engineering,bespecially chemistry-inspired ratios, added interpretability and small but meaningful performance gains.
 
<img width="1059" height="423" alt="image" src="https://github.com/user-attachments/assets/b1982274-06ec-40a4-83af-6c86f81d88af" />
