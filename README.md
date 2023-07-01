# Insurance-Fraud-Detection
Fraud is one of the largest and most well-known problems that insurers face. This article focuses on claim data of a car insurance company. Fraudulent claims can be highly expensive for each insurer. Therefore, it is important to know which claims are correct and which are not. It is not doable for insurance companies to check all claims personally since this will cost simply too much time and money. In this article, we will take advantage of the largest asset which insurers have in the fight against fraud: Data. We employ various attributes about the claims, insured people and other circumstances which are included in the data by the insurer. Separating different groups of claims and the corresponding rates of fraud within those groups provide new insights.
Furthermore, we use machine learning to predict which claims are likely to be fraudulent. This information can narrow down the list of claims that need a further check. It enables an insurer to detect more fraudulent claims.
# Problem Definition
The goal of this project is to build a model that can detect auto insurance fraud. The challenge behind fraud detection in machine learning is that frauds are far less common as compared to legit insurance claims.

Insurance fraud detection is a challenging problem, given the variety of fraud patterns and relatively small ratio of known frauds in typical samples. While building detection models, the savings from loss prevention needs to be balanced with the cost of false alerts. Machine learning techniques allow for improving predictive accuracy, enabling loss control units to achieve higher coverage with low false positive rates.

Insurance frauds cover the range of improper activities which an individual may commit in order to achieve a favourable outcome from the insurance company. This could range from staging the incident, misrepresenting the situation including the relevant actors and the cause of incident and finally the extent of damage caused.
# Data Analysis
In this project, we have a dataset which has the details of the insurance policy along with the customer details. It also has the details of the accident on the basis of which the claims have been made.

The given dataset contains 1000 rows and 40 columns. The column names like policy number, policy bind date, policy annual premium, incident severity, incident location, auto model, etc.

The obvious con of this data set is the small sample size. However, there are still many companies who do not have big data sets. The ability to work with what is available is crucial for any company looking to transition into leveraging data science.

![image](https://github.com/PurvaKolhe/Insurance-Fraud-Detection/assets/93534668/43cc4704-2f89-41bd-973c-6cfd03b0e5bd)
                                            Description of the data

Compared to a company that waits for the day when it has a huge data set, the company that started with a small data set and worked on it will more likely succeed earlier in its data science journey and reap its rewards.

There are some variables which contain the null values character ‘?’. The number of null values present is given below.

![image](https://github.com/PurvaKolhe/Insurance-Fraud-Detection/assets/93534668/e6a226d2-0bdf-4528-9f4a-8f954bf5e96e)
  Unique values

# Exploratory Data Analysis
* ### Dependent Variable:
  Exploratory data analysis was conducted starting with the dependent variable, Fraud_reported. There were 247 frauds and 753 non-frauds. 24.7% of the data were frauds while 75.3% were non-fraudulent claims.

  ![image](https://github.com/PurvaKolhe/Insurance-Fraud-Detection/assets/93534668/c3cca6f0-f1ec-4912-9abb-5860ddd8efd5)
 * ### Correlations among variables:
   Heatmap was plotted for variables with at least 0.3 Pearson’s correlation coefficient, including the DV. Month as customer and age had a correlation of 0.92. Probably because drivers buy auto insurance when they own a car and this time measure only increases with age. Apart from that, there don’t seem to be many correlations in the data. There don’t seem to be multicollinearity problems except maybe that all the claims are all correlated, and somehow total claims have accounted for them. However, the other claims provide some granularity that will not otherwise be captured by total claims. Thus, these variables were kept.
# Pre-processing Pipeline
Data preprocessing is a predominant step in machine learning to yield highly accurate and insightful results. Greater the quality of data, the greater is the reliability of the produced results. Incomplete, noisy, and inconsistent data are the inherent nature of real-world datasets. Data preprocessing helps in increasing the quality of data by filling in missing incomplete data, smoothing noise, and resolving inconsistencies.
* ### Incomplete data:
can occur due to many reasons. Appropriate data may not be persisted due to a misunderstanding, or because of instrument defects and malfunctions.
* ### Noisy data
can occur for a number of reasons (having incorrect feature values). The instruments used for the data collection might be faulty. Data entry may contain human or instrument errors. Data transmission errors might occur as well.
#### There are many stages involved in data preprocessing.
* Data cleaning
* Data integration
* Data transformation
* Data reduction
# Treating null values
Sometimes there are certain columns which contain the null value used to indicate missing or unknown values or maybe the value doesn’t exist. In our dataset the null values are present in columns collision_type, property_damage, police_report_available, and _c39 with 178, 360, 343 and 1000 number of null values.

There are different ways of replacing null values from the dataset, but we are using fillna to replace the null values from our data.
# Converting labels into numeric
In our data there are columns with categorical values. The columns like incident_severity, incident_state, incident_type, insured_hobbies, authorities_contacted, incident_city, police_report_available, auto_make, collision_type, auto_model, insured_occupation, insured_education_level, property_damage, insured_relationship, policy_state, insured_sex, fraud_reported. These columns have to be treated with one hot encoding or the label encoder. The target variable fraud_reported has to convert by using label encoder only.

Label Encoder refers to converting the labels into numeric form so as to convert it into the machine readable form. Machine learning algorithms can then decide in a better way on how those labels must be operated. It is an important preprocessing step for the structured dataset in supervised learning.
# Balancing our imbalanced data
There are different algorithms present to balance the target variable. We use the SMOTE() algorithm to make our data balance.
##### SMOTE algorithm works in 4 simple steps:
1. Choose a minority class as input vector.
2. Find its k-nearest neighbors.
3. Choose one of these neighbors and place a synthetic point anywhere on the line joining the point under consideration and its chosen neighbors.
4. Repeat the step until the data is balanced.
The original shape of our data was 753 for fraud_reported with NO value and 247 for YES. The SMOTE algorithm balances our data with the highest number of values present in it.
# Building machine learning models
I have applied the following supervised learning models.

(1) Logistic Regression

(2) Decision Tree

(3) Random Forest

(4) Gradient Boosting
# Conclusion
This project has built a model that can detect auto insurance fraud. In doing so, the model can reduce losses for insurance companies. The challenge behind fraud detection in machine learning is that frauds are far less common as compared to legit insurance claims.

Four different classifiers were used in this project: Logistic regression, Random forest, Decision tree, Gradient Boosting. Three different ways of handling imbalance classes were tested out with these four classifiers: oversampling with SMOTE, hyper parameter tuning, and plotting roc curve of the models.

The best and final fitted model was a weighted <b>Logistic Regression</b> that yelled a <b>F1 score of 76.8</b> and a <b>ROC AUC of 85.45</b>. The model performed excellent. The model’s F1 score and ROC AUC scores were the highest amongst the other models. In conclusion, the model was able to correctly distinguish between fraud claims and legit claims with high accuracy.
