# Predicting Customer Churn for SyriaTel Communications: A Data-Driven Approach to Retention Strategy
![hello](https://github.com/JimalShafi/SyriaTelCustomerChurn/blob/main/IMAGES/INTRODUCTION.png?raw=true)
# Project Title

# Table of Contents
* [Project Overview](#Project-Overview)
* [Problem Statement](#Problem-Statement)
* [Objective](#Objective)
* [Dataset Description](#Dataset-Description)
* [Exploratory Data Analysis (EDA)](#Exploratory-Data-Analysis-(EDA))
* [Is Calling Customer Service a Sign of Potential Churn?](#Is-Calling-Customer-Service-a-Sign-of-Potential-Churn?)
* [What Does Plan Usage Tell Us About Churn?](#What-Does-Plan-Usage-Tell-Us-About-Churn?)
* [Are Customers in Certain Areas More Likely to Churn?](#Are-Customers-in-Certain-Areas-More-Likely-to-Churn?)
* [EDA Conclusion](#EDA-Conclusion)
* [Model Analysis](#Model-Analysis)
* [Metric-Used](#Metric-Used)
* [Cost Benefit Analysis](#Cost-Benefit-Analysis)
* [Feature Importances](#Feature-Importances)
* [Model Fit & Score](#Model-Fit-&-Score)
* [Model-Conclusion](#Model-Conclusion)
* [Future Work](#Future-Work)
* [How-to-Run](#How-to-Run)
* [Dependencies](#Dependencies)
* [Credits & References](#Credits-&-References)


## Project Overview

SyriaTel Communications, a leading telecommunications company, faces a significant challenge: customer churn. Customer churn occurs when a customer discontinues their service, which is costly due to the loss of recurring revenue and the expense of acquiring new customers. This project aims to address this problem by predicting which customers are likely to churn, allowing SyriaTel to implement targeted retention strategies.

 ## Problem Statement

Customer churn is a critical issue for SyriaTel, as it directly impacts revenue and profitability. The challenge lies in accurately identifying at-risk customers and preventing them from leaving, which requires an effective predictive model that can guide retention efforts.

## Objective

The primary objective of this project is to build a machine learning model that accurately predicts customer churn, enabling SyriaTel to proactively engage with customers at risk of leaving. This will help reduce churn rates, lower customer acquisition costs, and increase overall profitability.

## Dataset Description

The dataset used in this project is the "Churn in Telecoms" dataset, which consists of 21 columns and 3333 unique customer records. The data was clean with no significant outliers or missing values, making it ideal for analysis. The dataset can be found in the data folder or via this Kaggle link [this link](https://www.kaggle.com/becksddf/churn-in-telecoms-dataset). 

# Exploratory Data Analysis (EDA)

## Is Calling Customer Service a Sign of Potential Churn?

**Findings**

The churn rate in the training dataset is approximately 14.5%.
The likelihood of churn increases with the number of customer service calls. Specifically, customers who make 4 or more calls have a 50% likelihood of churning.

![customer_service](https://github.com/JimalShafi/SyriaTelCustomerChurn/blob/main/IMAGES/Customer%20service%20calls%20and%20churn.png?raw=true)

- Customer service calls alone cannot guarantee that a customer will churn. In fact, the majority of customers who DID NOT churn made 1-2 customer service calls. However, it is important to note that the majority people who DID churn made 1-4 calls to customer service. Therefore, more than 3 calls to customer service should be a red flag that a customer is more likely to churn.

![customer_service_2](https://github.com/JimalShafi/SyriaTelCustomerChurn/blob/main/IMAGES/Distribution%20f%20customer%20service%20calls%20and%20churn.png?raw=true)

**Recommendation:** 

Revise the customer service protocol to offer incentives or discounts to customers making more than 3 calls, as they are more likely to churn.. 


## What Does Plan Usage Tell Us About Churn?

**Findings**

The usage across day, evening, night, and international calls is almost identical between customers who churned and those who did not.
Customers with international plans have a higher churn rate, likely due to the similar charge for international calls for both plan holders and non-plan holders (27 cents per minute).
It is also interesting to note that the percentage of customers who churned was higher for customers with international plans than for customers without international plans. Because of this similar charge for international calls, it is possible that the customers who had an international plan and churned did not feel that paying for the international plan was worth it.

![international_plan](https://github.com/JimalShafi/SyriaTelCustomerChurn/blob/main/IMAGES/International%20Plan.png?raw=true)

**Recommendations:**

Consider adjusting the rates for international calls to provide better value to customers with an international plan, potentially reducing their likelihood of churning.


## Are Customers in Certain Areas More Likely to Churn?

**Findings**

Churn rates vary significantly by state. For instance, Texas has the highest churn rate (27%), followed by New Jersey, Maryland, and California (over 23%).
States like Hawaii and Iowa have the lowest churn rates (under 0.05%).

![state_choropleth](https://github.com/JimalShafi/SyriaTelCustomerChurn/blob/main/IMAGES/churn%20by%20state.png?raw=true)

There could be a few reasons for this difference in churn in different states. One reason could be the lack of competitors in places like Hawaii and Iowa, which are more remote. States such as California, New Jersey or Texas could have many other big players in the market, which causes our customers to have other options when they feel inclined to leave. Another reason could be the lack of good service in certain areas in states with high churn.

**Recommendations**

Investigate competitors and signal strength in states with high churn, such as Texas, California, New Jersey, and other states with high churn to to identify potential factors contributing to customer loss; see if they are offering introductory offers that might compel some of our customers to churn. I also recommend looking into the cell signal in these states with higher churn to see if there are any deadzones contributing to the higher rates. 

## EDA Conclusion

In summary, frequent customer service calls and certain geographic regions are significant indicators of churn. Additionally, customers with international plans may not perceive enough value, leading to higher churn rates. These insights can guide SyriaTel in refining their customer retention strategies.


# Model Analysis

    
## Metric Used

The model was evaluated using the recall metric because false negatives (failing to identify a customer who will churn) are more costly than false positives. Misidentifying a loyal customer as a potential churner is less detrimental than losing a customer outright.

## Cost Benefit Analysis

The cost benefit analysis is crucial in understanding the financial implications of the model's predictions:

False Positive Cost (FP): $25 per customer (due to offering a discount unnecessarily).
False Negative Cost (FN): $100 per customer (loss of monthly payment and customer acquisition cost).
True Positive Benefit (TP): $25 per customer (retained with a discount).
True Negative Benefit (TN): $0 (no action needed)

![cb_analysis](https://github.com/JimalShafi/SyriaTelCustomerChurn/blob/main/ValidationMAIN.png?raw=true)

- Conclusion: The cost-benefit analysis reveals that the current strategy would result in a financial loss of $1.39 per customer per month. This negative expected value highlights a potential misalignment between the model's predictions and the customer retention strategy. While retaining customers who are falsely predicted to leave may not be inherently problematic, the overall financial impact indicates that adjustments are necessary.

To capitalize on the model's predictive strengths, it may be beneficial to revisit the retention strategy or refine the cost-benefit model to ensure it accurately reflects the business's objectives. With these adjustments, the model could contribute positively to the company’s long-term financial success.
 The good news here is that with this model predicting churn, we are not LOSING money! We can see the breakdown of each cost and benefit multiplied by the number of TP, TN, FP, FNs on the confusion matrix above. 


## Feature Importances

The most important features influencing churn prediction were:

Customer service calls
Total day charge
International plan
Number of voice mail messages

The final model's feature importances are graphed below. 

![feat_importances](https://github.com/JimalShafi/SyriaTelCustomerChurn/blob/main/IMAGES/FeatureImportanceMAIN.png?raw=true)


## Model Fit & Score

Validation Recall Score: 0.8
Training Recall Score: 0.84
The close recall scores indicate that the model is slightly overfitted but still generalizes well. The model's high recall ensures it effectively identifies customers at risk of churning.


## Model Conclusion

In conclusion, while the Gradient Boosting Classifier provides a robust predictive model for SyriaTel’s churn prediction needs, the cost-benefit analysis reveals that the current strategy would result in a financial loss of $1.39 per customer per month. This negative expected value highlights a potential misalignment between the model's predictions and the customer retention strategy. While retaining customers who are falsely predicted to leave may not be inherently problematic, the overall financial impact indicates that adjustments are necessary.

To capitalize on the model's predictive strengths, it may be beneficial to revisit the retention strategy or refine the cost-benefit model to ensure it accurately reflects the business's objectives. With these adjustments, the model could contribute positively to the company’s long-term financial success.


# Future Work

If given more time, the following areas could be explored:

- Competitor Analysis: Investigate competitors' offers in high-churn states.
- Signal Strength Analysis: Analyze cell signal quality in states with high churn to identify potential service issues.
- Voicemail Data: Assess whether voicemail usage could be an additional indicator of churn.
- Model Enhancement: Continue refining the model to improve recall and explore other algorithms like XGBoost or CatBoost.
- Automated Expected Value Integration: Develop a class for expected value and integrate it into the model pipeline for more accurate and automated cost-benefit analysis.

## How to Run

1. Clone the repository:

git clone https://github.com/JimalShafi/SyriaTelCustomerChurn.git
cd SyriaTelCustomerChurn

2. Set Up the Virtual Environment:

python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

3. Install Dependencies:

pip install -r requirements.txt

4. Run the Jupyter Notebook:

jupyter notebook

5. Execute the Notebook:

Open the notebook and run all cells to reproduce the analysis.


## Dependencies
To run the notebook, you will need the following libraries:

- **pandas**: For data manipulation and analysis
- **numpy**: For numerical operations
- **matplotlib**: For data visualization
- **seaborn**: For statistical data visualization
- **scikit-learn**: For machine learning algorithms and tools
- **jupyter**: To run the notebook

You can install these dependencies using the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

## Credits & References

This project was developed by Akoko Jim Alex. The code is based on the following resources:
Dataset: Churn in Telecoms Dataset from Kaggle.
Libraries: Pandas, NumPy, Matplotlib, [Seaborn](https://seaborn.pydata



