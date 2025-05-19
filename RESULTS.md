# Assignment 5: Health Data Classification Results

This file contains your manual interpretations and analysis of the model results from the different parts of the assignment.

## Part 1: Logistic Regression on Imbalanced Data

### Interpretation of Results

In this section, provide your interpretation of the Logistic Regression model's performance on the imbalanced dataset. Consider:

- Which metric performed best and why?
The best performing metric was accuracy (93.79%), largely because the dataset is imbalanced. The model achieved high accuracy by mostly predicting the majority class (non-positive cases).

- Which metric performed worst and why?
The worst performing metric was recall (48.59%). The model missed over half of the actual positive cases, which is a critical issue in many healthcare applications where identifying true positives is essential.

- How much did the class imbalance affect the results?
Class imbalance had a significant impact on model performance. It caused inflated accuracy and precision scores while low recall. The minority class was underrepresented during training.

- What does the confusion matrix tell you about the model's predictions?
The confusion matrix shows that the model made 1306 true negative predictions and 69 true positive predictions, but missed 73 actual positives (false negatives). This supports the low recall metric obtained, and the bias toward the majority class.


## Part 2: Tree-Based Models with Time Series Features

### Comparison of Random Forest and XGBoost

In this section, compare the performance of the Random Forest and XGBoost models:

- Which model performed better according to AUC score?
XGBoost outperformed Random Forest with an AUC score of 0.9969 vs. 0.9782 (on one run) and 0.9953 vs. 0.9735 (on another). XGBoost had better overall discriminative ability.

- Why might one model outperform the other on this dataset?
XGBoost likely outperformed Random Forest due to better handling of complex patterns and built-in regularization, which helps prevent overfitting but also still capturing informative splits in the data.

- How did the addition of time-series features (rolling mean and standard deviation) affect model performance?
Adding time-series features (rolling mean and standard deviation) improved model performance significantly. These features captured underlying temporal trends or fluctuations that static features alone could not, giving tree-based models more predictive power.


## Part 3: Logistic Regression with Balanced Data

### Improvement Analysis

In this section, analyze the improvements gained by addressing class imbalance:

- Which metrics showed the most significant improvement?
Recall improved dramatically from 48.59% to 84.51%, showing that addressing class imbalance helped the model identify far more true positives.

- Which metrics showed the least improvement?
AUC stayed nearly the same (0.9164 in both), suggesting that the model's overall ranking of predictions did not change despite the new balance in training data.

- Why might some metrics improve more than others?
Recall improved the most because SMOTE increased the minority class, giving the model more examples to learn from. Accuracy and F1 decreased slightly, and precision dropped significantly (from 0.7931 to 0.4054), probably because of an increase in false positives which is a tradeoff when made if we want to optimize for recall.

- What does this tell you about the importance of addressing class imbalance?
These results show that addressing class imbalance is crucial when the goal is to detect the minority class effectively. It trades off some precision and accuracy to improve recall, which something we usually want in healthcare settings.



## Overall Conclusions

Summarize your key findings from all three parts of the assignment:

- What were the most important factors affecting model performance?
- Which techniques provided the most significant improvements?
- What would you recommend for future modeling of this dataset?

Key Factors Affecting Performance:
The class distribution and feature engineering (especially time-series features) were the most influential factors. Class imbalance skewed evaluation metrics and model behavior, while time-series features significantly improved tree-based model performance. SMOTE helped to significantly improve recall. XGBoost with time-series features yielded the highest AUC and best overall discrimination. For future modeling of this dataset, I recommend addressing class imbalance when minority classes are critical, engineering temporal features when dealing with time-indexed health data, and using advanced ensemble methods like XGBoost when we want to maximize predictive performance. 