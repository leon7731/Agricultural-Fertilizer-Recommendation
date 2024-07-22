
<img align="top" width="80%" align="right" alt="Logo" src="https://drive.google.com/uc?export=view&id=1gngUiyEZcCaIvqHVn9yN5ZhdIiemiwpq"/>

# Introduction

In the realm of modern agriculture, the recommendation of appropriate fertilizers is indispensable, particularly with the increasing shift towards data-driven methodologies. Accurate fertilizer recommendations rely on comprehensive datasets that include soil nutrient composition, crop requirements, environmental factors, and historical fertilization practices. Analyzing and interpreting these datasets provide invaluable insights, which are crucial for optimizing fertilizer usage, enhancing crop productivity, and ensuring sustainable agricultural practices.

# Significance of Fertilizer Recommendation

- **Optimized Nutrient Management:** Precise fertilizer recommendations enable farmers to provide the exact nutrients needed by their crops. This ensures optimal growth and maximizes yield potential.

- **Resource Efficiency:** Data-driven fertilizer recommendations allow for the efficient use of fertilizers, reducing waste and minimizing the cost associated with over-application or under-application.

- **Environmental Protection:** By applying the right amount of fertilizers, farmers can significantly reduce the risk of nutrient runoff into water bodies, thus protecting the environment from pollution.

- **Soil Health Maintenance:** Appropriate fertilizer recommendations help maintain soil health by ensuring balanced nutrient levels, preventing soil degradation, and promoting long-term agricultural productivity.

- **Risk Mitigation:** Accurate recommendations can help mitigate risks associated with nutrient deficiencies or toxicities, which can adversely affect crop health and yield.

- **Sustainability:** Optimal fertilizer use promotes sustainable farming practices by reducing the environmental footprint of agriculture and conserving natural resources.

# Model Performance

Various machine learning models are employed to generate accurate fertilizer recommendations, each offering unique advantages. These include:

- **Decision Tree:** Provides interpretable rules for fertilizer application based on various factors like soil nutrient levels and crop type.
- **Random Forest:** An ensemble method that improves prediction accuracy and handles a large variety of input features effectively.
- **Support Vector Machine:** Classifies the optimal fertilizer requirements with high accuracy by finding the optimal boundaries between different nutrient levels.
- **K-Nearest Neighbors:** Recommends fertilizers by comparing current soil and crop conditions with historical data from similar conditions.
- **Neural Networks:** Captures complex relationships in the data to recommend precise fertilizer types and quantities even under varying conditions.
- **Gradient Boosting:** Combines the predictions of several weaker models to produce a powerful predictive model, enhancing accuracy.
- **Ensemble Models:** Integrates multiple machine learning models to improve overall prediction performance and robustness.

Additionally, all model hyperparameters are finely tuned using Optuna, an advanced optimization framework, to ensure optimal performance.

| Model | Precision Macro average (%) | Recall Macro average (%) | F1-Score Macro average (%) | Accuracy (%) |
|:-----------:|:------------:|:------------:|:-----------:|:-----------:|
| [XGBoost Classification](https://github.com/leon7731/Agricultural-Fertilizer-Recommendation/tree/main/XGBoost) | 97 | 97 | 97 | 97 |
| [CatBoost Classification](https://github.com/leon7731/Agricultural-Fertilizer-Recommendation/tree/main/CatBoost) | 96 | 96 | 96 | 96 |
| [Random Forest Classification](https://github.com/leon7731/Agricultural-Fertilizer-Recommendation/tree/main/Random%20Forest) | 96 | 96 | 96 | 96 |
| [KNN Classification](https://github.com/leon7731/Agricultural-Fertilizer-Recommendation/tree/main/KNN) | 96 | 96 | 96 | 96 |
| [Decision Tree Classification](https://github.com/leon7731/Agricultural-Fertilizer-Recommendation/tree/main/Decision%20Tree) | 95 | 95 | 95 | 95 |
| [Multi-Layer Perceptrons Classification](https://github.com/leon7731/Agricultural-Fertilizer-Recommendation/tree/main/MLP%20Classifier) | 89 | 89 | 89 | 89 |
| [SVM Classification](https://github.com/leon7731/Agricultural-Fertilizer-Recommendation/tree/main/SVM) | 86 | 85 | 83 | 85 |
| [AdaBoost Classification](https://github.com/leon7731/Agricultural-Fertilizer-Recommendation/tree/main/AdaBoost) | 17 | 25 | 19 | 25 |

# Conclusion

Accurate fertilizer recommendation is essential for the success of modern agriculture. It enables farmers to make informed decisions, optimize nutrient management, mitigate risks, and enhance sustainability. By leveraging comprehensive datasets and advanced analytical tools, farmers and agricultural experts can significantly improve production efficiency, soil health, and crop profitability. Ultimately, this approach ensures food security and fosters environmental stewardship within the agricultural sector.
This structure includes ensemble and gradient boosting models in the "Model Performance" section, highlighting their roles in enhancing prediction accuracy and robustness.
