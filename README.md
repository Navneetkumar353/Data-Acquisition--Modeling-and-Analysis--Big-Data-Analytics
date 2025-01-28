# Data Acquisition, Modeling and Analysis: Big Data Analytics
# Yahoo! Music Recommender System

This project focuses on building a music recommendation system using the Yahoo! Music Dataset. It employs various computational techniques, including baseline methods, matrix factorization, and machine learning models, to predict user preferences for tracks based on historical data.

## Project Overview

The recommendation system leverages hierarchical data structured by track, album, artist, and genre to generate accurate predictions. The project includes the following components:

1. **Baseline Model**:
   - Combines artist, album, and genre scores to predict user preferences.
   - Achieved a score of 0.856 using historical ratings.

2. **Advanced Feature Integration**:
   - Enhanced predictions by incorporating additional features like genre scores.
   - Improved accuracy slightly over the baseline model.

3. **Matrix Factorization with ALS**:
   - Decomposed the user-item matrix into latent factors using Alternating Least Squares (ALS).
   - Tuned hyperparameters (rank, iterations, regularization) for optimal performance.
   - Achieved robust predictions by uncovering latent user-item relationships.

4. **Machine Learning Models**:
   - Implemented Logistic Regression, Decision Tree, Random Forest, and Gradient-Boosted Trees.
   - Evaluated models with and without genre features, with Logistic Regression scoring highest (0.869).

5. **Ensemble Method**:
   - Combined predictions from multiple models using weighted ensembles.
   - Achieved a Kaggle leaderboard score of 0.869, demonstrating the strength of model diversity.

## Key Features

- **Dataset**: Hierarchical structure with user, track, album, artist, and genre data.
- **Algorithms**:
  - Baseline prediction using historical ratings.
  - ALS for latent factor discovery.
  - Machine learning models for enhanced predictions.
  - Ensemble techniques to combine model strengths.
- **Performance Metrics**:
  - Evaluated models using metrics like MSE and accuracy.
  - Compared individual and ensemble model scores.

## Results

- Baseline Model: Score of 0.856.
- ALS Matrix Factorization: Optimized performance with latent factors.
- Machine Learning Models:
  - Logistic Regression: Best individual score of 0.869.
  - Ensemble Method: Improved overall reliability with a score of 0.869.

## Tools & Technologies

- **Programming Language**: Python
- **Libraries**: PySpark, Pandas, NumPy, Scikit-learn, Matplotlib
- **Frameworks**: Spark ML for large-scale data processing
- **Dataset**: Yahoo! Music Dataset

## Future Work

- Explore advanced deep learning methods for recommendation.
- Implement real-time recommendations for dynamic user preferences.
- Incorporate contextual data, such as user location or time of listening.

## License

This project is licensed under the [MIT License](LICENSE).

