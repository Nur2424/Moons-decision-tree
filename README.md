# Decision Tree Classification on the Moons Dataset

## Overview

This project demonstrates an end-to-end machine learning workflow using a Decision Tree classifier on a noisy, non-linear synthetic dataset. The goal is to understand the behavior of Decision Trees, identify overfitting, and improve generalization through proper regularization and hyperparameter tuning.

The project follows the methodology described in Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow and is implemented using Scikit-Learn.

## Problem Description
	•	Task: Binary classification
	•	Dataset: Moons dataset (synthetic, non-linear)
	•	Challenge: High noise and overlapping classes
	•	Models:
		•	Decision Tree Classifier
		•	Manually constructed ensemble (Random Forest concept)

Decision Trees are flexible models that can easily overfit noisy data. This project explores that behavior and shows how regularization improves performance on unseen data.

## Dataset

The dataset is generated using sklearn.datasets.make_moons with the following parameters:

	•	Number of samples: 10,000
	•	Number of features: 2
	•	Number of classes: 2
	•	Noise level: 0.4

The dataset forms two interleaving half-moon shapes with significant overlap due to noise.

## Project Workflow

The project follows a standard machine learning pipeline:

	1.	Problem definition
	2.	Data generation and visualization
	3.	Train–test split
	4.	Baseline Decision Tree training
	5.	Overfitting analysis
	6.	Hyperparameter tuning using cross-validation
	7.	Final Decision Tree model training
	8.	Manual ensemble construction (Random Forest from scratch)
	9.	Majority voting and final evaluation
	10.	Decision boundary visualization and conclusions
	
## Model Training

	Decision Tree
	•	A baseline Decision Tree is trained using default hyperparameters.
	•	Severe overfitting is observed (very high training accuracy, lower test accuracy).
	•	Hyperparameters such as max_leaf_nodes are tuned using GridSearchCV.
	•	Cross-validation is used to select the best configuration.

	Ensemble Learning (Manual Random Forest)
	•	Multiple Decision Trees are trained on small random subsets of the training data.
	•	Each tree is intentionally weak and trained on limited data.
	•	Predictions from all trees are combined using majority voting.
	•	The ensemble achieves better generalization than a single tree.

## Evaluation Metric

	•	Accuracy is used as the primary evaluation metric.
	•	Final performance is measured on a held-out test set that is not used during training or tuning.
	•	Final ensemble test accuracy achieved:
	•	Test accuracy: 0.863

Expected test accuracy after tuning is approximately 85%–87%.

## Key Learnings

	•	Decision Trees are highly sensitive to noise and prone to overfitting.
	•	Regularization is essential for improving generalization.
	•	Cross-validation is critical for reliable hyperparameter selection.
	•	Ensemble methods reduce variance and improve stability.
	•	Majority voting allows weak learners to form a stronger model.
	•	Visualizing decision boundaries helps interpret model behavior.
	•	A structured ML workflow improves reproducibility and clarity.

## Technologies Used

	•	Python
	•	NumPy
	•	Matplotlib
	•	Scikit-Learn



## Notes

This project is designed as a learning-focused but professionally structured example of classical machine learning. It builds strong intuition for Decision Trees and ensemble methods and serves as preparation for built-in models such as Random Forests and Gradient Boosted Trees.

## Author

Created by Nur
Machine Learning Student and aspiring ML Engineer
