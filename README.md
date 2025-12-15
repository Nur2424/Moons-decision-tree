Decision Tree Classification on the Moons Dataset

Overview

This project demonstrates an end-to-end machine learning workflow using a Decision Tree classifier on a noisy, non-linear synthetic dataset. The goal is to understand the behavior of Decision Trees, identify overfitting, and improve generalization through proper regularization and hyperparameter tuning.

The project follows the methodology described in Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow and is implemented using Scikit-Learn.

⸻

Problem Description
	•	Task: Binary classification
	•	Dataset: Moons dataset (synthetic, non-linear)
	•	Challenge: High noise and overlapping classes
	•	Model: Decision Tree Classifier

Decision Trees are flexible models that can easily overfit noisy data. This project explores that behavior and shows how regularization improves performance on unseen data.

⸻

Dataset

The dataset is generated using sklearn.datasets.make_moons with the following parameters:
	•	Number of samples: 10,000
	•	Number of features: 2
	•	Number of classes: 2
	•	Noise level: 0.4

The dataset forms two interleaving half-moon shapes with significant overlap due to noise.

⸻

Project Workflow

The project follows a standard machine learning pipeline:
	1.	Problem definition
	2.	Data generation and visualization
	3.	Train–test split
	4.	Baseline Decision Tree model
	5.	Overfitting analysis
	6.	Hyperparameter tuning using cross-validation
	7.	Final model training
	8.	Evaluation on unseen test data
	9.	Decision boundary visualization
	10.	Summary and conclusions

⸻

Model Training
	•	A baseline Decision Tree is trained using default hyperparameters.
	•	Overfitting is observed through high training accuracy and lower test accuracy.
	•	Hyperparameters such as max_leaf_nodes are tuned using GridSearchCV.
	•	Cross-validation is used to select the best model configuration.

⸻

Evaluation Metric
	•	Accuracy is used as the primary evaluation metric.
	•	Final performance is measured on a held-out test set that is not used during training or tuning.

Expected test accuracy after tuning is approximately 85%–87%.

⸻

Key Learnings
	•	Decision Trees are highly sensitive to noise and prone to overfitting.
	•	Regularization is essential for good generalization.
	•	Cross-validation is critical for reliable hyperparameter selection.
	•	Visualizing decision boundaries helps understand model behavior.
	•	A systematic ML workflow improves both performance and reproducibility.

⸻

Technologies Used
	•	Python
	•	NumPy
	•	Matplotlib
	•	Scikit-Learn

⸻

How to Run
	1.	Clone the repository.
	2.	Install dependencies (see requirements.txt if provided).
	3.	Open the notebook in Jupyter or VS Code.
	4.	Run cells in order from top to bottom.

⸻

Notes

This project is intended as a learning-focused but professionally structured example of a classical machine learning workflow. It serves as preparation for more advanced models such as Random Forests and Gradient Boosted Trees.
