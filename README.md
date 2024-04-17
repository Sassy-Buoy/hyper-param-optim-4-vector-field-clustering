---
marp: true
theme: gaia
color: #000
colorSecondary: #333
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.jpg')
paginate: true
---

# ML for Vector Field Analysis

---

## What is Optuna?

- **Optuna** is an open-source hyperparameter optimization framework.
- It automates the process of tuning parameters for machine learning algorithms.

---

## Key Features

- **Distributed Computing**: Supports distributed optimization across multiple nodes.
- **Flexible**: Compatible with any Python-based machine learning framework.
- **Automatic**: Requires minimal user intervention, automating the parameter search process.
- **Optimization Algorithms**: Offers various algorithms like TPE, Grid Search, Random Search, etc.
- **Visualization**: Provides visualization tools to analyze optimization results.

---

## How does it work?

1. Define the search space.
2. Choose an optimization algorithm.
3. Define the objective function.
4. Run the optimization process.
5. Analyze the results and select the best set of parameters.

---

## Example Code

```python
import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

best_params = study.best_params
best_value = study.best_value

print("Best value:", best_value)
print("Best parameters:", best_params)
```

---

## Visualization

![Optuna Visualization](https://raw.githubusercontent.com/optuna/optuna/master/docs/image/parallel_coordinates_example.png)

---

## Use Cases

- **Model Training**: Optimize hyperparameters for machine learning models.
- **Deep Learning**: Tune neural network architectures and learning rates.
- **Feature Selection**: Optimize feature selection parameters.
- **Any Parameter Tuning**: Suitable for any parameter optimization task.

---
# 

## Auto encoder optimization :
- number of layers
- filters
- kernel size
- strides

---

- activation functions
- paddings
- kernel initializer       
- size of latent space (Done manually)

---

## Clustering optimization

Algorithms : DBSCAN, K-means, Ag-clustering

---
## Metrics for clustering
- Silhouette score(SS) modified : Multi-objective optimization where one objective is to maximize the SS and the second is to maximize the number of knife plots that cut through the average SS. 'Cutting' through the average SS means there is atleast one datapoint in each cluster whose SS is greater than the average SS. This ensures that all clusters perform well comparitively and one particularly good cluster does not inflate the SS.
- Centroid stand deviation : Minimize the standard deviation of the distance between the centroids of the clusters. This ensures that the centroids are evenly spaced in the latent space.
- KL divergence

---

## Open Questions

- Variational Auto encoder?
- Latent space dimensionality
- 

## 

