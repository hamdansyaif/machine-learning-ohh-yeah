# Chapter 9: Unsupervised Learning Techniques

**Author:** Hamdan Syaifuddin Zuhri  
**NIM:** 1103220220  
**Course:** Machine Learning / Deep Learning  
**Institution:** Telkom University  
**Source:** Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd Edition) by Aurélien Géron

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Chapter Structure](#chapter-structure)
4. [Key Concepts](#key-concepts)
5. [Algorithms Covered](#algorithms-covered)
6. [Datasets Used](#datasets-used)
7. [Requirements](#requirements)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Results Summary](#results-summary)
11. [Key Takeaways](#key-takeaways)
12. [References](#references)

---

## 📖 Overview

This notebook provides a **comprehensive, production-ready implementation** of unsupervised learning techniques covered in Chapter 9 of "Hands-On Machine Learning." Unsupervised learning involves discovering hidden patterns and structures in unlabeled data without explicit target variables.

The implementation emphasizes:
- ✅ **Professional code quality** with proper documentation
- ✅ **Theoretical foundations** with mathematical explanations
- ✅ **Practical applications** demonstrating real-world use cases
- ✅ **Comparative analysis** between different clustering algorithms
- ✅ **Best practices** for model selection and evaluation

---

## 🎯 Learning Objectives

By completing this notebook, you will be able to:

1. ✅ **Implement and optimize K-Means clustering** for various datasets
2. ✅ **Apply DBSCAN** for density-based clustering with arbitrary shapes
3. ✅ **Utilize Gaussian Mixture Models** for probabilistic clustering
4. ✅ **Perform model selection** using information criteria (BIC/AIC)
5. ✅ **Apply clustering for preprocessing** and feature engineering
6. ✅ **Detect anomalies** using clustering-based approaches
7. ✅ **Implement semi-supervised learning** with label propagation
8. ✅ **Visualize and interpret** clustering results effectively

---

## 📂 Chapter Structure

### **Part 1: Environment Setup** (Cells 1-17)
Comprehensive environment configuration including:
- Library imports and version verification
- Global configuration for reproducibility (Random Seed: 42)
- Helper functions for visualization and evaluation
- Quick functionality tests

### **Part 2: K-Means Clustering** (Cells 18-35)
In-depth exploration of K-Means algorithm:
- **Fundamentals**: Algorithm mechanics, initialization strategies (K-Means++)
- **Mathematical Foundation**: Inertia minimization, Voronoi tessellation
- **Implementation**: Hard vs soft clustering, centroid tracking
- **Optimization**: Finding optimal K using Elbow Method and Silhouette Analysis
- **Limitations**: Demonstration of failure cases (non-spherical, varying densities)
- **Complexity**: O(kmn) time complexity analysis

### **Part 3: Practical Applications** (Cells 36-47)
Real-world applications of K-Means:
- **Image Segmentation**: Color-based clustering for image compression
- **Preprocessing for Classification**: Feature engineering using cluster distances (Digits dataset)
- **Semi-Supervised Learning**: Three label propagation strategies
  - Representative Labeling (94.84% accuracy with only 2.8% labeled data)
  - Majority Vote Propagation (90.76% accuracy)
  - Self-Training with Iterative Propagation (90.61% accuracy)

### **Part 4: DBSCAN** (Cells 48-52)
Density-based clustering for complex structures:
- **Core Concepts**: Core points, border points, and noise detection
- **Algorithm Mechanics**: Density-reachability and connectivity
- **Parameter Tuning**: eps and min_samples sensitivity analysis
- **Advantages**: Handling arbitrary shapes, automatic outlier detection
- **Comparative Analysis**: DBSCAN vs K-Means on challenging datasets (Moons, Circles)

### **Part 5: Gaussian Mixture Models** (Cells 53-65)
Probabilistic clustering with GMM:
- **Mathematical Foundation**: EM algorithm, likelihood maximization
- **Covariance Types**: Full, tied, diagonal, and spherical covariances
- **Model Selection**: BIC and AIC for optimal component selection
- **Bayesian GMM**: Automatic component selection through weight pruning
- **Anomaly Detection**: Density-based outlier identification
- **Soft Clustering**: Probability distributions for cluster assignments

---

## 🔑 Key Concepts

### **1. K-Means Clustering**

**Algorithm:**
1. Initialize K centroids (K-Means++ recommended)
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat until convergence

**Objective Function (Inertia):**
```
Inertia = Σ min ||x_i - μ_j||²
```

**Strengths:**
- Fast and scalable: O(kmn) complexity
- Simple to understand and implement
- Works well with spherical clusters

**Weaknesses:**
- Requires pre-specifying K
- Sensitive to initialization (solved by K-Means++)
- Assumes spherical clusters of similar size
- Sensitive to outliers

---

### **2. DBSCAN (Density-Based Spatial Clustering)**

**Parameters:**
- **eps (ε)**: Maximum distance for neighborhood
- **min_samples**: Minimum points for dense region

**Point Classification:**
- **Core Point**: ≥ min_samples neighbors within ε
- **Border Point**: Within ε of core point, but not core
- **Noise Point**: Neither core nor border (label = -1)

**Strengths:**
- Discovers arbitrary cluster shapes
- Automatic outlier detection
- No need to specify K
- Robust to outliers

**Weaknesses:**
- Sensitive to parameter selection
- Struggles with varying densities
- Not suitable for high-dimensional data

---

### **3. Gaussian Mixture Models (GMM)**

**Model:**
```
p(x) = Σ φ_k N(x|μ_k, Σ_k)
```

**EM Algorithm:**
- **E-Step**: Calculate cluster responsibilities (probabilities)
- **M-Step**: Update parameters (weights, means, covariances)

**Covariance Types:**
- **Full**: Most flexible, each cluster has unique covariance matrix
- **Tied**: All clusters share same covariance
- **Diagonal**: Axes-aligned ellipsoids
- **Spherical**: Circular clusters (similar to K-Means)

**Strengths:**
- Soft clustering with probabilities
- Handles elliptical clusters
- Density estimation capability
- Anomaly detection through likelihood

**Weaknesses:**
- Slower than K-Means
- Requires specifying K (unless using Bayesian GMM)
- Can converge to local optima
- Sensitive to initialization

---

## 🤖 Algorithms Covered

| Algorithm | Type | Cluster Shape | K Required | Outlier Detection | Complexity |
|-----------|------|---------------|------------|-------------------|------------|
| **K-Means** | Centroid-based | Spherical | Yes | No | O(kmn) |
| **K-Means++** | Initialization | Spherical | Yes | No | O(kmn) |
| **DBSCAN** | Density-based | Arbitrary | No | Yes | O(n log n) |
| **GMM** | Probabilistic | Elliptical | Yes | Via density | O(kmn·iter) |
| **Bayesian GMM** | Probabilistic | Elliptical | No (auto) | Via density | O(kmn·iter) |

---

## 📊 Datasets Used

### **1. Synthetic Blob Data** (`sklearn.datasets.make_blobs`)
- **Purpose**: K-Means fundamentals demonstration
- **Samples**: 1,500
- **Features**: 2D (for visualization)
- **Clusters**: 5 well-separated Gaussian blobs

### **2. Moons Dataset** (`sklearn.datasets.make_moons`)
- **Purpose**: Demonstrating K-Means limitations and DBSCAN advantages
- **Samples**: 400-500
- **Features**: 2D non-linearly separable
- **Structure**: Two interleaving crescent shapes

### **3. Circles Dataset** (`sklearn.datasets.make_circles`)
- **Purpose**: Concentric structure demonstration
- **Samples**: 400
- **Features**: 2D
- **Structure**: Nested circles

### **4. Digits Dataset** (`sklearn.datasets.load_digits`)
- **Purpose**: Preprocessing for classification, semi-supervised learning
- **Samples**: 1,797
- **Features**: 64 (8×8 pixel images)
- **Classes**: 10 (digits 0-9)

### **5. Synthetic Image Data**
- **Purpose**: Image segmentation demonstration
- **Dimensions**: 200×300 pixels (RGB)
- **Regions**: 5 distinct color regions with added noise

---

## 🔧 Requirements

### **Python Version**
- Python 3.8 or higher

### **Core Libraries**
```python
numpy >= 1.26.4
pandas >= 2.3.3
matplotlib >= 3.10.8
seaborn >= 0.13.2
scikit-learn >= 1.7.2
scipy >= 1.14.1
```

### **Optional Libraries**
```python
pillow >= 11.3.0  # For image processing
imageio >= 2.37.2  # For image I/O operations
```

### **Hardware**
- **Minimum**: 4GB RAM, dual-core processor
- **Recommended**: 8GB+ RAM for larger datasets
- **GPU**: Not required (CPU-based algorithms)

---

## 📥 Installation

### **Option 1: Using Conda (Recommended)**
```bash
# Create new environment
conda create -n ml_chapter9 python=3.10

# Activate environment
conda activate ml_chapter9

# Install dependencies
conda install numpy pandas matplotlib seaborn scikit-learn scipy pillow imageio

# Or install via pip in conda environment
pip install numpy pandas matplotlib seaborn scikit-learn scipy pillow imageio
```

### **Option 2: Using pip**
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn scipy pillow imageio
```

### **Option 3: Install from requirements.txt**
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.26.4
pandas>=2.3.3
matplotlib>=3.10.8
seaborn>=0.13.2
scikit-learn>=1.7.2
scipy>=1.14.1
pillow>=11.3.0
imageio>=2.37.2
```

---

## 🚀 Usage

### **Running the Notebook**

1. **Clone the repository:**
```bash
git clone <repository-url>
cd handson-ml2/Chapter-09
```

2. **Launch Jupyter Notebook or VS Code:**
```bash
# Using Jupyter Notebook
jupyter notebook chapter_09_unsupervised_learning.ipynb

# Using VS Code
code chapter_09_unsupervised_learning.ipynb
```

3. **Execute cells sequentially:**
   - Run all cells from top to bottom
   - Each section builds upon previous concepts
   - All visualizations will be generated inline

### **Quick Start Example**
```python
# Import necessary libraries
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_blobs(n_samples=300, centers=3, random_state=42)

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], 
           kmeans.cluster_centers_[:, 1], 
           s=300, c='red', marker='X')
plt.title('K-Means Clustering')
plt.show()
```

---

## 📈 Results Summary

### **K-Means Performance (5-Cluster Blob Dataset)**
- **Inertia**: 1,048.18
- **Silhouette Score**: 0.7961 (Excellent)
- **Convergence**: 2 iterations
- **Training Time**: ~0.045 seconds
- **Cluster Balance**: Nearly perfect (19.9-20.1% per cluster)

### **Semi-Supervised Learning Results (Digits Dataset)**
**Scenario**: Only 50 labeled samples (2.8% of 1,797 total)

| Strategy | Accuracy | Improvement vs Baseline |
|----------|----------|------------------------|
| Baseline (50 labeled only) | 86.66% | - |
| Representative Labeling | **94.84%** | **+8.18%** |
| Majority Vote Propagation | 90.76% | +4.10% |
| Self-Training | 90.61% | +3.95% |
| Fully Supervised (all labels) | 100.00% | +13.34% |

**Key Finding**: Representative Labeling achieved 94.84% of fully supervised performance using only 2.8% labeled data!

### **DBSCAN vs K-Means (Moons Dataset)**
- **K-Means Silhouette**: 0.40 (Failed to separate moons)
- **DBSCAN Silhouette**: 0.82 (Successfully separated moons)
- **DBSCAN Noise Detection**: Automatically identified 8% of points as outliers

### **GMM Model Selection**
- **Optimal Components (BIC)**: 3 (matches true number)
- **Bayesian GMM**: Automatically selected 3 active components from 10 candidates
- **Best Covariance Type**: Full (most flexible for elliptical clusters)

### **Image Segmentation**
- **Original Colors**: Thousands of unique colors
- **K=5 Segmentation**: Reduced to 5 colors while preserving structure
- **Compression**: Significant reduction in color palette

---

## 💡 Key Takeaways

### **1. Algorithm Selection Guide**

**Use K-Means when:**
- ✅ Clusters are roughly spherical
- ✅ You know the number of clusters
- ✅ Speed is critical (large datasets)
- ✅ Hard clustering is sufficient

**Use DBSCAN when:**
- ✅ Clusters have arbitrary shapes
- ✅ Number of clusters is unknown
- ✅ Outlier detection is needed
- ✅ Data has varying densities (to some extent)

**Use GMM when:**
- ✅ Clusters are elliptical
- ✅ You need probability estimates
- ✅ Soft clustering is required
- ✅ Density estimation is needed
- ✅ Anomaly detection via likelihood

**Use Bayesian GMM when:**
- ✅ All GMM benefits plus
- ✅ Automatic model selection is needed
- ✅ You want to avoid manual K tuning

### **2. Best Practices**

**K-Means:**
- Always use K-Means++ initialization (default in scikit-learn)
- Run multiple initializations (`n_init=10`)
- Use Elbow Method AND Silhouette Analysis for K selection
- Scale features before clustering
- Be aware of outlier sensitivity

**DBSCAN:**
- Start with `min_samples = 5` for 2D data
- Use k-distance graph to select eps
- Try multiple parameter combinations
- Accept that some points will be noise (label=-1)

**GMM:**
- Use BIC for model selection (penalizes complexity)
- Try different covariance types
- Consider Bayesian GMM for automatic K selection
- Check convergence (`gmm.converged_`)
- Use for anomaly detection via `score_samples()`

### **3. Common Pitfalls to Avoid**

❌ **Using K-Means on non-spherical data**
✅ Use DBSCAN or GMM instead

❌ **Not scaling features**
✅ Use StandardScaler before clustering

❌ **Ignoring silhouette analysis**
✅ Always validate with multiple metrics

❌ **Using random initialization for K-Means**
✅ Use K-Means++ (default)

❌ **Treating all DBSCAN noise as errors**
✅ Noise detection is a feature, not a bug

❌ **Forgetting to check GMM convergence**
✅ Always verify `gmm.converged_`

### **4. Practical Applications**

✅ **Customer Segmentation**: Group customers for targeted marketing
✅ **Image Compression**: Reduce color palette while preserving quality
✅ **Anomaly Detection**: Identify unusual patterns in data
✅ **Semi-Supervised Learning**: Leverage unlabeled data effectively
✅ **Feature Engineering**: Create cluster-based features for classification
✅ **Data Exploration**: Discover hidden structure in unlabeled data
✅ **Recommender Systems**: Group similar users or items

---

## 📚 References

### **Primary Source**
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.
  - Chapter 9: Unsupervised Learning Techniques (Pages 265-306)
  - Appendix A: Exercise Solutions (Pages 749-783)

### **Academic Papers**
- MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations." *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*.
- Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The advantages of careful seeding." *Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms*.
- Ester, M., et al. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise." *KDD-96 Proceedings*.
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). "Maximum likelihood from incomplete data via the EM algorithm." *Journal of the Royal Statistical Society*.

### **Documentation**
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Scikit-learn K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Scikit-learn DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [Scikit-learn GMM](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)

### **Additional Resources**
- [Clustering Algorithms Explained](https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68)
- [Understanding DBSCAN](https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html)
- [Gaussian Mixture Models Explained](https://brilliant.org/wiki/gaussian-mixture-model/)

---

## 📝 Notes

### **Reproducibility**
- All random processes use `random_state=42` for reproducibility
- Results are consistent across different runs and environments
- Visualizations use consistent styling and color schemes

### **Performance Considerations**
- K-Means is highly scalable (tested up to 10,000+ samples)
- DBSCAN performance depends on eps and data structure
- GMM is slower than K-Means but provides richer information
- Consider MiniBatchKMeans for very large datasets (>100k samples)

### **File Size Optimization**
- Notebook optimized to stay under 10MB for GitHub preview
- Visualizations are efficient and informative
- Code is modular and reusable

---

## 🤝 Contributing

This notebook is part of an academic assignment for Machine Learning course at Telkom University. While direct contributions are not expected, feedback and suggestions are welcome.

For questions or discussions:
- **Name**: Hamdan Syaifuddin Zuhri
- **NIM**: 1103220220
- **Institution**: Telkom University

---

## 📄 License

This project is created for educational purposes as part of the Machine Learning course curriculum at Telkom University. The code implementations are based on "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron.

---

## 🙏 Acknowledgments

- **Aurélien Géron** for the excellent book "Hands-On Machine Learning"
- **Scikit-learn development team** for comprehensive machine learning library
- **Telkom University** Machine Learning course instructors
- **Open-source community** for various tools and libraries used

---

**Last Updated**: January 2026  
**Version**: 1.0  
**Status**: Complete ✅

---

*This README is part of Chapter 9 implementation for the Machine Learning course at Telkom University. All code, visualizations, and documentation follow professional software engineering and data science best practices.*