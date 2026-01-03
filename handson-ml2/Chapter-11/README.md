# 📚 Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow

**Comprehensive Implementation and Summary of Chapter 11: Training Deep Neural Networks**

---

## 👤 Student Information

| Field | Information |
|-------|-------------|
| **Name** | Hamdan Syaifuddin Zuhri |
| **Class** | TK-46-06 |
| **NIM** | 1103220220 |
| **University** | Telkom University |
| **Program** | Computer Engineering |

---

## 📖 Repository Purpose

This repository contains a comprehensive implementation and theoretical explanation of **Chapter 11: Training Deep Neural Networks** from the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"* by Aurélien Géron (O'Reilly).

The goal is to:
- ✅ Reproduce all key concepts and code examples from the chapter
- ✅ Provide detailed theoretical explanations for each technique
- ✅ Demonstrate practical applications on real datasets
- ✅ Solve all end-of-chapter exercises
- ✅ Create production-ready deep learning pipelines

---

## 🎯 Project Overview

This project systematically explores advanced techniques for training deep neural networks, addressing common challenges such as:
- **Vanishing/Exploding Gradients Problem**
- **Proper Weight Initialization Strategies**
- **Optimal Activation Function Selection**
- **Batch Normalization Implementation**
- **Transfer Learning Applications**
- **Advanced Optimizers (SGD, Adam, RMSprop, Nadam)**
- **Learning Rate Scheduling**
- **Regularization Techniques (Dropout, L1/L2, Max-Norm)**

---

## 📊 Experiments & Results Summary

### **Key Experiments Conducted:**

| Experiment | Topic | Key Finding | Performance Impact |
|------------|-------|-------------|-------------------|
| **1** | Vanishing Gradients | Poor init = 10% acc, He init = 85% acc | **+75 percentage points** |
| **2** | Weight Initialization | He initialization matches theoretical variance perfectly | Enables successful training |
| **3** | Initialization Impact | He init converges from epoch 1, Poor init stuck at 10% | **Dramatic difference** |
| **4** | Activation Functions | ReLU: 85.25%, SELU: 85.20%, ELU: 85.00% | **~1% difference** |
| **5** | Batch Normalization | BN gives +3.2% boost in epoch 1 | **Faster convergence** |
| **6** | Gradient Clipping | Clip by Value: 86.80% (best) | **+0.6% improvement** |
| **7** | Transfer Learning | From Scratch (95.1%) > Transfer (88.2%) | **Negative transfer** |
| **8** | Optimizers | SGD+Momentum: 87.85% (best), Adam: 87.65% | **SGD wins** |
| **9** | LR Scheduling | ReduceLROnPlateau: 89.25% (best) | **+0.25% improvement** |
| **10** | L1/L2 Regularization | No Reg: 87.60%, L2(0.01): 80.35% | **Too strong hurts** |
| **11** | Dropout | Dropout 10%: 88.80% > No Dropout: 86.65% | **+2.15% improvement** |
| **12** | Combined Techniques | Best Practices: 90.05% | **+2.45% vs baseline** |
| **13** | Fashion MNIST | Production model: **90.22% test accuracy** | **State-of-the-art** |
| **14** | CIFAR-10 | Deep network: **56.75% test accuracy** | Challenging dataset |

---

## 🏆 Final Model Performance

### **Fashion MNIST (Production Model)**

**Architecture:**
- 3 Hidden Layers: [300, 200, 100]
- Activation: ReLU
- Regularization: L2(0.0001) + Dropout(0.2) + Batch Normalization
- Optimizer: SGD + Nesterov Momentum (0.9)
- Callbacks: ReduceLROnPlateau + EarlyStopping

**Results:**
- ✅ **Test Accuracy: 90.22%**
- ✅ **Best Validation Accuracy: 90.48%**
- ✅ **Training Time: 5.42 minutes** (51 epochs)
- ✅ **Overfitting Gap: 4.26%** (well-controlled)

### **CIFAR-10 (Deep Network)**

**Architecture:**
- 4 Hidden Layers: [400, 300, 200, 100]
- Activation: ReLU
- Regularization: Batch Normalization + Dropout(0.3)
- Optimizer: Adam (lr=0.001)

**Results:**
- ✅ **Test Accuracy: 56.75%**
- ✅ **Best Validation Accuracy: 58.92%**
- ✅ **Training Time: 2.35 minutes** (30 epochs)

---

## 📁 Repository Structure
```
chapter-11-training-deep-neural-networks/
│
├── README.md                          # This file
├── chapter_11_notebook.ipynb          # Main Jupyter notebook with all experiments
│
├── results/                           # Experiment results
│   ├── figures/                       # Plots and visualizations
│   ├── models/                        # Saved model weights
│   └── metrics/                       # Performance metrics
│
└── docs/                              # Additional documentation
    ├── theory_notes.md                # Theoretical explanations
    └── best_practices.md              # Production guidelines
```

---

## 🔬 Models & Techniques Used

### **1. Weight Initialization Methods**
- ❌ **Poor Initialization:** RandomNormal(mean=0, std=1.0)
- ✅ **Glorot (Xavier):** For sigmoid/tanh activations
- ✅ **He Initialization:** For ReLU and variants (RECOMMENDED)
- ✅ **LeCun Initialization:** For SELU activation

### **2. Activation Functions**
- **ReLU:** Default choice (fast, effective)
- **Leaky ReLU:** Fixes dying ReLU problem
- **ELU:** Better performance, smoother gradients
- **SELU:** Self-normalizing for very deep networks

### **3. Optimizers Tested**
- **SGD (vanilla):** Baseline
- **SGD + Momentum (β=0.9):** 🏆 **BEST for production**
- **SGD + Nesterov:** Lookahead variant
- **Adagrad:** Adaptive learning rates
- **RMSprop:** Decay learning rate per parameter
- **Adam:** Momentum + RMSprop combined
- **Nadam:** Nesterov + Adam

### **4. Learning Rate Schedules**
- **Constant LR:** Baseline
- **Exponential Decay:** Smooth decrease
- **Piecewise Constant:** Step-wise reduction
- **ReduceLROnPlateau:** 🏆 **BEST (adaptive)**

### **5. Regularization Techniques**
- **L1 Regularization:** Sparse models
- **L2 Regularization:** Weight decay
- **Dropout (10-30%):** 🏆 **MOST EFFECTIVE**
- **Batch Normalization:** Faster convergence + regularization effect
- **Early Stopping:** Prevent overtraining

---

## 📊 Performance Metrics

All models evaluated using:
- **Accuracy:** Primary metric for classification
- **Loss:** Sparse categorical cross-entropy
- **Overfitting Gap:** Train accuracy - Validation accuracy
- **Convergence Speed:** Epochs to reach best performance
- **Training Time:** Wall-clock time

---

## 🚀 How to Navigate This Repository

### **Option 1: GitHub Web Interface**
1. Open `chapter_11_notebook.ipynb` directly on GitHub
2. GitHub will render the notebook with all outputs
3. Scroll through to see all experiments and results

### **Option 2: Local Jupyter Notebook**
```bash
# Clone repository
git clone <repository-url>
cd chapter-11-training-deep-neural-networks

# Create virtual environment (optional but recommended)
conda create -n ml_env python=3.9
conda activate ml_env

# Install dependencies
pip install tensorflow numpy pandas matplotlib seaborn jupyter

# Launch Jupyter
jupyter notebook chapter_11_notebook.ipynb
```

### **Option 3: Google Colab**
1. Upload `chapter_11_notebook.ipynb` to Google Colab
2. Run all cells sequentially
3. All datasets (Fashion MNIST, CIFAR-10) download automatically

---

## 📚 Chapter Structure

The notebook is organized into **15 comprehensive parts**:

### **Part 1-3: Foundations**
- Introduction & Problem Statement
- Vanishing/Exploding Gradients Problem
- Weight Initialization Strategies

### **Part 4-5: Activation & Normalization**
- Activation Functions Comparison
- Batch Normalization Theory & Implementation

### **Part 6-7: Advanced Techniques**
- Gradient Clipping
- Transfer Learning

### **Part 8-9: Optimization**
- Advanced Optimizers
- Learning Rate Scheduling

### **Part 10: Regularization**
- L1/L2 Regularization
- Dropout
- Combined Techniques

### **Part 11-12: Practical Applications**
- Best Practices & Decision Trees
- Fashion MNIST Production Pipeline
- CIFAR-10 Deep Network

### **Part 13-15: Exercises & Summary**
- All 10 Chapter Exercises Solved
- Comprehensive Summary
- Key Takeaways & Best Practices

---

## 🎓 Key Takeaways

### **Critical Learnings:**

1. **Initialization is CRITICAL**
   - Poor init → 10% accuracy (total failure)
   - He init → 85% accuracy (success)
   - **Impact: 75 percentage points difference!**

2. **Dropout > L2 Regularization**
   - Dropout 10%: 88.80% accuracy
   - No regularization: 87.60% accuracy
   - L2(0.01): 80.35% (too strong!)
   - **Dropout is superior for neural networks**

3. **SGD + Momentum > Adam for Production**
   - SGD+Momentum: 87.85% (best generalization)
   - Adam: 87.65% (faster convergence but overfits)
   - **Use Adam for prototyping, SGD for final models**

4. **Combined Techniques = Best Results**
   - Simple model: 87.60%
   - Best practices combined: 90.05%
   - **Improvement: +2.45% absolute**

---

## 🛠️ Default Production Configuration
```python
# Recommended architecture for most cases
model = keras.Sequential([
    layers.Dense(units, 
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
])

# Optimizer
optimizer = keras.optimizers.SGD(learning_rate=0.01, 
                                  momentum=0.9, 
                                  nesterov=True)

# Callbacks
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.5, 
                                       patience=5),
    keras.callbacks.EarlyStopping(monitor='val_loss', 
                                   patience=15, 
                                   restore_best_weights=True)
]
```

---

## 📖 Exercise Solutions

All **10 exercises** from Chapter 11 are solved with detailed explanations:

1. ❌ **Don't initialize all weights to same value** (symmetry problem)
2. ✅ **Biases can be initialized to 0** (no symmetry issue)
3. **SELU advantages:** Self-normalization, no dying neurons, smoother gradients
4. **Activation selection:** ReLU (default), ELU (better), SELU (very deep)
5. **Momentum β=0.9:** Best balance (considers ~10 past gradients)
6. **Sparse models:** L1 regularization, high dropout, magnitude pruning
7. **Dropout effects:** Slows training convergence, NO impact on inference
8. **Deep network implementation:** ✅ Completed in Experiment 13
9. **Regularization application:** ✅ Completed in Experiment 12
10. **Optimizer comparison:** ✅ Completed in Experiment 8

---

## 🔗 References

- **Book:** *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd Edition)
- **Author:** Aurélien Géron
- **Publisher:** O'Reilly Media
- **Chapter:** 11 - Training Deep Neural Networks
- **Official Repository:** [github.com/ageron/handson-ml3](https://github.com/ageron/handson-ml3)

---

## 📧 Contact

**Hamdan Syaifuddin Zuhri**
- **NIM:** 1103220220
- **Class:** TK-46-06
- **Email:** [Your Email]
- **LinkedIn:** [Your LinkedIn]
- **GitHub:** [Your GitHub]

---

## 📝 License

This project is for educational purposes as part of the Machine Learning coursework at Telkom University.

---

## 🙏 Acknowledgments

- **Aurélien Géron** for the excellent textbook and examples
- **Telkom University** for providing the learning opportunity
- **TensorFlow/Keras team** for the amazing framework
- **Machine Learning community** for continuous knowledge sharing

---

**⭐ If you find this repository helpful, please give it a star!**

**📌 Last Updated:** January 2026

---

## 🎯 Future Improvements

- [ ] Add CNN architectures for better CIFAR-10 performance
- [ ] Implement ResNet and skip connections
- [ ] Explore advanced techniques (Label Smoothing, Mixup)
- [ ] Add model interpretation and visualization
- [ ] Deploy best model as REST API

---

**🚀 Happy Learning!**