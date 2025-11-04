# AI Machine Learning Project Portfolio

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-yellow.svg)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit-Learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-green.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Welcome to my **AI Machine Learning Project Portfolio**! This repository showcases hands-on labs from the [AI for Beginners Curriculum](https://github.com/microsoft/ai-for-beginners), building from basic perceptrons to custom multi-layer neural networks. Each project demonstrates core ML/DL concepts: data handling, model implementation, training, evaluation, and optimization. 

**Ongoing Projects**: Currently features 2 completed labs (MNIST perceptron & custom NN). More coming: Linear regression, CNNs, RL agents—stay tuned!

## 🚀 Why This Portfolio?
- **From Scratch Focus**: Implement algorithms without high-level libraries (e.g., custom backprop)—proves deep understanding.
- **Progressive Complexity**: Starts simple (linear classifiers) → advanced (multi-layer with activations, overfitting analysis).
- **Skills Showcased**:
  - **Core ML**: Perceptron rule, SGD, loss functions, gradients.
  - **DL Fundamentals**: Forward/backward prop, modular nets, activations (Tanh/ReLU), weight tracking.
  - **Evaluation**: Accuracy, confusion matrices, overfitting detection (train-valid gaps), hyperparameter tuning (LR, hidden units).
  - **Tools**: NumPy for math, Scikit-Learn for data, Matplotlib for viz (boundaries, curves).
- **Results**: Achieved competitive MNIST accuracies (85% perceptron → 97% custom NN)—with analysis of weights/gradients.

## 📁 Project Overview
| # | Project | Description | Key Metrics | Skills Highlighted |
|---|---------|-------------|-------------|--------------------|
| **01** | [Perceptron MNIST](01-perceptron-mnist/) | One-vs-All perceptrons for digit classification. Implements classic perceptron learning rule. | **Test Acc: 85.2%**<br>Best Digit: 1 (95.3%) | Binary classifiers, OVO/OVA strategy, confusion matrices, per-class metrics. |
| **02** | [Custom NN MNIST](02-custom-nn-mnist/) | Multi-layer perceptron (1-3 layers) with custom framework. Experiments with activations & depth. | **Valid Acc: 97.27%** (3-layer ReLU)<br>Train Acc: 99.36% | Backprop, modular nets, activations (Tanh/ReLU), overfitting tuning, weight norm plots. |
| **03+** | **Ongoing** | Linear regression, CNNs, RL—coming soon! | N/A | Advanced: Conv layers, policy gradients, etc. |

- **Dataset**: MNIST (70k 28x28 grayscale digits 0-9; splits: 50k train, 10k val, 10k test).
- **Common Tools**: NumPy (core math), Scikit-Learn (data split), Matplotlib (viz).

## 🛠️ Installation & Setup (Global)
1. **Clone Repo**:
   ```
   git clone https://github.com/XEnvisage/AI-Machine-Learning-Project-Portfolio.git
   cd AI-Machine-Learning-Project-Portfolio
   ```

2. **Environment**:
   - Python 3.7+.
   - Install deps:
     ```
     pip install -r requirements.txt  # numpy, scikit-learn, matplotlib
     ```

3. **Dataset**:
   - Download `mnist.pkl` from [AI for Beginners](https://github.com/microsoft/ai-for-beginners/tree/main/data).
   - Place in root or let notebooks auto-fetch.

4. **Run a Project**:
   - Jupyter: `jupyter notebook 01-perceptron-mnist/perceptron_mnist.ipynb`
   - Or: `python run_experiment.py --project 02` (add scripts for automation).

## 📖 Quick Start
### Project 01: Perceptron
```python
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```

### Project 02: Custom NN
```python
# MyFW_MNIST.ipynb
net = Net()
net.add(Linear(784, 128))
net.add(ReLU())
net.add(Linear(128, 10))
net.add(Softmax())
loss = CrossEntropyLoss()
res = train_and_plot(20, net, loss)  # Plots progress, ~97% valid acc
```

## 📊 Key Results Across Projects
| Project | Model Depth | Activation | Train Acc | Valid/Test Acc | Overfit Gap | Epochs to Converge |
|---------|-------------|------------|-----------|----------------|-------------|--------------------|
| **01** | 1-Layer (Perceptron) | - | 88.5% | 85.2% | 3.3% | 50 (converges) |
| **02** | 1-Layer (NN) | - | 92.5% | 91.2% | 1.3% | 5 |
| **02** | 2-Layer | ReLU | 97.2% | 96.0% | 1.2% | 8 |
| **02** | 3-Layer | ReLU | 99.36% | 97.27% | 2.09% | 10 |

- **Trends**: Depth + ReLU boosts acc (+6% vs. baseline); 3-layer overfits slightly (gap=2%)—tuned with lower LR/hidden units.
- **Weights**: Max abs(W) stabilizes (1.2→1.1 early, plateau)—no vanishing/explosion.

## 💡 Skills & Insights Showcased
This portfolio builds **progressive ML/DL proficiency**:
- **Fundamentals**: Perceptron rule → full backprop (chain rule, gradients).
- **Design**: Modular classes (Net, Linear)—scalable to deeper nets.
- **Analysis**: Confusion matrices, per-class acc, train-valid gaps (overfitting detection); weight norm plots (stability).
- **Tuning**: LR (0.01-0.1), hidden (64-256), activations (ReLU > Tanh for speed/acc).
- **Challenges Overcome**: Overfitting in deeper models (smaller LR reduces gap); vanishing grads in Tanh (ReLU fixes).

**Portfolio Highlights**:
- **Accuracy Gains**: 85% (simple perceptron) → 97% (custom 3-layer)—shows depth/non-linearity value.
- **Activations Impact**: ReLU +2% valid acc, faster (no saturation).
- **Depth Needs**: 2-layer optimal for MNIST (3-layer overfits on this data).
- **Training Issues**: Deeper = slower convergence (lr=0.05 helps); small batches stable.

## 📝 License
MIT License—free to use, modify, distribute.

## 🤝 Contribute & Contact
- **Issues/PRs**: Welcome! Add conv layers or RL labs.
- **Fork & Star**: Experiment—try CIFAR-10!
- **About Me**: [Your GitHub/LinkedIn] | AI Enthusiast | Ongoing: 10+ labs from Microsoft AI Curriculum.

Built with ❤️ for hands-on AI. Questions? Open an issue. ⭐ Thanks!
