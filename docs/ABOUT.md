# Confusion Matrix Explorer

**Confusion Matrix Explorer** is an interactive learning app that helps users *see* how changing a decision threshold affects model performance.
It demonstrates, in real time, how **raising** or **lowering the bar** shifts the balance between sensitivity, specificity, and other diagnostic metrics.

Built with [PyShiny](https://shiny.posit.co/py/), [Plotly](https://plotly.com/python/), and [NumPy](https://numpy.org/), the app provides a visual way to understand one of the most important tradeoffs in classification modeling.

---

## Concept Overview

In any binary classification task - whether detecting diseases, fraud, or spam - the model assigns each case a **score** (often a probability).  
To make a final decision, we choose a **threshold**.

- If the score ≥ threshold > we predict **positive**
- If the score < threshold > we predict **negative**

Changing that threshold is what we call **raising or lowering the bar**.

---

### Raising the Bar (More Specific)

**Raising the bar** means making it harder to call something positive. 
The threshold line moves to the **right**.

- **Fewer positives overall**
- **More False Negatives (missed detections)**
- **Higher Specificity (fewer false alarms)**
- **Lower Sensitivity (catch fewer real cases)**  
- Precision (PPV) often increases - you're more confident in what you call positive.

This is like a very strict test: you'll rarely call someone "positive," but when you do, you're pretty sure.

---

### Lowering the Bar (More Sensitive)

**Lowering the bar** means making it easier to call something "positive."  
The threshold line moves to the **left**.

- **More positives overall**
- **More False Positives (false alarms)**
- **Lower Specificity**
- **Higher Sensitivity (catch nearly everyone with the condition)**  
- Precision (PPV) often decreases - more of your "positives" turn out to be wrong.

This is like a very sensitive test: you'll catch almost everyone who's sick, but you'll also flag some healthy people.

---

## The Confusion Matrix

Every threshold creates a **confusion matrix**, a 2×2 table that summarizes prediction outcomes:

|                | **Disease Present** | **Disease Absent** |
|----------------|---------------------|--------------------|
| **Test Positive** | True Positive (TP) | False Positive (FP) |
| **Test Negative** | False Negative (FN) | True Negative (TN) |

From these four counts, we derive all the key diagnostic metrics.

---

## Calculations

Let **a = TP**, **b = FN**, **c = FP**, **d = TN**, and **N = a + b + c + d**.

| Metric | Formula | Meaning |
|--------|----------|----------|
| **Sensitivity (TPR)** | a / (a + b) | Probability the test is positive given disease is present |
| **Specificity (TNR)** | d / (c + d) | Probability the test is negative given disease is absent |
| **Precision (PPV)** | a / (a + c) | Probability the disease is present given a positive test |
| **Negative Predictive Value (NPV)** | d / (b + d) | Probability the disease is absent given a negative test |
| **Accuracy** | (a + d) / N | Overall proportion correctly classified |
| **Prevalence** | (a + b) / N | Fraction of total population that truly has the disease |

> Note: These values depend on both model behavior and population prevalence.  
> A highly accurate test in one population may perform differently in another.

---

## Visualization

The app displays two overlapping **score distributions**:

- One showing counts where the disease is **present**  
- One showing counts where the disease is **absent**

The vertical dashed line ("T line) marks the **decision threshold**.

- Moving the line **right** > raises the bar > increases specificity, decreases sensitivity.  
- Moving it **left** > lowers the bar > increases sensitivity, decreases specificity.

Each update recomputes the confusion matrix and metrics to illustrate tradeoffs.

---

## See Also

1. [PyCM (Python Confusion Matrix)](https://github.com/sepandhaghighi/pycm)
2. [Scikit-learn confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
3. [Evidently: Confusion Matrix Dashboard](https://www.evidentlyai.com/classification-metrics/confusion-matrix)

---

## Author and License

Developed by **Denise Case**  
© 2025 MIT License
