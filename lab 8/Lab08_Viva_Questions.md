# Lab 08 Viva Questions — Neural Networks
**CS201L: Artificial Intelligence Laboratory | IIT Dharwad**

---

## 1. Fundamentals of Neural Networks

**Q1. What is a neural network and how is it inspired by the human brain?**
A neural network is a computational model made up of layers of interconnected nodes (neurons). Each connection has a weight, and each neuron applies an activation function to its input. It is inspired by biological neurons that fire electrical signals when stimulated beyond a threshold.

**Q2. What is the role of the input layer, hidden layer, and output layer?**
- **Input layer:** Receives raw feature values and passes them forward without transformation.
- **Hidden layer:** Extracts intermediate representations by applying weights and activation functions.
- **Output layer:** Produces the final prediction; in classification, each neuron corresponds to a class.

**Q3. Why do we need activation functions? What happens if we don't use them?**
Activation functions introduce non-linearity into the network. Without them, stacking multiple layers is equivalent to a single linear transformation, making deeper architectures pointless — the network cannot learn complex, non-linear decision boundaries.

**Q4. What is the tanh activation function? What are its properties?**
`tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
- Output range: **(-1, 1)**
- Zero-centred (unlike Sigmoid), which helps gradient flow.
- Saturates at extreme values, which can cause vanishing gradients in deep networks.

**Q5. Why is Softmax used in the output layer for multi-class classification?**
Softmax converts raw logits into a probability distribution over all classes, ensuring all outputs sum to 1. This makes the output interpretable as class probabilities and aligns with Cross-Entropy loss.

---

## 2. Training Process

**Q6. What is forward propagation?**
Forward propagation is the process of passing the input through the network layer by layer — applying weights, biases, and activation functions — to produce a prediction (output).

**Q7. What is back propagation? Why is it important?**
Backpropagation computes the gradient of the loss with respect to each weight using the chain rule of calculus. These gradients indicate how much each weight contributed to the error, allowing the optimizer to update weights and reduce the loss.

**Q8. What is the Cross-Entropy loss function? Why is it preferred for classification?**
Cross-Entropy measures the difference between the predicted probability distribution and the true label distribution:
`L = -Σ y_true * log(y_pred)`
It penalises confident wrong predictions heavily and is well-suited for probabilistic outputs like Softmax.

**Q9. What is Stochastic Gradient Descent (SGD)?**
SGD updates model weights using the gradient computed on the entire training batch (in batch-SGD) or individual samples. It iteratively moves weights in the direction that reduces the loss, guided by the learning rate.

**Q10. What is the learning rate and how does it affect training?**
The learning rate controls the step size of weight updates.
- Too **high** → overshoots minima, training diverges.
- Too **low** → training is very slow, may get stuck in local minima.
- A good range in this lab is `0.001 ≤ lr ≤ 0.01`.

**Q11. What is meant by convergence in training? How is it detected in this lab?**
Convergence is when the training loss stops decreasing meaningfully. In this lab it is detected by tracking how much the loss changes between epochs — if the change is below a `threshold` (1e-3) for `patience` consecutive epochs, training stops early.

**Q12. What is the purpose of `optimizer.zero_grad()` in PyTorch?**
PyTorch accumulates gradients by default. Calling `zero_grad()` before each training step clears old gradients so they don't add up across iterations, which would corrupt weight updates.

---

## 3. Architecture Choices

**Q13. How does increasing the number of neurons in a hidden layer affect the model?**
More neurons increase the model's capacity to learn complex patterns. However, too many neurons can lead to overfitting, where the model memorises the training data but generalises poorly.

**Q14. What is the difference between a single hidden layer and a two hidden layer network?**
A two hidden layer network can learn more hierarchical and abstract features. The first hidden layer may learn low-level patterns while the second learns higher-level combinations. However, it also has more parameters and is harder to train.

**Q15. Why might Architecture (64, 16) perform differently from (34, 34)?**
(64, 16) has more neurons in the first layer to capture broad feature combinations, then compresses to 16 for refinement — a funnel structure. (34, 34) maintains a uniform width. Depending on the data, one may generalise better than the other.

**Q16. In PyTorch, why does `nn.CrossEntropyLoss` not require an explicit Softmax in the output layer?**
`nn.CrossEntropyLoss` internally combines `LogSoftmax` and `NLLLoss`. Adding a Softmax before it would result in applying Softmax twice, which distorts the probabilities and degrades training.

---

## 4. Data Normalisation

**Q17. What is Z-score normalisation and why is it applied?**
Z-score normalisation rescales each feature to have zero mean and unit standard deviation:
`z = (x - μ) / σ`
It ensures all features are on the same scale, preventing features with large magnitudes from dominating learning and generally leading to faster, more stable convergence.

**Q18. Why must the mean and standard deviation be computed only from the training data?**
Using validation or test statistics would constitute **data leakage** — the model would have indirect knowledge of unseen data during training. The same training statistics are then applied to validation and test sets to simulate real-world conditions.

**Q19. In PyTorch, how do you efficiently apply Z-score normalisation to an entire matrix?**
Using broadcasting:
```python
mean = X_train.mean(dim=0)   # shape: (34,)
std  = X_train.std(dim=0)    # shape: (34,)
X_normalised = (X - mean) / std
```
PyTorch broadcasts the mean/std vectors across all rows automatically.

**Q20. What could go wrong if a feature has zero standard deviation during normalisation?**
Division by zero would occur, producing `NaN` values. This is handled by clamping `std` to a small minimum value (e.g., `std.clamp(min=1e-8)`).

---

## 5. Evaluation Metrics

**Q21. What is a confusion matrix? What information does it provide?**
A confusion matrix is a table where rows represent actual classes and columns represent predicted classes. The diagonal shows correct predictions; off-diagonal entries show misclassification patterns between specific class pairs.

**Q22. What is the difference between micro and macro averages?**
- **Macro average:** Computes the metric independently per class, then averages equally across classes. Treats all classes equally regardless of size.
- **Micro average:** Aggregates all true positives, false positives, etc. across classes before computing the metric. Favours the performance on larger classes.

**Q23. When would macro average be more informative than micro average?**
When classes are imbalanced. Macro average highlights poor performance on minority classes that micro average might mask due to the dominance of larger classes.

**Q24. What is the F1-score and when is it preferred over accuracy?**
F1-score is the harmonic mean of precision and recall:
`F1 = 2 * (Precision * Recall) / (Precision + Recall)`
It is preferred when classes are imbalanced, since accuracy can be misleadingly high if the model simply predicts the majority class.

**Q25. Can accuracy alone be a sufficient metric for this dataset? Why or why not?**
Not necessarily. If the 7 fruit classes are unevenly distributed, a model predicting the dominant class always could achieve high accuracy while being useless. Precision, recall, and F1 per class provide a more complete picture.

---

## 6. PyTorch Specifics

**Q26. What does `model.train()` vs `model.eval()` do?**
- `model.train()` enables layers like Dropout and BatchNorm to behave in training mode (weights can be updated).
- `model.eval()` switches these layers to inference mode (Dropout is disabled, BatchNorm uses running statistics). This ensures reproducible predictions on validation/test data.

**Q27. Why is `torch.no_grad()` used during evaluation?**
It disables gradient computation, saving memory and computation time. During inference, gradients are not needed since weights are not being updated.

**Q28. What does `torch.argmax(outputs, dim=1)` return?**
It returns the index of the maximum value across the class dimension (dim=1) for each sample — i.e., the predicted class label for each input.

**Q29. How are models saved and loaded in PyTorch?**
- Save: `torch.save(model.state_dict(), 'model.pth')`
- Load: `model.load_state_dict(torch.load('model.pth'))`
`state_dict()` stores only the learnable parameters (weights and biases), not the architecture itself.

**Q30. What is `nn.Sequential` in PyTorch?**
`nn.Sequential` is a container that chains layers in order. When you call `forward(x)`, the input passes through each layer sequentially. It simplifies model definition when there are no branching or skip connections.

---

## 7. Conceptual / Analytical Questions

**Q31. Why does normalised data generally produce better results with this dataset?**
The Date Fruit dataset has features with very different scales (large integers, large floats, small floats). Without normalisation, features with large values dominate gradient updates, causing slow or unstable learning. Z-score puts all features on equal footing.

**Q32. What is overfitting and how can you detect it using validation data?**
Overfitting is when the model performs well on training data but poorly on unseen data. It can be detected when training loss keeps decreasing but validation accuracy stops improving or starts dropping.

**Q33. What is the role of the validation set in this lab?**
The validation set is used to compare architectures and select the best one — without ever touching the test set. This avoids selection bias, ensuring the test set gives an unbiased final performance estimate.

**Q34. Why should the test set only be evaluated once, at the very end?**
Repeated evaluation on the test set and using its results to make decisions (like tuning hyperparameters) causes the model to indirectly overfit to the test set, making it an unreliable estimate of real-world performance.

**Q35. If both architectures give similar validation accuracy, what other factors might guide your choice?**
- Training time (fewer parameters = faster)
- Convergence speed (fewer epochs needed)
- Stability (consistent results across multiple runs)
- Generalisation gap (difference between training and validation loss)

---

*Good luck with your viva!*
