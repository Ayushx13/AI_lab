# Lab 08 - Neural Networks: Potential Viva Questions
### CS201L: Artificial Intelligence Laboratory | IIT Dharwad

---

## 1. Conceptual / Theory

1. What is the role of an activation function in a neural network? Why can't we just use linear neurons throughout?
2. Why is `tanh` used as the hidden layer activation function in this assignment? What are its properties?
3. What is the vanishing gradient problem? How does `tanh` relate to it, and why is ReLU often preferred in deeper networks?
4. Why do we use Cross Entropy loss for multiclass classification instead of Mean Squared Error (MSE)?
5. What is Softmax, and why is it suitable for the output layer in a multiclass classification problem?
6. Why does PyTorch's `CrossEntropyLoss` not require you to explicitly apply Softmax in the output layer?
7. What is the difference between a single hidden layer and a two hidden layer neural network in terms of representational capacity?
8. What does it mean for a neural network to "converge"? How is convergence detected in this assignment?
9. What is the role of the learning rate in SGD? What happens if it is too high or too low?
10. What is the difference between Stochastic Gradient Descent (SGD) and full-batch Gradient Descent? Which one is used here, and why?

---

## 2. Dataset & Preprocessing

11. What is the Human Activity Recognition (HAR) dataset? What are the six activity classes?
12. Why was the `subject` column removed from the dataset before training?
13. What is the purpose of standardizing (scaling) the dataset before feeding it to a neural network?
14. What is PCA (Principal Component Analysis)? Why is it used here?
15. What is the difference between the PCA-All dataset and the PCA-99% dataset? How many features does each have?
16. Why does retaining 99% variance in PCA reduce the input from 561 to 156 features? What do the remaining components represent?
17. How were the training, validation, and test sets split in this assignment (ratio)?
18. Why do we use a validation set separate from the test set during training?

---

## 3. Architecture & Implementation

19. How do you define a neural network in PyTorch? What is the role of `nn.Module` and the `forward()` method?
20. What does `nn.Sequential` do? How is it used in this assignment to stack layers?
21. What is `LabelEncoder` used for? Why do we need to convert string class labels to integers?
22. Why do we call `model.train()` before training and `model.eval()` before evaluation?
23. What does `torch.no_grad()` do, and why is it used during evaluation?
24. What does `optimizer.zero_grad()` do? What would happen if we skipped this step?
25. What does `loss.backward()` compute? What is the mathematical operation happening under the hood?
26. What does `optimizer.step()` do after `loss.backward()`?
27. How does `torch.argmax(outputs, dim=1)` give us the predicted class label?
28. How are the trained models saved and loaded in PyTorch? What does `torch.save(model.state_dict(), path)` store?

---

## 4. Training & Hyperparameters

29. What is the `patience` parameter in early stopping? How was it set in this assignment?
30. What is the `threshold` parameter used for in the convergence check? What value was used here?
31. Why was a learning rate of `0.01` chosen? What range of values is generally considered good for SGD?
32. What would happen if `max_epochs` is set too low? And too high?
33. How does the training loss curve (epoch vs loss plot) help you understand the training process?
34. What does it mean if the training loss curve is oscillating and not decreasing smoothly?
35. If two architectures have similar training loss but very different validation accuracy, what might be happening?

---

## 5. Evaluation Metrics

36. What is a confusion matrix? How do you read it for a multiclass problem?
37. What is accuracy, and when can it be a misleading metric?
38. What is the difference between Precision and Recall? Give an intuitive explanation.
39. What is the F1-score, and when is it more useful than accuracy alone?
40. What is the difference between micro-average and macro-average for precision, recall, and F1-score?
41. When would you prefer macro-average over micro-average for an imbalanced dataset?
42. Looking at the class distribution table in the assignment (LAYING: 18.88%, WALKING DOWNSTAIRS: 13.65%), is this dataset balanced? How might imbalance affect evaluation?

---

## 6. Comparison Across Datasets & Architectures

43. Which dataset (Original / Scaled / PCA-All / PCA-99) gave the best test accuracy and why do you think that is?
44. Did adding a second hidden layer consistently improve performance? Why or why not?
45. Did increasing the number of neurons in the hidden layer always lead to better accuracy? What tradeoff does this introduce?
46. Why might the PCA-99 dataset (156 features) perform comparably or better than the full 561-feature dataset?
47. What are the advantages of using PCA before training a neural network?
48. Compare the convergence speed across different datasets. Which converged fastest and why?

---

## 7. Practical / Code-Based

49. If you got an error like `RuntimeError: mat1 and mat2 shapes cannot be multiplied`, what does it likely mean in the context of your neural network?
50. What change would you make to the code if you wanted to use ReLU instead of tanh as the hidden layer activation?
51. How would you modify the `SingleHiddenLayerNN` class to add dropout regularization?
52. How would you add a learning rate scheduler to reduce the learning rate over time?
53. The assignment uses full-batch training (all samples at once). How would you modify the code to use mini-batches with `DataLoader`?
54. What would you change in the code to use the Adam optimizer instead of SGD?
55. If you wanted to load a saved model and run predictions on new data, what steps would you follow?

---

## 8. Tricky / Insight Questions

56. The output layer has no explicit Softmax in the model definition. Does this mean the model outputs probabilities? Explain.
57. If training accuracy is very high but test accuracy is low, what is the likely problem? How would you fix it?
58. Why might a very deep or very wide network perform worse than a simpler one on this dataset?
59. Is it possible for a model with lower training loss to have lower test accuracy than a model with higher training loss? Explain.
60. If you were to improve this assignment further, what would you try — more layers, different activation functions, batch normalization, dropout, or a different optimizer? Justify your choice.

---

*Good luck with your viva!*
