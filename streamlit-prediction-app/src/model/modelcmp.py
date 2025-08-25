accuracies = {
    "Catboost": 81.30,
    "Random Forest": 79.47,
    "LightGBM": 80.65,
    "SVM": 79.72,
    "XGBoost": 81.91
}
import matplotlib.pyplot as plt

models = list(accuracies.keys())
scores = list(accuracies.values())

plt.figure(figsize=(8, 6))
plt.bar(models, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
for i, score in enumerate(scores):
    plt.text(i, score + 0.1, f'{score}%', ha='center', va='bottom')
plt.ylim(78, 84)
plt.yticks(range(78, 85, 2))
plt.ylabel("Accuracy")
plt.title("Comparison of Model Accuracies")
plt.show()
