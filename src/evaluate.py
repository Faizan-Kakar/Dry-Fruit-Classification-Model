# Selecting a batch of test images
imgs , test_labels = next(test_batches)

# Converting the data from one-hot-encoded to argmax
test_labels = np.argmax(test_labels , axis=1)

predicted_labels = model.predict(imgs)

# Converting probabilities into argmax
predicted_labels = np.argmax(predicted_labels, axis=1)

# Checking how many images classified as true
# Valus on diagnol are classified true other than diagonal is false
cm = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(7), yticklabels=range(7))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()