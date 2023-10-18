from sklearn.metrics import classification_report


predictions = model.predict(labeled_images_val)
predicted_labels = (predictions ).astype(int)  


report = classification_report(labels_val, predicted_labels)


print(report)