# prediction outputs

def rf_prediction(X, best_rf_classifier, class_descriptions):
    pred_class = best_rf_classifier.predict(X)[0]
    pred_probs = best_rf_classifier.predict_proba(X)[0]
    
    predicted_class_probability = pred_probs[best_rf_classifier.classes_.tolist().index(pred_class)]
    predicted_class_description = class_descriptions.get(pred_class, 'Unknown')

    class_probabilities = [
        {'class': int(cls), 'description': class_descriptions.get(cls, 'Unknown'), 'probability': f"{prob * 100:.2f}%"}
        for cls, prob in zip(best_rf_classifier.classes_, pred_probs)
    ]

    return int(pred_class), predicted_class_probability, predicted_class_description, class_probabilities


