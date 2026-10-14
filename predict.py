import io
import joblib
import requests

# Your Raw GitHub model URL
MODEL_URL = (
    "https://raw.githubusercontent.com/"
    "Kalaiyarasan1810/ML-Text-Classification/"
    "main/20newsgroups_model.joblib"
)

print("Fetching the trained model from GitHub...")

try:
    # Download the model
    response = requests.get(MODEL_URL)
    response.raise_for_status()

    # Load model directly from memory
    model = joblib.load(io.BytesIO(response.content))

    print("Model loaded successfully!\n")

    # New text samples
    new_samples = [
        "The local team won the championship game in overtime last night.",
        "Scientists have discovered a new planet orbiting a distant star.",
        "My computer graphics card is not working properly.",
    ]

    # Make predictions
    predictions = model.predict(new_samples)

    print("----- Prediction Results -----")

    for text, prediction in zip(new_samples, predictions):
        print(f"\nText: {text}")
        print(f"Predicted Category ID: {prediction}")

except requests.exceptions.RequestException as e:
    print(f"Error downloading model from GitHub: {e}")

except Exception as e:
    print(f"Error loading or using the model: {e}")