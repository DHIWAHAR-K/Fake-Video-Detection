from config import IMG_SIZE
from data_loader import load_dataset
from model_builder import build_model
from sklearn.model_selection import train_test_split


print("Loading dataset...")
x, y = load_dataset()
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

print("Building model...")
model = build_model()

print("Training model...")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val), batch_size=16)

print("Evaluating model...")
val_loss, val_accuracy = model.evaluate(x_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

print("Saving model...")
model.save('saved_model/deepfake_detection_model.h5')