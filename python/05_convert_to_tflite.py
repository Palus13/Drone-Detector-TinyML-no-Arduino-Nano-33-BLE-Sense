import tensorflow as tf

# Carrega o modelo fine-tunado
model = tf.keras.models.load_model("models/finetuned_lightweight_drone_arduino_mic.keras")

# Converte para TFLite (sem otimizações, puro float32)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Salva
with open("models/finetuned_lightweight_drone_arduino_mic.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo TFLite float32 salvo em: models/finetuned_lightweight_drone_arduino_mic.tflite")
print(f"Tamanho do arquivo: {len(tflite_model) / 1024:.2f} KB")