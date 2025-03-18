import sys
import types

# Si tensorflow_decision_forests no existe, creamos un módulo dummy
if 'tensorflow_decision_forests' not in sys.modules:
    sys.modules['tensorflow_decision_forests'] = types.ModuleType('tensorflow_decision_forests')

import tensorflow as tf
import tensorflowjs as tfjs

# Cargar el modelo entrenado
model = tf.keras.models.load_model("chatbot_model.h5")

# Convertir y guardar el modelo en el formato de TensorFlow.js
tfjs.converters.save_keras_model(model, 'model')
