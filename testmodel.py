from tensorflow.keras.models import load_model
model = load_model('models\savedmodel.h5')
print(model.summary())