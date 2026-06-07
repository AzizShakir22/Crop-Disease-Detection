import tensorflow as tf

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("✅ GPU detected:", gpus)
    print("🚀 RTX 5080 is ready!")
else:
    print("❌ GPU not found - will use CPU")