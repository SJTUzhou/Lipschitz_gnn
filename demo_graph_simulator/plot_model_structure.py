import tensorflow as tf

if __name__ == "__main__":
    i = 0
    model = tf.keras.models.load_model("saved_model/fit{}_model_with_Lip_constr.h5".format(i))
    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True,
        show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96
    )