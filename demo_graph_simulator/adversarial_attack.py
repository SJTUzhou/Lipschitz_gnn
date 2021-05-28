from numpy.core.fromnumeric import shape
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape
tf.compat.v1.disable_eager_execution()
import numpy as np
import pandas as pd
import ast
import re
from sklearn.preprocessing import OneHotEncoder
from art.attacks.evasion import DeepFool
from art.estimators.classification import KerasClassifier
import complex_generator



def load_raw_result_csv(result_csvfile):
    df_result = pd.read_csv(result_csvfile)
    array_std = df_result["std"].to_numpy()
    array_mean_values = np.array([ast.literal_eval(re.sub(r'[ary()]',"",element)) for element in df_result["means"].tolist()])
    array_overlap_ratio = df_result["overlap ratio"].to_numpy()
    return array_std, array_mean_values, array_overlap_ratio



def reconstruct_test_data(std, mean_values, Ad, num_graph):
    num_class = mean_values.shape[0]
    variance = std**2
    cov = np.diag([variance,variance,variance])
    Attributes, Labels = complex_generator.generate_dataset(A=Ad, num_class=num_class, size=num_graph, means=mean_values, cov=cov)
    Labels = np.expand_dims(Labels, axis=2)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded_labels = np.zeros(shape=(Labels.shape[0],Labels.shape[1],num_class))
    for i in range(num_graph):
        onehot_encoded_labels[i,:,:] = onehot_encoder.fit_transform(Labels[i,:,:])
    print("node attribute shape: ", Attributes.shape)
    print("node label shape: ", onehot_encoded_labels.shape)
    return Attributes, onehot_encoded_labels




def main():
    RAW_RESULT_FILE = "./result.csv"
    Ad = np.load("./data/Ad.npy")
    NUM_TEST = 1
    NUM_GRAPH = 100
    array_std, array_mean_values, array_overlap_ratio = load_raw_result_csv(RAW_RESULT_FILE)
    NUM_CLASS = array_mean_values.shape[1]
    print(array_mean_values.shape)
    for i in range(NUM_TEST):
        x_test, y_test = reconstruct_test_data(array_std[i], array_mean_values[i], Ad, NUM_GRAPH)
        model_with_Lip_constr = tf.keras.models.load_model("saved_model/fit{}_model_with_Lip_constr.h5".format(i))
        print(model_with_Lip_constr.summary())
        model_without_Lip_constr = tf.keras.models.load_model("saved_model/fit{}_model_without_Lip_constr.h5".format(i))
        
        # y_predict = model_with_Lip_constr.predict(x_test)
        # print(y_predict)
        print("Evaluation of model WITH Lipschitz constant constraint on TEST data")
        loss_test_L, acc_test_L = model_with_Lip_constr.evaluate(x_test, y_test, batch_size=x_test.shape[0], verbose=0)
        print("Loss: {:.4f}, accuracy: {:.4f}".format(loss_test_L, acc_test_L))

        print("Evaluation of model WITHOUT Lipschitz constant constraint on TEST data")
        loss_test_WL, acc_test_WL = model_without_Lip_constr.evaluate(x_test, y_test, batch_size=x_test.shape[0], verbose=0)
        print("Loss: {:.4f}, accuracy: {:.4f}".format(loss_test_WL, acc_test_WL))

        # Reshape model output
        reshape_with_Lip = Reshape((x_test.shape[1]*NUM_CLASS,),name="added_reshape_layer_L")(model_with_Lip_constr.output)
        new_model_with_Lip = Model(inputs=model_with_Lip_constr.input, outputs=reshape_with_Lip)
        reshape_without_Lip = Reshape((x_test.shape[1]*NUM_CLASS,),name="added_reshape_layer_WL")(model_without_Lip_constr.output)
        new_model_without_Lip = Model(inputs=model_without_Lip_constr.input, outputs=reshape_without_Lip)

        min_value = np.min(array_mean_values[i]) - 100*array_std[i]
        max_value = np.max(array_mean_values[i]) + 100*array_std[i]
        classifier_with_Lip = KerasClassifier(model=new_model_with_Lip, clip_values=(min_value, max_value), use_logits=False)
        classifier_without_Lip = KerasClassifier(model=new_model_without_Lip, clip_values=(min_value, max_value), use_logits=False)
        attack1 = DeepFool(classifier=classifier_with_Lip, epsilon=0.2, batch_size=10)
        attack2 = DeepFool(classifier=classifier_without_Lip, epsilon=0.2,  batch_size=10)
        x_test_adv1 = attack1.generate(x=x_test)
        x_test_adv2 = attack2.generate(x=x_test)
        y_predict_adv_with_Lip = classifier_with_Lip.predict(x_test_adv1)
        y_predict_adv_without_Lip = classifier_without_Lip.predict(x_test_adv2)

        y_predict_adv_with_Lip = y_predict_adv_with_Lip.reshape((y_test.shape))
        y_predict_adv_without_Lip = y_predict_adv_without_Lip.reshape((y_test.shape))
        acc_adv_with_Lip = np.sum(np.argmax(y_predict_adv_with_Lip,axis=2)==np.argmax(y_test,axis=2)) / (y_test.shape[0]*y_test.shape[1])
        print("Accuracy on adversarial test examples with Lipschitz constraint: {:.2f}%".format(acc_adv_with_Lip * 100))
        acc_adv_without_Lip = np.sum(np.argmax(y_predict_adv_without_Lip,axis=2)==np.argmax(y_test,axis=2)) / (y_test.shape[0]*y_test.shape[1])
        print("Accuracy on adversarial test examples without Lipschitz constraint: {:.2f}%".format(acc_adv_without_Lip * 100))
        

if __name__ == "__main__":
    main()