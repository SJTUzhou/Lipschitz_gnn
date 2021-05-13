import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    RESULT_FILE = "./result.csv"
    RESULT_FILE2 = "./.old/result-measurement-2dense.csv"
    RESULT_FILE3 = "./.old/result-measurement-3dense.csv"
    df2 = pd.read_csv(RESULT_FILE2, delimiter=",")
    df3 = pd.read_csv(RESULT_FILE3, delimiter=",")

    array_ratio2 = df2["overlap ratio"].to_numpy()
    array_ratio3 = df3["overlap ratio"].to_numpy()

    array_train_loss_with_Lip2 = df2["train loss with L"].to_numpy()
    array_test_loss_with_Lip2 = df2["test loss with L"].to_numpy()
    array_train_loss_with_Lip3 = df3["train loss with L"].to_numpy()
    array_test_loss_with_Lip3 = df3["test loss with L"].to_numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(array_ratio2**2, array_train_loss_with_Lip2, label="training_loss_with_Lipschitz_constraint  (2-dense model)")
    ax1.scatter(array_ratio2**2, array_test_loss_with_Lip2, label="test_loss_with_Lipschitz_constraint  (2-dense model)")
    ax1.scatter(array_ratio3**2, array_train_loss_with_Lip3, label="training_loss_with_Lipschitz_constraint  (3-dense model)")
    ax1.scatter(array_ratio3**2, array_test_loss_with_Lip3, label="test_loss_with_Lipschitz_constraint  (3-dense model)")

    ax1.legend()
    ax1.grid(True)
    ax1.set_xlabel("std_dev / ||Mu_0-Mu_1||")
    ax1.set_ylabel("loss")
    ax1.set_xlim([0,3.6])
    ax1.set_ylim([0.2,0.7])

    array_train_acc_with_Lip2 = df2["train acc with L"].to_numpy()
    array_test_acc_with_Lip2 = df2["test acc with L"].to_numpy()
    array_train_acc_with_Lip3 = df3["train acc with L"].to_numpy()
    array_test_acc_with_Lip3 = df3["test acc with L"].to_numpy()

    ax2.scatter(array_ratio2, array_train_acc_with_Lip2, label="training_accuracy_with_Lipschitz_constraint (2-dense model)")
    ax2.scatter(array_ratio2, array_test_acc_with_Lip2, label="test_accuracy_with_Lipschitz_constraint (2-dense model)")
    ax2.scatter(array_ratio3, array_train_acc_with_Lip3, label="training_accuracy_with_Lipschitz_constraint (3-dense model)")
    ax2.scatter(array_ratio3, array_test_acc_with_Lip3, label="test_accuracy_with_Lipschitz_constraint (3-dense model)")

    ax2.legend()
    ax2.grid(True)
    ax2.set_xlabel("std_dev / ||Mu_0-Mu_1||")
    ax2.set_ylabel("accuracy")
    ax2.set_xlim([0,3.6])
    ax2.set_ylim([0.5,1.0])





    plt.show()