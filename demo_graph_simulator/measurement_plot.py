import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    RESULT_FILE = "./result.csv"
    RESULT_FILE = "./.old/result-measurement-2dense.csv"
    df = pd.read_csv(RESULT_FILE, delimiter=",")
    array_ratio = df["overlap ratio"].to_numpy()
    array_train_loss_with_Lip = df["train loss with L"].to_numpy()
    array_test_loss_with_Lip = df["test loss with L"].to_numpy()
    array_train_loss_without_Lip = df["train loss without L"].to_numpy()
    array_test_loss_without_Lip = df["test loss without L"].to_numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(array_ratio, array_train_loss_with_Lip, label="training_loss_with_Lipschitz_constraint")
    ax1.scatter(array_ratio, array_test_loss_with_Lip, label="test_loss_with_Lipschitz_constraint")
    ax1.scatter(array_ratio, array_train_loss_without_Lip, label="training_loss_without_Lipschitz_constraint")
    ax1.scatter(array_ratio, array_test_loss_without_Lip, label="test_loss_without_Lipschitz_constraint")

    ax1.legend()
    ax1.grid(True)
    ax1.set_xlabel("std_dev / ||Mu_0-Mu_1||")
    ax1.set_ylabel("loss")
    ax1.set_xlim([0,3.6])
    ax1.set_ylim([0,1.5])

    array_train_acc_with_Lip = df["train acc with L"].to_numpy()
    array_test_acc_with_Lip = df["test acc with L"].to_numpy()
    array_train_acc_without_Lip = df["train acc without L"].to_numpy()
    array_test_acc_without_Lip = df["test acc without L"].to_numpy()

    ax2.scatter(array_ratio, array_train_acc_with_Lip, label="training_accuracy_with_Lipschitz_constraint")
    ax2.scatter(array_ratio, array_test_acc_with_Lip, label="test_accuracy_with_Lipschitz_constraint")
    ax2.scatter(array_ratio, array_train_acc_without_Lip, label="training_accuracy_without_Lipschitz_constraint")
    ax2.scatter(array_ratio, array_test_acc_without_Lip, label="test_accuracy_without_Lipschitz_constraint")

    ax2.legend()
    ax2.grid(True)
    ax2.set_xlabel("std_dev / ||Mu_0-Mu_1||")
    ax2.set_ylabel("accuracy")
    ax2.set_xlim([0,3.6])
    ax2.set_ylim([0.5,1.1])





    plt.show()