import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_result_of_one_model():
    # Directory to the results
    RESULT_FILE = "./result.csv"
    df = pd.read_csv(RESULT_FILE, delimiter=",")

    array_ratio = 1./df["overlap ratio"].to_numpy()

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
    ax1.set_xlabel("||Mu_0-Mu_1|| / std_dev")
    ax1.set_ylabel("loss")
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

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
    ax2.set_xlabel("||Mu_0-Mu_1|| / std_dev")
    ax2.set_ylabel("accuracy")
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    plt.show()



def plot_result_comparison():
    # Directory to the results
    RESULT_FILE2 = "./results/result_2dense_diff-cut.csv"
    RESULT_FILE3 = "./results/result_3dense_diff-cut.csv"
    df2 = pd.read_csv(RESULT_FILE2, delimiter=",")
    df3 = pd.read_csv(RESULT_FILE3, delimiter=",")

    array_ratio2 = 1./df2["overlap ratio"].to_numpy()
    array_ratio3 = 1./df3["overlap ratio"].to_numpy()

    array_train_loss_with_Lip2 = df2["train loss with L"].to_numpy()
    array_test_loss_with_Lip2 = df2["test loss with L"].to_numpy()
    array_train_loss_with_Lip3 = df3["train loss with L"].to_numpy()
    array_test_loss_with_Lip3 = df3["test loss with L"].to_numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(array_ratio2, array_train_loss_with_Lip2, label="training_loss_with_Lipschitz_constraint  (2-dense model)")
    ax1.scatter(array_ratio2, array_test_loss_with_Lip2, label="test_loss_with_Lipschitz_constraint  (2-dense model)")
    ax1.scatter(array_ratio3, array_train_loss_with_Lip3, label="training_loss_with_Lipschitz_constraint  (3-dense model)")
    ax1.scatter(array_ratio3, array_test_loss_with_Lip3, label="test_loss_with_Lipschitz_constraint  (3-dense model)")

    ax1.legend()
    ax1.grid(True)
    ax1.set_xlabel("||Mu_0-Mu_1|| / std_dev")
    ax1.set_ylabel("loss")
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

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
    ax2.set_xlabel("||Mu_0-Mu_1|| / std_dev")
    ax2.set_ylabel("accuracy")
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    plt.show()


if __name__ == "__main__":
    plot_result_of_one_model()
    # plot_result_comparison()