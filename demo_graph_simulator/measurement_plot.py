import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    RESULT_FILE = "./result.csv"
    df = pd.read_csv(RESULT_FILE, delimiter=",")
    array_ratio = df["overlap ratio"].to_numpy()
    array_train_loss_with_Lip = df["train loss with L"].to_numpy()
    array_test_loss_with_Lip = df["test loss with L"].to_numpy()
    array_train_loss_without_Lip = df["train loss without L"].to_numpy()
    array_test_loss_without_Lip = df["test loss without L"].to_numpy()
    
    fig, ax = plt.subplots()
    ax.scatter(array_ratio, array_train_loss_with_Lip, label="train_loss_with_Lipschitz_constr")
    ax.scatter(array_ratio, array_test_loss_with_Lip, label="test_loss_with_Lipschitz_constr")
    ax.scatter(array_ratio, array_train_loss_without_Lip, label="train_loss_without_Lipschitz_constr")
    ax.scatter(array_ratio, array_test_loss_without_Lip, label="test_loss_without_Lipschitz_constr")

    ax.legend()
    ax.grid(True)
    ax.set_xlabel("std_dev / ||Mu_0-Mu_1||")
    ax.set_ylabel("loss")
    ax.set_xlim([0.2,1])
    ax.set_ylim([0,0.8])
    plt.show()