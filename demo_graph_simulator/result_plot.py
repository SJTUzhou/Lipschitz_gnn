import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

def plot_result_of_one_model(result_file):
    # Directory to the results
    df = pd.read_csv(result_file, delimiter=",")

    array_ratio = 1./df["overlap ratio"].to_numpy()

    array_train_loss_with_Lip = df["train loss with L"].to_numpy()
    array_test_loss_with_Lip = df["test loss with L"].to_numpy()
    array_train_loss_without_Lip = df["train loss without L"].to_numpy()
    array_test_loss_without_Lip = df["test loss without L"].to_numpy()
    
    fig, ax1 = plt.subplots()
    ax1.scatter(array_ratio, array_train_loss_with_Lip, label="Train loss of model with constraint")
    ax1.scatter(array_ratio, array_test_loss_with_Lip, label="Test loss of model with constraint")
    ax1.scatter(array_ratio, array_train_loss_without_Lip, label="Train loss of model without constraint")
    ax1.scatter(array_ratio, array_test_loss_without_Lip, label="Test loss of model without constraint")

    ax1.legend()
    ax1.grid(True)
    ax1.set_xlabel(r"$\eta = (\sum ||\mu_i-\mu_j||)/\sigma $")
    ax1.set_ylabel("Loss")
    ax1.set_xlim()
    ax1.set_ylim()
    seaborn.despine(ax=ax1, offset=0)
    plt.show()

    fig, ax2 = plt.subplots()
    array_train_acc_with_Lip = df["train acc with L"].to_numpy()
    array_test_acc_with_Lip = df["test acc with L"].to_numpy()
    array_train_acc_without_Lip = df["train acc without L"].to_numpy()
    array_test_acc_without_Lip = df["test acc without L"].to_numpy()

    ax2.scatter(array_ratio, array_train_acc_with_Lip, label="Train accuracy of model with constraint")
    ax2.scatter(array_ratio, array_test_acc_with_Lip, label="Test accuracy of model with constraint")
    ax2.scatter(array_ratio, array_train_acc_without_Lip, label="Train accuracy of model without constraint")
    ax2.scatter(array_ratio, array_test_acc_without_Lip, label="Test accuracy of model without constraint")

    ax2.legend()
    ax2.grid(True)
    ax2.set_xlabel(r"$\eta = (\sum ||\mu_i-\mu_j||)/\sigma $")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlim()
    ax2.set_ylim()
    seaborn.despine(ax=ax2, offset=0)
    plt.show()



def plot_complex_models_comparison(result_file_1, result_file_2):
    # Directory to the results
    df2 = pd.read_csv(result_file_1, delimiter=",")
    df3 = pd.read_csv(result_file_2, delimiter=",")

    array_ratio2 = 1./df2["overlap ratio"].to_numpy()
    array_ratio3 = 1./df3["overlap ratio"].to_numpy()

    array_train_loss_with_Lip2 = df2["train loss with L"].to_numpy()
    array_test_loss_with_Lip2 = df2["test loss with L"].to_numpy()
    array_train_loss_with_Lip3 = df3["train loss with L"].to_numpy()
    array_test_loss_with_Lip3 = df3["test loss with L"].to_numpy()
    
    fig, ax1 = plt.subplots()
    ax1.scatter(array_ratio2, array_train_loss_with_Lip2, label="Train loss of model with constraint (2-dense-layer)")
    ax1.scatter(array_ratio2, array_test_loss_with_Lip2, label="Test loss of model with constraint (2-dense-layer)")
    ax1.scatter(array_ratio3, array_train_loss_with_Lip3, label="Train loss of model with constraint (3-dense-layer)")
    ax1.scatter(array_ratio3, array_test_loss_with_Lip3, label="Test loss of model with constraint (3-dense-layer)")

    ax1.legend()
    ax1.grid(True)
    ax1.set_xlabel(r"$\eta = ||\mu_0-\mu_1||/\sigma $")
    ax1.set_ylabel("Loss")
    ax1.set_xlim()
    ax1.set_ylim()
    seaborn.despine(ax=ax1, offset=0)
    plt.show()



    array_train_acc_with_Lip2 = df2["train acc with L"].to_numpy()
    array_test_acc_with_Lip2 = df2["test acc with L"].to_numpy()
    array_train_acc_with_Lip3 = df3["train acc with L"].to_numpy()
    array_test_acc_with_Lip3 = df3["test acc with L"].to_numpy()

    fig, ax2 = plt.subplots()

    ax2.scatter(array_ratio2, array_train_acc_with_Lip2, label="Train accuracy of model with constraint (2-dense-layer)")
    ax2.scatter(array_ratio2, array_test_acc_with_Lip2, label="Test accuracy of model with constraint (2-dense-layer)")
    ax2.scatter(array_ratio3, array_train_acc_with_Lip3, label="Train accuracy of model with constraint (3-dense-layer)")
    ax2.scatter(array_ratio3, array_test_acc_with_Lip3, label="Test accuracy of model with constraint (3-dense-layer)")

    ax2.legend()
    ax2.grid(True)
    ax2.set_xlabel(r"$\eta = ||\mu_0-\mu_1||/\sigma $")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlim()
    ax2.set_ylim()
    seaborn.despine(ax=ax2, offset=0)
    plt.show()


def plot_robust_test_results(result_file, process_type="train", result_type="loss", test_type="noised"):
    """
    process_type: string, "train" or "test"
    result_type: string, "loss" or "acc"
    test_type: string, "noised" or "disturbed"
    """
    df = pd.read_csv(result_file, delimiter=",")

    array_ratio = 1./df["overlap ratio"].to_numpy()

    array_original_with_Lip = df["{} {} with L".format(process_type,result_type)].to_numpy()
    array_original_without_Lip = df["{} {} without L".format(process_type,result_type)].to_numpy()
    array_robust_test_with_Lip = df["{} {} {} with L".format(test_type,process_type,result_type)].to_numpy()
    array_robust_test_without_Lip = df["{} {} {} without L".format(test_type,process_type,result_type)].to_numpy()
    
    ax = plt.subplot()
    ax.scatter(array_ratio, array_original_with_Lip, label="Original {} {} of model with constraint".format(process_type,result_type))
    ax.scatter(array_ratio, array_robust_test_with_Lip, label="{} {} {} of model with constraint".format(test_type.capitalize(),process_type,result_type))
    ax.scatter(array_ratio, array_original_without_Lip, label="Original {} {} of model without constraint".format(process_type,result_type))
    ax.scatter(array_ratio, array_robust_test_without_Lip, label="{} {} {} of model without constraint".format(test_type.capitalize(),process_type,result_type))

    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r"$\eta = (\sum ||\mu_i-\mu_j||)/\sigma $")
    ax.set_ylabel(result_type.capitalize())
    ax.set_xlim()
    ax.set_ylim()
    seaborn.despine(ax=ax, offset=0)
    plt.show()


def plot_adversarial_attack_result(result_file):
    df = pd.read_csv(result_file, delimiter=",")
    array_ratio = 1./df["overlap ratio"].to_numpy()
    acc_with_Lip = df["acc_test_L"].to_numpy()
    acc_without_Lip = df["acc_test_WL"].to_numpy()
    acc_adv_with_Lip = df["acc_adv_with_Lip"].to_numpy()
    acc_adv_without_Lip = df["acc_adv_without_Lip"].to_numpy()
    ax = plt.subplot()
    # ax.scatter(array_ratio, acc_with_Lip, label="Test accuracy of model with constraint")
    # ax.scatter(array_ratio, acc_without_Lip, label="Test accuracy of model without constraint")
    # ax.scatter(array_ratio, acc_adv_with_Lip, label="Test accuracy of model with constraint on adversarial samples")
    # ax.scatter(array_ratio, acc_adv_without_Lip, label="Test accuracy of model without constraint on adversarial samples")

    DELTA = 0.005
    array_ratio = np.log(array_ratio)
    for i in range(array_ratio.shape[0]):
        ax.broken_barh([(array_ratio[i],DELTA)], (acc_adv_with_Lip[i],acc_with_Lip[i]-acc_adv_with_Lip[i]), facecolors="tab:blue")
        ax.broken_barh([(array_ratio[i]-DELTA,DELTA)], (acc_adv_without_Lip[i],acc_without_Lip[i]-acc_adv_without_Lip[i]), facecolors="tab:orange")

    # ax.legend()
    ax.grid(True)
    ax.set_xlabel(r"$ln \eta = ln( (\sum ||\mu_i-\mu_j||)/\sigma )$")
    ax.set_ylabel("Accuracy decrease")
    ax.set_xlim()
    ax.set_ylim()
    seaborn.despine(ax=ax, offset=0)
    plt.show()



if __name__ == "__main__":
    # RESULT_FILE = "./results/result_dense=3_diff-cut_class=2_lip=1.csv"
    # plot_result_of_one_model(RESULT_FILE)


    # result_file_1 = "./results/result_dense=2_diff-cut_class=2_lip=1.csv"
    # result_file_2 = "./results/result_dense=3_diff-cut_class=2_lip=1.csv"
    # plot_complex_models_comparison(result_file_1, result_file_2)



    # RESULT_FILE = "./results/result_dense=2_diff-cut_class=5_lip=5_more-info_try=2.csv"
    # plot_robust_test_results(RESULT_FILE, process_type="train", result_type="loss", test_type="noised")
    # plot_robust_test_results(RESULT_FILE, process_type="test", result_type="loss", test_type="noised")
    # plot_robust_test_results(RESULT_FILE, process_type="train", result_type="acc", test_type="noised")
    # plot_robust_test_results(RESULT_FILE, process_type="test", result_type="acc", test_type="noised")
    # plot_robust_test_results(RESULT_FILE, process_type="train", result_type="loss", test_type="disturbed")
    # plot_robust_test_results(RESULT_FILE, process_type="test", result_type="loss", test_type="disturbed")
    # plot_robust_test_results(RESULT_FILE, process_type="train", result_type="acc", test_type="disturbed")
    # plot_robust_test_results(RESULT_FILE, process_type="test", result_type="acc", test_type="disturbed")


    RESULT_FILE =  "./result_PGD.csv" # "./result_DeepFool.csv" # 
    plot_adversarial_attack_result(RESULT_FILE)
