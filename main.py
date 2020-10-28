import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from scipy.stats import norm


def gen_signal(Es):
    # Generate signal image
    k = sqrt(Es / 1024)
    diag = k * np.ones(1024)
    signal = np.diag(diag)
    return signal


def gen_noise(variance):
    # Generate Gaussian Noise
    return np.random.normal(0, variance, (1024, 1024))


def ln_likelihood_ratio(X, variance, Es):
    k = sqrt(Es / 1024)
    x = np.diag(X)
    S_diag = x - k
    ln_lambda = sum(np.multiply(x, x) - np.multiply(S_diag, S_diag)) / (2 * variance)
    return ln_lambda


def Pr(pdf, beta):
    N = 500
    c = sum(pdf > beta)
    return c / N


if __name__ == '__main__':
    # un-comment the following code to generate the example image
    # Es = 1024
    # S = gen_signal(Es)
    # plt.imshow(S, cmap='gray')
    # plt.show()
    # N = gen_noise(0.1)
    # plt.imshow(N, cmap='gray')
    # plt.show()
    # SN = S + N
    # plt.imshow(SN, cmap='gray')
    # plt.show()

    # let variance=1, Es = 1, 2, 4, 16
    # calculate the ln likelihood ratio
    Es_list = np.array([1, 2, 4, 16])
    v = 1
    ln_lambda_H0_list = np.zeros((4, 500))
    ln_lambda_H1_list = np.zeros((4, 500))
    for i, Es in zip(range(4), Es_list):
        lambda_H0 = np.zeros(500)
        lambda_H1 = np.zeros(500)
        S = gen_signal(Es)
        for j in range(500):
            N = gen_noise(1)
            SN = S + N
            lambda_H0[j] = ln_likelihood_ratio(N, v, Es)
            lambda_H1[j] = ln_likelihood_ratio(SN, v, Es)
        ln_lambda_H0_list[i] = lambda_H0
        ln_lambda_H1_list[i] = lambda_H1

    # plot histgram of ln(lambda)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(ln_lambda_H0_list[0], alpha=0.5, label='H0')
    axs[0, 0].hist(ln_lambda_H1_list[0], alpha=0.5, label='H1')
    axs[0, 0].set_title('Es = 1')
    leg = axs[0, 0].legend(loc='upper right', shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    axs[0, 1].hist(ln_lambda_H0_list[1], alpha=0.5, label='H0')
    axs[0, 1].hist(ln_lambda_H1_list[1], alpha=0.5, label='H1')
    axs[0, 1].set_title('Es = 2')
    leg = axs[0, 1].legend(loc='upper right', shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    axs[1, 0].hist(ln_lambda_H0_list[2], alpha=0.5, label='H0')
    axs[1, 0].hist(ln_lambda_H1_list[2], alpha=0.5, label='H1')
    axs[1, 0].set_title('Es = 4')
    leg = axs[1, 0].legend(loc='upper right', shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    axs[1, 1].hist(ln_lambda_H0_list[3], alpha=0.5, label='H0')
    axs[1, 1].hist(ln_lambda_H1_list[3], alpha=0.5, label='H1')
    axs[1, 1].set_title('Es = 16')
    leg = axs[1, 1].legend(loc='upper right', shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.show()

    num_points = 100  # set number of points in AOC curve as 100
    PF_list = np.zeros((4, num_points))
    PD_list = np.zeros((4, num_points))
    real_PF_list = np.zeros((4, num_points))
    real_PD_list = np.zeros((4, num_points))
    for i in range(4):
        k = sqrt(Es_list[i] / 1024)
        muPF = -1024 * k * k / (2 * v)
        muPD = 1024 * k * k / (2 * v)
        vPF = 1024 * k * k / v
        vPD = 1024 * k * k / v
        # generate threshold list
        min_b = min(ln_lambda_H0_list[i].min(), ln_lambda_H1_list[i].min())
        max_b = max(ln_lambda_H0_list[i].max(), ln_lambda_H1_list[i].max())
        betas = np.linspace(min_b, max_b, num_points)
        # for each threshold simulate ROC and calculate the theoretical results
        for j, beta in zip(range(num_points), betas):
            PF_list[i, j] = Pr(ln_lambda_H0_list[i], beta)
            PD_list[i, j] = Pr(ln_lambda_H1_list[i], beta)
            # calculate the theoretical ROC
            real_PF_list[i, j] = norm.sf(beta, muPF, sqrt(vPF))
            real_PD_list[i, j] = norm.sf(beta, muPD, sqrt(vPD))

    plt.plot(PF_list[0], PD_list[0], 'b', label='Es = 1, simulation')
    plt.plot(real_PF_list[0], real_PD_list[0], 'b--', label='Es = 1, theoretical')
    plt.plot(PF_list[1], PD_list[1], 'y', label='Es = 2, simulation')
    plt.plot(real_PF_list[1], real_PD_list[1], 'y--', label='Es = 2, theoretical')
    plt.plot(PF_list[2], PD_list[2], 'g', label='Es = 4, simulation')
    plt.plot(real_PF_list[2], real_PD_list[2], 'g--', label='Es = 4, theoretical')
    plt.plot(PF_list[3], PD_list[3], 'r', label='Es = 16, simulation')
    plt.plot(real_PF_list[3], real_PD_list[3], 'r--', label='Es = 16, theoretical')
    plt.xlabel('PF')
    plt.ylabel('PD')
    leg = plt.legend(loc='lower right', shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.show()


