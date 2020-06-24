# Created by Xingyu Lin, 2019-07-25
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import axes3d
import copy
from scipy.stats import multivariate_normal
import os.path as osp
from matplotlib import animation

save_path = './results/'
title_size = 20
title_shift = 0.75
color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]


def l1(x, y):
    return x * x + y * y


def dl1(x, y):
    # Gradient of the main task loss
    return np.array([2 * x, 2 * y])


def l2(x, y):
    return (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)


def dl2(x, y):
    return np.array([(2 * x - 1), (2 * y - 1)])


def l3(x, y):
    return -l1(x, y)


def dl3(x, y):
    return -dl1(x, y)


def v(x, y):
    dx = -y / (x * x + y * y) - 2 * x
    dy = x / (x * x + y * y)
    return np.array([dx, dy])


def adaptive_training(theta0, w0, grad_aux_funcs, variants, beta_momentum=0.0):
    theta = copy.copy(theta0)
    w = copy.copy(w0)
    alpha = variants['alpha']
    beta = variants['beta']
    N = variants['N']
    dot_product_history = []
    w_history = [copy.copy(w)]
    theta_history = [copy.copy(theta0)]
    l_history = [copy.copy(l1(*theta0))]
    vel = np.zeros(w.shape)
    for i in range(variants['training_iter']):
        grad_main = dl1(*theta)

        grad_auxs = []
        for idx, grad_aux_func in enumerate(grad_aux_funcs):
            grad_auxs.append(w[idx] * grad_aux_func(*theta))
        grad = grad_main + np.sum(np.array(grad_auxs), axis=0)
        vel = beta_momentum * vel + (1 - beta_momentum) * grad
        theta -= alpha * vel
        dot_product = (np.array(grad_auxs) @ np.expand_dims(grad_main, 1)).reshape((-1,))
        dot_product_history.append(dot_product)
        if (i + 1) % N == 0 and not variants['fixed_weight']:
            grad_w = np.sum(np.array(dot_product_history[-N:]), 0)
            w += beta * grad_w
            w_history.append(w.copy())
        l_history.append(copy.copy(l1(*theta)))
        theta_history.append(theta.copy())
    return {
        'theta': np.array(theta_history),
        'w': np.array(w_history),
        'l': np.array(l_history),
    }


def visualize(info_history, save_suffix=''):
    l_history = info_history['l']
    plt.figure()
    plt.plot(list(range(len(l_history))), l_history)
    plt.savefig(osp.join(save_path, 'l1_loss_' + save_suffix + '.png'))

    w_history = info_history['w']
    plt.figure()
    for i in range(len(w_history[0])):
        plt.plot(list(range(len(w_history))), w_history[:, i])
    plt.savefig(osp.join(save_path, 'w_' + save_suffix + '.png'))

    theta_history = info_history['theta']
    x = [theta[0] for theta in theta_history]
    y = [theta[1] for theta in theta_history]
    plt.figure()
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.grid()
    plt.plot(x, y)
    plt.savefig(osp.join(save_path, 'theta_' + save_suffix + '.png'))


def plot_contour(info_historys):
    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.mgrid[-2:2:30j, -2:2:30j]
    Z1 = np.zeros(X.shape)
    Z2 = np.zeros(X.shape)
    Z3 = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z1[i, j] = l1(X[i, j], Y[i, j])
            Z2[i, j] = l2(X[i, j], Y[i, j])
            Z3[i, j] = l3(X[i, j], Y[i, j])
    ax.view_init(elev=25)
    ax.set_zlim(-1, 1)
    cmap = "viridis"
    ax.contour(X, Y, Z1, 10, lw=10, cmap=cmap, linestyles="solid", offset=0)
    ax.contour(X, Y, Z2, 10, lw=10, cmap=cmap, linestyles="solid", offset=1)
    ax.contour(X, Y, Z3, 10, lw=10, cmap=cmap, linestyles="solid", offset=-1)

    for idx, info_history in enumerate(info_historys):
        theta_history = info_history['theta']
        x = [theta[0] for theta in theta_history]
        y = [theta[1] for theta in theta_history]
        ax.plot(x, y, zs=0, color=color_defaults[idx], label='Seed ' + str(idx))
    plt.plot([0], [0], zs=[0], marker='*', markersize=10, color='b')
    ax.set_ylabel('y')
    ax.set_xlabel('x')

    # Turn off tick labels
    ax.set_zticklabels([])

    plt.legend(loc='upper right')
    plt.savefig(osp.join(save_path, 'surface.png'))

    plt.figure()
    for idx, info_history in enumerate(info_historys):
        w_history = info_history['w']
        plt.plot(list(range(len(w_history))), w_history[:, 0], color=color_defaults[idx], label='Seed ' + str(idx))
    plt.title('Weight of auxiliary loss L1')
    plt.legend()
    plt.savefig(osp.join(save_path, 'w1.png'))

    plt.figure()
    for idx, info_history in enumerate(info_historys):
        w_history = info_history['w']
        plt.plot(list(range(len(w_history))), w_history[:, 1], color=color_defaults[idx], label='Seed ' + str(idx))
    plt.title('Weight of auxiliary loss L2')
    plt.legend()
    plt.savefig(osp.join(save_path, 'w2.png'))

    plt.figure()
    for idx, info_history in enumerate(info_historys):
        l_history = info_history['l']
        plt.plot(list(range(len(l_history))), l_history, color=color_defaults[idx], label='Seed ' + str(idx))
        plt.legend()

    plt.title('Main task loss')
    plt.savefig(osp.join(save_path, 'loss.png'))


def plot_loss_funcs(info_historys, ax, title):
    X, Y = np.mgrid[-2:2:30j, -2:2:30j]
    Z1 = np.zeros(X.shape)
    Z2 = np.zeros(X.shape)
    Z3 = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z1[i, j] = l1(X[i, j], Y[i, j])
            Z2[i, j] = l2(X[i, j], Y[i, j])
            Z3[i, j] = l3(X[i, j], Y[i, j])
    ax.view_init(elev=25)
    ax.set_zlim(-1, 1)
    cmap = "viridis"
    ax.contour(X, Y, Z1, 10, lw=10, cmap=cmap, linestyles="solid", offset=0)
    ax.contour(X, Y, Z2, 10, lw=10, cmap=cmap, linestyles="solid", offset=1)
    ax.contour(X, Y, Z3, 10, lw=10, cmap=cmap, linestyles="solid", offset=-1)

    for idx, info_history in enumerate(info_historys):
        theta_history = info_history['theta']
        x = [theta[0] for theta in theta_history]
        y = [theta[1] for theta in theta_history]
        ax.plot(x, y, zs=0, color=color_defaults[idx], label='Start point ' + str(idx))
    plt.plot([0], [0], zs=[0], marker='*', markersize=10, color='b')
    plt.title(title, fontsize=title_size)
    ax.set_ylabel('y')
    ax.set_xlabel('x')

    # Turn off tick labels
    ax.set_zticklabels([])


def plot_loss_funcs_2d(info_historys, lines, points, time_step):
    for idx, info_history in enumerate(info_historys):
        theta_history = info_history['theta']
        x = [theta[0] for theta in theta_history]
        y = [theta[1] for theta in theta_history]
        lines[idx].set_data(x[:time_step + 1], y[:time_step + 1])
        points[idx].set_data(x[time_step], y[time_step])


def plot_weights(info_historys, lines_w1, lines_w2, time_step):
    for idx, info_history in enumerate(info_historys):
        w_history = info_history['w']
        lines_w1[idx].set_data(list(range(time_step + 1)), w_history[:time_step + 1, 0])
        lines_w2[idx].set_data(list(range(time_step + 1)), w_history[:time_step + 1, 1])


def plot_main_loss(info_historys, lines3, time_step):
    for idx, info_history in enumerate(info_historys):
        l_history = info_history['l']
        lines3[idx].set_data(list(range(time_step + 1)), l_history[:time_step + 1])


def init():
    ax1.plot([0], [0], marker='*', markersize=23, color='b')
    ax2.plot([0], [0], marker='*', markersize=23, color='b')
    ax1.plot([0.5], [0.5], marker='^', markersize=20, color='b')
    ax2.plot([0.5], [0.5], marker='^', markersize=20, color='b')
    X, Y = np.mgrid[-2:2:30j, -2:2:30j]
    Z1 = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z1[i, j] = l1(X[i, j], Y[i, j])
    cmap = "viridis"
    ax1.contour(X, Y, Z1, 7, lw=10, cmap=cmap, linestyles="solid", offset=0)
    ax2.contour(X, Y, Z1, 7, lw=10, cmap=cmap, linestyles="solid", offset=0)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend(loc='upper right', prop={'size': 16})
    ax1.grid()
    ax1.set_title('Fixed weight, w1=w2=1', fontsize=title_size)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    blue_star = mlines.Line2D([], [], color='blue', marker='*', linestyle='None',
                              markersize=10, label='Optimum of main task')
    purple_triangle = mlines.Line2D([], [], color='blue', marker='^', linestyle='None',
                                    markersize=10, label='Optimum of auxiliary task L1')

    ax2.legend(loc='upper left', prop={'size': 16}, handles=[blue_star, purple_triangle])
    ax2.grid()
    ax2.set_title('OL-AUX (Ours)', fontsize=title_size)

    # weight and loss plots
    ax3.set_xlabel('Training Iteration', position=(0.8, 1.))
    ax3.set_title('Main task loss', fontsize=title_size)
    ax3.grid()
    ax3.legend(loc='upper right')
    ax3.set_yscale('log')
    ax3.set_ylim(0.001, 10)
    ax4.set_ylim(0, 5)
    ax5.set_ylim(0, 5)

    ax4.set_title('Weight of auxiliary task L1', y=title_shift, fontsize=title_size)
    ax4.grid()

    ax5.set_title('Weight of auxiliary task L2', y=title_shift, fontsize=title_size)
    ax5.grid()

    for line1, line2, point1, point2, line3, line4, line5 in zip(lines1, lines2, points1, points2, lines3, lines4, lines5):
        line1.set_data([], [])
        line2.set_data([], [])
        point1.set_data([], [])
        point2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        line5.set_data([], [])

    return lines1, lines2, points1, points2, lines3, lines4, lines5


def animate(i, info_historys, info_historys_fixed_weight):
    if i > 150:
        i = (i - 150) * 10 + 150
    if i < 1000:
        plot_loss_funcs_2d(info_historys, lines2, points2, i)

        plot_loss_funcs_2d(info_historys_fixed_weight, lines1, points1, i)

        plot_main_loss(info_historys, lines3, i)

        plot_weights(info_historys, lines4, lines5, i)

        # ax3.autoscale(enable=True, axis='both', tight=None)
        x_max = 1000
        ax3.set_xlim(0, x_max)
        ax4.set_xlim(0, x_max)
        ax5.set_xlim(0, x_max)

    return lines1, lines2, points1, points2, lines3, lines4, lines5


def plot_all(info_historys, info_historys_fixed_weight, plot3d=False):
    global ax1, ax2, ax3, ax4, ax5, lines1, points1, lines2, points2, lines3, lines4, lines5
    lines1, lines2, points1, points2 = [], [], [], []
    lines3, lines4, lines5 = [], [], []
    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(333)
    ax4 = fig.add_subplot(336)
    ax5 = fig.add_subplot(339)

    for i in range(len(info_historys)):
        line1, = ax1.plot([], [], '-', linewidth=4, color=color_defaults[i], label='Start point ' + str(i))
        point1, = ax1.plot([], [], 'o', linewidth=4, color=color_defaults[i], markersize=10)
        line2, = ax2.plot([], [], '-', linewidth=4, color=color_defaults[i], label='Start point ' + str(i))
        point2, = ax2.plot([], [], 'o', linewidth=4, color=color_defaults[i], markersize=10)
        lines1.append(line1), lines2.append(line2)
        points1.append(point1), points2.append(point2)

        line3, = ax3.plot([], [], linewidth=4, color=color_defaults[i], label='Start point ' + str(i))
        line4, = ax4.plot([], [], linewidth=4, color=color_defaults[i], label='Start point ' + str(i))
        line5, = ax5.plot([], [], linewidth=4, color=color_defaults[i], label='Start point ' + str(i))
        lines3.append(line3), lines4.append(line4), lines5.append(line5)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=300, fargs=(info_historys, info_historys_fixed_weight), interval=20,
                                   blit=False)
    anim.save('results/quadratic.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    # plt.savefig(osp.join('surface_all.png'))


if __name__ == '__main__':
    variants = dict()
    variants['training_iter'] = 1000
    variants['N'] = 1
    variants['alpha'] = 1e-2
    variants['beta'] = 1e-2
    variants['fixed_weight'] = False
    grad_aux_funcs = [dl2, dl3]
    w0 = np.ones(len(grad_aux_funcs))
    theta0 = [[-0.8, -1.6], [1.8, 1.], [-1.8, 1], [1, -1]]
    info_historys = []
    for i in range(len(theta0)):
        info_history = adaptive_training(theta0[i], w0, grad_aux_funcs, variants)
        info_historys.append(copy.copy(info_history))

    variants['fixed_weight'] = True
    info_historys_fixed_weight = []
    for i in range(len(theta0)):
        info_history = adaptive_training(theta0[i], w0, grad_aux_funcs, variants)
        info_historys_fixed_weight.append(copy.copy(info_history))

    plot_all(info_historys, info_historys_fixed_weight)
