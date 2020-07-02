import numpy as np
import matplotlib.pyplot as plt

nx = 201
xmin = -5.0e-4
xmax = 5.0e-4
dx = (xmax - xmin) / (nx - 1)

ny = 201
ymin = -5.0e-4
ymax = 5.0e-4
dy = (ymax - ymin) / (ny - 1)

c = 29979245800.0  # speed of light in cm/s
e_0 = 8.85418782e-12  # permittivity of free space in m^-3 kg^-1 s^4 A^2
m_e = 9.10938356e-31  # electron mass in kg
e_c = 1.60217662e-19  # electron charge in C

lamb = 1.053e-4 / 3.0  # wavelength of light, in cm. This is frequency-tripled "3w" or "blue" (UV) light
freq = c / lamb  # frequency of light, in Hz
omega = 2 * np.pi * freq  # frequency of light, in rad/s
ncrit = 1e-6 * (omega ** 2.0 * m_e * e_0 / e_c ** 2.0)

xx = np.linspace(xmin, xmax, nx)
yy = np.linspace(ymin, ymax, ny)

x, y = np.meshgrid(xx, yy)
cmap = 'jet'


def import_beam(filename):
    output = []
    try:
        rays = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
    except:
        return []

    ray_num = rays[0][0]
    temp = []
    for point in rays:
        if point[0] == ray_num:
            temp.append((point[1], point[2]))
        else:
            ray_num = point[0]
            output.append(temp)
            temp = [(point[1], point[2])]

    output.append(temp)
    return output


def import_intersections(filename):
    x_ints = []
    y_ints = []
    try:
        inters = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
        # x_ints, y_ints = zip(*inters)
        x_ints, y_ints = inters
    except:
        return [], []

    return x_ints, y_ints


def plot_rays(b1, b2, x1_ints, y1_ints):
    plt.figure()
    plt.pcolormesh(x, y, eden / ncrit, cmap=cmap)
    plt.plot(x - (dx / 2), y - (dy / 2), 'k:')
    plt.plot(y - (dy / 2), x - (dx / 2), 'k:')

    plt.plot(x - (dx / 2), y + (dy / 2), 'k:')
    plt.plot(y + (dy / 2), x - (dx / 2), 'k:')

    plt.plot(x + (dx / 2), y - (dy / 2), 'k:')
    plt.plot(y - (dy / 2), x + (dx / 2), 'k:')

    plt.plot(x + (dx / 2), y + (dy / 2), 'k:')
    plt.plot(y + (dy / 2), x + (dx / 2), 'k:')

    plt.plot(x, y, 'k--')
    plt.plot(y, x, 'k--')

    plt.colorbar()

    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title('n_e_/n_crit_')

    if b1:
        for r1 in b1:
            x1, y1 = zip(*r1)
            plt.plot(x1, y1, 'm.')

    if b2:
        for r2 in b2:
            x2, y2 = zip(*r2)
            plt.plot(x2, y2, 'c.')

    plt.plot(x1_ints, y1_ints, 'bo')

    plt.show(block=True)


def plot_intensity(e_b1, e_b2):
    if e_b1.size != 0 and e_b2.size != 0:
        combined_edep = np.add(e_b1, e_b2)
    elif e_b1.size != 0:
        combined_edep = e_b1
    elif e_b2.size != 0:
        combined_edep = e_b2
    else:
        return

    edep_min = np.min(combined_edep)
    edep_max = np.max(combined_edep)
    norm_edep = (combined_edep - edep_min) / (edep_max - edep_min)

    plt.figure()
    # clo = 0.0
    # chi = np.max(combined_edep)
    # plt.pcolormesh(x, y, combined_edep, cmap=cmap, vmin=clo, vmax=chi)
    plt.pcolormesh(x, y, norm_edep, cmap=cmap)
    plt.colorbar()
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title('Overlapped intensity')
    plt.show(block=False)


def plot_var1(e_b1, e_b2):
    if e_b1.size != 0 and e_b2.size != 0:
        var1 = 8.053e-10 * np.sqrt(e_b1 + e_b2 + 1.0e-10) * (1.053 / 3.0)
    else:
        return

    vmin = np.min(var1)
    vmax = np.max(var1)
    norm_var1 = (var1 - vmin) / (vmax - vmin)

    plt.figure()
    # plt.pcolormesh(x, y, var1, cmap=cmap, vmin=np.min(var1), vmax=np.max(var1))
    plt.pcolormesh(x, y, norm_var1, cmap=cmap)

    plt.colorbar()
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title('Total original field amplitude (a0)')
    plt.show(block=False)


def plot_a0(e_b1_new, e_b2_new):
    if e_b1_new.size != 0 and e_b2_new.size != 0:
        e_b1_new[e_b1_new < 1.0e-10] = 1.0e-10
        e_b2_new[e_b2_new < 1.0e-10] = 1.0e-10
        a0 = 8.053e-10 * np.sqrt(e_b1_new + e_b2_new + 1.0e-10) * (1.053 / 3.0)
    else:
        return

    amin = np.min(a0)
    amax = np.max(a0)
    norm_a0 = (a0 - amin) / (amax - amin)

    plt.figure()
    # plt.pcolormesh(x, y, a0, cmap=cmap, vmin=np.min(a0), vmax=np.max(a0))
    plt.pcolormesh(x, y, norm_a0, cmap=cmap)

    plt.colorbar()
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title('Total CBET new field amplitude (a0)')
    plt.show(block=False)

    plt.figure()
    plt.plot(xx, a0[1, :], ',-b')
    plt.plot(xx, a0[ny - 2, :], ',-r')
    plt.plot(xx, a0[ny // 2, :], ',-g')
    plt.xlabel('X (cm)')
    plt.ylabel('a0')
    plt.title('a0(x) at z_min, z_0, z_max')
    plt.grid(linestyle='--')
    plt.show(block=False)

    plt.figure()
    plt.plot(yy, a0[:, 1], ',-b')
    plt.plot(yy, a0[:, nx - 2], ',-r')
    plt.plot(yy, a0[:, nx // 2], ',-g')
    plt.xlabel('Y (cm)')
    plt.ylabel('a0')
    plt.title('a0(y) at x_min, x_0, x_max')
    plt.grid(linestyle='--')
    plt.show(block=False)


if __name__ == '__main__':
    output_path = './Outputs/'
    beam1 = import_beam(output_path + 'beam1.csv')
    beam2 = import_beam(output_path + 'beam2.csv')
    eden = np.genfromtxt(output_path + 'eden.csv', delimiter=',', dtype=np.float32)
    ix1, iy1 = import_intersections(output_path + "beam1_intersections.csv")

    try:
        i_b1 = np.genfromtxt(output_path + 'beam1_edep.csv', delimiter=',', dtype=np.float32)
    except:
        i_b1 = np.zeros((nx, ny))
    try:
        i_b2 = np.genfromtxt(output_path + 'beam2_edep.csv', delimiter=',', dtype=np.float32)
    except:
        i_b2 = np.zeros((nx, ny))
    try:
        i_b1_new = np.genfromtxt(output_path + 'beam1_edep_new.csv', delimiter=',', dtype=np.float32)
    except:
        i_b1_new = np.zeros((nx, ny))
    try:
        i_b2_new = np.genfromtxt(output_path + 'beam2_edep_new.csv', delimiter=',', dtype=np.float32)
    except:
        i_b2_new = np.zeros((nx, ny))

    plot_rays(beam1, beam2, ix1, iy1)
    # plot_intensity(i_b1, i_b2)
    # plot_var1(i_b1, i_b2)
    # plot_a0(i_b1_new, i_b2_new)
