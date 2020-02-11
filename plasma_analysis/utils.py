import numpy as np
import math


def get_peak_to_valley(file):
    temp_data = np.genfromtxt(file, skip_header=3)
    peak_to_valley = temp_data[0]
    return peak_to_valley


def image_to_double(im):
    info = np.iinfo(im.dtype)
    return im.astype(np.float)/info.max


def apply_threshold(image, t):
    """
    image: image represented as a 2D array
    T: number between 0 and 1
    """
    image_max = np.amax(image)
    image_min = np.amin(image)
    diff = image_max-image_min
    threshold = image_min+diff*t
    thresholded_image = np.copy(image)
    thresholded_image[thresholded_image < threshold] = 0
    return thresholded_image


def symmetrize_image(image, i):
    """
    image: image represented as a 2D array
    i: index of center of mass
    returns one half of the symmetrized image
    """
    n_pixels, _ = np.shape(image)
    symmetrized_half = image[i, :]
    j = 1
    while j <= i and i+j < n_pixels:
        temp = np.stack((image[i-j, :], image[i+j, :]))
        average = np.mean(temp, axis=0)
        symmetrized_half = np.vstack((symmetrized_half, average))
        j += 1
    return symmetrized_half


def get_index_of_center_of_mass_by_line(image_column):
    """
    image_column: a column of an image represented as a 1D array
    returns the index of the center of mass
    """
    f = image_column
    x = np.arange(1, len(f) + 1)
    first_moment = calculate_moment(f, x, 1, 0)
    total_mass = calculate_moment(f, x, 0, 0)
    # The first moment divided by the total mass is the center of mass, i.e. a point in the interval x.
    # Since x = [1, len(f)], the index of the center of mass should be calculated as below.
    center_of_mass = first_moment/total_mass
    index_of_center_of_mass = np.rint(center_of_mass-1)
    return int(index_of_center_of_mass)


def get_index_of_center_of_mass(thresholded_image):
    """
    thresholded_image: an image where a threshold has been applied
    returns the row index that corresponds to the center of mass of the thresholded image
    """
    return get_index_of_center_of_mass_by_line(np.sum(thresholded_image, axis=1))


def calculate_moment(f, x, n, c):
    """
    f: function f(x) represented as an 1D array
    n: order of moment
    c: center value (0)
    returns the moment calculated with the formula on https://en.wikipedia.org/wiki/Moment_(mathematics)
    """
    integrand = np.multiply(np.power(x-c, n), f)
    return np.trapz(integrand)


def rebuild_image(symmetrized_half):
    n_rows, n_columns = np.shape(symmetrized_half)
    full_image = np.zeros((2*n_rows-1, n_columns))
    full_image[0:n_rows, :] = symmetrized_half
    full_image[n_rows:, :] = np.flipud(symmetrized_half[0:-1, :])
    return full_image


def abel_inversion(phase_map, thresholded_phase_map, y_calibration, wavelength):
    index_of_center_of_mass = get_index_of_center_of_mass(thresholded_phase_map)
    symmetrized_half_phase_map = symmetrize_image(phase_map, index_of_center_of_mass)
    n_rows, n_columns = np.shape(symmetrized_half_phase_map)
    abel_inverted_half_map = np.zeros((n_rows, n_columns))
    i = 0
    for column in symmetrized_half_phase_map.T:
        abel_inverted_half_map[:, i] = abel_inversion_by_line(column, y_calibration)
        i += 1
    half_density_map = get_density_map(np.flipud(abel_inverted_half_map), wavelength)
    return half_density_map, rebuild_image(half_density_map)


def get_density_map(m, wavelength):
    c = 0.299792458  # [µm/fs]
    pulsation = wavelength_to_pulsation(wavelength)
    m = m*(c/pulsation)
    m = 1+m
    m = np.power(m, 2)
    m = 1-m
    density_map = m*pulsation_to_critical_density(pulsation)
    return density_map


def wavelength_to_pulsation(wavelength):
    """
    wavelength: wavelength in nanometer
    """
    c = 299.792458  # [nm/s]
    pulsation = 2*math.pi*c/wavelength  # [rad/fs]
    return pulsation


def pulsation_to_critical_density(pulsation):
    c = 299792458  # [m/s]
    epsilon_0 = 1/(math.pow(c, 2)*4*math.pi*1e-7)  # [C^2N^-1m^-2]
    m_e = 9.10938215e-31  # [kg]
    e = 1.602176487e-19  # [C]
    temp = (epsilon_0*m_e)/math.pow(e, 2)*1e24*1e-18
    return math.pow(pulsation, 2)*temp


def abel_inversion_by_line(phase_half_map_column, dy):
    phase_array = np.asarray(phase_half_map_column)
    n_pixels = phase_array.size
    y = np.linspace(0, (n_pixels-1)*dy, n_pixels)
    s = derivative(phase_array, 0, 0, dy)
    inverted_column = []
    i = 0
    for r in y:
        sub_y = y[i:]
        sub_s = s[i:]
        temp = np.power(sub_y, 2)-math.pow(r, 2)
        temp = np.sqrt(temp)
        temp = np.reciprocal(temp, out=np.zeros_like(temp), where=temp != 0)
        integrand = np.multiply(sub_s, temp)
        integral = np.trapz(integrand, dx=dy)
        inverted_column.append(-integral/math.pi)
        i += 1
    return inverted_column


def integrate(f, dt):
    n = f.size
    integral = 0
    for i in range(0, n):
        if i == 0 or i == n-1:
            integral += f[i]
        else:
            integral += 2*f[i]
    return integral*dt/2


def derivative(x, ic, fc, dt):
    n = x.size
    y = []
    for i in range(n):
        if i == 0:
            d = (x[i+1]-ic)/(2*dt)
        elif i == n-1:
            d = (fc-x[i-1])/(2*dt)
        else:
            d = (x[i+1]-x[i-1])/(2*dt)
        y.append(d)
    return y
