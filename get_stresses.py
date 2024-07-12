import numpy as np


def calculate_radial_stress_softening(
        r_i, p_0, UCS, GSI, D, m_i, E_rm, nu, epsilon_1e, alpha, h, eta, CI, CD, sigma_T, m_b, s_b, m_r, s_r, G_rm, M_e, sigma_re, f, plastic_radius_final, k_range):
    delta = 0.001  # Increment step for strains
    n_iterations = 1000  # Number of iteration steps

    epsilon_theta = np.zeros(k_range)
    epsilon_r = np.zeros(k_range)
    sigma_r = np.zeros(k_range)
    sigma_theta = np.zeros(k_range)
    mp = np.zeros(k_range)
    sp = np.zeros(k_range)
    lamda = np.ones(k_range)
    u_p = np.zeros(k_range)

    epsilon_theta[0] = M_e * UCS / (2 * G_rm)
    epsilon_r[0] = -epsilon_theta[0]
    sigma_r[0] = sigma_re
    sigma_theta[0] = 2 * p_0 - sigma_r[0]
    mp[0] = m_b
    sp[0] = s_b
    r_j = np.zeros(k_range)
    dr_j = (plastic_radius_final-r_i)/k_range

    Nr = 2/(m_r*UCS)*(m_r*UCS*p_0+s_r*UCS**2-m_r*(UCS**2)*M_e)**0.5
    r_e = r_i * np.exp(Nr - 2/(m_r*UCS)*(s_r*UCS**2)**0.5)
    dr_je = (r_e-r_i)/k_range

    def ring_radius_softening(num_rings):
        # start the strain softening calculation below
        for k in range(k_range):
            r_j[k] = plastic_radius_final - k * dr_j
        return r_j

    def radial_stress_softening(num_rings):

        for k in range(k_range):
            # dεθ[j] = Δ * εθ[j - 1]
            epsilon_theta[k] = epsilon_theta[k - 1] + delta * \
                epsilon_theta[k - 1] if k != 0 else epsilon_theta[0]
            if k != 0:
                epsilon_r[k] = epsilon_r[k - 1] - f * (delta * epsilon_theta[k - 1]) if epsilon_theta[k] > alpha * \
                    epsilon_1e else epsilon_r[k - 1] - h * (delta * epsilon_theta[k - 1])
            else:
                epsilon_r[k] = epsilon_r[0]

            lamda[k] = (2 * epsilon_theta[k - 1] - epsilon_r[k - 1] - epsilon_r[k]) / \
                (2 * epsilon_theta[k] - epsilon_r[k - 1] -
                 epsilon_r[k]) * lamda[k-1]if k != 0 else lamda[0]

            if k != 0:
                if epsilon_theta[k] <= alpha * epsilon_1e:
                    mp[k] = m_b + (m_r - m_b) * (epsilon_theta[k] - epsilon_theta[0]) / \
                        ((alpha - 1) * epsilon_theta[0])
                    sp[k] = s_b + (s_r - s_b) * (epsilon_theta[k] - epsilon_theta[0]) / \
                        ((alpha - 1) * epsilon_theta[0])
                else:
                    mp[k] = m_r
                    sp[k] = s_r
            else:
                mp[k] = mp[0]
                sp[k] = sp[0]
            ma = (mp[k - 1] + mp[k]) / 2 if k != 0 else mp[k]
            sa = (sp[k - 1] + sp[k]) / 2 if k != 0 else sp[k]

            k_s = ((lamda[k-1]-lamda[k])/(lamda[k-1]+lamda[k]))**2
            sigma_r[k - 1] = sigma_r[k - 1] if k != 0 else sigma_r[0]
            a = (sigma_r[k - 1]) ** 2 - 4 * k_s * \
                (0.5 * ma * UCS * sigma_r[k - 1] + sa * (UCS ** 2))
            b = sigma_r[k - 1] + k_s * ma * UCS

            sigma_r[k] = b - np.sqrt(b ** 2 - a)
            sigma_theta[k] = sigma_r[k] + \
                np.sqrt(mp[k] * sigma_r[k] * UCS + sp[k] * UCS**2)
            u_p[k] = lamda[k]*epsilon_theta[k] * \
                (r_i * np.exp(Nr - 2/(m_r*UCS) *
                 (m_r*UCS*sigma_r[k]+s_r*UCS**2)**0.5))

        return sigma_r, sigma_theta, u_p

    return ring_radius_softening, radial_stress_softening
