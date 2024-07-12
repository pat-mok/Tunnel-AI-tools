import numpy as np


def calculate_ground_reaction_curve_strain_softening(
        r_i, p_0, UCS, GSI, D, m_i, E_rm, nu, epsilon_1e, alpha, h, eta, CI, CD, sigma_T, m_b, s_b, m_r, s_r, G_rm, M_e, sigma_re, f):
    delta = 0.001  # Increment step for strains
    n_iterations = 1000  # Number of iteration steps

    epsilon_theta = np.zeros(n_iterations + 1)
    epsilon_r = np.zeros(n_iterations + 1)
    sigma_r = np.zeros(n_iterations + 1)
    sigma_theta = np.zeros(n_iterations + 1)
    mp = np.zeros(n_iterations + 1)
    sp = np.zeros(n_iterations + 1)
    lamda = np.ones(n_iterations + 1)
    r_p = np.zeros(n_iterations+1)
    u_p = np.zeros(n_iterations+1)

    # get the elastic brittle function here

    def plastic_radius(p_i):
        # Solution Constant
        Nr = 2/(m_r*UCS)*(m_r*UCS*p_0+s_r*UCS**2-m_r*(UCS**2)*M_e)**0.5
        # Calculate plastic radius
        if p_i >= sigma_re:
            return r_i
        else:
            return r_i * np.exp(Nr - 2/(m_r*UCS)*(m_r*UCS*p_i+s_r*UCS**2)**0.5)

    # start the strain softening calculation below

    epsilon_theta[0] = M_e * UCS / (2 * G_rm)
    epsilon_r[0] = -epsilon_theta[0]
    sigma_r[0] = sigma_re
    sigma_theta[0] = 2 * p_0 - sigma_r[0]
    mp[0] = m_b
    sp[0] = s_b

    def plastic_radius_softening(p_i):
        # calculate solution for each internal pressure value
        for j in range(n_iterations):

            # dεθ[j] = Δ * εθ[j - 1]
            epsilon_theta[j] = epsilon_theta[j - 1] + delta * \
                epsilon_theta[j - 1] if j != 0 else epsilon_theta[0]
            if j != 0:
                epsilon_r[j] = epsilon_r[j - 1] - f*(delta * epsilon_theta[j - 1]) if epsilon_theta[j] > alpha * \
                    epsilon_1e else epsilon_r[j - 1] - h * (delta * epsilon_theta[j - 1])
            else:
                epsilon_r[j] = epsilon_r[0]
            lamda[j] = (2 * epsilon_theta[j - 1] - epsilon_r[j - 1] - epsilon_r[j]) / \
                (2 * epsilon_theta[j] - epsilon_r[j - 1] -
                 epsilon_r[j]) * lamda[j-1]if j != 0 else lamda[0]
            u_p[j] = lamda[j]*epsilon_theta[j]*plastic_radius(p_i)

            if j != 0:
                if epsilon_theta[j] <= alpha * epsilon_1e:
                    mp[j] = m_b + (m_r - m_b) * (epsilon_theta[j] - epsilon_theta[0]) / \
                        ((alpha - 1) * epsilon_theta[0])
                    sp[j] = s_b + (s_r - s_b) * (epsilon_theta[j] - epsilon_theta[0]) / \
                        ((alpha - 1) * epsilon_theta[0])
                else:
                    mp[j] = m_r
                    sp[j] = s_r
            else:
                mp[j] = mp[0]
                sp[j] = sp[0]
            ma = (mp[j - 1] + mp[j]) / 2 if j != 0 else mp[j]
            sa = (sp[j - 1] + sp[j]) / 2 if j != 0 else sp[j]
            # k = ((1/lamda[j] - 1) / (1/lamda[j] + 1)) ** 2
            k = ((lamda[j-1]-lamda[j])/(lamda[j-1]+lamda[j]))**2
            sigma_r[j - 1] = sigma_r[j - 1] if j != 0 else sigma_r[0]
            a = (sigma_r[j - 1]) ** 2 - 4 * k * \
                (0.5 * ma * UCS * sigma_r[j - 1] + sa * (UCS ** 2))
            b = sigma_r[j - 1] + k * ma * UCS

            sigma_r[j] = b - np.sqrt(b ** 2 - a)
            sigma_theta[j] = sigma_r[j] + \
                np.sqrt(mp[j]*sigma_r[j]*UCS+sp[j]*UCS**2)
            r_p[j] = plastic_radius(p_i)/lamda[j]
            # test = lamda[j]
            if sigma_r[j] < p_i:
                break
        return r_p[j] if j != 0 else r_i

    def tunnel_convergence_softening(p_i):
        # calculate solution for each internal pressure value
        for j in range(n_iterations):

            # dεθ[j] = Δ * εθ[j - 1]
            epsilon_theta[j] = epsilon_theta[j - 1] + delta * \
                epsilon_theta[j - 1] if j != 0 else epsilon_theta[0]
            if j != 0:
                epsilon_r[j] = epsilon_r[j - 1] - f*(delta * epsilon_theta[j - 1]) if epsilon_theta[j] > alpha * \
                    epsilon_1e else epsilon_r[j - 1] - h * (delta * epsilon_theta[j - 1])
            else:
                epsilon_r[j] = epsilon_r[0]
            lamda[j] = (2 * epsilon_theta[j - 1] - epsilon_r[j - 1] - epsilon_r[j]) / \
                (2 * epsilon_theta[j] - epsilon_r[j - 1] -
                 epsilon_r[j]) * lamda[j-1]if j != 0 else lamda[0]
            u_p[j] = lamda[j]*epsilon_theta[j]*plastic_radius(p_i)

            if j != 0:
                if epsilon_theta[j] <= alpha * epsilon_1e:
                    mp[j] = m_b + (m_r - m_b) * (epsilon_theta[j] - epsilon_theta[0]) / \
                        ((alpha - 1) * epsilon_theta[0])
                    sp[j] = s_b + (s_r - s_b) * (epsilon_theta[j] - epsilon_theta[0]) / \
                        ((alpha - 1) * epsilon_theta[0])
                else:
                    mp[j] = m_r
                    sp[j] = s_r
            else:
                mp[j] = mp[0]
                sp[j] = sp[0]
            ma = (mp[j - 1] + mp[j]) / 2 if j != 0 else mp[j]
            sa = (sp[j - 1] + sp[j]) / 2 if j != 0 else sp[j]
            # k = ((1/lamda[j] - 1) / (1/lamda[j] + 1)) ** 2
            k = ((lamda[j-1]-lamda[j])/(lamda[j-1]+lamda[j]))**2
            sigma_r[j - 1] = sigma_r[j - 1] if j != 0 else sigma_r[0]
            a = (sigma_r[j - 1]) ** 2 - 4 * k * \
                (0.5 * ma * UCS * sigma_r[j - 1] + sa * (UCS ** 2))
            b = sigma_r[j - 1] + k * ma * UCS

            sigma_r[j] = b - np.sqrt(b ** 2 - a)
            sigma_theta[j] = sigma_r[j] + \
                np.sqrt(mp[j]*sigma_r[j]*UCS+sp[j]*UCS**2)
            r_p[j] = plastic_radius(p_i)/lamda[j]
            # test = lamda[j]
            if sigma_r[j] <= p_i:
                break

        return u_p[j] if j != 0 else r_i*(p_0 - p_i) / (2 * G_rm)

    def num_rings_softening(p_i):
        # calculate solution for each internal pressure value
        for j in range(n_iterations):

            # dεθ[j] = Δ * εθ[j - 1]
            epsilon_theta[j] = epsilon_theta[j - 1] + delta * \
                epsilon_theta[j - 1] if j != 0 else epsilon_theta[0]
            if j != 0:
                epsilon_r[j] = epsilon_r[j - 1] - f*(delta * epsilon_theta[j - 1]) if epsilon_theta[j] > alpha * \
                    epsilon_1e else epsilon_r[j - 1] - h * (delta * epsilon_theta[j - 1])
            else:
                epsilon_r[j] = epsilon_r[0]
            lamda[j] = (2 * epsilon_theta[j - 1] - epsilon_r[j - 1] - epsilon_r[j]) / \
                (2 * epsilon_theta[j] - epsilon_r[j - 1] -
                 epsilon_r[j]) * lamda[j-1]if j != 0 else lamda[0]
            u_p[j] = lamda[j]*epsilon_theta[j]*plastic_radius(p_i)

            if j != 0:
                if epsilon_theta[j] <= alpha * epsilon_1e:
                    mp[j] = m_b + (m_r - m_b) * (epsilon_theta[j] - epsilon_theta[0]) / \
                        ((alpha - 1) * epsilon_theta[0])
                    sp[j] = s_b + (s_r - s_b) * (epsilon_theta[j] - epsilon_theta[0]) / \
                        ((alpha - 1) * epsilon_theta[0])
                else:
                    mp[j] = m_r
                    sp[j] = s_r
            else:
                mp[j] = mp[0]
                sp[j] = sp[0]
            ma = (mp[j - 1] + mp[j]) / 2 if j != 0 else mp[j]
            sa = (sp[j - 1] + sp[j]) / 2 if j != 0 else sp[j]
            # k = ((1/lamda[j] - 1) / (1/lamda[j] + 1)) ** 2
            k = ((lamda[j-1]-lamda[j])/(lamda[j-1]+lamda[j]))**2
            sigma_r[j - 1] = sigma_r[j - 1] if j != 0 else sigma_r[0]
            a = (sigma_r[j - 1]) ** 2 - 4 * k * \
                (0.5 * ma * UCS * sigma_r[j - 1] + sa * (UCS ** 2))
            b = sigma_r[j - 1] + k * ma * UCS

            sigma_r[j] = b - np.sqrt(b ** 2 - a)
            sigma_theta[j] = sigma_r[j] + \
                np.sqrt(mp[j]*sigma_r[j]*UCS+sp[j]*UCS**2)
            r_p[j] = plastic_radius(p_i)/lamda[j]
            # test = lamda[j]
            if sigma_r[j] < p_i:
                break
        return j if j != 0 else 0
    return plastic_radius_softening, tunnel_convergence_softening, num_rings_softening
