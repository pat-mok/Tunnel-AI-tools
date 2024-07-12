# elastic_brittle.py

import numpy as np


def calculate_ground_reaction_curve_elastic_brittle(r_i, p_0, UCS, GSI, D, m_i, E_rm, nu, epsilon_1e, alpha, h, eta, CI, CD, sigma_T, G_rm, m_r, s_r, M_e, f, sigma_re):

    def plastic_radius(p_i):
        # Solution Constant
        Nr = 2/(m_r*UCS)*(m_r*UCS*p_0+s_r*UCS**2-m_r*(UCS**2)*M_e)**0.5
        # Calculate plastic radius
        if p_i >= sigma_re:
            return r_i
        else:
            return r_i * np.exp(Nr - 2/(m_r*UCS)*(m_r*UCS*p_i+s_r*UCS**2)**0.5)

    def tunnel_convergence(p_i):
        if p_i >= sigma_re:
            return r_i * (p_0 - p_i) / (2 * G_rm)
        else:
            r_e = plastic_radius(p_i)
            return r_i * (M_e * UCS / (G_rm * (f + 1))) * ((f-1)/2 + (r_e/r_i)**(f+1))

    return plastic_radius, tunnel_convergence
