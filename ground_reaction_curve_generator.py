import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from elastic_brittle import calculate_ground_reaction_curve_elastic_brittle
from strain_softening import calculate_ground_reaction_curve_strain_softening
from get_stresses import calculate_radial_stress_softening

# Custom CSS to adjust the width of the expander and chart, and to keep the title on one line
css = """
<style>
    .main .block-container {
        max-width: 1200px;
        padding: 1rem 1rem 1rem 1rem;
    }
    .streamlit-expander {
        width: 100% !important;
    }
    .streamlit-expanderHeader {
        width: 100% !important;
    }
</style>
"""

# Inject the custom CSS
st.markdown(css, unsafe_allow_html=True)

# Streamlit app
st.title('Tunnel Ground Reaction Curve Analysis')

# Sidebar for input parameters
st.sidebar.header('Input Parameters')
col1, col2 = st.sidebar.columns(2)

# Input parameters
with col1:
    r_i = st.number_input('Initial tunnel radius (m):',
                          value=7.5, min_value=0.0, format="%.1f")
    p_0 = st.number_input('Initial in-situ stress (MPa):',
                          value=2.4, min_value=0.0, format="%.1f")
    UCS = st.number_input('Uniaxial compressive strength (MPa):',
                          value=10.0, min_value=0.0, format="%.2f")
    GSI = st.number_input('Geological Strength Index (GSI):',
                          value=50.0, min_value=0.0, format="%.1f")
    D = st.number_input('Disturbance factor:', value=0.0,
                        min_value=0.0, format="%.1f")
    m_i = st.number_input('Initial value of material constant (m_i):',
                          value=12.0, min_value=0.0, format="%.1f")
    E_rm = st.number_input('Rock mass Modulus (MPa):',
                           value=700.0, min_value=0.0, format="%.2f")
    nu = st.number_input('Poisson\'s ratio:', value=0.25,
                         min_value=0.1, max_value=0.5, step=0.01, format="%.2f")

with col2:
    epsilon_1e = st.number_input('Peak elastic strain (epsilon_1e):',
                                 value=0.0020, min_value=0.0, step=0.0001, format="%.4f")
    alpha = st.number_input('Residual strain ratio:',
                            value=4.0, min_value=0.0, format="%.1f")
    h = st.number_input('Post-peak strain ratio:',
                        value=3.0, min_value=0.0, format="%.1f")
    eta = st.number_input('Jointed rock mass modification factor:',
                          value=1.5, min_value=0.0, format="%.1f")
    CI = st.number_input('Crack initiation Threshold (MPa):',
                         value=0.5 * UCS, min_value=0.0, format="%.2f")
    CD = st.number_input('Crack damage Threshold (MPa):',
                         value=0.8 * UCS, min_value=0.0, format="%.2f")
    sigma_T = st.number_input(
        'Rock tensile strength (MPa):', value=UCS / m_i, min_value=0.0, format="%.2f")

# Hoek-Brown Parameters for peak and residual rock mass properties
a_b = 0.25 if GSI >= 65 else 0.5 + (1/6)*(np.exp(-GSI/15) - np.exp(-20/3))
s_b = 0.25 if GSI >= 65 else np.exp((GSI-100)/(9-3*D))
m_b = (s_b*UCS/sigma_T) ** (0.5/a_b) if GSI >= 65 else m_i * \
    np.exp((GSI-100)/(28-14*D))
G_rm = E_rm / (2 * (1 + nu))
M_e = 0.5 * ((m_b / 4) ** 2 + m_b * p_0 / UCS + s_b) ** 0.5 - m_b / 8
sigma_re = p_0 - M_e * UCS

a_r = 0.75 if GSI >= 50 else 0.5 + 1 / 6 * \
    (np.exp(-0.7 * GSI / 15) - np.exp(-20 / 3)) if GSI >= 30 else a_b
m_r = 9 ** (0.5 / a_r) if GSI >= 70 else 15 ** (0.5 / a_r) if GSI >= 50 else m_i * \
    np.exp((0.7 * GSI - 100) / (28 - 14 * D)) if GSI >= 30 else m_b
s_r = 0.0001 if GSI >= 50 else np.exp(
    (0.7 * GSI - 100) / (9 - 3 * D)) if GSI >= 30 else s_b
F_r = m_b / (2 * (m_b * sigma_re / UCS + s_b) ** 0.5)
f = 1 + F_r

print(M_e, G_rm, M_e * UCS / (2 * G_rm))

# Calculation of ground reaction curves
num_points = 50
p_i_values = np.linspace(p_0, 0, num_points)


plastic_radius_elastic, tunnel_convergence_elastic = calculate_ground_reaction_curve_elastic_brittle(
    r_i, p_0, UCS, GSI, D, m_i, E_rm, nu, epsilon_1e, alpha, h, eta, CI, CD, sigma_T, G_rm, m_r, s_r, M_e, f, sigma_re
)
plastic_radius_elastic = np.array(
    [plastic_radius_elastic(p_i) for p_i in p_i_values])
tunnel_convergence_elastic = np.array(
    [tunnel_convergence_elastic(p_i) for p_i in p_i_values])

# get plastic radius at elastic boundary for strain softening calculation
r_e = plastic_radius_elastic
u_e = tunnel_convergence_elastic[-1]

plastic_radius_softening, tunnel_convergence_softening, num_rings_softening = calculate_ground_reaction_curve_strain_softening(
    r_i, p_0, UCS, GSI, D, m_i, E_rm, nu, epsilon_1e, alpha, h, eta, CI, CD, sigma_T, m_b, s_b, m_r, s_r, G_rm, M_e, sigma_re, f)

plastic_radius_softening = np.array(
    [plastic_radius_softening(p_i) for p_i in p_i_values])
tunnel_convergence_softening = np.array(
    [tunnel_convergence_softening(p_i) for p_i in p_i_values])
num_rings_softening = np.array(
    [num_rings_softening(p_i) for p_i in p_i_values])

# get radial stresses

num_rings = int(num_rings_softening[-1])
num_rings_values = np.linspace(0, num_rings, num_rings+1)
plastic_radius_final = plastic_radius_softening[-1]

k_range = 1 if num_rings == 0 else num_rings
ring_radius_softening, radial_stress_softening = calculate_radial_stress_softening(
    r_i, p_0, UCS, GSI, D, m_i, E_rm, nu, epsilon_1e, alpha, h, eta, CI, CD, sigma_T, m_b, s_b, m_r, s_r, G_rm, M_e, sigma_re, f, plastic_radius_final, k_range)
ring_radius_softening = ring_radius_softening(num_rings=1)
r_stress_softening, t_stress_softening, radial_deformation = radial_stress_softening(
    num_rings=1)


# concatenate elastic stress profile beyond plastic zone
radius_in_elastic_zone = np.linspace(
    plastic_radius_final+5*r_i, plastic_radius_final, num_points+1)
r_stress_in_elastic_zone = np.array([
    p_0-(p_0-sigma_re)*(plastic_radius_final/r_ez)**2 for r_ez in radius_in_elastic_zone])
t_stress_in_elastic_zone = np.array([
    p_0+(p_0-sigma_re)*(plastic_radius_final/r_ez)**2 for r_ez in radius_in_elastic_zone])
elastic_deformation = np.array(
    [(p_0-(p_0-(p_0-sigma_re)*(plastic_radius_final/r_ez)**2))/(2*G_rm)*r_i for r_ez in radius_in_elastic_zone])
radius_profile_softening = np.append(
    radius_in_elastic_zone, ring_radius_softening)
radial_stress_profile_softening = np.append(
    r_stress_in_elastic_zone, r_stress_softening)
tangetnial_stress_profile_softening = np.append(
    t_stress_in_elastic_zone, t_stress_softening)
deformation_profile_softening = np.append(
    elastic_deformation, radial_deformation)

# Longitudinal deformation profiles


def calculate_longitudinal_profile(plastic_radius, convergence, r_i, eta, k=50, k_step=0.2):
    u_0 = (1/3) * np.exp(-0.15 * plastic_radius[-1] / r_i)
    x_values = np.zeros(k)
    u_values = np.zeros(k)
    for x in range(0, k):
        x_values[x] = -5 + k_step * x
        u_values[x] = u_0 * np.exp(eta * x_values[x]) if x_values[x] <= 0 else 1 - (
            1 - u_0) * np.exp(-eta * 3 * x_values[x] / (2 * plastic_radius[-1] / r_i))
    return x_values, u_values


x_e, u_elastic = calculate_longitudinal_profile(
    plastic_radius_elastic, tunnel_convergence_elastic, r_i, eta)
x_r, u_softening = calculate_longitudinal_profile(
    plastic_radius_softening, tunnel_convergence_softening, r_i, eta)

# # Create a DataFrame for the results checking
# results_df = pd.DataFrame({

#     'ring radius': ring_radius_softening,
#     'radial stress': radial_stress_softening,


# })

# # Display the results as a table
# st.table(results_df)


# Plotting
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(tunnel_convergence_elastic, p_i_values,
         label='Ground Reaction Curve - Elastic Brittle Rock Mass')
ax1.plot(tunnel_convergence_softening, p_i_values,
         label='Ground Reaction Curve - Strain Softening Rock Mass', color='red')
ax1.set_ylabel('Internal Support Pressure (MPa)')
ax1.set_xlabel('Tunnel Convergence (m)')
ax1.set_title('Internal Support Pressure vs. Tunnel Convergence')
ax1.legend()
ax1.grid(True)
plt.legend(fontsize='small')

# Secondary y-axis for longitudinal profile
ax2 = ax1.twinx()
ax2.plot(u_elastic * tunnel_convergence_elastic[-1], x_e * r_i,
         label='Longitudinal Deformation - Elastic Brittle Rock Mass', color='green')
ax2.plot(u_softening * tunnel_convergence_softening[-1], x_r * r_i,
         label='Longitudinal Deformation - Strain Softening Rock Mass', color='purple')
ax2.set_ylabel('Distance from advancing face (m)', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc='upper left')
plt.legend(fontsize='small')
# Save the chart as an image file
chart_path = "chart.png"
fig1.savefig(chart_path)
st.pyplot(fig1)

# Inject custom CSS to adjust the width of columns
st.markdown(
    """
    <style>
    .col-style-1 {
        flex: 1 1 40%;  /* Adjust the width as needed */
    }
    .col-style-2 {
        flex: 1 1 20%;  /* Adjust the width as needed */
    }
    .col-style-3 {
        flex: 1 1 20%;  /* Adjust the width as needed */
    }
    .col-style-4 {
        flex: 1 1 20%;  /* Adjust the width as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Expander for output query
col1, col2, col3, col4 = st.columns([4, 2, 2, 2])

output_query = st.expander('Output Query')
with output_query:
    advance_length = st.number_input('Advance Length:')
    st.write('Results based on Advance Length:')
    with col1:
        x_e_input = advance_length / r_i
        u_elastic_output = np.interp(
            x_e_input, x_e, u_elastic) * tunnel_convergence_elastic[-1]
        st.info(f'Tunnel convergence-elastic brittle: {u_elastic_output:.3f}m')
        x_r_input = advance_length / r_i
        u_softening_output = np.interp(
            x_r_input, x_r, u_softening) * tunnel_convergence_softening[-1]
        st.info(
            f'Tunnel convergence-strain softening: {u_softening_output:.3f}m')
    with col2:
        p_e_output = np.interp(
            u_elastic_output, tunnel_convergence_elastic, p_i_values)
        st.info(f'Internal pressure: {p_e_output:.2f}MPa')
        p_r_output = np.interp(
            u_softening_output, tunnel_convergence_softening, p_i_values)
        st.info(f'Internal pressure: {p_r_output:.2f}MPa')
    with col3:
        st.info(f'Relaxation:{(1-p_e_output/p_0)*100:.0f}%')
        st.info(f'Relaxation:{(1-p_r_output/p_0)*100:.0f}%')
    with col4:
        st.info(f'Plastic radius:{plastic_radius_elastic[-2]:.2f}m')
        st.info(f'Plastic radius:{plastic_radius_softening[-2]:.2f}m')

# DataFrames for CSV export
df_elastic = pd.DataFrame({
    'Internal Support Pressure (MPa)': p_i_values,
    'Tunnel Convergence Elastic (m)': tunnel_convergence_elastic,
    'Plastic Radius Elastic': plastic_radius_elastic,
    'Longitudinal Deformation Elastic': u_elastic * tunnel_convergence_elastic[-1],
    'Distance from advancing face Elastic (m)': x_e * r_i
})

df_softening = pd.DataFrame({
    'Internal Support Pressure (MPa)': p_i_values,
    'Tunnel Convergence Softening (m)': tunnel_convergence_softening,
    'Plastic Radius Softening': plastic_radius_softening,
    'Longitudinal Deformation Softening': u_softening * tunnel_convergence_softening[-1],
    'Distance from advancing face Softening (m)': x_r * r_i,
})


df_radial_stress = pd.DataFrame({
    'radius (m)': radius_profile_softening,
    'radial stress (MPa)': radial_stress_profile_softening,
    'tangential stress (MPa)': tangetnial_stress_profile_softening,
    'radial deformation(m)': deformation_profile_softening
})
# 'tangential stress (MPa)': tangetnial_stress_profile_softening

# Export to CSV
csv_elastic = df_elastic.to_csv(index=False).encode('utf-8')
csv_softening = df_softening.to_csv(index=False).encode('utf-8')
csv_stresses = df_radial_stress.to_csv(index=False).encode('utf-8')

# Download buttons
st.sidebar.download_button(label="Download Elastic Data as CSV",
                           data=csv_elastic, file_name='elastic_data.csv', mime='text/csv')
st.sidebar.download_button(label="Download Softening Data as CSV",
                           data=csv_softening, file_name='softening_data.csv', mime='text/csv')
st.sidebar.download_button(label="Download Stresses Data as CSV",
                           data=csv_stresses, file_name='stresses_data.csv', mime='text/csv')

# Provide a download button for the chart
with open(chart_path, "rb") as file:
    btn = st.sidebar.download_button(
        label="Download Chart",
        data=file,
        file_name="chart.png",
        mime="image/png"
    )
