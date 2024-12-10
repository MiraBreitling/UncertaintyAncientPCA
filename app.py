import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
import sys
import pathlib
sys.path.append(str(pathlib.Path().absolute()).split("/ancientPCA")[0] + "/ancientPCA")

from ancientPCA.PCA.ancient import *
from ancientPCA.preprocessing.parse_eigenstrat import *
from ancientPCA.PCA.ancient import PMPs_drift

############################################################
# Standard files 
stats = pd.read_csv('../ancientPCA/database/mean_percentile_pc1_pc2_distances.csv', header=0)
P =  pd.read_csv('../ancientPCA/database/P.csv', header=0)
modern = '../ancientPCA/database/coordinates_MWE.csv'
indices = pd.read_csv('../ancientPCA/database/SNPs_mwe.csv', header=0)

############################################################
# Helper functions
def get_nonvariant_geno(geno, indices):
    indices = indices["x"].values - 1
    return geno[:, indices]

def update_selected_percentiles():
    st.session_state["selected_percentiles"] = st.session_state["percentile_selection"]

def missing_statistics(geno, ind):
    total_positions = geno.shape[1]
    nines_counts = [{"Name": ind.iloc[i].values[2], "NaNs": np.isnan(geno[i, :]).sum()} for i in range(geno.shape[0])]

    missing_data_percentage = [
        {"Name": item["Name"], "Percentage": (item["NaNs"] / total_positions) * 100} for item in nines_counts
    ]

    return pd.DataFrame(missing_data_percentage).sort_values(by="Percentage", ascending=False)

def plot_missing(sample_names, missing_percentage):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sample_names,
        y=missing_percentage,
        marker=dict(color='blue'),
        name="Missing Data (%)",
        hoverinfo="x+y",
    ))
    fig.add_trace(go.Scatter(
        x=sample_names,
        y=[100] * len(sample_names),
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="100% (Total Positions)",
        hoverinfo="skip",
    ))
    fig.update_layout(
        title="Missing Data per Sample",
        xaxis_title="Samples",
        yaxis_title="Missing Data (%)",
        xaxis=dict(tickangle=45),
        template="plotly_white",
        height=500,
        width=800,
        showlegend=True,
    )
    st.plotly_chart(fig)


@st.cache_data
def generate_plot_data(modern, ancients, inds, P, nines):
    # Gecachte Berechnung für den Plot (ohne Ellipsen)
    fig = go.Figure()

    # Modern samples hinzufügen
    mwe = pd.read_csv(modern)
    coords_mwe = mwe[["PC1", "PC2"]]
    fig.add_trace(go.Scatter(
        x=coords_mwe["PC1"],
        y=coords_mwe["PC2"],
        mode='markers',
        marker=dict(color='rgb(127, 127, 127)'),
        name='Modern Samples',
    ))

    # Ancient samples hinzufügen
    for t, line in enumerate(ancients):
        name = inds.iloc[t].values[2]
        tau = PMPs_drift(line, P[["PC1", "PC2"]], P["genomean"])
        percentage = nines.loc[nines['Name'] == name, 'Percentage'].iloc[0] if not nines.loc[nines['Name'] == name].empty else None
        sample_text = f"{name} \n ({percentage:.2f}% missing)" if percentage is not None else name

        # Sample hinzufügen
        fig.add_trace(go.Scatter(
            x=[tau[0]],
            y=[tau[1]],
            mode='markers+text',
            marker=dict(color='blue'),
            name=name,
            text=sample_text,
            textposition="top center",
            legendgroup=name,
        ))

    return fig


@st.cache_data
def add_uncertainty_ellipses(fig, ancients, inds, P, stats, nines, selected_percentiles):
    # Ellipsen hinzufügen, falls `uncertainty=True`
    for t, line in enumerate(ancients):
        name = inds.iloc[t].values[2]
        tau = PMPs_drift(line, P[["PC1", "PC2"]], P["genomean"])
        percentage = nines.loc[nines['Name'] == name, 'Percentage'].iloc[0] if not nines.loc[nines['Name'] == name].empty else None

        if selected_percentiles and percentage is not None:
            alpha = 0.2
            row = stats[stats["rate"] > float(percentage)].iloc[0]

            for percentile in selected_percentiles:
                numeric_percentile = int(percentile[:-1])
                pc1_col = f"mean_percentile_{numeric_percentile}_pc1_dist"
                pc2_col = f"mean_percentile_{numeric_percentile}_pc2_dist"

                if pc1_col in row and pc2_col in row:
                    angles = np.linspace(0, 2 * np.pi, 100)
                    x = tau[0] + row[pc1_col] * np.cos(angles)
                    y = tau[1] + row[pc2_col] * np.sin(angles)

                    fig.add_trace(go.Scatter(
                        x=np.append(x, x[0]),
                        y=np.append(y, y[0]),
                        mode='none',
                        fill='toself',
                        fillcolor=f'rgba(255, 0, 0, {alpha})',
                        name=f"{percentile}% Ellipse",
                        legendgroup=name,
                        showlegend=False,
                    ))

    return fig


def plot_samples(modern, ancients, inds, P, stats, uncertainty=False, nines=None, selected_percentiles=None):
    # Basis-Plot aus Cache laden oder generieren
    fig = generate_plot_data(modern, ancients, inds, P, nines)

    # Unsicherheits-Ellipsen hinzufügen, falls erforderlich
    if uncertainty:
        fig = add_uncertainty_ellipses(fig, ancients, inds, P, stats, nines, selected_percentiles)

    # Layout aktualisieren
    fig.update_layout(
        title="Projections of Ancient Samples based on Modern West-Eurasians",
        xaxis_title="PC1",
        yaxis_title="PC2",
        template="plotly_white",
    )

    return fig



############################################################
# App Layout
st.set_page_config(page_title="Ancient PCA Uncertainty Analysis")
#st.title("Uncertainty Analysis for Ancient Genomic Data Principal Component Projections")
st.markdown("<h3 style='text-align: left; margin-top: 0;'>Uncertainty Analysis for Ancient Genomic Data Principal Component Projections</h3>", unsafe_allow_html=True)


# Initialize session state
if "geno" not in st.session_state:
    st.session_state["geno"] = None
if "ind" not in st.session_state:
    st.session_state["ind"] = None
if "nines" not in st.session_state:
    st.session_state["nines"] = None
if "selected_percentiles" not in st.session_state:
    st.session_state["selected_percentiles"] = ["99%", "90%", "75%", "50%"]
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Genotype Data"

# Sidebar file uploads
st.sidebar.header("Please upload your EIGENSTRAT files")
geno_file = st.sidebar.file_uploader("GENO file", type=["csv", "txt", "geno"])
ind_file = st.sidebar.file_uploader("IND file", type=["csv", "txt", "ind"])
example_data = st.sidebar.button("Use Example Data")
if example_data:
    st.info("Using example data...")
    st.session_state["geno"] = parse_geno_file_path("../data/Data_HO/geno_files/ancients_for_tooltesting/test_ancient.geno")
    nonv_geno = get_nonvariant_geno(st.session_state["geno"], indices)
    st.session_state["nonvariant_geno"] = nonv_geno
    st.session_state["ind"] = parse_ind("../data/Data_HO/geno_files/ancients_for_tooltesting/test_ancient.ind")
    st.session_state["nines"] = missing_statistics(st.session_state["geno"], st.session_state["ind"])
elif geno_file and ind_file:
    st.session_state["geno"] = parse_geno_from_file(StringIO(geno_file.getvalue().decode("utf-8")))
    nonv_geno = get_nonvariant_geno(st.session_state["geno"], indices)
    st.session_state["nonvariant_geno"] = nonv_geno
    st.session_state["ind"] = parse_ind(ind_file)
    st.session_state["nines"] = missing_statistics(st.session_state["geno"], st.session_state["ind"])


# Tabs
tab1, tab2, tab3 = st.tabs(["Genotype Data", "Uncertainty Analysis", "Information"])

# Tab 1: Genotype Data
with tab1:
    st.header("Genotype Data")

    if st.session_state["geno"] is not None and st.session_state["ind"] is not None:
        st.success("GENO and IND data uploaded successfully!")

        # Datenvorschau
        with st.expander("Data Preview", expanded=False):
            st.subheader("Data preview")
            st.write("Geno (First 50x50)")
            st.write(st.session_state["geno"][0:50, 0:50])
            st.write("Ind (First Rows)")
            st.write(st.session_state["ind"].head())

        # Missing Data Statistics und Plot
        st.subheader("Missing Data Statistics")
        missing_data = st.session_state["nines"]

        st.write("Missing Data per Sample (in %)")
        st.table(missing_data)

        # Missing Data Plot
        sample_names = missing_data["Name"].tolist()
        missing_percentage = missing_data["Percentage"].tolist()
        plot_missing(sample_names, missing_percentage)
        
        # PCA-Platzierung
        st.subheader("Sample Placement Based on SmartPCA")
        fig = plot_samples(modern, st.session_state["nonvariant_geno"], st.session_state["ind"], P, stats, uncertainty=False, nines=st.session_state["nines"], selected_percentiles=None)
        st.plotly_chart(fig)
    else:
        st.warning("Please upload your GENO and IND files.")

# Tab 2: Uncertainty Analysis
with tab2:
    st.header("Uncertainty Analysis")

    if st.session_state["geno"] is None or st.session_state["ind"] is None:
        st.warning("Please upload your GENO, SNP, and IND files.")
    else:
        # Dropdown-Menü für Percentile-Auswahl mit Callback
        st.multiselect(
            "Select Percentiles to Display:",
            options=["99%", "90%", "75%", "50%"],
            default=st.session_state["selected_percentiles"],
            key="percentile_selection",  # Temporärer Key
            on_change=update_selected_percentiles,  # Aktualisierung des Session States
        )

        # Plot generieren
        fig = plot_samples(
            modern,
            st.session_state["nonvariant_geno"],
            st.session_state["ind"],
            P,
            stats,
            uncertainty=True,
            nines=st.session_state["nines"],
            selected_percentiles=st.session_state["selected_percentiles"],  # Aktualisierter Wert
        )
        st.plotly_chart(fig)

# Tab 3: Information
with tab3:
    st.header("Information")
    st.write("""
        This platform provides statistics and visualization to assess the uncertainty of genotype sample projections in a Principal Component Analysis (PCA).
        Derived from the SmartPCA algorithm, the placement variability of ancient genomic data points in the feature space is quantified and displayed based on modern West-Eurasian samples.
    """)


    # Hochzuladende Dateien
    st.subheader("Data")
    st.write("To assess the uncertainty in your PCA placement, please upload the following data:")
    st.markdown("""
    - **GENO-Datei**: The genotype data in EIGENSTRAT format (other formats will be supported soon)
    - **IND-Datei**: Information on the individuals (e.g. name, population).
    - **SNP-Datei**: The SNP names and positions.
    """)

