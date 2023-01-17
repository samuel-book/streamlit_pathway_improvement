"""
Streamlit app template.

Because a long app quickly gets out of hand,
try to keep this document to mostly direct calls to streamlit to write
or display stuff. Use functions in other files to create and
organise the stuff to be shown. In this example, most of the work is
done in functions stored in files named container_(something).py
"""
# ----- Imports -----
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import matplotlib

# Add an extra bit to the path if we need to.
# Try importing something as though we're running this from the same
# directory as the landing page.
try:
    from utilities_pathway.fixed_params import page_setup
except ModuleNotFoundError:
    # If the import fails, add the landing page directory to path.
    # Assume that the script is being run from the directory above
    # the landing page directory, which is called
    # streamlit_pathway_improvement.
    import sys
    sys.path.append('./streamlit_pathway_improvement/')

# For importing data:
try:
    stroke_teams_test = pd.read_csv('./data_pathway/scenario_results.csv')
    dir = './'
except FileNotFoundError:
    dir = 'streamlit_pathway_improvement/'


# Custom functions:
from utilities_pathway.fixed_params import page_setup
from utilities_pathway.inputs import \
    write_text_from_file


def main():
    # ###########################
    # ##### START OF SCRIPT #####
    # ###########################
    page_setup()

    # Title:
    st.markdown('# Pathway improvement')


    # ###########################
    # ########## SETUP ##########
    # ###########################

    df = pd.read_csv(
        dir + 'data_pathway/scenario_results.csv',
        # index_col=index_col,
        header='infer'
        )

    # Baseline_good_outcomes_(median)
    # Baseline_good_outcomes_per_1000_patients_(low_5%)
    # Baseline_good_outcomes_per_1000_patients_(high_95%)
    # Baseline_good_outcomes_per_1000_patients_(mean)
    # Baseline_good_outcomes_per_1000_patients_(stdev)
    # Baseline_good_outcomes_per_1000_patients_(95ci)
    # Percent_Thrombolysis_(median%)
    # Percent_Thrombolysis_(low_5%)
    # Percent_Thrombolysis_(high_95%)
    # Percent_Thrombolysis_(mean)
    # Percent_Thrombolysis_(stdev)
    # Percent_Thrombolysis_(95ci)
    # Additional_good_outcomes_per_1000_patients_(median)
    # Additional_good_outcomes_per_1000_patients_(low_5%)
    # Additional_good_outcomes_per_1000_patients_(high_95%)
    # Additional_good_outcomes_per_1000_patients_(mean)
    # Additional_good_outcomes_per_1000_patients_(stdev)
    # Additional_good_outcomes_per_1000_patients_(95ci)
    # Onset_to_needle_(mean)
    # calibration
    # scenario
    # stroke_team

    scenarios = [
        'base',
        'speed',
        'onset',
        'benchmark',
        'speed_onset',
        'speed_benchmark',
        'onset_benchmark',
        'speed_onset_benchmark',
        # 'same_patient_characteristics'
    ]

    # Pull out the list of stroke teams:
    stroke_team_list = sorted(set(df['stroke_team'].to_numpy()))

    # Select a team
    highlighted_teams = st.multiselect(
        'Pick teams to highlight',
        stroke_team_list
        )
    highlighted_colours = px.colors.qualitative.Plotly + list(matplotlib.colors.cnames.values())




    # ###########################
    # ######### RESULTS #########
    # ###########################

    # Make a bar chart of the mean values:
    import plotly.graph_objects as go

    # Find max y values across all teams.
    max_percent_thrombolysis_mean = max(df['Percent_Thrombolysis_(mean)'])
    max_additional_good_mean = max(df['Additional_good_outcomes_per_1000_patients_(mean)'])

    for team in highlighted_teams:
        st.markdown('__Team '+team+'__')
        # Pick out just the data for the chosen team:
        df_here = df[df['stroke_team'] == team]
        cols_bar = st.columns(2)
        with cols_bar[0]:
            # Percentage thrombolysis use:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_here['scenario'],
                y=df_here['Percent_Thrombolysis_(mean)']
            ))

            fig.update_xaxes(title='Scenario')
            fig.update_yaxes(title='Percent Thrombolysis (mean)')

            fig.update_yaxes(range=[0, max_percent_thrombolysis_mean*1.05])

            st.plotly_chart(fig, use_container_width=True)

        with cols_bar[1]:
            # Additional good outcomes
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_here['scenario'],
                y=df_here['Additional_good_outcomes_per_1000_patients_(mean)'],
                marker=dict(color='red')
            ))

            fig.update_xaxes(title='Scenario')
            fig.update_yaxes(title='Additional good outcomes<br>per 1000 patients<br>(mean)')

            fig.update_yaxes(range=[0, max_additional_good_mean*1.05])

            st.plotly_chart(fig, use_container_width=True)


    def box_and_whisker(df, scenarios):
        # Box and whisker plot:
        fig = go.Figure()

        med_vals = df['Percent_Thrombolysis_(median%)'].values
        for x, scenario in enumerate(scenarios):

            df_scenario = df[df['scenario'] == scenario]
            med_vals = df_scenario['Percent_Thrombolysis_(median%)'].values

            fig.add_trace(go.Box(
                y=med_vals,
                x=[scenario]*len(med_vals),
                name=scenario,
                boxpoints=False,
                marker=dict(color='grey'),
                showlegend=False
                ))

        st.plotly_chart(fig, use_container_width=True)



    # Violin plot:

    def find_offsets_for_scatter(n_points, y_gap=0.05, y_max=0.2):
        # Where to scatter the team markers:
        y_offsets_scatter = [0.0]
        while len(y_offsets_scatter) < n_points:
            y_extra = np.arange(y_gap, y_max, y_gap)
            y_extra = np.stack((
                y_extra, -y_extra
            )).T.flatten()
            y_offsets_scatter = np.append(y_offsets_scatter, y_extra)
            y_gap = 0.5 * y_gap
        return y_offsets_scatter

    fig = go.Figure()

    x_offsets_scatter = find_offsets_for_scatter(len(highlighted_teams))

    med_vals = df['Percent_Thrombolysis_(median%)'].values
    for x, scenario in enumerate(scenarios):

        df_scenario = df[df['scenario'] == scenario]
        med_vals = df_scenario['Percent_Thrombolysis_(median%)'].values

        fig.add_trace(go.Violin(
            y=med_vals,
            x=[x]*len(med_vals),
            name=scenario,
            points=False,
            marker=dict(color='grey'),
            showlegend=False
            ))

        # Add scatter markers and line for the min/max/median:
        y_vals = [
            np.max(med_vals),
            np.median(med_vals),
            np.min(med_vals)
            ]
        fig.add_trace(go.Scatter(
            y=y_vals,
            x=[x]*len(y_vals),
            name=scenario,
            line_color='black',
            marker=dict(size=20, symbol='line-ew-open'),
            mode='markers+lines',
            showlegend=False,
            hoverinfo='skip',
        ))

        # Add scatter markers for highlighted teams
        showlegend_scatter = False if x > 0 else True
        for t, team in enumerate(highlighted_teams):
            y_team = df_scenario['Percent_Thrombolysis_(median%)'][df_scenario['stroke_team'] == team].values
            fig.add_trace(go.Scatter(
                y=y_team,
                x=[x+x_offsets_scatter[t]],
                name=team,
                mode='markers',
                marker=dict(color=highlighted_colours[t],
                            line=dict(color='black', width=1.0)),
                showlegend=showlegend_scatter
            ))

    fig.update_xaxes(
        tickmode='array',
        tickvals=np.arange(len(scenarios)),
        ticktext=scenarios
    )
    st.plotly_chart(fig, use_container_width=True)

















    show_box = False
    if show_box is True:
        # Box and whisker plot:
        fig = go.Figure()

        data = np.array([
            # df_here['Percent_Thrombolysis_(median%)'][df_here['scenario'] == s].values for s in df_here['scenario']
            0 for s in df_here['scenario']
            ]).T
        st.text(data)
        fig.add_trace(go.Box(
            y=data,
            x=df_here['scenario'],
            # name='boxes'
            # boxpoints=False,
            ))
        # Percent_Thrombolysis_(median%)
        # Percent_Thrombolysis_(low_5%)
        # Percent_Thrombolysis_(high_95%)
        # Percent_Thrombolysis_(mean)
        # Percent_Thrombolysis_(stdev)
        # Percent_Thrombolysis_(95ci)
        fig.update_traces(
            # q1=df_here['Percent_Thrombolysis_(low_5%)'],
            median=df_here['Percent_Thrombolysis_(median%)'],
            # q3=df_here['Percent_Thrombolysis_(high_95%)'],
            lowerfence=df_here['Percent_Thrombolysis_(low_5%)'],
            upperfence=df_here['Percent_Thrombolysis_(high_95%)'],
            mean=df_here['Percent_Thrombolysis_(mean)'],
            sd=df_here['Percent_Thrombolysis_(stdev)'],
            # notchspan=df_here['Percent_Thrombolysis_(median)']
            )


        st.plotly_chart(fig, use_container_width=True)

    st.write('Full data for this team: ')
    st.write(df_here)

    # ----- The end! -----


if __name__ == '__main__':
    main()
