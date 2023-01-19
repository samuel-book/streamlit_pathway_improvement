import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utilities_pathway.fixed_params import scenarios, scenarios_dict2

def plot_bars_sorted_rank(df_all, scenario, scenario_for_rank, n_teams='all'):
    # Use this string for labels:
    scenario_str = scenarios_dict2[scenario]
    # List of all the separate traces:
    hb_teams_input = st.session_state['hb_teams_input']
    highlighted_colours = st.session_state['highlighted_teams_colours']

    # change_colour = 'rgba(255,255,255,0.8)'

    fig = go.Figure()
    for n, name in enumerate(hb_teams_input):
        df = df_all[df_all['HB_team'] == name]

        colour = highlighted_colours[name]

        # Percentage thrombolysis use:
        custom_data = np.stack((
            # Name of the stroke team:
            df['stroke_team'][df['scenario'] == 'base'],
            # Effect of scenario:
            # (round this now so we can use the +/- sign format later)
            np.round(
                df['Percent_Thrombolysis_(mean)_diff']\
                    [df['scenario'] == scenario], 1),
            # Final prob:
            # (round this now so we can use the +/- sign format later)
            df['Percent_Thrombolysis_(mean)'][df['scenario'] == scenario],
            ), axis=-1)

        fig.add_trace(go.Bar(
            x=df['Sorted_rank!'+scenario_for_rank][df['scenario'] == 'base'],
            y=df['Percent_Thrombolysis_(mean)'][df['scenario'] == 'base'],
            name=name,
            width=1,
            marker=dict(color=colour),
                # line=dict(color=colour)),  #'rgba(0,0,0,0.5)'),
            customdata=custom_data
        ))


        if scenario != 'base':
            # y_diffs = (
            #     df['Percent_Thrombolysis_(mean)'][df['scenario'] == scenario].values -
            #     df['Percent_Thrombolysis_(mean)'][df['scenario'] == 'base'].values
            # )
            # The differences are already in the dataframe:
            y_diffs = (
                df['Percent_Thrombolysis_(mean)_diff'][df['scenario'] == scenario]
            )

            show_leggy = False if n > 0 else True
            leg_str = scenario_str.replace('+ ', '+<br>')
            leg_str_full = f'Difference due to<br>"{leg_str}"'
            fig.add_trace(go.Bar(
                x=df['Sorted_rank!'+scenario_for_rank][df['scenario'] == scenario],
                y=y_diffs,
                name=leg_str_full,
                width=0.3,
                marker=dict(color=colour, #opacity=0.2,
                    line=dict(color='black', width=0.1)),
                # customdata=custom_data
                hoverinfo='skip',
                showlegend=show_leggy
            ))
        else:
            leg_str_full = 'dummy'


    # Update hover label info *before* adding traces that have
    # hoverinfo='skip', otherwise this step will overwrite them.
    # Change the hover format
    fig.update_layout(hovermode='x')

    # Define the hover template:
    if scenario == 'base':
        # No change bars here.
        ht = (
            '%{customdata[0]}' +
            '<br>' +
            'Base probability: %{y:.1f}%' +
            '<br>' +
            'Rank: %{x} of ' + f'{n_teams} teams'
            '<extra></extra>'
        )
    else:
        # Add messages to reflect the change bars.
        ht = (
            '%{customdata[0]}' +
            '<br>' +
            'Base probability: %{y:.1f}%' +
            '<br>' +
            f'Effect of {scenario}: ' + '%{customdata[1]:+}%' +
            '<br>' +
            f'Final probability: ' + '%{customdata[2]:.1f}%' +
            '<br>' +
            'Rank: %{x} of ' + f'{n_teams} teams'
            '<extra></extra>'
        )

    # Update the hover template only for the bars that aren't
    # marking the changes, i.e. the bars that have the name
    # in the legend that we defined earlier.
    fig.for_each_trace(
        lambda trace: trace.update(hovertemplate=ht)
        # if trace.marker.color != change_colour
        if trace.name != leg_str_full
        else (),
    )

    # Change the bar mode
    fig.update_layout(barmode='stack')

    fig.update_layout(
        title=f'{scenario_str}',
        xaxis_title=f'Rank sorted by {scenario_for_rank}',
        yaxis_title='Percent Thrombolysis (mean)',
        legend_title='Highlighted team'
    )


    fig.update_yaxes(range=[0, max(df_all['Percent_Thrombolysis_(mean)'])*1.05])
    fig.update_xaxes(range=[
        min(df_all['Sorted_rank!'+scenario_for_rank])-1,
        max(df_all['Sorted_rank!'+scenario_for_rank])+1
        ])

    st.plotly_chart(fig, use_container_width=True)


def plot_bars_for_single_team(df, team):
    # For y-limits:
    # Find max y values across all teams.
    max_percent_thrombolysis_mean = max(df['Percent_Thrombolysis_(mean)'])
    min_percent_thrombolysis_mean = 0  # min(df['Percent_Thrombolysis_(mean)'])
    max_additional_good_mean = max(df['Additional_good_outcomes_per_1000_patients_(mean)'])
    min_additional_good_mean = 0  # min(df['Additional_good_outcomes_per_1000_patients_(mean)'])


    # max_percent_thrombolysis_mean_diff = max(df['Percent_Thrombolysis_(mean)_diff'].values)
    # min_percent_thrombolysis_mean_diff = min(df['Percent_Thrombolysis_(mean)_diff'].values)
    # max_additional_good_mean_diff = max(df['Additional_good_outcomes_per_1000_patients_(mean)_diff'].values)
    # min_additional_good_mean_diff = min(df['Additional_good_outcomes_per_1000_patients_(mean)_diff'].values)

    scenarios_str_list = []
    for s in scenarios:
        scenarios_str_list.append(scenarios_dict2[s])

    # Pick out just the data for the chosen team:
    df_here = df[df['stroke_team'] == team]
    # Pick out base values:
    base_percent_thromb_here = df_here['Percent_Thrombolysis_(mean)'][df_here['scenario'] == 'base'].values[0]
    base_additional_good_here = df_here['Additional_good_outcomes_per_1000_patients_(mean)'][df_here['scenario'] == 'base'].values[0]


    # cols_bar = st.columns(2)
    # with cols_bar[0]:

    subplot_titles = [
        'Percent Thrombolysis',
        'Additional good outcomes<br>per 1000 patients'
    ]
    fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles)

    fig.update_layout(title='<b>Team ' + team)
    # Percentage thrombolysis use:
    # fig = go.Figure()

    custom_data_mean = np.stack((
        # Difference between this scenario and base:
        np.round(df_here['Percent_Thrombolysis_(mean)_diff'], 1),
        # Base value:
        np.full(df_here['scenario'].values.shape, df_here['Percent_Thrombolysis_(mean)'][df_here['scenario'] == 'base'])
    ), axis=-1)

    fig.add_trace(go.Bar(
        x=df_here['scenario'],
        # y=df_here['Percent_Thrombolysis_(mean)'],
        y=df_here['Percent_Thrombolysis_(mean)'],
        customdata=custom_data_mean,
        showlegend=False
    ),
        row=1, col=1)

    fig.update_yaxes(title='Percent Thrombolysis', row=1, col=1)

    fig.update_yaxes(range=[
        min_percent_thrombolysis_mean*1.05,
        max_percent_thrombolysis_mean*1.05
        ],
        row=1, col=1)


    # --- Additional good outcomes ---
    
    custom_data_add = np.stack((
        # Difference between this scenario and base:
        np.round(df_here['Additional_good_outcomes_per_1000_patients_(mean)_diff'], 1),
        # Base value:
        np.full(df_here['scenario'].values.shape, df_here['Additional_good_outcomes_per_1000_patients_(mean)'][df_here['scenario'] == 'base'])
    ), axis=-1)

    fig.add_trace(go.Bar(
        x=df_here['scenario'],
        # y=df_here['Additional_good_outcomes_per_1000_patients_(mean)'],
        y=df_here['Additional_good_outcomes_per_1000_patients_(mean)'],
        marker=dict(color='red'),
        customdata=custom_data_add,
        showlegend=False
    ), row=1, col=2)


    fig.update_yaxes(title='Additional good outcomes',
        row=1, col=2)

    fig.update_yaxes(range=[
        min_additional_good_mean*1.05,
        max_additional_good_mean*1.05
        ],
        row=1, col=2)

    # Shared formatting:
    cols = [1, 2]
    base_values = [base_percent_thromb_here, base_additional_good_here]
    for c, col in enumerate(cols):
        # Update x-axis title and tick labels:
        fig.update_xaxes(title='Scenario', row=1, col=col)
        fig.update_xaxes(
            tickmode='array',
            tickvals = np.arange(len(scenarios_str_list)),
            ticktext=scenarios_str_list,
            row=1, col=col
        )
        # Draw a horizontal line at the base value:
        fig.add_hline(
            y=base_values[c],
            line=dict(color='silver', width=1.0),
            layer='above',  # Puts it above the bars
            row=1, col=col)
        # Write the size of each bar within the bar:
        fig.update_traces(text='%{customdata[0]}', row=1, col=col)
        # Set text position to "auto" so it sits inside the bar if there's
        # room, and outside the bar if there's not enough room.
        # For everything inside and auto-size the text, use "inside".
        fig.update_traces(textposition='outside', row=1, col=col)
        # Explicitly ask for + and - signs:
        fig.update_traces(texttemplate='%{customdata[0]:+}%', row=1, col=col)
        # Update hover template:
        perc_str = '%' if col == 1 else ''
        fig.update_traces(hovertemplate=(
            # 'Value for %{x}: %{y}%' +
            'Value: %{y}' + perc_str +
            '<br>' +
            'Difference from base: %{customdata[0]:+}' + perc_str +
            '<br>' +
            '<extra></extra>'
        ),
            row=1, col=col)



    st.plotly_chart(fig, use_container_width=True)
