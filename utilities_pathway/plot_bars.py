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

    # Format legend so newest teams appear at bottom:
    fig.update_layout(legend=dict(traceorder='normal'))

    fig.update_yaxes(range=[0, max(df_all['Percent_Thrombolysis_(mean)'])*1.05])
    fig.update_xaxes(range=[
        min(df_all['Sorted_rank!'+scenario_for_rank])-1,
        max(df_all['Sorted_rank!'+scenario_for_rank])+1
        ])

    st.plotly_chart(fig, use_container_width=True)



def plot_scatter_sorted_rank(df_all, scenario, scenario_for_rank, n_teams='all'):
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
            # Base prob:
            # (round this now so we can use the +/- sign format later)
            df['Percent_Thrombolysis_(mean)'][df['scenario'] == 'base'],
            # Final prob:
            # (round this now so we can use the +/- sign format later)
            df['Percent_Thrombolysis_(mean)'][df['scenario'] == scenario],
            ), axis=-1)


        # Setup assuming scenario == 'base':
        x_for_scatter = df['Sorted_rank!'+scenario_for_rank][df['scenario'] == 'base']
        y_for_scatter = df['Percent_Thrombolysis_(mean)'][df['scenario'] == 'base']
        mode = 'markers'
        symbols = 'circle'
        leg_str_full = 'dummy'  # Is this needed?
        size = 4

        if scenario != 'base':
            y_scenario = (
                df['Percent_Thrombolysis_(mean)'][df['scenario'] == scenario]
            )
            # The differences are already in the dataframe:
            y_diffs = (
                df['Percent_Thrombolysis_(mean)_diff'][df['scenario'] == scenario]
            )

            # Base symbols:
            symbols = ['circle'] * len(x_for_scatter)
            # Scenario symbols: 
            symbols_scen = np.full(len(symbols), 'circle', dtype='U20')
            symbols_scen[np.where(y_diffs >= 0)] = 'triangle-up'
            symbols_scen[np.where(y_diffs < 0)] = 'triangle-down'
            symbols = np.array([symbols, symbols_scen])

            x_for_scatter = np.array([x_for_scatter, x_for_scatter])
            y_for_scatter = np.array([y_for_scatter, y_scenario])

            mode = 'markers+lines'


            leg_str = scenario_str.replace('+ ', '+<br>')
            leg_str_full = f'Difference due to<br>"{leg_str}"'


        if scenario == 'base':
            fig.add_trace(go.Scatter(
                x=x_for_scatter,
                y=y_for_scatter,
                name=name,
                mode=mode,
                # width=1,
                marker=dict(color=colour, symbol=symbols, size=size),
                    # line=dict(color=colour)),  #'rgba(0,0,0,0.5)'),
                customdata=custom_data
            ))
        else:
            for t, team in enumerate(range(x_for_scatter.shape[1])):
                showlegend = False if t > 0 else True

                # st.write(custom_data)
                custom_data_here = custom_data[t, :]
                # st.text(custom_data_here)
                custom_data_here = np.transpose(np.stack((custom_data_here, custom_data_here), axis=-1))
                # st.text(custom_data_here)
                # st.text(custom_data_here[:, 0])

                fig.add_trace(go.Scatter(
                    x=x_for_scatter[:, t],
                    y=y_for_scatter[:, t],
                    name=name,
                    mode=mode,
                    # width=1,
                    marker=dict(color=colour, symbol=symbols[:, t], size=[4, 10]),
                        # line=dict(color=colour)),  #'rgba(0,0,0,0.5)'),
                    customdata=custom_data_here,
                    showlegend=showlegend
                ))



    # Update hover label info *before* adding traces that have
    # hoverinfo='skip', otherwise this step will overwrite them.
    # Change the hover format
    fig.update_layout(hovermode='closest')
    # # Reduce hover distance to prevent multiple labels popping
    # # up for each x:
    # fig.update_layout(hoverdistance=1)
    # ^ this isn't good enough - still get three labels

    # Define the hover template:
    if scenario == 'base':
        # No change bars here.
        ht = (
            '%{customdata[0]}' +
            '<br>' +
            'Base probability: %{customdata[2]:.1f}%' +
            '<br>' +
            'Rank: %{x} of ' + f'{n_teams} teams'
            '<extra></extra>'
        )
    else:
        # Add messages to reflect the change bars.
        ht = (
            '%{customdata[0]}' +
            '<br>' +
            'Base probability: %{customdata[2]:.1f}%' +
            '<br>' +
            f'Effect of {scenario}: ' + '%{customdata[1]:+}%' +
            '<br>' +
            f'Final probability: ' + '%{customdata[3]:.1f}%' +
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

    fig.update_layout(
        title=f'{scenario_str}',
        xaxis_title=f'Rank sorted by {scenario_for_rank}',
        yaxis_title='Percent Thrombolysis (mean)',
        legend_title='Highlighted team'
    )

    # Format legend so newest teams appear at bottom:
    fig.update_layout(legend=dict(traceorder='normal'))

    fig.update_yaxes(range=[0, max(df_all['Percent_Thrombolysis_(mean)'])*1.05])
    fig.update_xaxes(range=[
        min(df_all['Sorted_rank!'+scenario_for_rank])-1,
        max(df_all['Sorted_rank!'+scenario_for_rank])+1
        ])

    st.plotly_chart(fig, use_container_width=True)



def plot_scatter_base_vs_scenario(df_all, scenario, scenario_for_rank, n_teams='all'):
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
            # Base prob:
            # (round this now so we can use the +/- sign format later)
            df['Percent_Thrombolysis_(mean)'][df['scenario'] == 'base'],
            # Final prob:
            # (round this now so we can use the +/- sign format later)
            df['Percent_Thrombolysis_(mean)'][df['scenario'] == scenario],
            ), axis=-1)


        # Setup assuming scenario == 'base':
        x_for_scatter = df['Percent_Thrombolysis_(mean)'][df['scenario'] == 'base']
        y_for_scatter = df['Percent_Thrombolysis_(mean)'][df['scenario'] == 'base']
        mode = 'markers'
        symbols = 'circle'
        leg_str_full = 'dummy'  # Is this needed?
        size = 4

        if scenario != 'base':
            y_scenario = (
                df['Percent_Thrombolysis_(mean)'][df['scenario'] == scenario]
            )
            # The differences are already in the dataframe:
            y_diffs = (
                df['Percent_Thrombolysis_(mean)_diff'][df['scenario'] == scenario]
            )

            # Base symbols:
            symbols = ['circle'] * len(x_for_scatter)
            # Scenario symbols: 
            symbols_scen = np.full(len(symbols), 'circle', dtype='U20')
            symbols_scen[np.where(y_diffs >= 0)] = 'triangle-up'
            symbols_scen[np.where(y_diffs < 0)] = 'triangle-down'
            symbols = np.array([symbols, symbols_scen])

            x_for_scatter = np.array([x_for_scatter, x_for_scatter])
            y_for_scatter = np.array([y_for_scatter, y_scenario])

            mode = 'markers+lines'


            leg_str = scenario_str.replace('+ ', '+<br>')
            leg_str_full = f'Difference due to<br>"{leg_str}"'


        if scenario == 'base':
            fig.add_trace(go.Scatter(
                x=x_for_scatter,
                y=y_for_scatter,
                name=name,
                mode=mode,
                # width=1,
                marker=dict(color=colour, symbol=symbols, size=size),
                    # line=dict(color=colour)),  #'rgba(0,0,0,0.5)'),
                customdata=custom_data
            ))
        else:
            for t, team in enumerate(range(x_for_scatter.shape[1])):
                showlegend = False if t > 0 else True

                # st.write(custom_data)
                custom_data_here = custom_data[t, :]
                # st.text(custom_data_here)
                custom_data_here = np.transpose(np.stack((custom_data_here, custom_data_here), axis=-1))
                # st.text(custom_data_here)
                # st.text(custom_data_here[:, 0])

                fig.add_trace(go.Scatter(
                    x=x_for_scatter[:, t],
                    y=y_for_scatter[:, t],
                    name=name,
                    mode=mode,
                    # width=1,
                    marker=dict(color=colour, symbol=symbols[:, t], size=[4, 10]),
                        # line=dict(color=colour)),  #'rgba(0,0,0,0.5)'),
                    customdata=custom_data_here,
                    showlegend=showlegend
                ))



    # Update hover label info *before* adding traces that have
    # hoverinfo='skip', otherwise this step will overwrite them.
    # Change the hover format
    fig.update_layout(hovermode='closest')
    # # Reduce hover distance to prevent multiple labels popping
    # # up for each x:
    # fig.update_layout(hoverdistance=1)
    # ^ this isn't good enough - still get three labels

    # Define the hover template:
    if scenario == 'base':
        # No change bars here.
        ht = (
            '%{customdata[0]}' +
            '<br>' +
            'Base probability: %{customdata[2]:.1f}%' +
            '<br>' +
            'Rank: %{x} of ' + f'{n_teams} teams'
            '<extra></extra>'
        )
    else:
        # Add messages to reflect the change bars.
        ht = (
            '%{customdata[0]}' +
            '<br>' +
            'Base probability: %{customdata[2]:.1f}%' +
            '<br>' +
            f'Effect of {scenario}: ' + '%{customdata[1]:+}%' +
            '<br>' +
            f'Final probability: ' + '%{customdata[3]:.1f}%' +
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

    fig.update_layout(
        title=f'{scenario_str}',
        xaxis_title='Thrombolysis use (%)',
        yaxis_title='Thrombolysis use (%)',
        legend_title='Highlighted team'
    )

    # Format legend so newest teams appear at bottom:
    fig.update_layout(legend=dict(traceorder='normal'))

    fig.update_yaxes(range=[0, max(df_all['Percent_Thrombolysis_(mean)'])*1.05])
    fig.update_xaxes(range=[0, max(df_all['Percent_Thrombolysis_(mean)'])*1.05],
                     constrain='domain')  # For aspect ratio.)

    # Set aspect ratio:
    fig.update_yaxes(
        scaleanchor='x',
        scaleratio=1.0,
        constrain='domain'
    )

    st.plotly_chart(fig, use_container_width=True)



def plot_bar_scatter_sorted_rank(df_all, scenario, scenario_for_rank, n_teams='all'):
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
            # Base prob:
            # (round this now so we can use the +/- sign format later)
            df['Percent_Thrombolysis_(mean)'][df['scenario'] == 'base'],
            # Final prob:
            # (round this now so we can use the +/- sign format later)
            df['Percent_Thrombolysis_(mean)'][df['scenario'] == scenario],
            ), axis=-1)

        # Setup assuming scenario == 'base':
        x_for_scatter = df['Sorted_rank!'+scenario_for_rank][df['scenario'] == 'base']
        y_for_scatter = df['Percent_Thrombolysis_(mean)'][df['scenario'] == 'base']
        mode = 'markers'
        symbols = 'circle'
        leg_str_full = 'dummy'  # Is this needed?
        size = 4

        # Add bar chart:

        fig.add_trace(go.Bar(
            x=x_for_scatter,
            y=y_for_scatter,
            name=name,
            width=1,
            marker=dict(color=colour),
                # line=dict(color=colour)),  #'rgba(0,0,0,0.5)'),
            customdata=custom_data
        ))


        if scenario != 'base':
            y_scenario = (
                df['Percent_Thrombolysis_(mean)'][df['scenario'] == scenario]
            )
            # The differences are already in the dataframe:
            y_diffs = (
                df['Percent_Thrombolysis_(mean)_diff'][df['scenario'] == scenario]
            )

            # Base symbols:
            symbols = ['line-ew-open'] * len(x_for_scatter)
            # Scenario symbols: 
            symbols_scen = np.full(len(symbols), 'circle', dtype='U20')
            symbols_scen[np.where(y_diffs >= 0)] = 'triangle-up'
            symbols_scen[np.where(y_diffs < 0)] = 'triangle-down'
            symbols = np.array([symbols, symbols_scen])

            x_for_scatter = np.array([x_for_scatter, x_for_scatter])
            y_for_scatter = np.array([y_for_scatter, y_scenario])

            mode = 'markers+lines'


            leg_str = scenario_str.replace('+ ', '+<br>')
            leg_str_full = f'Difference due to<br>"{leg_str}"'


        if scenario == 'base':
            fig.add_trace(go.Scatter(
                x=x_for_scatter,
                y=y_for_scatter,
                name=name,
                mode=mode,
                # width=1,
                marker=dict(color=colour, symbol=symbols, size=size),
                    # line=dict(color=colour)),  #'rgba(0,0,0,0.5)'),
                customdata=custom_data
            ))
        else:
            for t, team in enumerate(range(x_for_scatter.shape[1])):
                # showlegend = False if t > 0 else True

                # st.write(custom_data)
                custom_data_here = custom_data[t, :]
                # st.text(custom_data_here)
                custom_data_here = np.transpose(np.stack((custom_data_here, custom_data_here), axis=-1))
                # st.text(custom_data_here)
                # st.text(custom_data_here[:, 0])

                fig.add_trace(go.Scatter(
                    x=x_for_scatter[:, t],
                    y=y_for_scatter[:, t],
                    name=name,
                    mode=mode,
                    # width=1,
                    marker=dict(color=colour, symbol=symbols[:, t], size=[0, 10]),
                        # line=dict(color=colour, width=0.2)),  #'rgba(0,0,0,0.5)'),
                    line=dict(width=0.5),
                    customdata=custom_data_here,
                    showlegend=False
                ))



    # Update hover label info *before* adding traces that have
    # hoverinfo='skip', otherwise this step will overwrite them.
    # Change the hover format
    fig.update_layout(hovermode='closest')
    # # Reduce hover distance to prevent multiple labels popping
    # # up for each x:
    # fig.update_layout(hoverdistance=1)
    # ^ this isn't good enough - still get three labels

    # Define the hover template:
    if scenario == 'base':
        # No change bars here.
        ht = (
            '%{customdata[0]}' +
            '<br>' +
            'Base probability: %{customdata[2]:.1f}%' +
            '<br>' +
            'Rank: %{x} of ' + f'{n_teams} teams'
            '<extra></extra>'
        )
    else:
        # Add messages to reflect the change bars.
        ht = (
            '%{customdata[0]}' +
            '<br>' +
            'Base probability: %{customdata[2]:.1f}%' +
            '<br>' +
            f'Effect of {scenario}: ' + '%{customdata[1]:+}%' +
            '<br>' +
            f'Final probability: ' + '%{customdata[3]:.1f}%' +
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

    fig.update_layout(
        title=f'{scenario_str}',
        xaxis_title=f'Rank sorted by {scenario_for_rank}',
        yaxis_title='Percent Thrombolysis (mean)',
        legend_title='Highlighted team'
    )

    # Format legend so newest teams appear at bottom:
    fig.update_layout(legend=dict(traceorder='normal'))

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
    min_percent_thrombolysis_mean = 0
    max_additional_good_mean = max(df['Additional_good_outcomes_per_1000_patients_(mean)'])
    min_additional_good_mean = 0

    scenarios_str_list = []
    for s in scenarios:
        s_label = scenarios_dict2[s]
        if '+' in s_label:
            s_label = '(' + s_label + ')'
        if len(s_label) > 20:
            # Onset + Speed + Benchmark is too long, so move the
            # "+ Benchmark" onto its own line.
            s_label = ' +<br> Benchmark'.join(s_label.split(' + Benchmark'))
        scenarios_str_list.append(s_label)

    # Pick out just the data for the chosen team:
    df_here = df[df['stroke_team'] == team]
    # Pick out base values:
    base_percent_thromb_here = df_here['Percent_Thrombolysis_(mean)'][df_here['scenario'] == 'base'].values[0]
    base_additional_good_here = df_here['Additional_good_outcomes_per_1000_patients_(mean)'][df_here['scenario'] == 'base'].values[0]


    subplot_titles = [
        'Thrombolysis use (%)',
        'Additional good outcomes<br>per 1000 admissions'
    ]
    fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles)

    fig.update_layout(title='<b>Team ' + team)  # <b> for bold

    # --- Percentage thrombolysis use ---
    diff_vals_mean = np.round(df_here['Percent_Thrombolysis_(mean)_diff'].values, 1)[1:]
    sign_list_mean = np.full(diff_vals_mean.shape, '+')
    sign_list_mean[np.where(diff_vals_mean < 0)[0]] = '-'
    bar_text_mean = []
    for i, diff in enumerate(diff_vals_mean):
        diff_str = ''.join([sign_list_mean[i], str(diff)])
        bar_text_mean.append(diff_str)
    bar_text_mean = [np.round(base_percent_thromb_here, 1).astype(str)] + bar_text_mean

    custom_data_mean = np.stack((
        # Difference between this scenario and base:
        np.round(df_here['Percent_Thrombolysis_(mean)_diff'], 1),
        # Base value:
        np.full(
            df_here['scenario'].values.shape, 
            df_here['Percent_Thrombolysis_(mean)']\
                [df_here['scenario'] == 'base']
            ),
        # Text for top of bars:
        bar_text_mean
    ), axis=-1)

    fig.add_trace(go.Bar(
        x=df_here['scenario'],
        # y=df_here['Percent_Thrombolysis_(mean)'],
        y=df_here['Percent_Thrombolysis_(mean)'],
        customdata=custom_data_mean,
        showlegend=False
    ),
        row=1, col=1)

    fig.update_yaxes(title='Thrombolysis use (%)', row=1, col=1)

    fig.update_yaxes(range=[
        min_percent_thrombolysis_mean*1.1,
        max_percent_thrombolysis_mean*1.1
        ],
        row=1, col=1)


    # --- Additional good outcomes ---
    diff_vals_add = np.round(df_here['Additional_good_outcomes_per_1000_patients_(mean)_diff'].values, 1)[1:]
    sign_list_add = np.full(diff_vals_add.shape, '+')
    sign_list_add[np.where(diff_vals_add < 0)[0]] = '-'
    bar_text_add = []
    for i, diff in enumerate(diff_vals_add):
        diff_str = ''.join([sign_list_add[i], str(diff)])
        bar_text_add.append(diff_str)
    bar_text_add = [np.round(base_additional_good_here, 1).astype(str)] + bar_text_add

    custom_data_add = np.stack((
        # Difference between this scenario and base:
        np.round(df_here['Additional_good_outcomes_per_1000_patients_(mean)_diff'], 1),
        # Base value:
        np.full(
            df_here['scenario'].values.shape,
            df_here['Additional_good_outcomes_per_1000_patients_(mean)']\
                [df_here['scenario'] == 'base']
            ),
        # Text for top of bars:
        bar_text_add
    ), axis=-1)

    fig.add_trace(go.Bar(
        x=df_here['scenario'],
        # y=df_here['Additional_good_outcomes_per_1000_patients_(mean)'],
        y=df_here['Additional_good_outcomes_per_1000_patients_(mean)'],
        # marker=dict(color='red'),
        customdata=custom_data_add,
        showlegend=False
    ), row=1, col=2)

    fig.update_yaxes(title='Additional good outcomes', row=1, col=2)

    # Update y-axis limits with breathing room for labels above bars.
    fig.update_yaxes(range=[
        min_additional_good_mean*1.1,
        max_additional_good_mean*1.1
        ],
        row=1, col=2)

    # Shared formatting:
    cols = [1, 2]
    base_values = [base_percent_thromb_here, base_additional_good_here]
    for c, col in enumerate(cols):
        # Add some extra '%' for the first subplot but not the second.
        perc_str = '%' if col == 1 else ''
        # Update x-axis title and tick labels:
        fig.update_xaxes(title='Scenario', row=1, col=col)
        fig.update_xaxes(
            tickmode='array',
            tickvals = np.arange(len(scenarios_str_list)),
            ticktext=scenarios_str_list,
            tickangle=90,
            row=1, col=col
        )
        # Draw a horizontal line at the base value:
        fig.add_hline(
            y=base_values[c],
            line=dict(color='silver', width=1.0),
            layer='above',  # Puts it above the bars
            row=1, col=col)
        # Write the size of each bar within the bar:
        fig.update_traces(text='%{customdata[2]}', row=1, col=col)
        # Set text position to "auto" so it sits inside the bar if there's
        # room, and outside the bar if there's not enough room.
        # For everything inside and auto-size the text, use "inside".
        fig.update_traces(textposition='outside', row=1, col=col)
        # Explicitly ask for + and - signs:
        fig.update_traces(texttemplate='%{customdata[2]}' + perc_str, row=1, col=col)
        # Update hover template:
        fig.update_traces(hovertemplate=(
            # 'Value for %{x}: %{y}%' +
            'Value: %{y}' + perc_str +
            '<br>' +
            'Difference from base: %{customdata[0]:+}' + perc_str +
            '<br>' +
            '<extra></extra>'
        ),
            row=1, col=col)

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True)
