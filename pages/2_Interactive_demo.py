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

import plotly.graph_objects as go
# For importing colours:
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

# Custom functions:
from utilities_pathway.fixed_params import \
    page_setup, scenarios, scenarios_dict, plotly_colours
import utilities_pathway.inputs
import utilities_pathway.plot_bars
import utilities_pathway.plot_violins
import utilities_pathway.plot_hist
from utilities_pathway.plot_utils import \
    remove_old_colours_for_highlights, choose_colours_for_highlights


def main():
    # ###########################
    # ##### START OF SCRIPT #####
    # ###########################
    page_setup()

    # Title:
    st.markdown('# Pathway improvement')
    st.warning(''.join([
        ':warning: __Work in progress__ ',
        'This app runs but still needs explanatory text.'
        ]))

    container_highlighted_input = st.container()

    tabs_results = st.tabs(['All teams', 'Highlighted teams'])
    with tabs_results[0]:
        container_percentage_thrombolysed = st.container()
        container_sort_input = st.container()
        container_additional_good = st.container()


    # ###########################
    # ########## SETUP ##########
    # ###########################

    stroke_teams_list = utilities_pathway.inputs.\
        import_lists_from_data()

    # User inputs:
    with container_sort_input:
        st.markdown('__:warning: TO DO:__ implement sorting better.')
        scenario, scenario_for_rank = utilities_pathway.inputs.\
            inputs_for_bar_chart()

    with container_highlighted_input:
        # Receive the user inputs now and show this container now:
        # with container_bar_chart:
        st.markdown(''.join([
            'To highlight stroke teams on the following charts, ',
            'select them in this box. ',
            '__:warning: TO DO:__ make charts clickable.'
            # ' or click on them in the charts.'
        ]))
        # Pick teams to highlight on the bar chart:
        highlighted_teams_input = utilities_pathway.inputs.\
            highlighted_teams(stroke_teams_list)


    # #######################################
    # ########## MAIN CALCULATIONS ##########
    # #######################################

    df = utilities_pathway.inputs.\
        import_stroke_data(
            stroke_teams_list, scenarios, highlighted_teams_input
            )

    # Sort the data according to this input:
    for each_scenario in scenarios + [scenario_for_rank]:
        df = utilities_pathway.inputs.add_sorted_rank_column_to_df(
            df, each_scenario, len(stroke_teams_list), len(scenarios))

    # Find colour lists for plotting (saved to session state):
    hb_teams_input = st.session_state['hb_teams_input']
    remove_old_colours_for_highlights(hb_teams_input)
    choose_colours_for_highlights(hb_teams_input)

    highlighted_colours = st.session_state['highlighted_teams_colours']

    # ###########################
    # ######### RESULTS #########
    # ###########################

    with container_percentage_thrombolysed:
        st.markdown('## Percentage of patients thrombolysed')
        cols_perc = st.columns(2)
        with cols_perc[0]:
            st.markdown('__Histogram for all teams__')
            utilities_pathway.plot_hist.plot_hist(
                df,
                ['base', scenario],
                highlighted_teams_input,
                highlighted_colours,
                len(stroke_teams_list),
                df_column='Percent_Thrombolysis_(mean)', 
                y_title='Thrombolysis use (%)'
            )
        with cols_perc[1]:
            st.markdown('__Change for each team__')
            utilities_pathway.plot_bars.plot_scatter_sorted_rank(
                df,
                scenario,
                scenario_for_rank,
                n_teams=len(stroke_teams_list),
                y_str='Percent_Thrombolysis_(mean)',
                showlegend_col=True
                )


    with container_additional_good:
        st.markdown('## Additional good outcomes')
        cols_add = st.columns(2)
        with cols_add[0]:
            st.markdown('__Histogram for all teams__')
            utilities_pathway.plot_hist.plot_hist(
                df,
                ['base', scenario],
                highlighted_teams_input,
                highlighted_colours,
                len(stroke_teams_list),
                df_column='Additional_good_outcomes_per_1000_patients_(mean)', 
                y_title='Additional good outcomes<br>per 1000 admissions'
            )

        with cols_add[1]:
            st.markdown('__Change for each team__')
            utilities_pathway.plot_bars.plot_scatter_sorted_rank(
                df,
                scenario,
                scenario_for_rank,
                n_teams=len(stroke_teams_list),
                y_str='Additional_good_outcomes_per_1000_patients_(mean)',
                showlegend_col=True
                )


    with tabs_results[1]:
        # Bar chart for individual team:
        if len(highlighted_teams_input) < 1:
            st.markdown('No teams are highlighted.')
        else:
            for team in highlighted_teams_input:
                cols_single_bars = st.columns(2)
                with cols_single_bars[0]:
                    utilities_pathway.plot_bars.plot_bars_for_single_team(
                            df, team,
                            df_column='Percent_Thrombolysis_(mean)',
                            y_label='Thrombolysis use (%)',
                            bar_colour=plotly_colours[0]
                            )
                with cols_single_bars[1]:
                    utilities_pathway.plot_bars.\
                        plot_bars_for_single_team(
                            df, team,
                            df_column='Additional_good_outcomes_per_1000_patients_(mean)',
                            y_label='Additional good outcomes<br>per 1000 admissions',
                            bar_colour=plotly_colours[1]
                            )
                # Write an empty header for breathing room;
                st.markdown('# ')

    # ----- The end! -----


if __name__ == '__main__':
    main()
