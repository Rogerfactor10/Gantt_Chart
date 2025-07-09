import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gantt Project Planner",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Project Timeline and Gantt Chart Tool")
st.write("Create, edit, and visualize your project schedule. Add tasks, milestones, and track their progress.")

# --- Mapping of color names and icons to Plotly formats (GLOBAL) ---
color_map = {
    'red': 'rgb(220, 50, 50)',      # Rojo más vibrante
    'green': 'rgb(50, 180, 50)',    # Verde más vibrante
    'blue': 'rgb(50, 100, 220)',    # Azul más profundo
    'orange': 'rgb(255, 140, 0)',   # Naranja más fuerte
    'purple': 'rgb(150, 50, 200)',  # Púrpura más oscuro y saturado
    'black': 'rgb(100, 100, 100)',  # Gris más oscuro para "black" original
    'gray': 'rgb(150, 150, 150)',   # Gris medio más usable
    'yellow': 'rgb(220, 220, 50)',  # Amarillo más intenso
    'cyan': 'rgb(50, 200, 200)',    # Cian más fuerte
    'magenta': 'rgb(200, 50, 200)'  # Magenta más fuerte
}


icon_map = {
    'star': 'star',
    'circle': 'circle',
    'square': 'square',
    'diamond': 'diamond',
    'flag': 'triangle-right',
    'checkmark': 'circle-open',
    'x': 'x',
    'none': ''
}

def darken_rgb(rgb_str, factor=0.7):
    """
    Darkens an RGB color string. Handles both 'rgb(R,G,B)' and color names
    by converting names to RGB first using color_map.
    """
    try:
        # If it's a color name, convert to RGB using the color_map
        if not str(rgb_str).lower().startswith('rgb('):
            rgb_str = color_map.get(str(rgb_str).lower(), 'rgb(100,100,100)') # Default to a neutral gray

        if not rgb_str.startswith('rgb(') or not rgb_str.endswith(')'):
             raise ValueError("RGB string format invalid")

        parts_str = rgb_str.replace('rgb(', '').replace(')', '')
        parts = [int(p.strip()) for p in parts_str.split(',')]
        r, g, b = parts

        r_new = int(r * factor)
        g_new = int(g * factor)
        b_new = int(b * factor)
        return f'rgb({r_new},{g_new},{b_new})'
    except Exception as e:
        # Fallback for any parsing errors
        return 'rgb(50,50,50)'

# --- SESSION STATE INITIALIZATION ---
# Define expected dtypes globally so both init and cleanup can use them
expected_dtypes = {
    'Type': str,
    'Name': str,
    'Start': 'datetime64[ns]', # Pandas internal datetime format
    'Finish': 'datetime64[ns]',
    'Resource': str,
    'Progress': str,
    'Color': str,
    'Icon': str,
    'Associated_Task': 'object'
}

if 'project_items_df' not in st.session_state:
    initial_data = [
        {'Type': 'Task', 'Name': 'Requirements Analysis', 'Start': date(2025, 1, 1), 'Finish': date(2025, 1, 15), 'Resource': 'Analyst', 'Progress': '0%', 'Color': 'blue', 'Icon': None, 'Associated_Task': None},
        {'Type': 'Task', 'Name': 'UI/UX Design', 'Start': date(2025, 1, 16), 'Finish': date(2025, 2, 10), 'Resource': 'Designer', 'Progress': '25%', 'Color': 'green', 'Icon': None, 'Associated_Task': None},
        {'Type': 'Task', 'Name': 'Backend Development', 'Start': date(2025, 2, 11), 'Finish': date(2025, 4, 10), 'Resource': 'Backend', 'Progress': '50%', 'Color': 'orange', 'Icon': None, 'Associated_Task': None},
        {'Type': 'Task', 'Name': 'Frontend Development', 'Start': date(2025, 2, 11), 'Finish': date(2025, 4, 20), 'Resource': 'Frontend', 'Progress': '75%', 'Color': 'purple', 'Icon': None, 'Associated_Task': None},
        {'Type': 'Task', 'Name': 'Testing & QA', 'Start': date(2025, 4, 21), 'Finish': date(2025, 5, 10), 'Resource': 'QA', 'Progress': '100%', 'Color': 'red', 'Icon': None, 'Associated_Task': None},
        {'Type': 'Task', 'Name': 'Deployment', 'Start': date(2025, 5, 11), 'Finish': date(2025, 5, 15), 'Resource': 'DevOps', 'Progress': '0%', 'Color': 'gray', 'Icon': None, 'Associated_Task': None},
        {'Type': 'Milestone', 'Name': 'Project Kick-off', 'Start': date(2025, 1, 1), 'Finish': date(2025, 1, 1), 'Resource': None, 'Progress': '0%', 'Color': 'red', 'Icon': 'star', 'Associated_Task': 'Requirements Analysis'},
        {'Type': 'Milestone', 'Name': 'Prototype Approved', 'Start': date(2025, 2, 10), 'Finish': date(2025, 2, 10), 'Resource': None, 'Progress': '0%', 'Color': 'green', 'Icon': 'circle', 'Associated_Task': 'UI/UX Design'},
        {'Type': 'Milestone', 'Name': 'Lanzamiento Beta', 'Start': date(2025, 5, 15), 'Finish': date(2025, 5, 15), 'Resource': None, 'Progress': '0%', 'Color': 'blue', 'Icon': 'square', 'Associated_Task': 'Deployment'},
        {'Type': 'Milestone', 'Name': 'Hito Sin Tarea', 'Start': date(2025, 3, 1), 'Finish': date(2025, 3, 1), 'Resource': None, 'Progress': '0%', 'Color': 'orange', 'Icon': 'diamond', 'Associated_Task': None}
    ]
    st.session_state.project_items_df = pd.DataFrame(initial_data)
    
    # Ensure all expected columns are present and set initial types for the *initial* DataFrame
    for col, dtype in expected_dtypes.items():
        if col not in st.session_state.project_items_df.columns:
            if 'date' in str(dtype):
                st.session_state.project_items_df[col] = pd.NaT
            elif dtype == str:
                st.session_state.project_items_df[col] = ''
            else:
                st.session_state.project_items_df[col] = None
        
        # Convert to datetime if it's a date column and not already
        if 'date' in str(dtype):
            st.session_state.project_items_df[col] = pd.to_datetime(st.session_state.project_items_df[col], errors='coerce')
        else:
            if dtype == str:
                # Ensure string columns are actually strings and handle potential NaN/None
                st.session_state.project_items_df[col] = st.session_state.project_items_df[col].astype(str).replace('nan', '').replace('None', '')
            else:
                # Attempt to convert to other dtypes, ignore errors if mixed types
                st.session_state.project_items_df[col] = st.session_state.project_items_df[col].astype(dtype, errors='ignore')

    # Convert dates to Python date objects after initial DataFrame setup for consistency with chart plotting
    st.session_state.project_items_df['Start'] = st.session_state.project_items_df['Start'].dt.date
    st.session_state.project_items_df['Finish'] = st.session_state.project_items_df['Finish'].dt.date

# --- Data Cleanup and Update Function (Callable as Callback) ---
# The function now accepts the edited DataFrame as an argument
def clean_and_update_df(edited_df_from_editor):
    """
    This function processes the edited DataFrame from st.data_editor and updates
    st.session_state.project_items_df. It's designed to be called after st.data_editor returns.
    """
    # Use the DataFrame passed directly from the data_editor's return value
    edited_df = edited_df_from_editor.copy() 

    # Ensure all expected columns exist in the edited_df
    # This addresses the KeyError. When a new row is added, it might not initially
    # have all columns populated by default by st.data_editor.
    for col, dtype in expected_dtypes.items():
        if col not in edited_df.columns:
            if 'date' in str(dtype):
                edited_df[col] = pd.NaT # Not a Time for missing dates
            elif dtype == str:
                edited_df[col] = '' # Empty string for missing text
            else: # For object types like Associated_Task (can be None)
                edited_df[col] = None
        
        # Ensure column types are correct right after creation/fill_na for the edited df
        if 'date' in str(dtype):
            edited_df[col] = pd.to_datetime(edited_df[col], errors='coerce')
        elif dtype == str:
            # Handle potential non-string types that need to be coerced to string
            edited_df[col] = edited_df[col].astype(str).replace('nan', '').replace('None', '')
        # For 'object' types, direct assignment of None/str is often fine.


    # Convert dates ensuring they are actual date objects or NaT (Python date objects for plotting)
    # The .dt.date conversion can result in NaT if the input was invalid, so check that.
    edited_df['Start'] = edited_df['Start'].apply(lambda x: x.date() if pd.notna(x) else pd.NaT)
    edited_df['Finish'] = edited_df['Finish'].apply(lambda x: x.date() if pd.notna(x) else pd.NaT)


    # Fill NaNs/None for string columns with empty string and ensure type
    edited_df['Name'] = edited_df['Name'].fillna('').astype(str)
    edited_df['Type'] = edited_df['Type'].fillna('Task').astype(str)
    edited_df['Color'] = edited_df['Color'].fillna('blue').astype(str)
    edited_df['Progress'] = edited_df['Progress'].fillna('0%').astype(str)
    edited_df['Icon'] = edited_df['Icon'].fillna('none').astype(str)
    edited_df['Resource'] = edited_df['Resource'].apply(lambda x: '' if pd.isna(x) else str(x))

    # Apply logic based on item Type for each row
    for idx, row in edited_df.iterrows():
        item_type = str(row['Type']).strip().lower()

        if item_type == 'milestone':
            edited_df.at[idx, 'Resource'] = None
            edited_df.at[idx, 'Progress'] = '0%' # Milestones always 0% progress
            edited_df.at[idx, 'Icon'] = str(row['Icon']).lower()
            
            # For milestones, Start and Finish should be the same date
            if pd.notna(row['Start']):
                edited_df.at[idx, 'Finish'] = row['Start']
            else: # If Start is missing, default to today
                edited_df.at[idx, 'Start'] = date.today()
                edited_df.at[idx, 'Finish'] = date.today()
            
            # Get valid task names *from the current edited_df* to ensure dropdown consistency
            current_valid_task_names = edited_df[
                (edited_df['Type'].astype(str).str.lower() == 'task') & (edited_df['Name'].astype(bool))
            ]['Name'].unique().tolist()

            # Ensure Associated_Task is a valid task name or None
            if row['Associated_Task'] not in current_valid_task_names:
                edited_df.at[idx, 'Associated_Task'] = None
            else:
                edited_df.at[idx, 'Associated_Task'] = str(row['Associated_Task'])

        else: # Type is 'Task' or any other unexpected value, default to Task behavior
            edited_df.at[idx, 'Icon'] = 'none' # Tasks don't have icons
            edited_df.at[idx, 'Associated_Task'] = None # Tasks don't associate with other tasks
            
            # Ensure resource is string
            if pd.isna(row['Resource']) or str(row['Resource']).strip() == '':
                edited_df.at[idx, 'Resource'] = ''
            else:
                edited_df.at[idx, 'Resource'] = str(row['Resource'])
            
            # Ensure progress is one of the valid options
            if str(row['Progress']) not in progress_options:
                edited_df.at[idx, 'Progress'] = '0%'
            
            # Default dates for tasks if invalid or missing
            if pd.isna(row['Start']) and pd.isna(row['Finish']):
                edited_df.at[idx, 'Start'] = date.today()
                edited_df.at[idx, 'Finish'] = date.today() + timedelta(days=7)
            elif pd.isna(row['Start']):
                # If only Start is missing, infer from Finish
                edited_df.at[idx, 'Start'] = edited_df.at[idx, 'Finish'] - timedelta(days=7) if pd.notna(edited_df.at[idx, 'Finish']) else date.today()
            elif pd.isna(row['Finish']):
                # If only Finish is missing, infer from Start
                edited_df.at[idx, 'Finish'] = edited_df.at[idx, 'Start'] + timedelta(days=7) if pd.notna(edited_df.at[idx, 'Start']) else date.today() + timedelta(days=7)
            
            # Ensure Finish is not before Start for tasks (at least 1 day duration)
            if pd.notna(edited_df.at[idx, 'Start']) and pd.notna(edited_df.at[idx, 'Finish']):
                if edited_df.at[idx, 'Finish'] < edited_df.at[idx, 'Start']:
                    edited_df.at[idx, 'Finish'] = edited_df.at[idx, 'Start'] + timedelta(days=1)
    
    # Filter out rows where 'Name' is empty *after* processing defaults
    final_cleaned_df = edited_df[edited_df['Name'].astype(bool)].reset_index(drop=True)

    # Finally, update the session state
    st.session_state.project_items_df = final_cleaned_df

# --- Gantt Chart Function ---
def create_gantt_chart(project_items_df):
    today = datetime.now().date()

    DEFAULT_Y_MIN_EMPTY = -0.5
    DEFAULT_Y_MAX_EMPTY = 1.5

    Y_AXIS_BUFFER_TOP = 1.0
    Y_AXIS_BUFFER_BOTTOM = 0.5

    tasks_df_for_chart = project_items_df[project_items_df['Type'] == 'Task'].copy()
    milestones_df_for_chart = project_items_df[project_items_df['Type'] == 'Milestone'].copy()

    # Ensure date columns are datetime objects for plotting, coercing errors
    tasks_df_for_chart['Start'] = pd.to_datetime(tasks_df_for_chart['Start'], errors='coerce')
    tasks_df_for_chart['Finish'] = pd.to_datetime(tasks_df_for_chart['Finish'], errors='coerce')
    tasks_df_for_chart.dropna(subset=['Name', 'Start', 'Finish'], inplace=True) # Drop rows with NaT dates here

    milestones_df_for_chart['Start'] = pd.to_datetime(milestones_df_for_chart['Start'], errors='coerce')
    milestones_df_for_chart['Finish'] = pd.to_datetime(milestones_df_for_chart['Finish'], errors='coerce')
    milestones_df_for_chart.dropna(subset=['Name', 'Start', 'Finish'], inplace=True) # Drop rows with NaT dates here

    # Determine chart date range
    all_dates = []
    
    # Use .dropna() when converting to list to filter out NaT/None
    if not tasks_df_for_chart.empty:
        all_dates.extend(tasks_df_for_chart['Start'].dropna().tolist())
        all_dates.extend(tasks_df_for_chart['Finish'].dropna().tolist())
    if not milestones_df_for_chart.empty:
        all_dates.extend(milestones_df_for_chart['Start'].dropna().tolist())
        all_dates.extend(milestones_df_for_chart['Finish'].dropna().tolist())

    chart_min_date = None
    chart_max_date = None

    if all_dates:
        valid_dates_for_min_max = []
        for d in all_dates:
            # pd.notna(d) handles both NaT and None/np.nan
            if pd.notna(d): 
                if isinstance(d, datetime):
                    valid_dates_for_min_max.append(d)
                elif isinstance(d, date): # If it's a date.date object, convert to datetime
                    valid_dates_for_min_max.append(datetime.combine(d, datetime.min.time()))
                # Add a check for Timestamp (Pandas' internal datetime type)
                elif isinstance(d, pd.Timestamp):
                    valid_dates_for_min_max.append(d.to_pydatetime())
        
        if valid_dates_for_min_max:
            chart_min_date = min(valid_dates_for_min_max) - timedelta(days=7) # A week buffer
            chart_max_date = max(valid_dates_for_min_max) + timedelta(days=7) # A week buffer
        else: # Fallback if after all filtering, no valid dates remain
            chart_min_date = today - timedelta(days=30)
            chart_max_date = today + timedelta(days=30)
    else: # If all_dates is completely empty
        chart_min_date = today - timedelta(days=30)
        chart_max_date = today + timedelta(days=30)

    # Handle empty DataFrame case explicitly to show an empty chart with today's line
    if tasks_df_for_chart.empty and milestones_df_for_chart.empty:
        fig = go.Figure()
        y_range_bottom_for_full_line_empty = DEFAULT_Y_MAX_EMPTY + Y_AXIS_BUFFER_BOTTOM
        y_range_top_for_full_line_empty = DEFAULT_Y_MIN_EMPTY - Y_AXIS_BUFFER_TOP

        fig.update_layout(
            xaxis=dict(
                side="top",
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1w", step="day", stepmode="todate"),
                        dict(count=1, label="1m", step="month", stepmode="todate"),
                        dict(count=6, label="6m", step="month", stepmode="todate"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="todate"),
                        dict(step="all")
                    ]),
                    y=1.1,
                    yanchor='bottom'
                ),
                rangeslider=dict(visible=False),
                type="date",
                range=[chart_min_date, chart_max_date]
            ),
            yaxis=dict(
                visible=False, # No task bars, so hide y-axis labels
                autorange="reversed",
                range=[y_range_top_for_full_line_empty, y_range_bottom_for_full_line_empty]
            ),
            showlegend=False, # Will add custom legend items later
        )
        
        # Add 'Today' line for empty chart
        fig.add_trace(go.Scatter(
            x=[today, today],
            y=[y_range_top_for_full_line_empty, y_range_bottom_for_full_line_empty],
            mode='lines',
            line=dict(color='red', width=3, dash='solid'),
            name='We are Here',
            showlegend=True,
            hoverinfo='text',
            text=f"Today: {today.strftime('%d-%m-%Y')}" # Changed format here too for consistency
        ))

        # Add annotation for 'Today'
        annotation_y_pos_today_empty = y_range_top_for_full_line_empty - 0.2
        fig.add_annotation(
            x=today,
            y=annotation_y_pos_today_empty,
            text=f"We are Here<br>({today.strftime('%d-%m-%Y')})", # Changed format here too
            showarrow=False,
            yshift=10,
            font=dict(color='red', size=12, weight='bold'),
            align="center",
        )

        # Add milestones even if tasks are empty
        if not milestones_df_for_chart.empty:
             for index, row in milestones_df_for_chart.iterrows():
                milestone_date = row['Start']
                milestone_color = color_map.get(str(row['Color']).lower(), 'rgb(100, 100, 100)')
                milestone_icon_symbol = icon_map.get(str(row['Icon']).lower(), 'circle')

                # Milestone line
                fig.add_trace(go.Scatter(
                    x=[milestone_date, milestone_date],
                    y=[y_range_top_for_full_line_empty, y_range_bottom_for_full_line_empty],
                    mode='lines',
                    line=dict(color=milestone_color, width=2, dash='dot'),
                    name=row['Name'],
                    showlegend=True,
                    hoverinfo='text',
                    text=f"{row['Name']}<br>{milestone_date.strftime('%d-%m-%Y')}" # Changed format here too
                ))

                milestone_y_pos_icon_text_empty = y_range_top_for_full_line_empty + 0.1

                # Milestone icon
                if milestone_icon_symbol:
                    fig.add_trace(go.Scatter(
                        x=[milestone_date],
                        y=[milestone_y_pos_icon_text_empty],
                        mode='markers',
                        marker=dict(
                            symbol=milestone_icon_symbol,
                            size=14,
                            color=milestone_color,
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        name=f"Icon {row['Name']}", # Name for hover, not legend
                        showlegend=False,
                        hoverinfo='text',
                        text=f"Milestone: {row['Name']}<br>Date: {milestone_date.strftime('%d-%m-%Y')}" # Changed format here too
                    ))

                # Milestone annotation
                milestone_annotation_text = f"{row['Name']}<br>({milestone_date.strftime('%d-%m-%Y')})" # Changed format here too
                fig.add_annotation(
                    x=milestone_date,
                    y=milestone_y_pos_icon_text_empty,
                    text=milestone_annotation_text,
                    showarrow=False,
                    yshift=25,
                    font=dict(color=milestone_color, size=11),
                    align="center",
                )
        return fig, chart_min_date, chart_max_date

    # --- Create Gantt Chart for tasks using Plotly Figure Factory ---
    tasks_for_ff = tasks_df_for_chart.rename(columns={'Name': 'Task'}).copy()
    tasks_for_ff['Complete'] = tasks_for_ff['Color'].apply(lambda x: color_map.get(str(x).lower(), 'rgb(100,100,100)'))

    fig = ff.create_gantt(
        tasks_for_ff.to_dict('records'),
        index_col='Task',
        colors=tasks_for_ff['Complete'].tolist(),
        group_tasks=False,
        showgrid_x=True,
        showgrid_y=True,
    )

    # Get the order of tasks on the y-axis from the generated Gantt chart
    if hasattr(fig.layout.yaxis, 'categoryarray') and fig.layout.yaxis.categoryarray is not None:
        plotly_y_categories = fig.layout.yaxis.categoryarray
    else:
        plotly_y_categories = []

    task_to_y_index = {task_name: i for i, task_name in enumerate(plotly_y_categories)}

    # Bar dimensions for progress overlay
    bar_actual_half_width = 0.4
    progress_bar_padding_y = 0.05
    
    progress_bar_y0_offset = -bar_actual_half_width + progress_bar_padding_y
    progress_bar_y1_offset = bar_actual_half_width - progress_bar_padding_y

    num_categories = len(plotly_y_categories)

    # Calculate y-axis range based on number of tasks
    y_plot_min_for_tasks = -0.5
    y_plot_max_for_tasks = (num_categories - 1) + 0.5 if num_categories > 0 else -0.5

    y_range_top_for_full_line = y_plot_min_for_tasks - Y_AXIS_BUFFER_TOP
    y_range_bottom_for_full_line = y_plot_max_for_tasks + Y_AXIS_BUFFER_BOTTOM

    # If no tasks, revert to default empty chart range for vertical lines
    if tasks_df_for_chart.empty:
        y_range_top_for_full_line = DEFAULT_Y_MIN_EMPTY - Y_AXIS_BUFFER_TOP
        y_range_bottom_for_full_line = DEFAULT_Y_MAX_EMPTY + Y_AXIS_BUFFER_BOTTOM

    # Update layout for combined chart
    fig.update_layout(
        yaxis=dict(
            autorange="reversed",
            categoryorder='array',
            categoryarray=plotly_y_categories,
            range=[y_range_top_for_full_line, y_range_bottom_for_full_line]
        ),
        xaxis_title="Date",
        yaxis_title="Tasks / Milestones",
        hovermode='x unified', # Shows consolidated hover info for all traces at an x-coordinate
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),

        xaxis=dict(
            side="top",
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1w", step="day", stepmode="todate"),
                    dict(count=1, label="1m", step="month", stepmode="todate"),
                    dict(count=6, label="6m", step="month", stepmode="todate"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="todate"),
                    dict(step="all")
                ]),
                y=1.1,
                yanchor='bottom'
            ),
            rangeslider=dict(visible=False),
            type="date",
            range=[chart_min_date, chart_max_date]
        )
    )

    # --- Add Progress Bars and Percentage Text ---
    for index, row in tasks_df_for_chart.iterrows():
        task_name = row['Name']
        start_date = row['Start']
        finish_date = row['Finish']
        progress_str = str(row['Progress'])
        base_color_name = str(row['Color'])

        try:
            progress_percentage = float(progress_str.strip('%')) / 100.0
        except ValueError:
            progress_percentage = 0.0 # Default to 0 if parsing fails

        if progress_percentage > 0 and task_name in task_to_y_index:
            total_duration_seconds = (finish_date - start_date).total_seconds()
            progress_duration_seconds = total_duration_seconds * progress_percentage
            
            # Ensure progress end date doesn't exceed finish date
            progress_end_date = min(start_date + timedelta(seconds=progress_duration_seconds), finish_date)

            y_index = task_to_y_index[task_name]

            y0_progress_rect = y_index + progress_bar_y0_offset
            y1_progress_rect = y_index + progress_bar_y1_offset

            # Get the darker shade for the progress bar
            darker_rgb_color = darken_rgb(color_map.get(base_color_name.lower(), 'rgb(100,100,100)'))

            # Add the progress rectangle
            fig.add_shape(
                type="rect",
                x0=start_date,
                y0=y0_progress_rect,
                x1=progress_end_date,
                y1=y1_progress_rect,
                fillcolor=darker_rgb_color,
                line=dict(width=0),
                layer="above", # Ensure it's on top of the task bar
                name=f"Progress: {task_name}",
                hoverinfo="skip" # Hover info handled by annotation/main bar
            )
            
            # Add progress percentage text
            if progress_percentage > 0:
                # Calculate luminance to decide text color (white or black)
                # Convert 'rgb(R,G,B)' string to R, G, B components
                r_val, g_val, b_val = [int(p) for p in darker_rgb_color.replace('rgb(', '').replace(')', '').split(',')]
                luminance = (0.299 * r_val + 0.587 * g_val + 0.114 * b_val) / 255
                text_color = "white" if luminance < 0.5 else "black" # Dark background -> white text, else black

                mid_progress_date = start_date + timedelta(seconds=progress_duration_seconds / 2)
                
                fig.add_annotation(
                    x=mid_progress_date,
                    y=y_index, # Centered vertically on the bar
                    text=f"{int(progress_percentage * 100)}%",
                    showarrow=False,
                    font=dict(
                        color=text_color,
                        size=12,      # Slightly larger font
                        weight='bold' # Make it bold
                    ),
                    yanchor="middle",
                    xanchor="center",
                    hovertext=f"Progress: {int(progress_percentage * 100)}% ({row['Name']})",
                    hoverlabel=dict(bgcolor=darker_rgb_color, font=dict(color=text_color))
                )

    # --- Add Milestone lines and icons ---
    # Milestones are added after tasks so they appear on top
    milestone_fixed_y_pos_top = y_range_top_for_full_line + 0.3 # Fixed Y-position for milestone icon/text
    
    for index, row in milestones_df_for_chart.iterrows():
        milestone_name = row['Name']
        milestone_date = row['Start']
        milestone_color = color_map.get(str(row['Color']).lower(), 'rgb(100, 100, 100)')
        
        milestone_icon_symbol = icon_map.get(str(row['Icon']).lower(), 'circle')
        associated_task_name = row['Associated_Task']

        # Determine the vertical extent of the milestone line
        line_y_start = y_range_top_for_full_line
        line_y_end = y_range_bottom_for_full_line
        
        # Default Y position for milestone icon/text
        milestone_y_position_for_icon_text = milestone_fixed_y_pos_top

        # If associated with a task, draw line from task to top and place icon/text above
        if associated_task_name and associated_task_name in task_to_y_index:
            task_y_index = task_to_y_index[associated_task_name]
            line_y_start = task_y_index - bar_actual_half_width # Start line from bottom of task bar
            line_y_end = y_range_top_for_full_line # Extend to the top of chart area
            milestone_y_position_for_icon_text = milestone_fixed_y_pos_top
        
        # Add milestone vertical line
        fig.add_trace(go.Scatter(
            x=[milestone_date, milestone_date],
            y=[line_y_end, line_y_start],
            mode='lines',
            line=dict(color=milestone_color, width=2, dash='dot'),
            name=milestone_name,
            showlegend=True,
            hoverinfo='text',
            text=f"Milestone: {milestone_name}<br>Date: {milestone_date.strftime('%d-%m-%Y')}" # Changed format here too
        ))

        # Add milestone icon marker
        if milestone_icon_symbol:
            fig.add_trace(go.Scatter(
                x=[milestone_date],
                y=[milestone_y_position_for_icon_text],
                mode='markers',
                marker=dict(
                    symbol=milestone_icon_symbol,
                    size=14,
                    color=milestone_color,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name=f"Icon {milestone_name}", # Name for hover, not legend
                showlegend=False,
                hoverinfo='text',
                text=f"Milestone: {milestone_name}<br>Date: {milestone_date.strftime('%d-%m-%Y')}" # Changed format here too
            ))

        # Add milestone annotation (text label)
        milestone_annotation_text = f"{milestone_name}<br>({milestone_date.strftime('%d-%m-%Y')})" # Changed format here too
        fig.add_annotation(
            x=milestone_date,
            y=milestone_y_position_for_icon_text,
            text=milestone_annotation_text,
            showarrow=False,
            yshift=25,
            font=dict(color=milestone_color, size=11),
            align="center",
        )

    # --- Add 'Today' line ---
    fig.add_trace(go.Scatter(
        x=[today, today],
        y=[y_range_top_for_full_line, y_range_bottom_for_full_line],
        mode='lines',
        line=dict(color='red', width=3, dash='solid'),
        name='We are Here',
        showlegend=True,
        hoverinfo='text',
        text=f"Today: {today.strftime('%d-%m-%Y')}" # Changed format here too
    ))

    # Add 'Today' annotation
    annotation_y_pos_today = y_range_top_for_full_line + 0.1 # Position above the tasks
    fig.add_annotation(
        x=today,
        y=annotation_y_pos_today,
        text=f"We are Here<br>({today.strftime('%d-%m-%Y')})", # Changed format here too
        showarrow=False,
        yshift=10,
        font=dict(color='red', size=12, weight='bold'),
        align="center",
    )

    return fig, chart_min_date, chart_max_date

# --- STREAMLIT USER INTERFACE ---
st.subheader("📝 Edit Project Items (Tasks & Milestones)")

progress_options = ['0%', '25%', '50%', '75%', '100%']
item_type_options = ['Task', 'Milestone']
icon_options = list(icon_map.keys())
color_name_options = list(color_map.keys())

default_start_date_new = datetime.now().date()
default_finish_date_new = datetime.now().date() + timedelta(days=7)

# Get current task names for Associated_Task dropdown
valid_task_names = st.session_state.project_items_df[
    st.session_state.project_items_df['Type'] == 'Task'
]['Name'].dropna().unique().tolist()
associated_task_options = [None] + sorted(valid_task_names)


project_items_column_config = {
    "Type": st.column_config.SelectboxColumn(
        "Type",
        options=item_type_options,
        required=True,
        default='Task',
        help="Select 'Task' or 'Milestone'."
    ),
    "Name": st.column_config.TextColumn("Name", required=True, help="Name of the Task or Milestone."),
    "Start": st.column_config.DateColumn(
        "Start Date",
        format="DD-MM-YYYY",
        required=True,
        default=default_start_date_new,
        help="Start date for tasks, or date for milestones."
    ),
    "Finish": st.column_config.DateColumn(
        "End Date",
        format="DD-MM-YYYY",
        required=True,
        default=default_finish_date_new,
        help="End date for tasks. For milestones, keep same as Start Date."
    ),
    "Resource": st.column_config.TextColumn(
        "Responsible/Resource",
        help="Persona o equipo responsable de la tarea. N/A para hitos.", # Updated help text
        default=""
    ),
    "Progress": st.column_config.SelectboxColumn(
        "Progress",
        options=progress_options,
        required=True,
        default='0%',
        help="Completion percentage for tasks. N/A for milestones (use 0%)."
    ),
    "Color": st.column_config.SelectboxColumn(
        "Color",
        options=color_name_options,
        required=True,
        default='blue',
        help="Color for the item bar/marker."
    ),
    "Icon": st.column_config.SelectboxColumn(
        "Icon",
        options=icon_options,
        default='none',
        help="Icon for milestones. Leave as 'none' for tasks."
    ),
    "Associated_Task": st.column_config.SelectboxColumn(
        "Associated Task (Milestone Only)",
        options=associated_task_options,
        required=False,
        default=None,
        help="If a milestone, associate it with a specific task."
    ),
}

edited_project_df = st.data_editor(
    st.session_state.project_items_df,
    num_rows="dynamic",
    column_config=project_items_column_config,
    use_container_width=True,
    key="project_items_editor" 
)

clean_and_update_df(edited_project_df)


# --- CHART RENDERING ---
st.subheader("Schedule Visualization")

gantt_fig, chart_min_date, chart_max_date = create_gantt_chart(st.session_state.project_items_df)

if not st.session_state.project_items_df.empty and chart_min_date is not None and chart_max_date is not None:
    st.markdown(f"<h3 style='text-align: center;'>Project Schedule<br><span style='font-size: 0.7em;'>{chart_min_date.strftime('%d-%m-%Y')} to {chart_max_date.strftime('%d-%m-%Y')}</span></h3>", unsafe_allow_html=True)
else:
    st.markdown("<h3 style='text-align: center;'>No tasks or milestones to display. Add items above.</h3>", unsafe_allow_html=True)

st.plotly_chart(gantt_fig, use_container_width=True)

# --- DOWNLOAD OPTIONS ---
st.subheader("⬇️ Download Chart")

html_bytes = gantt_fig.to_html().encode('utf-8')

col_down1, col_down2 = st.columns(2)

with col_down1:
    st.download_button(
        label="Download as Interactive HTML",
        data=html_bytes,
        file_name="gantt_chart.html",
        mime="text/html"
    )