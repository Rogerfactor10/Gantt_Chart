import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gantt Project Planner",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Project Timeline and Gantt Chart Tool")
st.write("Create, edit, and visualize your project schedule. Add tasks, milestones, and track their progress.")

# --- SESSION STATE INITIALIZATION ---
if 'tasks_df' not in st.session_state:
    st.session_state.tasks_df = pd.DataFrame([
        {'Task': 'Requirements Analysis', 'Start': '2025-01-01', 'Finish': '2025-01-15', 'Resource': 'Analyst'},
        {'Task': 'UI/UX Design', 'Start': '2025-01-16', 'Finish': '2025-02-10', 'Resource': 'Designer'},
        {'Task': 'Backend Development', 'Start': '2025-02-11', 'Finish': '2025-04-10', 'Resource': 'Backend'},
        {'Task': 'Frontend Development', 'Start': '2025-02-11', 'Finish': '2025-04-20', 'Resource': 'Frontend'},
        {'Task': 'Testing & QA', 'Start': '2025-04-21', 'Finish': '2025-05-10', 'Resource': 'QA'},
        {'Task': 'Deployment', 'Start': '2025-05-11', 'Finish': '2025-05-15', 'Resource': 'DevOps'}
    ])
    st.session_state.tasks_df['Start'] = pd.to_datetime(st.session_state.tasks_df['Start'])
    st.session_state.tasks_df['Finish'] = pd.to_datetime(st.session_state.tasks_df['Finish'])

if 'milestones_df' not in st.session_state:
    st.session_state.milestones_df = pd.DataFrame([
        {'Milestone': 'Project Kick-off', 'Date': '2025-01-01', 'Color': 'red', 'Icon': 'star', 'Associated_Task': 'Requirements Analysis'},
        {'Milestone': 'Prototype Approved', 'Date': '2025-02-10', 'Color': 'green', 'Icon': 'circle', 'Associated_Task': 'UI/UX Design'},
        {'Milestone': 'Lanzamiento Beta', 'Date': '2025-05-15', 'Color': 'blue', 'Icon': 'square', 'Associated_Task': 'Deployment'},
        {'Milestone': 'Hito Sin Tarea', 'Date': '2025-03-01', 'Color': 'orange', 'Icon': 'diamond', 'Associated_Task': None}
    ])
    st.session_state.milestones_df['Date'] = pd.to_datetime(st.session_state.milestones_df['Date'])

# --- Mapping of color names and icons to Plotly formats (GLOBAL) ---
color_map = {
    'red': 'rgb(255, 0, 0)',
    'green': 'rgb(0, 128, 0)',
    'blue': 'rgb(0, 0, 255)',
    'orange': 'rgb(255, 165, 0)',
    'purple': 'rgb(128, 0, 128)',
    'black': 'rgb(0, 0, 0)'
}

# Plotly marker symbols: https://plotly.com/python/marker-symbols/
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

def create_gantt_chart(tasks_df, milestones_df):
    today = datetime.now().date()

    # Default Y-range for empty charts (these are visual positions, not categories)
    DEFAULT_Y_MIN_EMPTY = -0.5 
    DEFAULT_Y_MAX_EMPTY = 1.5 
    
    # Margin to extend beyond the task bars for the full height line
    Y_AXIS_BUFFER_TOP = 1.0 # Extra space at the visual top for milestones/WeAreHere
    Y_AXIS_BUFFER_BOTTOM = 0.5 # Extra space at the visual bottom

    # --- Handling Empty Task DataFrame Separately ---
    if tasks_df.empty:
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
                    # --- AQUI EL CAMBIO PARA MOVER EL RANAGESELECTOR ---
                    y=1.1,  # Ajusta este valor. 1.0 es el tope del área de trazado. 1.1 lo sube más.
                    yanchor='bottom' # Ancla la parte inferior del rangeselector a la coordenada y
                    # ----------------------------------------------------
                ),
                rangeslider=dict(visible=False),
                type="date"
            ),
            yaxis=dict(
                visible=False,
                autorange="reversed", 
                range=[y_range_top_for_full_line_empty, y_range_bottom_for_full_line_empty] 
            ),
            showlegend=False,
        )
        
        if not milestones_df.empty:
             for index, row in milestones_df.iterrows():
                milestone_date = row['Date']
                milestone_color = color_map.get(row['Color'].lower(), 'rgb(100, 100, 100)')
                milestone_icon_symbol = icon_map.get(row['Icon'].lower(), 'circle')
                
                fig.add_trace(go.Scatter(
                    x=[milestone_date, milestone_date],
                    y=[y_range_top_for_full_line_empty, y_range_bottom_for_full_line_empty], 
                    mode='lines',
                    line=dict(color=milestone_color, width=2, dash='dot'),
                    name=row['Milestone'],
                    showlegend=True,
                    hoverinfo='text',
                    text=f"{row['Milestone']}<br>{milestone_date.strftime('%Y-%m-%d')}"
                ))

                milestone_y_pos_icon_text_empty = y_range_top_for_full_line_empty + 0.1 
                
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
                        name=f"Icon {row['Milestone']}",
                        showlegend=False,
                        hoverinfo='text',
                        text=f"Milestone: {row['Milestone']}<br>Date: {milestone_date.strftime('%Y-%m-%d')}"
                    ))
                
                milestone_annotation_text = f"{row['Milestone']}<br>({milestone_date.strftime('%Y-%m-%d')})"
                fig.add_annotation(
                    x=milestone_date,
                    y=milestone_y_pos_icon_text_empty, 
                    text=milestone_annotation_text,
                    showarrow=False,
                    yshift=25, 
                    font=dict(color=milestone_color, size=11),
                    align="center",
                )
        
        # "We are Here" line always added for empty chart
        fig.add_trace(go.Scatter(
            x=[today, today],
            y=[y_range_top_for_full_line_empty, y_range_bottom_for_full_line_empty], 
            mode='lines',
            line=dict(color='red', width=3, dash='solid'),
            name='We are Here',
            showlegend=True,
            hoverinfo='text',
            text=f"Today: {today.strftime('%Y-%m-%d')}"
        ))
        
        annotation_y_pos_today_empty = y_range_top_for_full_line_empty - 0.2 
        fig.add_annotation(
            x=today,
            y=annotation_y_pos_today_empty, 
            text=f"We are Here<br>({today.strftime('%Y-%m-%d')})",
            showarrow=False,
            yshift=10,
            font=dict(color='red', size=12, weight='bold'),
            align="center",
        )
        return fig, None, None

    # --- If tasks_df is NOT empty, proceed with Gantt Chart creation ---

    current_tasks_df = st.session_state.tasks_df.sort_values(by=['Start', 'Task'], ascending=[True, True]).copy()

    resource_base_colors = {}
    unique_resources = current_tasks_df['Resource'].unique()
    for i, res in enumerate(unique_resources):
        r = (i * 50) % 255
        g = (i * 100) % 255
        b = (i * 150) % 255
        resource_base_colors[res] = f'rgb({r},{g},{b})'

    task_colors = {}
    for _, row in current_tasks_df.iterrows():
        task_name = row['Task']
        resource_name = row['Resource']
        task_colors[task_name] = resource_base_colors.get(resource_name, 'rgb(100,100,100)') 

    min_date = current_tasks_df['Start'].min()
    max_date = current_tasks_df['Finish'].max()
    
    fig = ff.create_gantt(
        current_tasks_df.to_dict('records'), 
        index_col='Task', 
        colors=task_colors, 
        group_tasks=False, 
        showgrid_x=True,
        showgrid_y=True,
    )

    if hasattr(fig.layout.yaxis, 'categoryarray') and fig.layout.yaxis.categoryarray is not None:
        plotly_y_categories = fig.layout.yaxis.categoryarray
    else:
        plotly_y_categories = [] 

    num_categories = len(plotly_y_categories)
    
    y_plot_min_for_tasks = -0.5 # Numerical Y for the top of the first task bar
    y_plot_max_for_tasks = (num_categories - 1) + 0.5 if num_categories > 0 else -0.5 

    y_range_top_for_full_line = y_plot_min_for_tasks - Y_AXIS_BUFFER_TOP 
    y_range_bottom_for_full_line = y_plot_max_for_tasks + Y_AXIS_BUFFER_BOTTOM 
    
    fig.update_layout(
        yaxis=dict(
            autorange="reversed", 
            categoryorder='array', 
            categoryarray=plotly_y_categories, 
            range=[y_range_top_for_full_line, y_range_bottom_for_full_line]
        ),
        xaxis_title="Date",
        yaxis_title="Tasks",
        hovermode='x unified',
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
                # --- AQUI EL CAMBIO PARA MOVER EL RANAGESELECTOR ---
                y=1.1,  # Ajusta este valor. 1.0 es el tope del área de trazado. 1.1 lo sube más.
                yanchor='bottom' # Ancla la parte inferior del rangeselector a la coordenada y
                # ----------------------------------------------------
            ),
            rangeslider=dict(visible=False),
            type="date"
        )
    )

    # --- Add Milestones (similar to "We are Here") ---
    milestone_fixed_y_pos = y_range_top_for_full_line + 0.3 

    if not milestones_df.empty: 
        for index, row in milestones_df.iterrows():
            milestone_date = row['Date']
            milestone_color = color_map.get(row['Color'].lower(), 'rgb(100, 100, 100)')
            milestone_icon_symbol = icon_map.get(row['Icon'].lower(), 'circle')
            
            fig.add_trace(go.Scatter(
                x=[milestone_date, milestone_date],
                y=[y_range_top_for_full_line, y_range_bottom_for_full_line], 
                mode='lines',
                line=dict(color=milestone_color, width=2, dash='dot'),
                name=row['Milestone'],
                showlegend=True,
                hoverinfo='text',
                text=f"{row['Milestone']}<br>{milestone_date.strftime('%Y-%m-%d')}"
            ))

            if milestone_icon_symbol:
                fig.add_trace(go.Scatter(
                    x=[milestone_date],
                    y=[milestone_fixed_y_pos], 
                    mode='markers',
                    marker=dict(
                        symbol=milestone_icon_symbol,
                        size=14,
                        color=milestone_color,
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    name=f"Icon {row['Milestone']}",
                    showlegend=False,
                    hoverinfo='text',
                    text=f"Milestone: {row['Milestone']}<br>Date: {milestone_date.strftime('%Y-%m-%d')}"
                ))
            
            milestone_annotation_text = f"{row['Milestone']}<br>({milestone_date.strftime('%Y-%m-%d')})"
            
            fig.add_annotation(
                x=milestone_date,
                y=milestone_fixed_y_pos, 
                text=milestone_annotation_text,
                showarrow=False, 
                yshift=20, 
                font=dict(color=milestone_color, size=11),
                align="center",
            )

    # "We are Here" line
    fig.add_trace(go.Scatter(
        x=[today, today],
        y=[y_range_top_for_full_line, y_range_bottom_for_full_line], 
        mode='lines',
        line=dict(color='red', width=3, dash='solid'), 
        name='We are Here',
        showlegend=True,
        hoverinfo='text',
        text=f"Today: {today.strftime('%Y-%m-%d')}"
    ))

    annotation_y_pos_today = y_range_top_for_full_line + 0.1 
    
    fig.add_annotation(
        x=today,
        y=annotation_y_pos_today,
        text=f"We are Here<br>({today.strftime('%Y-%m-%d')})",
        showarrow=False, 
        yshift=10, 
        font=dict(color='red', size=12, weight='bold'),
        align="center",
    )
    
    return fig, min_date, max_date

# --- STREAMLIT USER INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Edit Tasks")
    edited_tasks = st.data_editor(
        st.session_state.tasks_df,
        num_rows="dynamic",
        column_config={
            "Task": st.column_config.TextColumn("Task", required=True),
            "Start": st.column_config.DateColumn("Start Date", format="YYYY-MM-DD", required=True),
            "Finish": st.column_config.DateColumn("End Date", format="YYYY-MM-DD", required=True),
            "Resource": st.column_config.TextColumn("Responsible/Resource", required=True),
        },
        use_container_width=True
    )
    if edited_tasks is not None:
        st.session_state.tasks_df = edited_tasks


with col2:
    st.subheader("🚩 Edit Milestones")
    task_options = st.session_state.tasks_df['Task'].tolist()
    task_options.insert(0, None) 
    
    edited_milestones = st.data_editor(
        st.session_state.milestones_df,
        num_rows="dynamic",
        column_config={
            "Milestone": st.column_config.TextColumn("Milestone", required=True),
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
            "Color": st.column_config.SelectboxColumn("Color", options=list(color_map.keys())),
            "Icon": st.column_config.SelectboxColumn("Icon", options=list(icon_map.keys()),
                                                    help="Select an icon for the milestone"),
            "Associated_Task": st.column_config.SelectboxColumn(
                "Associated Task", 
                options=task_options, 
                required=False, 
                help="Select a task to associate this milestone with. Leave blank for a general milestone."
            )
        },
        use_container_width=True
    )
    if edited_milestones is not None:
        st.session_state.milestones_df = edited_milestones

# --- CHART RENDERING ---
st.subheader("Schedule Visualization")

gantt_fig, chart_min_date, chart_max_date = create_gantt_chart(st.session_state.tasks_df, st.session_state.milestones_df)

if not st.session_state.tasks_df.empty:
    st.markdown(f"<h3 style='text-align: center;'>Project Schedule<br><span style='font-size: 0.7em;'>{chart_min_date.strftime('%Y-%m-%d')} to {chart_max_date.strftime('%Y-%m-%d')}</span></h3>", unsafe_allow_html=True)
else:
    st.markdown("<h3 style='text-align: center;'>No tasks to display. Add tasks above.</h3>", unsafe_allow_html=True)


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