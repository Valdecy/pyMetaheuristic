############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Graphs #from ..utils import *

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import itertools
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio           

from scipy.sparse.linalg import svds

############################################################################

# Function: Solution Plot 
def plot_single_function(min_values, max_values, target_function, step = [0.1, 0.1], solution = [ ], proj_view = '3D', view = 'browser'):
    if (view == 'browser' ):
        pio.renderers.default = 'browser'
    x    = []
    data = []
    for j in range(0, len(min_values)):
        values = np.arange(min_values[j], max_values[j], step[j])
        x.append(values)
    cartesian_product = list(itertools.product(*x)) 
    front             = np.array(cartesian_product, dtype = np.dtype('float'))
    front             = np.c_[ front, np.zeros(len(cartesian_product)) ]
    value             = [target_function(item) for item in cartesian_product]
    front[:, -1]      = value
    nid_list          = [ 'f(x) = '+str(round(item, 4)) for item in value] 
    if (proj_view == '1D' or proj_view == '1d' or proj_view == '2D' or proj_view == '2d'):
        if (len(solution) > 0):
            sol     = np.array(solution)
            c_sol   = ['f(x) = '+str(round(target_function(item), 2)) for item in sol]
            if ( (proj_view == '1D' or proj_view == '1d') and len(min_values) > 1):
                if (sol.shape[0] == 1):
                    sol = np.vstack([sol, sol])
                sol, _, _ = svds(sol, k = 1)
            if ( (proj_view == '2D' or proj_view == '2d') and len(min_values) > 2):
                if (sol.shape[0] == 1):
                    sol = np.vstack([sol, sol])
                sol, _, _ = svds(sol, k = 2)
            s_trace = go.Scatter(x         = sol[:, 0],
                                 y         = sol[:, 1],
                                 opacity   = 1,
                                 mode      = 'markers+text',
                                 marker    = dict(symbol = 'circle-dot', size = 10, color = 'red'),
                                 hovertext = c_sol,
                                 name      = ''
                                 )
            data.append(s_trace)
        if ( len(min_values) > 2):
            sol, _, _  = svds(front[:,:-1], k = 2)
            front[:,0] = sol[:,0]
            front[:,1] = sol[:,1]
        n_trace = go.Scatter(x         = front[:, 0],
                             y         = front[:, 1],
                             opacity   = 0.5,
                             mode      = 'markers+text',
                             marker    = dict(symbol = 'circle-dot', size = 5, color = -front[:,-1]),
                             hovertext = nid_list,
                             name      = ''
                             )
        data.append(n_trace)
        layout  = go.Layout(showlegend   = False,
                            hovermode    = 'closest',
                            margin       = dict(b = 10, l = 5, r = 5, t = 10),
                            plot_bgcolor = 'white',
                            xaxis        = dict(  showgrid       = False, 
                                                  zeroline       = False, 
                                                  showticklabels = True, 
                                                  tickmode       = 'array', 
                                               ),
                            yaxis        = dict(  showgrid       = False, 
                                                  zeroline       = False, 
                                                  showticklabels = True,
                                                  tickmode       = 'array', 
                                                )
                            )
        fig_aut = go.Figure(data = data, layout = layout)
        fig_aut.update_traces(textfont_size = 10, textfont_color = 'white') 
        fig_aut.show() 
    elif (proj_view == '3D' or proj_view == '3d'):
        if (len(solution) > 0):
            sol   = np.array(solution)
            c_val = [target_function(item) for item in sol]
            if ( len(min_values) > 2):
                if (sol.shape[0] == 1):
                    sol = np.vstack([sol, sol])
                sol, _, _ = svds(sol, k = 2)
            s_trace = go.Scatter3d(x         = sol[:, 0],
                                   y         = sol[:, 1],
                                   z         = c_val,
                                   opacity   = 1,
                                   mode      = 'markers+text',
                                   marker    = dict(size = 10, color = 'red'),
                                   name      = ''
                                   )
            data.append(s_trace)
        if ( len(min_values) > 2):
            sol, _, _  = svds(front[:,:-1], k = 2)
            front[:,0] = sol[:,0]
            front[:,1] = sol[:,1]
        n_trace = go.Scatter3d(x         = front[:, 0],
                               y         = front[:, 1],
                               z         = front[:,-1],
                               opacity   = 0.5,
                               mode      = 'markers+text',
                               marker    = dict(size = 5, color = -front[:,-1]),
                               name      = ''
                               )
        data.append(n_trace)
        layout  = go.Layout(showlegend   = False,
                            hovermode    = 'closest',
                            margin       = dict(b = 10, l = 5, r = 5, t = 10),
                            plot_bgcolor = 'white',
                            )
        fig_aut = go.Figure(data = data, layout = layout)
        fig_aut.update_traces(textfont_size = 10, textfont_color = 'white') 
        fig_aut.update_scenes(xaxis_visible = False, yaxis_visible = False, zaxis_visible = False)
        fig_aut.show() 
    else:
        dict_list = []
        color_lst = [0]*front.shape[0]  
        if (len(solution) > 0):
            for i in range(0, len(solution)):
                sol   = np.array(solution[i])
                front = np.r_[front, np.zeros((1, front.shape[1])) ]
                color_lst.append(1)
                for j in range(0, sol.shape[0]):
                    front[-1, j] = sol[j]     
        for j in range(0, len(min_values)):
            dict_list.append(dict(range = [min_values[j]*1.00, max_values[j]*1.00], label = 'x'+str(j+1), values = front[:,j]))
        lines = go.Parcoords(
                             line       = dict(color = color_lst, colorscale = [[0,'lightblue'], [0.5,'lightblue'], [1,'red']]),
                             dimensions = dict_list
                            )
        fig_aut = go.Figure(data = lines)
        fig_aut.update_layout(font = dict(family = 'Arial Black', size = 15, color = 'black'))
        fig_aut.show()
    return

############################################################################
