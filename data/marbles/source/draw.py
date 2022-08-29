import math

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import pandas as pd 

import cairo

stimuli_counter = 0
stimuli = []
WIDTH, HEIGHT = 256, 256

marble_colors = pd.DataFrame()
from scipy.stats import dirichlet

all_colors = []
CURRENT_SET = "stimuli"
# r,g,b
alpha = [1, 2, 3]
seed = 0 
probability = dirichlet.rvs(alpha, size=1, random_state=1)[0]
while(len(all_colors) < 100):
    np.random.seed(seed)
    colors = np.random.choice(3, 9, p=probability)
    seed += 1

    if(len(all_colors) > 0 and any(np.array_equal(colors, i) for i in all_colors)):
        seed += 1
        continue 

    all_colors.append(colors)
    marble_colors = marble_colors.append({"colors":colors.tolist()}, ignore_index=True)

    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, WIDTH, HEIGHT)
    ctx = cairo.Context(surface)
    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.set_source_rgba(1, 1, 1, 1)
    ctx.rectangle(0, 0, WIDTH, HEIGHT)
    ctx.fill()

    xys = [ [WIDTH//2, HEIGHT//2], 
            [WIDTH//5, HEIGHT//5],
            [WIDTH//2, HEIGHT//5],
            [WIDTH - 50, HEIGHT//5],
            [WIDTH//5, HEIGHT//2],
            [WIDTH//2, HEIGHT - 50],
            [WIDTH - 50, HEIGHT//2],
            [WIDTH - 50, HEIGHT - 50],
            [WIDTH//5, HEIGHT - 50],
            ]

    for color_index, xy in enumerate(xys):
        color = colors[color_index]
        x,y = xy

        # orange, green, purple 
        # 1, 2, 3: proportions 
        #colors = np.array([[128,64,0], [64,128,0], [64,0,128]])

        orange = 0
        purple = 2
        green = 1 
        

        ctx.set_line_width(9)
        if color == orange:
            #ctx.set_source_rgb(256, 0, 0)
            #ctx.set_source_rgb(128,64,0)
            ctx.set_source_rgb(0.75, 0.5, 0)
        elif color == green:
            #ctx.set_source_rgb(0, 256, 0)
            #ctx.set_source_rgb(64,128,0)
            ctx.set_source_rgb(0.5, 0.75, 0)
        else:   
            #ctx.set_source_rgb(0, 0, 256) 
            #ctx.set_source_rgb(64,0,128)
            ctx.set_source_rgb(0.5, 0, 0.75)

        ctx.arc(x,y, 30, 0, 2*math.pi)
        ctx.stroke_preserve()

        if color == orange:
            #ctx.set_source_rgb(0.8, 0.4, 0.4)
            ctx.set_source_rgb(0.5, 0.25, 0)
        elif color == green:
            #ctx.set_source_rgb(0.4, 0.8, 0.4)
            ctx.set_source_rgb(0.25, 0.5, 0)
        else:   
            #ctx.set_source_rgb(0.4, 0.4, 0.8)
            ctx.set_source_rgb(0.25, 0, 0.5)
        ctx.fill()

    ctx.set_line_width(5)
    ctx.set_source_rgb(0,0,0)
    ctx.rectangle(10, 10, WIDTH-20, HEIGHT-20)   
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.stroke_preserve()

    surface.write_to_png("./" + CURRENT_SET  + "/stimulus" + str(stimuli_counter) + ".png")  # Output to PNG
    stimuli_counter += 1



marble_colors.to_csv("./colors.csv")
print(marble_colors)

for vec_a_idx, vec_a in enumerate(marble_colors):
    for vec_b_idx, vec_b in enumerate(marble_colors):
        if(np.array_equal(vec_a, vec_b) and vec_a_idx != vec_b_idx):
            print("equal")