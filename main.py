import cv2
import itertools
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import datetime
import tkinter as tk

import glass as g
import mustache as m 
import oil_paint as o 
import dog_filter as d 
import fire_eyes as f 
import brightener as b 
import bw 
import blur as bl 

#g.apply_glass()
#m.apply_mustache()
#o.apply_filter()
#d.apply_dogfilter()
#f.apply_fire_eyes()
#b.adjust_brightness()
#bw.apply_bw()
#bl.apply_blur()



root = tk.Tk()
root.title("SNAPCHAT_FILTERS")

# Create buttons
button1 = tk.Button(root, text="GLASSES", command=g.apply_glass)
button1.pack()

button2 = tk.Button(root, text="MUSTACHE", command=m.apply_mustache)
button2.pack()

button3 = tk.Button(root, text="OIL_PAINT", command=o.apply_oilpaint)
button3.pack()

button4 = tk.Button(root, text="DOG_FILTER", command=d.apply_dogfilter)
button4.pack()

button5 = tk.Button(root, text="FIRE_EYES", command=f.apply_fire_eyes)
button5.pack()

button6 = tk.Button(root, text="BRIGHTNER", command=b.apply_brighten)
button6.pack()

button7 = tk.Button(root, text="BW", command=bw.apply_bw)
button7.pack()

button8 = tk.Button(root, text="BLUR", command=bl.apply_blur)
button8.pack()

# Start the main event loop
root.mainloop()



