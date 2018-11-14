import Tkinter as tk

import collections

color = Color = collections.namedtuple('Color', ('r', 'g', 'b'))

def hex(clr):
    return '#{:02x}{:02x}{:02x}'.format(*clr)

class Canvas(object):
    
    def __init__(self, w, h):
        self.win = tk.Tk()
        self.ctx = tk.Canvas(self.win, width=w, height=h)
        self.ctx.pack()
    
    def line(self, x1, y1, x2, y2, stroke=Color(0,0,0), strokewidth=1):
        self.ctx.create_line(x1, y1, x2, y2, fill=hex(stroke), width=strokewidth)

    def rect(self, x, y, w, h):
        self.ctx.create_rectangle(x, y, w, h)

    def run(self):
        self.win.mainloop()
        #self.win.update()
        
    def save(self):
        self.ctx.postscript(file="test.eps")

#w.create_line(0, 100, 200, 0, fill="red", dash=(4, 4))
#w.create_rectangle(50, 25, 150, 75, fill="blue")

canvas = Canvas(400, 400)
canvas.line(0, 0, 100, 200, stroke=Color(255,0,0), strokewidth=0.5)
canvas.rect(10, 10, 30, 40)
canvas.run()
canvas.save()
