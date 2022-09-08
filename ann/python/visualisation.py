import pyglet
from pyglet.window import key, mouse
from pyglet import shapes

import numpy as np
import math 
from ann import ANN, ANN_Hidden_Layer
from ann_af import ANN_Sigmoid_Activation

sa  = ANN_Sigmoid_Activation()
ann = ANN(2, 2, sa)
ann.add_hidden_layer(ANN_Hidden_Layer(2, 2, sa))
#ann.add_hidden_layer(ANN_Hidden_Layer(10, 10, sa))
#ann.add_hidden_layer(ANN_Hidden_Layer(10, 10, sa))
#ann.set_biases_vector([[0, 0]])
#ann.set_weight_matrix([[1, 1],
#                       [1, 1],
#                       [1, 1],
#                       [1, 1]])
print(ann)

print("starting...")

window = pyglet.window.Window(resizable=True)#, fullscreen=True)
batch = pyglet.graphics.Batch()
window.set_caption("ANN Visualisation")

class ANNVisualisation:
    """ A class responsible for the visualisation of the ANN itself.
    """
    def __init__(self, ann, window, batch):
        self.margin = 150
        self.layer_spacing = 100
        self.node_radius = 25
        self.window_width = window.get_size()[0]
        self.window_height = window.get_size()[1]
        
        self.ann = ann
        self.window = window
        self.batch = batch

    def _line_width(self, v, max):
        return 10*abs(v)/max + 1

    def _line_color(self, v):
        if v > 0:
            return (200, 20, 20)
        return (20, 20, 200)

    def draw_ann(self):
        self.weight_lines = []
        if len(ann.hidden_layers) > 0:
            for h in range(len(self.ann.hidden_layers)):
                sw = (self.window_width - 2*self.margin) / (len(self.ann.hidden_layers) + 1)
                hl = self.ann.hidden_layers[h]

                hidden_weight_lines = []
                for x in range(hl.num_input_nodes):
                    si = (self.window_height - 2*self.margin) / (hl.num_input_nodes - 1)
                    for y in range(hl.num_hidden_nodes):
                        so = (self.window_height - 2*self.margin) / (hl.num_hidden_nodes - 1)
                        m = max(abs(hl.Wh.min()), hl.Wh.max())
                        line_width = self._line_width(hl.Wh[x,y], m)
                        line_color = self._line_color(hl.Wh[x,y])
                        hidden_weight_lines.append( shapes.Line(self.margin+(h)*sw, self.margin+x*si, self.margin+(h+1)*sw, self.margin+y*so, width=line_width, color=line_color, batch=self.batch) )

                self.weight_lines.append(hidden_weight_lines)
        else:
            print("NOT IMPLEMENTED YET!")

        self.weight_lines_output = []
        if len(self.ann.hidden_layers) > 0:
            h  = len(self.ann.hidden_layers) - 1
            hl = self.ann.hidden_layers[h] # last hidden layer
            sw = (self.window_width - 2*self.margin) / (len(self.ann.hidden_layers) + 1)
            for x in range(hl.num_hidden_nodes):
                si = (self.window_height - 2*self.margin) / (hl.num_hidden_nodes - 1)

                output_weight_lines = []
                for y in range(ann.num_output_nodes):
                    so = (self.window_height - 2*self.margin) / (self.ann.num_output_nodes - 1)
                    m = max(abs(self.ann.Wy.min()), self.ann.Wy.max())
                    line_width = self._line_width(self.ann.Wy[x,y], m)
                    line_color = self._line_color(self.ann.Wy[x,y])
                    output_weight_lines.append( shapes.Line(self.margin+(h+1)*sw, self.margin+x*si, self.margin+(h+2)*sw, self.margin+y*so, width=line_width, color=line_color, batch=self.batch) )

                self.weight_lines_output.append(output_weight_lines)   

        else:
            print("NOT IMPLEMENTED YET!")

        # draw input nodes
        self.input_nodes = []
        for i in range(self.ann.num_input_nodes):
            s = (self.window_height - 2*self.margin) / (self.ann.num_input_nodes - 1)
            self.input_nodes.append( shapes.Circle(self.margin, self.margin+i*s, self.node_radius, color=(50, 225, 30), batch=self.batch) )

        #draw output nodes
        self.output_nodes = []
        for i in range(ann.num_output_nodes):
            s = (self.window_height - 2*self.margin) / (self.ann.num_output_nodes - 1)
            self.output_nodes.append( shapes.Circle(self.window_width-self.margin, self.margin+i*s, self.node_radius, color=(50, 225, 30), batch=self.batch) )

        #draw hidden layers
        self.hidden_nodes = []
        for h in range(len(self.ann.hidden_layers)):
            sw = (self.window_width - 2*self.margin) / (len(self.ann.hidden_layers) + 1)
            hl = self.ann.hidden_layers[h]

            hidden_layer_nodes = []
            for i in range(hl.num_hidden_nodes):
                s = (self.window_height - 2*self.margin) / (hl.num_hidden_nodes - 1)
                hidden_layer_nodes.append( shapes.Circle(self.margin+(h+1)*sw, self.margin+i*s, self.node_radius, color=(50, 225, 30), batch=self.batch) )

            self.hidden_nodes.append(hidden_layer_nodes)

    def mouse_click(self, x, y):
        #print("x: " + x + ", y: " + y)
        #self.weight_lines
        #self.weight_lines_output
        #self.input_nodes
        #self.output_nodes
        #self.hidden_nodes
        for n in self.input_nodes:
            n.color = (50, 225, 30)
            if x > (n.x - (n.radius/2)) and x < (n.x + (n.radius/2)) and y > (n.y - (n.radius/2)) and y < (n.y + (n.radius/2)):
                n.color = (255, 30, 30)

    def mouse_move(self, x, y):
        #print("x: " + x + ", y: " + y)
        #self.weight_lines
        #self.weight_lines_output
        #self.input_nodes
        #self.output_nodes
        #self.hidden_nodes
        for n in self.input_nodes:
            n.color = (50, 225, 30)
            if x > (n.x - n.radius) and x < (n.x + n.radius) and y > (n.y - n.radius) and y < (n.y + n.radius):
                n.color = (255, 30, 30)
        for n in self.output_nodes:
            n.color = (50, 225, 30)
            if x > (n.x - n.radius) and x < (n.x + n.radius) and y > (n.y - n.radius) and y < (n.y + n.radius):
                n.color = (255, 30, 30)
        for layer_nodes in self.hidden_nodes:
            for n in layer_nodes:
                n.color = (50, 225, 30)
                if x > (n.x - n.radius) and x < (n.x + n.radius) and y > (n.y - n.radius) and y < (n.y + n.radius):
                    n.color = (255, 30, 30)
        for ln in range(len(self.weight_lines)):
            lines = self.weight_lines[ln]
            for i in range(len(lines)):
                l = lines[i]
                print("(%d) %d => line(%d,%d,%d,%d)" % (ln, i, l.x, l.y, l.x2, l.y2) )
                hl = self.ann.hidden_layers[ln] # get the hidden layer
                hx = i % hl.num_hidden_nodes
                hy = int(i / hl.num_hidden_nodes)
                print("Number of nodes: %d" % hl.num_hidden_nodes)
                print("X: %d, y: %d" % (hx, hy))
                m = max(abs(hl.Wh.min()), hl.Wh.max())
                l.color = self._line_color(hl.Wh[hx,hy])
                l.width = self._line_width(hl.Wh[hx,hy], m)
                if x > l.x and x < l.x2:# and y > l.y and y < l.y2:
                    dx = l.x2 - l.x
                    dy = l.y2 - l.y + 0.0000001
                    fa = dy/dx
                    ga = -dx/dy
                    fb = l.y - (fa*l.x)
                    gb = y - (ga*x)
                    sx = (gb - fb) / (fa - ga)
                    sy = fa*sx + fb
                    distance = math.sqrt( (x-sx)**2 + (y-sy)**2 )
                    print("distance: " + str(distance))
                    if distance < 10:
                        l.color = (0, 255, 0)
        for ln in range(len(self.weight_lines_output)):
            lines = self.weight_lines_output[ln]
            for i in range(len(lines)):
                l = lines[i]
                print("(%d) %d => line(%d,%d,%d,%d)" % (ln, i, l.x, l.y, l.x2, l.y2) )
                hx = i % self.ann.num_output_nodes
                hy = int(i / self.ann.num_output_nodes)
                print("Number of nodes: %d" % self.ann.num_output_nodes)
                print("X: %d, y: %d" % (hx, hy))
                m = max(abs(self.ann.Wy.min()), self.ann.Wy.max())
                l.width = self._line_width(self.ann.Wy[hx,hy], m)
                l.color = self._line_color(self.ann.Wy[hx,hy])
                if x > l.x and x < l.x2:# and y > l.y and y < l.y2:
                    dx = l.x2 - l.x
                    dy = l.y2 - l.y + 0.0000001
                    fa = dy/dx
                    ga = -dx/dy
                    fb = l.y - (fa*l.x)
                    gb = y - (ga*x)
                    sx = (gb - fb) / (fa - ga)
                    sy = fa*sx + fb
                    distance = math.sqrt( (x-sx)**2 + (y-sy)**2 )
                    print("distance: " + str(distance))
                    if distance < 10:
                        l.color = (0, 255, 0)


#circle = shapes.Circle(700, 150, 100, color=(50, 225, 30), batch=batch)
#square = shapes.Rectangle(200, 200, 200, 200, color=(55, 55, 255), batch=batch)
#rectangle = shapes.Rectangle(250, 300, 400, 200, color=(255, 22, 20), batch=batch)
#rectangle.opacity = 128
#rectangle.rotation = 33
#line = shapes.Line(100, 100, 100, 200, width=19, batch=batch)
#line2 = shapes.Line(150, 150, 444, 111, width=4, color=(200, 20, 20), batch=batch)
#star = shapes.Star(800, 400, 60, 40, num_spikes=20, color=(255, 255, 0), batch=batch)

ann_visualisation = ANNVisualisation(ann, window, batch)
ann_visualisation.draw_ann()

@window.event
def on_resize(width, height):
    print('The window was resized to %dx%d' % (width, height))
    windows_width = 640
    window_height = 480

@window.event
def on_key_press(symbol, modifiers):
    #print("press" + str(symbol))
    pass

@window.event
def on_key_release(symbol, modifiers):
    #print("release: " + str(symbol))
    pass

@window.event
def on_mouse_motion(x, y, dx, dy):
    ann_visualisation.mouse_move(x, y)

@window.event
def on_mouse_press(x, y, button, modifiers):
    if button & pyglet.window.mouse.LEFT:
        ann_visualisation.mouse_click(x, y)

@window.event
def on_mouse_release(x, y, button, modifiers):
    pass

@window.event
def on_draw():
    window.clear()
    batch.draw()

pyglet.app.run()

