import pyglet
from pyglet.window import key, mouse
from pyglet import shapes

import numpy as np
import math 
from ann import ANN, ANN_Hidden_Layer
from ann_af import ANN_Sigmoid_Activation

sa  = ANN_Sigmoid_Activation()
ann = ANN(4, 2, sa)
ann.add_hidden_layer(ANN_Hidden_Layer(4, 4, sa))
ann.add_hidden_layer(ANN_Hidden_Layer(4, 4, sa))
#ann.add_hidden_layer(ANN_Hidden_Layer(10, 10, sa))
#ann.set_biases_vector([[0, 0]])
#ann.set_weight_matrix([[1, 1],
#                       [1, 1],
#                       [1, 1],
#                       [1, 1]])
print(ann)

print("starting...")

window = pyglet.window.Window(resizable=True, fullscreen=True)
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
        self.input_labels = []
        for i in range(self.ann.num_input_nodes):
            s = (self.window_height - 2*self.margin) / (self.ann.num_input_nodes - 1)
            self.input_nodes.append( shapes.Circle(self.margin, self.margin+i*s, self.node_radius, color=(50, 225, 30), batch=self.batch) )
            self.input_labels.append( pyglet.text.Label("%f" % self.ann.x[0,i], x=self.margin-self.node_radius*2, y=self.margin+i*s-self.node_radius*2, batch=self.batch) )

        #draw output nodes
        self.output_nodes = []
        self.output_labels = []
        for i in range(ann.num_output_nodes):
            s = (self.window_height - 2*self.margin) / (self.ann.num_output_nodes - 1)
            self.output_nodes.append( shapes.Circle(self.window_width-self.margin, self.margin+i*s, self.node_radius, color=(50, 225, 30), batch=self.batch) )
            self.output_labels.append( pyglet.text.Label("%f" % self.ann.y[0,i], x=self.window_width-self.margin-self.node_radius*2, y=self.margin+i*s-self.node_radius*2, batch=self.batch) )

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
        x = np.array([[0.1, 0.1, 0.1, 0.1]])
        print( ann.forward_propagation(x) )

        x = np.array([[1, 1, 1, 1]])
        y = np.array([[1, 0]])
        alpha = 0.01
        result = ann.back_propagation(x, y)
        ann.set_weight_matrix(ann.get_weight_matrix() - alpha*result['dJ_dWy'])
        for i in range(ann.get_total_hidden_layers()):
            hl = ann.get_hidden_layer(i)
            wm = result['dJ_dWh'][i]
            hl.set_weight_matrix( hl.get_weight_matrix() - alpha*wm )

    def mouse_move(self, x, y):
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
                #print("(%d) %d => line(%d,%d,%d,%d)" % (ln, i, l.x, l.y, l.x2, l.y2) )
                hl = self.ann.hidden_layers[ln] # get the hidden layer
                hx = int(i / hl.num_hidden_nodes)
                hy = int(i % hl.num_hidden_nodes)
                #print("Number of nodes: %d" % hl.num_hidden_nodes)
                #print("X: %d, y: %d" % (hx, hy))
                m = max(abs(hl.Wh.min()), hl.Wh.max())
                #print(hl.Wh)
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
                    #print("distance: " + str(distance))
                    if distance < 10:
                        l.color = (0, 255, 0)
                        print("X: %d, y: %d" % (hx, hy))
        for ln in range(len(self.weight_lines_output)):
            lines = self.weight_lines_output[ln]
            for i in range(len(lines)):
                l = lines[i]
                #print("(%d) %d => line(%d,%d,%d,%d)" % (ln, i, l.x, l.y, l.x2, l.y2) )
                #print("Number of nodes: %d" % self.ann.num_output_nodes)
                #print("X: %d, y: %d" % (hx, hy))
                m = max(abs(self.ann.Wy.min()), self.ann.Wy.max())
                l.width = self._line_width(self.ann.Wy[ln,i], m)
                l.color = self._line_color(self.ann.Wy[ln,i])
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
                    #print("distance: " + str(distance))
                    if distance < 10:
                        l.color = (0, 255, 0)
                        print("(%d,%d)" % (ln, i))

    def update(self):
        """Update the visual based on values of ann..."""
        for i in range(self.ann.num_input_nodes):
            print(self.ann.x)
            v = self.ann.x[0,i]
            print("value: " + str(v))
            print("%d => %f" % (i, v))
            self.input_labels[i].text = "%f" % self.ann.x[0,i]
        for i in range(self.ann.num_output_nodes):
            print(self.ann.y)
            v = self.ann.y[0,i]
            print("value: " + str(v))
            print("%d => %f" % (i, v))
            self.output_labels[i].text = "%f" % self.ann.y[0,i]

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
    ann_visualisation.update()
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

