import torch
import numpy as np
import time
import tkinter
from tkinter import messagebox
from collections import defaultdict
from itertools import product
import matplotlib.pyplot as plt
import matplotlib
import argparse

from pycolab.cropping import ObservationCropper
from rl_pysc2.agents.a2c.model import A2C
from graph_rl.tools.config import Config
from graph_rl.environments.warehouse import VariationalWarehouse


class OurAttnVisualizer():

    DEFAULT_COLOR = "#CCCCCC"

    def __init__(self, config_path, width = 1000, height = 600):
        self.root = tkinter.Tk()
        self.root.title("OurAttnNet Warehouse")
        self.cropper = ObservationCropper()
        
        
        self.canvas = tkinter.Canvas(self.root, width=width, height=height, bg="gray")
        self.canvas.pack()
        self.canvas_height = height
        self.canvas_width = width

        self.config = Config.load(config_path)
        self.n_entity = self.config.model_kwargs['n_entity']

        self.self_attn_cells = self._init_self_attn()
        self.entity_cells = self._init_entities()

        reset_button = tkinter.Button(
            self.root,
            text="Reset",
            command=self.reset())
        reset_button.pack()
        reset_button.place(x=width//20, y=height//60,
                           height=height//20, width=width//12)

        step_button = tkinter.Button(
            self.root,
            text="Step",
            command=self.step())
        step_button.pack()
        step_button.place(x=width//5, y=height//60,
                          height=height//20, width=width//12)

    def _init_entities(self):
        x_begin = self.canvas_width//20
        width = (self.canvas_width*0.9)//(self.n_entity*2)
        height_begin = self.canvas_height//6*5
        height = self.canvas_height//12
        cells = []
        for e in range(self.n_entity):
            cell = tkinter.Button(
                self.root,
                text="entity " + str(e + 1))
            cell.pack()
            cell.place(x=x_begin, y=height_begin,
                                         width=width, height=height)
            
            cell.bind("<Enter>", self.hover(e))
            cell.bind("<Leave>", lambda event: self.paint_environment(self.map_cells, self.env.observation))

            x_begin += width*2
            cells.append(cell)
        return cells

    def _init_self_attn(self):
        canvas_size = self.canvas_height//6*4
        height_begin = self.canvas_height//12
        width_begin = self.canvas_width//20*11
        cell_size = canvas_size//self.n_entity
        cells = [self.canvas.create_rectangle(x, y, x+cell_size, y+cell_size)
                     for y, x in product(np.arange(self.n_entity)*cell_size + height_begin,
                                         np.arange(self.n_entity)*cell_size + width_begin)]
        return cells


    def _init_map(self,border_ratio=0.05):
        """ Initialize the renderer and pop ups the window. Create each cell
        in each croppers. While doing so leaving a single cell sized gap
        between the cells of different croppers.
        Return:
            - List of cell for all croppers.
        """
        rows = self.cropper.rows
        cols = self.cropper.cols
        canvas_size = self.canvas_height//6*4
        cell_size = canvas_size//rows

        b_w = int(cell_size*border_ratio)
        b_h = int(cell_size*border_ratio)

        width_offset = self.canvas_width // 20
        height_offset = self.canvas_height // 12

        cells = [self.canvas.create_rectangle(x*cell_size + b_w + width_offset,
                                                y*cell_size + b_h + height_offset,
                                                (x+1)*cell_size - b_w + width_offset,
                                                (y+1)*cell_size - b_h+ height_offset)
                    for y, x in product(range(rows), range(cols))]

        return cells

    def initial_reset(self):

        network = self.config.initiate_model()
        device = self.config.hyperparams['device']
        optimizer = torch.optim.Adam(network.parameters(),
                                 lr=self.config.hyperparams["lr"])
        agent = A2C(network, optimizer)
        agent.to(device)
        agent.eval()

        agent.load_state_dict(self.config.model_params['agent'])
        optimizer.load_state_dict(self.config.model_params['optimizer'])

        self.agent = agent
        self.env = self.config.initiate_env()
        # Initial step
        self.state = self.env.reset()
        self.done = False

        self.colors = defaultdict(lambda: self.DEFAULT_COLOR)
        for key, value in self.env.renderer_kwargs["colors"].items():
            self.colors[ord(key)] = value

        self.cropper.set_engine(self.env.game)

    def paint_map(self, map_cells, vis_attn, cmap="gray"):
        attn_cmap = plt.cm.get_cmap(cmap, 100)
        assert len(vis_attn.shape) == 2

        board = self.cropper.crop(self.env.observation).board

        for cell, attn_value, cell_value in zip(map_cells,
                                    vis_attn.flatten(),
                                    board.flatten()):

            gray_rgb = attn_cmap(int(attn_value*100))[:3]
            grey_color = np.array(tuple(gray_rgb))

            cell_color = self.colors[cell_value]
            cell_color = np.array([int(cell_color[i+1:i+3], 16) for i in (0, 2, 4)])/255

            color = cell_color*attn_value.item() + np.zeros(3)*(1-attn_value.item())

            self.canvas.itemconfig(cell,
                                   fill=matplotlib.colors.rgb2hex(tuple(color)))
        self.root.update()

    def paint_environment(self, map_cells, board):
       
        cropped_board = self.cropper.crop(board).board
        for cell, value in zip(map_cells, cropped_board.flatten()):
            self.canvas.itemconfig(cell, fill=self.colors[value])
        self.root.update()

    def paint_self_attn(self, attn_cells, attn_head, cmap="viridis"):
        attn_cmap = plt.cm.get_cmap(cmap, 100)
        for cell, attn_value in zip(attn_cells, attn_head.flatten()):
            rgb = attn_cmap(int(attn_value*100))[:3]
            color = matplotlib.colors.rgb2hex(rgb)
            self.canvas.itemconfig(cell, fill=color)
        self.root.update()

    def reset(self):
        def reset_callback():
            self.initial_reset()
            self.map_cells = self._init_map()
            board = self.env.observation
            self.paint_environment(self.map_cells, board)
            attn_head = np.zeros((self.n_entity, self.n_entity))
            self.paint_self_attn(self.self_attn_cells, attn_head)

        return reset_callback

    def step(self):

        def to_torch(array):
            return torch.from_numpy(array).to(self.config.hyperparams['device']).float()

        def step_callback():
            if self.done is True:
                messagebox.showinfo("Error", "Press reset before step")
                return None
            state = to_torch(self.state)
            action, log_prob, value, entropy = self.agent(state.unsqueeze(0))
            action = action.item()
            self.state, reward, self.done, _ = self.env.step(action)

            board = self.env.observation
            self.paint_environment(self.map_cells, board)
            try:
                self.paint_self_attn(
                    self.self_attn_cells,
                    self.agent.network.attn_module.self_attn.self_attn)
            except AttributeError:
                print("Warning!!!! Attention module not present!")
        return step_callback

    def hover(self, index):
        
        def hover_callback(event):
            self.paint_map(
                self.map_cells,
                self.agent.network.attn_module.vis_attn_features[0, index])
        
        return hover_callback

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("--name", help="Config file name to load",
                        action="store", dest="config_path")
    parser.add_argument("--width", help="number of tests",
                    action='store', type=int, default=1000)
    parser.add_argument("--height", help="number of tests",
                    action='store', type=int, default=600)

    kwargs = vars(parser.parse_args())
    kwargs["config_path"] = "configs/configs/" + kwargs["config_path"] 
    app = OurAttnVisualizer(**kwargs)
    app.root.mainloop()
