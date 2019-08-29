import torch
import numpy as np
import time
import tkinter
from tkinter import messagebox
from collections import defaultdict
from itertools import product

from pycolab.cropping import ObservationCropper
import matplotlib.pyplot as plt
import matplotlib
from environment import VariationalWarehouse
from rl_pysc2.agents.a2c.model import A2C
from relationalnet import RelationalNet
from main import warehouse_setting


class AttentionVisualizer():

    DEFAULT_COLOR = "#CCCCCC"

    def __init__(self, cell_size, colors, cmap, balls, bucket,
                 worldmaps, pairing, load_param_path,
                 device="cuda", border_ratio=0.05):
        self.root = tkinter.Tk()
        self.root.title("Relational Warehouse")
        self.croppers = [
            ObservationCropper(),
            ObservationCropper(),
        ]

        self.initial_reset(balls, bucket,
                           worldmaps, pairing,
                           load_param_path, device)

        width = ((sum(cropper.cols for cropper in self.croppers) +
                  len(self.croppers) - 1) * cell_size)
        self.width_offset = (self.croppers[0].cols // 2) * cell_size
        width += self.width_offset
        height = max(cropper.rows for cropper in self.croppers) * cell_size

        self.canvas = tkinter.Canvas(
            self.root, width=width, height=height, bg="gray")
        self.canvas.pack()
        self.canvas_height = height
        self.canvas_widht = width
        self.cell_size = cell_size

        self.border_ratio = border_ratio
        colors = colors or self.env.renderer_kwargs["colors"]
        self.colors = defaultdict(lambda: DEFAULT_COLOR)
        for key, value in colors.items():
            self.colors[ord(key)] = value
        self.cmap = cmap

        self.cell_list = self._init_map()

        # TODO: Add button for step and bind
        # TODO: Add button for reset and bind
        reset_button = tkinter.Button(
            self.root,
            text="Reset",
            command=self.reset(balls, bucket,
                               worldmaps, pairing,
                               load_param_path, device))
        reset_button.pack()
        reset_button.place(x=cell_size, y=cell_size,
                           height=cell_size, width=cell_size*3)

        step_button = tkinter.Button(
            self.root,
            text="Step",
            command=self.step(device))
        step_button.pack()
        step_button.place(x=cell_size, y=cell_size*3,
                          height=cell_size, width=cell_size*3)

        # TODO: Add hover callback

    def _init_model(self, balls, bucket,
                    worldmaps, pairing,
                    load_param_path, device):
        env = VariationalWarehouse(balls, bucket, worldmaps, pairing)
        in_channel, mapsize, _ = env.observation_space.shape
        n_act = 4
        network = RelationalNet(in_channel, mapsize, n_act)
        agent = A2C(network, None)
        agent.to(device)
        agent.eval()
        agent.load_model(load_param_path)
        return agent, env

    def _init_map(self):
        """ Initialize the renderer and pop ups the window. Create each cell
        in each croppers. While doing so leaving a single cell sized gap
        between the cells of different croppers.
        Return:
            - List of cell for all croppers.
        """
        cell_list = []
        global_col = self.width_offset//self.cell_size
        for cropper in self.croppers:
            rows = cropper.rows
            cols = cropper.cols

            b_w = int(self.cell_size*self.border_ratio)
            b_h = int(self.cell_size*self.border_ratio)

            cells = [self.canvas.create_rectangle(x*self.cell_size + b_w,
                                                  y*self.cell_size + b_h,
                                                  (x+1)*self.cell_size - b_w,
                                                  (y+1)*self.cell_size - b_h)
                     for x, y in product(range(global_col, cols + global_col),
                                         range(rows))]
            cell_list.append(cells)
            global_col += (1 + cropper.cols)
        
        for cell in cell_list[0]:
            self.canvas.tag_bind(cell, "<Enter>", self.hover_callback)
            self.canvas.tag_bind(cell, "<Leave>", lambda event: None)

        return cell_list

    def _paint_environment(self, board):
        cropped_board = self.croppers[0].crop(board).board
        for i, v in enumerate(cropped_board.flatten("F")):
            self.canvas.itemconfig(self.cell_list[0][i], fill=self.colors[v])
        self.root.update()

    def _paint_attention(self, attention_row):
        attn_cmap = plt.cm.get_cmap(self.cmap, 100)
        if isinstance(attention_row, torch.Tensor):
            attention_row = attention_row.cpu().detach().numpy()
        for i, v in enumerate(attention_row.flatten("F")):
            rgb = attn_cmap(int(v*100))[:3]
            color = matplotlib.colors.rgb2hex(rgb)
            self.canvas.itemconfig(self.cell_list[1][i],
                                   fill=color)
        self.root.update()

    def initial_reset(self, balls, bucket,
                      worldmaps, pairing,
                      load_param_path, device):
        agent, env = self._init_model(balls, bucket,
                                      worldmaps, pairing,
                                      load_param_path, device)
        self.agent = agent
        self.env = env

        # Initial step
        self.state = env.reset()
        self.done = False

        for cropper in self.croppers:
            cropper.set_engine(self.env.game)

    def reset(self, balls, bucket,
              worldmaps, pairing,
              load_param_path, device):
        def reset_callback():
            self.initial_reset(balls, bucket,
                               worldmaps, pairing,
                               load_param_path, device)
            board = self.env.observation
            self._paint_environment(board)
            attn = np.zeros(self.state.shape[1]*self.state.shape[2])
            self._paint_attention(attn)

        return reset_callback

    # Bind to step button
    # Color envrionment map
    def step(self, device):

        def to_torch(array):
            return torch.from_numpy(array).to(device).float()

        def step_callback():
            if self.done is True:
                messagebox.showinfo("Error", "Press reset before step")
                return None
            state = to_torch(self.state)
            action, log_prob, value, entropy = self.agent(state.unsqueeze(0))
            action = action.item()
            self.state, reward, self.done, _ = self.env.step(action)

            board = self.env.observation
            self._paint_environment(board)
        return step_callback

    # Bind to hover
    # color attention map
    def hover_callback(self, event):
        grid_x = (event.x - self.width_offset)//self.cell_size
        grid_y = event.y//self.cell_size
        cell_num = grid_y * self.croppers[0].cols + grid_x

        attn_row = self.agent.network.attn_weights[0][cell_num]
        self._paint_attention(attn_row)
        print(torch.sum(attn_row))

if __name__ == "__main__":
    BALL_COUNT = {"b": 1}
    BALLS = "bcd"
    N_MAPS = 100
    BUCKET = "B"
    PARAM_PATH = "experiments/Relational_a2c_maxpool_concat_attn_entropy/0/param.b"
    worldmaps, pairing, _ = warehouse_setting(
        BALL_COUNT, BALLS, N_MAPS, BUCKET)
    kwargs = dict(
        colors=None,
        balls=BALLS,
        bucket=BUCKET,
        load_param_path=PARAM_PATH,
        cell_size=40,
        cmap="viridis",
        worldmaps=worldmaps,
        pairing=pairing,
        device="cuda",
        border_ratio=0.05
    )
    app = AttentionVisualizer(**kwargs)
    app.root.mainloop()
