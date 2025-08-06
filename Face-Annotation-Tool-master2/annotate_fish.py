#!/usr/bin/env python
"""
Facial landmark annotation tool
... (original docstring unchanged)
"""
from __future__ import print_function
from __future__ import division
import os
import argparse
import warnings

import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle
import matplotlib.cbook

#remap matplotlib panning action so right click becomes drag to change position and avoids placing label while moving
matplotlib.rcParams['toolbar'] = 'toolmanager'
matplotlib.rcParams['backend'] = 'TkAgg'
# Remap mouse buttons (1 = left, 2 = middle, 3 = right)
# 1: pan, 3: zoom originally; here swapped
matplotlib.rcParams['keymap.pan'] = ['1']    # left-click to pan
matplotlib.rcParams['keymap.zoom'] = []      # disable right-click zoom



import numpy as np


def enum(**enums):
    return type('Enum', (), enums)


class InteractiveViewer(object):
    def __init__(self, img_path):
        self.img_path = img_path
        self.key_pressed = False
        self.key_event = None
        self.rect_clicked = False

        #self.annotations = [None for _ in range(4)]      start with how many annotation buttons, here 4
        self.annotations = [None]

        self.image = cv2.imread(img_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        self.orig_height, self.orig_width = self.image.shape[:2]  # save original size
        # Auto downscale large images else it lags hard
        max_dim = 2550  # Change to control max image height/width, its the original resolutions 1/4 and coordinates set needs to be accordingly inflated
        if max(self.orig_height, self.orig_width) > max_dim:
            scale = max_dim / max(self.orig_height, self.orig_width)
            self.image = cv2.resize(self.image, (int(self.orig_width * scale), int(self.orig_height * scale)), interpolation=cv2.INTER_AREA)

        self.clone = self.image.copy()
        self.img_height, self.img_width = self.image.shape[:2]
        # Compute scaling factors
        self.scale_x = self.orig_width / self.img_width
        self.scale_y = self.orig_height / self.img_height

        self.fig = None
        self.im_ax = None

        self.buttons = []
        self.button_done = None
        self.button_skip = None
        self.button_add = None
        self.button_delete = None
        self.button_ruler = None
        self.button_ruler_reset = None

        self.highlight_patch = None
        #self.curr_state = 0
        self.curr_state = -1  #start with no annotation selected, change back to 0 when want to start with annotation 1
        self.is_finished = False
        self.is_skipped = False

        self.ruler_points = []
        self.ruler_artist = []
        self.using_ruler = False

        self.calibration_mode = False
        self.pixels_per_cm = None

        self.zoom_slider = None
        self.zoom_level = 1.0

        # Check for existing TPS file
        img_name = os.path.basename(self.img_path)
        fish_id = os.path.splitext(img_name)[0]
        tps_path = os.path.join(os.path.dirname(__file__), "labels", "annotated", f"{fish_id}.tps")

        if os.path.isfile(tps_path):
            with open(tps_path, "r") as f:
                lines = [line.strip() for line in f.readlines()]
                lm_line = next((line for line in lines if line.startswith("LM=")), None)
                if lm_line:
                    lm_count = int(lm_line.split("=")[1])
                    coords = lines[1:1 + lm_count]
                    self.annotations = [None] * lm_count  # Resize annotation list

                    for idx, line in enumerate(coords):
                        try:
                            x_str, y_str = line.strip().split()
                            x_orig = float(x_str)
                            y_orig = float(y_str)
                            x_disp = int(x_orig / self.scale_x)
                            y_disp = int(y_orig / self.scale_y)
                            self.annotations[idx] = (x_disp, y_disp)
                        except ValueError:
                            continue  # skip invalid lines


    def write_tps_file(self): #new storage of annotations in tps form
        self.fill_default_coords()
        scaled_annotations = [(int(x * self.scale_x), int(y * self.scale_y)) if x >= 0 else (-1, -1) for x, y in
                              self.annotations]
        valid_coords = [coord for coord in scaled_annotations if coord != (-1, -1)]
        if not valid_coords:
            return  # nothing to save

        lm_count = len(valid_coords)
        lines = [f"LM={lm_count}"]
        lines += [f"{x:.4f} {y:.4f}" for x, y in valid_coords]

        img_name = os.path.basename(self.img_path)
        fish_id = os.path.splitext(img_name)[0]
        lines.append(f"IMAGE = Face-Annotation-Tool-master\\fish\\{img_name}")
        lines.append(f"ID = {fish_id}")
        lines.append("SCALE = -")

        out_dir = os.path.join(os.path.dirname(__file__), "labels", "annotated")
        os.makedirs(out_dir, exist_ok=True)
        tps_path = os.path.join(out_dir, f"{fish_id}.tps")
        with open(tps_path, "w") as f:
            f.write("\n".join(lines))

    def redraw_annotations(self):
        self.image = self.clone.copy()
        radius = 2   # how big the label red point appears on the image. default 4 too big
        for coords in self.annotations:
            if coords is not None:
                cv2.circle(self.image, coords, radius, (255, 0, 0), -1)
        self.image_artist.set_data(self.image)

    def update_button_labels(self):
        for idx, btn in enumerate(self.buttons):
            if idx < len(self.annotations):
                btn.label.set_text(f'Annotation {idx + 1}\n{self.annotations[idx]}')

    def highlight_active_button(self):
        if self.highlight_patch:
            try:
                self.highlight_patch.remove()
            except NotImplementedError:
                pass
            self.highlight_patch = None

        if 0 <= self.curr_state < len(self.buttons):
            btn = self.buttons[self.curr_state]
            bbox = btn.ax.get_position()
            self.highlight_patch = Rectangle(
                (bbox.x0, bbox.y0), bbox.width, bbox.height,
                transform=self.fig.transFigure,
                linewidth=1.5, edgecolor='red', facecolor='none', zorder=1000
            )
            self.fig.add_artist(self.highlight_patch)
            self.fig.canvas.draw_idle()

    def draw_ruler(self):
        for artist in self.ruler_artist:
            artist.remove()
        self.ruler_artist = []

        if len(self.ruler_points) == 2:
            pt1, pt2 = self.ruler_points
            line = self.im_ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-')[0]
            dot1 = self.im_ax.plot(pt1[0], pt1[1], 'bo')[0]
            dot2 = self.im_ax.plot(pt2[0], pt2[1], 'bo')[0]
            dist_px = np.linalg.norm(np.array(pt1) - np.array(pt2))

            if self.calibration_mode:
                self.pixels_per_cm = dist_px
                label = self.im_ax.text((pt1[0] + pt2[0]) / 2,
                                        (pt1[1] + pt2[1]) / 2,
                                        f"1.0 cm (set)",
                                        color='green')
            else:
                if self.pixels_per_cm and self.pixels_per_cm > 0:
                    cm = (dist_px / self.pixels_per_cm)                 #change this part for different ruler scaling. if scaling on 10cm then cm = (dist_px / self.pixels_per_cm)*10
                    label = self.im_ax.text((pt1[0] + pt2[0]) / 2,
                                            (pt1[1] + pt2[1]) / 2,
                                            f"{cm:.2f} cm",
                                            color='blue')
                else:
                    label = self.im_ax.text((pt1[0] + pt2[0]) / 2,
                                            (pt1[1] + pt2[1]) / 2,
                                            f"{dist_px:.1f}px",
                                            color='blue')

            self.ruler_artist.extend([line, dot1, dot2, label])
            self.fig.canvas.draw_idle()
            self.calibration_mode = False

    def reset_ruler(self):
        self.ruler_points = []
        for artist in self.ruler_artist:
            artist.remove()
        self.ruler_artist = []
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.im_ax or event.button != 1:
            return

        # Ruler mode
        if self.using_ruler:
            if len(self.ruler_points) < 2:
                self.ruler_points.append((event.xdata, event.ydata))
            if len(self.ruler_points) == 2:
                self.draw_ruler()
            return

        # Prevent placing if no active annotation is selected
        if not (0 <= self.curr_state < len(self.annotations)):
            return

        self.annotations[self.curr_state] = (int(event.xdata), int(event.ydata))
        self.buttons[self.curr_state].label.set_text(
            f'Annotation {self.curr_state + 1}\n{self.annotations[self.curr_state]}')
        self.redraw_annotations()

    def simulate_add_annotation(self): #for keypress detection for hotkeys
        if len(self.annotations) < 21:
            self.annotations.append(None)
            col = (len(self.annotations) - 1) % 2
            row = (len(self.annotations) - 1) // 2
            x = 0.76 + 0.12 * col
            y = 0.87 - 0.06 * row
            new_btn = Button(plt.axes([x, y, 0.12, 0.06]), f'Annotation {len(self.annotations)}')
            new_btn.on_clicked(self.button_event)
            self.buttons.append(new_btn)
            self.curr_state = len(self.annotations) - 1
            self.highlight_active_button()
            self.update_button_labels()

    def on_key_press(self, event):
        self.key_event = event
        self.key_pressed = True
        #hotkey n for new annotation
        if event.key == 'n':
            self.simulate_add_annotation()

    def connect(self):
        self.fig.canvas.mpl_connect('button_press_event', self.mouse_override)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def mouse_override(self, event):
        toolbar = self.fig.canvas.toolbar
        if event.button == 3 and toolbar:
            toolbar.pan()
        #Deselect annotation when clicking outside of any axes
        if event.inaxes is None and self.curr_state != -1:
            self.curr_state = -1
            self.highlight_active_button()

    def button_event(self, event):
        self.key_pressed = False
        for idx, btn in enumerate(self.buttons):
            if event.inaxes == btn.ax:
                self.annotations[idx] = None
                self.curr_state = idx  #activate selected label
                self.using_ruler = False
                self.highlight_active_button()
                self.redraw_annotations()
                self.update_button_labels()
                return

        #If clicked elsewhere, clear label selection
        self.curr_state = -1
        self.highlight_active_button()

        if event.inaxes == self.button_done.ax:
            self.is_finished = True
        elif event.inaxes == self.button_skip.ax:
            self.is_skipped = True
        elif event.inaxes == self.button_add.ax:
            if len(self.annotations) < 21:
                self.annotations.append(None)
                col = (len(self.annotations) - 1) % 2
                row = (len(self.annotations) - 1) // 2
                x = 0.76 + 0.12 * col
                y = 0.87 - 0.06 * row
                new_btn = Button(plt.axes([x, y, 0.12, 0.06]), f'Annotation {len(self.annotations)}')
                new_btn.on_clicked(self.button_event)
                self.buttons.append(new_btn)
                self.curr_state = len(self.annotations) - 1
                self.highlight_active_button()
                self.update_button_labels()
        elif event.inaxes == self.button_delete.ax:
            if len(self.annotations) > 0:
                self.annotations.pop()
                btn = self.buttons.pop()
                btn.ax.remove()
                self.curr_state = max(0, len(self.annotations) - 1)
                self.highlight_active_button()
                self.redraw_annotations()
                self.update_button_labels()
        elif event.inaxes == self.button_ruler.ax:
            self.using_ruler = True
            self.ruler_points = []
        elif event.inaxes == self.button_ruler_reset.ax:
            self.using_ruler = False
            self.reset_ruler()
        elif event.inaxes == self.button_ruler_calibrate.ax:
            self.calibration_mode = True
            self.using_ruler = True
            self.ruler_points = []

    def init_subplots(self):
        # Create figure and image axis, change figsize here
        self.fig = plt.figure(os.path.basename(self.img_path), figsize=(10.5, 7))
        self.im_ax = self.fig.add_axes([0.05, 0.20, 0.68, 0.75])  #size of the main img/plot, change for adjustment
        self.im_ax.set_title('Input')
        #self.image_artist = self.im_ax.imshow(self.image, interpolation='nearest', animated=True)
        self.image_artist = self.im_ax.imshow(self.image, interpolation='nearest') #removed animated=True for performance
        for i in range(len(self.annotations)):
            col = i % 2
            row = i // 2
            x = 0.76 + 0.12 * col
            y = 0.87 - 0.06 * row
            btn = Button(plt.axes([x, y, 0.12, 0.06]), f'Annotation {i + 1}')
            btn.on_clicked(self.button_event)
            self.buttons.append(btn)

        self.button_add = Button(plt.axes([0.52, 0.07, 0.2, 0.05]), 'New Annotation (n)')
        self.button_add.on_clicked(self.button_event)

        self.button_delete = Button(plt.axes([0.74, 0.07, 0.2, 0.05]), 'Delete Annotation')
        self.button_delete.on_clicked(self.button_event)

        self.button_ruler = Button(plt.axes([0.1, 0.06, 0.3, 0.05]), 'Ruler')
        self.button_ruler.on_clicked(self.button_event)

        self.button_ruler_reset = Button(plt.axes([0.1, 0.005, 0.3, 0.05]), 'Reset Ruler')
        self.button_ruler_reset.on_clicked(self.button_event)

        self.button_skip = Button(plt.axes([0.52, 0.01, 0.2, 0.05]), 'Skip')
        self.button_skip.on_clicked(self.button_event)

        self.button_done = Button(plt.axes([0.74, 0.01, 0.2, 0.05]), 'Done')
        self.button_done.on_clicked(self.button_event)

        self.button_ruler_calibrate = Button(plt.axes([0.1, 0.115, 0.3, 0.04]), 'Ruler Calibrate 1 cm')
        self.button_ruler_calibrate.on_clicked(self.button_event)

        #zoom_ax = plt.axes([0.1, 0.12, 0.3, 0.03])
        zoom_ax = plt.axes([0.1, 0.165, 0.3, 0.03])
        self.zoom_slider = Slider(zoom_ax, 'Zoom', 1.0, 20.0, valinit=1.0)  #zoom slider variable adjusted
        self.zoom_slider.on_changed(self.apply_zoom)

        self.redraw_annotations()
        self.update_button_labels()
        self.highlight_active_button()

    def fill_default_coords(self):
        for i in range(len(self.annotations)):
            if self.annotations[i] is None:
                self.annotations[i] = (-1, -1)

    def save_annotations(self):
        self.fill_default_coords()
        scaled_annotations = [(int(x * self.scale_x), int(y * self.scale_y)) if x >= 0 else (-1, -1) for x, y in
                              self.annotations]
        coords_str_tab = '\t'.join([f'{x}\t{y}' for x, y in scaled_annotations])
        coords_str_csv = ','.join([f'{x},{y}' for x, y in scaled_annotations])
        #f_winner.write(f'{self.img_path}\t{coords_str_tab}\n')
        print(f"{os.path.basename(self.img_path)},{coords_str_csv}")

    def apply_zoom(self, val):
        self.zoom_level = val
        # get current center of view (not image)
        try:
            curr_xlim = self.im_ax.get_xlim()
            curr_ylim = self.im_ax.get_ylim()
            center_x = (curr_xlim[0] + curr_xlim[1]) / 2
            center_y = (curr_ylim[0] + curr_ylim[1]) / 2
        except Exception:
            center_x = self.img_width / 2
            center_y = self.img_height / 2

        width = self.img_width / val
        height = self.img_height / val

        self.im_ax.set_xlim(center_x - width / 2, center_x + width / 2)
        self.im_ax.set_ylim(center_y + height / 2, center_y - height / 2)
        self.fig.canvas.draw_idle()

    def run(self):
        self.init_subplots()
        self.connect()
        while True:
            plt.pause(0.03)
            if self.is_finished or (self.key_pressed and self.key_event.key == 'q') or self.is_skipped:
                break
        plt.close()
        if self.is_finished:
            self.save_annotations()
            self.write_tps_file()
            return 0

        elif self.is_skipped:
            return 0
        else:
            return 1


def parse_arguments():
    parser = argparse.ArgumentParser(description='Annotate one or more face images. Output to stdout.')
    base_group = parser.add_mutually_exclusive_group()
    base_group.add_argument('-d', '--dirimgs', type=str, help='dir with images')
    base_group.add_argument('-i', '--img', type=str, help='single image')
    parser.add_argument('-n', '--nimgs', type=int, help='number of images for -d mode', default=1)
    args = parser.parse_args()

    #Default to ./fish/ if no input is given, changed from other code so no need to set run parameters
    if args.dirimgs is None and args.img is None:
        print("No input path provided. Defaulting to: ./fish/")
        args.dirimgs = './fish/'

    return args



def main(args):
    if args.dirimgs is not None:
        flist = sorted([
            f for f in os.listdir(args.dirimgs)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
        ])
        print('image_fname,' + ','.join([f'ann{i}_x,ann{i}_y' for i in range(1, 21)]))
        for curr_file in flist:
            img_path = os.path.join(args.dirimgs, curr_file)
            viewer = InteractiveViewer(img_path)
            if viewer.run() == 1:
                break
    elif args.img is not None:
        print('image_fname,' + ','.join([f'ann{i}_x,ann{i}_y' for i in range(1, 21)]))
        img_path = args.img
        viewer = InteractiveViewer(img_path)
        viewer.run()


if __name__ == '__main__':
    #f_winner = open('landmark_output.txt', 'a')
    main(parse_arguments())
    #f_winner.close()
