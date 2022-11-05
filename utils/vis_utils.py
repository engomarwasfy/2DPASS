import os

import matplotlib.pyplot as plt
import matplotlib as matplotlib
import numpy as np
from IPython.core.display_functions import display

from utils.turbo_cmap import interpolate_or_clip, turbo_colormap_data
import numpy as np
import plotly
import plotly.graph_objs as go

import torch
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate

COLOR_MAP = np.array(['#f59664', '#f5e664', '#963c1e', '#b41e50', '#ff0000',
                      '#1e1eff', '#c828ff', '#5a1e96', '#ff00ff', '#ff96ff',
                      '#4b004b', '#4b00af', '#00c8ff', '#3278ff', '#00af00',
                      '#003c87', '#50f096', '#96f0ff', '#0000ff', '#ffffff'])

LABEL_MAP = np.array([19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 0, 1, 19,
                      19, 19, 2, 19, 19, 3, 19, 4, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 5, 6, 7, 19, 19, 19, 19, 19, 19,
                      19, 8, 19, 19, 19, 9, 19, 19, 19, 10, 11, 12, 13,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 14, 15, 16, 19, 19, 19, 19, 19,
                      19, 19, 17, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19])

# all classes
NUSCENES_COLOR_PALETTE = [
    (255, 158, 0),  # car
    (255, 158, 0),  # truck
    (255, 158, 0),  # bus
    (255, 158, 0),  # trailer
    (255, 158, 0),  # construction_vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # motorcycle
    (255, 61, 99),  # bicycle
    (0, 0, 0),  # traffic_cone
    (0, 0, 0),  # barrier
    (200, 200, 200),  # background
]

NUSCENSE_LIDARSEG_PALETTE = [
    (0, 0, 0),  # noise
    (112, 128, 144),  # barrier
    (220, 20, 60),  # bicycle
    (255, 127, 80),  # bus
    (255, 158, 0),  # car
    (233, 150, 70),  # construction_vehicle
    (255, 61, 99),  # motorcycle
    (0, 0, 230),  # pedestrian
    (47, 79, 79),  # traffic_cone
    (255, 140, 0),  # trailer
    (255, 99, 71),  # Tomato
    (0, 207, 191),  # nuTonomy green
    (175, 0, 75),
    (75, 0, 75),
    (112, 180, 60),
    (222, 184, 135),  # Burlywood
    (0, 175, 0)
]

# classes after merging (as used in xMUDA)
NUSCENES_COLOR_PALETTE_SHORT = [
    (255, 158, 0),  # vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # bike
    (0, 0, 0),  # traffic boundary
    (200, 200, 200),  # background
]

# all classes
A2D2_COLOR_PALETTE_SHORT = [
    (255, 0, 0),  # car
    (255, 128, 0),  # truck
    (182, 89, 6),  # bike
    (204, 153, 255),  # person
    (255, 0, 255),  # road
    (150, 150, 200),  # parking
    (180, 150, 200),  # sidewalk
    (241, 230, 255),  # building
    (147, 253, 194),  # nature
    (255, 246, 143),  # other-objects
    (0, 0, 0)  # ignore
]

# colors as defined in https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
SEMANTIC_KITTI_ID_TO_BGR = {  # bgr
  0: [0, 0, 0],
  1: [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0],
}
SEMANTIC_KITTI_COLOR_PALETTE = [SEMANTIC_KITTI_ID_TO_BGR[id] if id in SEMANTIC_KITTI_ID_TO_BGR.keys() else [0, 0, 0]
                                for id in range(list(SEMANTIC_KITTI_ID_TO_BGR.keys())[-1] + 1)]


# classes after merging (as used in xMUDA)
SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR = [
    [245, 150, 100],  # car
    [180, 30, 80],  # truck
    [150, 60, 30],  # bike
    [30, 30, 255],  # person
    [255, 0, 255],  # road
    [255, 150, 255],  # parking
    [75, 0, 75],  # sidewalk
    [0, 200, 255],  # building
    [0, 175, 0],  # nature
    [255, 255, 50],  # other-objects
    [0, 0, 0],  # ignore
]
SEMANTIC_KITTI_COLOR_PALETTE_SHORT = [(c[2], c[1], c[0]) for c in SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR]
def save_prediction(prediction,path,path_name):
    pred_np = prediction.reshape((-1)).astype(np.int32)
    # save scan
    path = os.path.join(path,"predictions", path_name)
    pred_np.tofile(path)
def write_obj(points, file, rgb=False):
    fout = open('%s.obj' % file, 'w')
    for i in range(points.shape[0]):
        if not rgb:
            fout.write('v %f %f %f %d %d %d\n' % (
                points[i, 0], points[i, 1], points[i, 2], 255, 255, 0))
        else:
            fout.write('v %f %f %f %d %d %d\n' % (
                points[i, 0], points[i, 1], points[i, 2], points[i, -3] * 255, points[i, -2] * 255,
                points[i, -1] * 255))


def draw_points_image_labels(img, img_indices, seg_labels, show=True, color_palette_type='NuScenes', point_size=3.5):
    if color_palette_type == 'NuScenes':
        color_palette = NUSCENSE_LIDARSEG_PALETTE
    elif color_palette_type == 'A2D2':
        color_palette = A2D2_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI_long':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE
    else:
        raise NotImplementedError('Color palette type not supported')

    color_palette = np.array(color_palette) / 255.
    # seg_labels[seg_labels == -100] = len(color_palette) - 1
    colors = color_palette[seg_labels[:, 0]]
    colors = colors[:, [2, 1, 0]]
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5, s=point_size)

    plt.axis('off')

    if show:
        plt.show()


def normalize_depth(depth, d_min, d_max):
    # normalize linearly between d_min and d_max
    data = np.clip(depth, d_min, d_max)
    return (data - d_min) / (d_max - d_min)


def draw_points_image_depth(img, img_indices, depth, show=True, point_size=0.5):
    # depth = normalize_depth(depth, d_min=3., d_max=50.)
    depth = normalize_depth(depth, d_min=depth.min(), d_max=depth.max())
    colors = []
    for depth_val in depth:
        colors.append(interpolate_or_clip(colormap=turbo_colormap_data, x=depth_val))
    # ax5.imshow(np.full_like(img, 255))
    plt.imshow(img)
    plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5, s=point_size)

    plt.axis('off')

    if show:
        plt.show()


def draw_bird_eye_view(data_dict,prediction,labels,full_scale=4096):

    color_palette = SEMANTIC_KITTI_COLOR_PALETTE
    color_palette = np.array(color_palette) / 255.
    # seg_labels[seg_labels == -100] = len(color_palette) - 1
    colors = color_palette[data_dict['labels'].cpu().detach().numpy()]
    colors = colors[:, [2, 1, 0]]
    axes = [4096, 4096, 4096]
    data = data_dict['full_coors1']

    # Control Transparency
    alpha = 0.9

    # Control colour
    # Plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

# Voxels is used to customizations of the
# sizes, positions and colors.
    ax.voxels(data)
    plt.figure(figsize=(10, 6))
    '''
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.5, s=0.1)
    plt.xlim([0, full_scale])
    plt.ylim([0, full_scale])
    plt.zlim([0, full_scale])
    plt.show()
    '''
    plt.savefig('figure.png')
def configure_plotly_browser_state(coords,labels):

    trace = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker={
            'size': 1,
            'opacity': 0.8,
            'color': COLOR_MAP[labels].tolist(),
        }
    )
    import IPython
    display(IPython.core.display.HTML('''
            <script src="/static/components/requirejs/require.js"></script>
            <script>
              requirejs.config({
                paths: {
                  base: '/static/base',
                  plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
                },
              });
            </script>
            '''))
    plotly.offline.init_notebook_mode(connected=False)

    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.2))
    )

    plotly.offline.iplot(go.Figure(data=[trace], layout=layout))