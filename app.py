from ast import Interactive
from xml.sax.xmlreader import InputSource
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import os
import cv2
import pathlib
import math
import nibabel as nib 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from predictor import Predictor
from volume_processor import VolumeProcessor

# 初始化处理器
volume_processor = VolumeProcessor()
display_height = 600
H = 256
W = 256

test_example_dir = pathlib.Path("./test_examples")
test_examples = [str(test_example_dir / x) for x in sorted(os.listdir(test_example_dir))]

default_example = test_examples[0]
exp_dir = pathlib.Path('./checkpoints')
default_model = 'ScribblePrompt-Unet'

model_dict = {
    'ScribblePrompt-Unet': 'ScribblePrompt_unet_v1_nf192_res128.pt'
}

# =============================================================================
# Model initialization functions
# =============================================================================
def load_model(exp_key: str = default_model):
    fpath = exp_dir / model_dict.get(exp_key)
    exp = Predictor(fpath)
    return exp, None

# =============================================================================
# Vizualization functions
# =============================================================================
def _get_overlay(img, lay, const_color="l_blue"):
    """
    Helper function for preparing overlay
    """
    assert lay.ndim == 2, "Overlay must be 2D, got shape: " + str(lay.shape)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    assert img.ndim == 3, "Image must be 3D, got shape: " + str(img.shape)
    if const_color == "blue":
        const_color = 255 * np.array([0, 0, 1])
    elif const_color == "green":
        const_color = 255 * np.array([0, 1, 0])
    elif const_color == "red":
        const_color = 255 * np.array([1, 0, 0])
    elif const_color == "l_blue":
        const_color = np.array([31, 119, 180])
    elif const_color == "orange":
        const_color = np.array([255, 127, 14])
    else:
        raise NotImplementedError

    x, y = np.nonzero(lay)
    for i in range(img.shape[-1]):
        img[x, y, i] = const_color[i]
    return img

def image_overlay(img, mask=None, scribbles=None, contour=False, alpha=0.5,target_size=(1000, 1000)):
    """
    Overlay the ground truth mask and scribbles on the image if provided
    """
    assert img.ndim == 2, "Image must be 2D, got shape: " + str(img.shape)
    # Resize image and mask to target size
    img_resized = cv2.resize(img, target_size)
    
    if mask is not None:
        assert mask.ndim == 2, "Mask must be 2D, got shape: " + str(mask.shape)
        mask_resized = cv2.resize(mask, target_size)  # Ensure mask is the same size as the image
    else:
        mask_resized = None

    output = np.repeat(img_resized[..., None], 3, axis=-1)

    if mask is not None:
        
        if contour:
            contours = cv2.findContours((mask[..., None] > 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours[0], -1, (0, 255, 0), 2)
        else:
            mask_overlay = _get_overlay(img, mask)
            mask2 = 0.5 * np.repeat(mask[..., None], 3, axis=-1)
            output = cv2.convertScaleAbs(mask_overlay * mask2 + output * (1 - mask2))
    if scribbles is not None:
        pos_scribble_overlay = _get_overlay(output, scribbles[0, ...], const_color="green")
        cv2.addWeighted(pos_scribble_overlay, alpha, output, 1 - alpha, 0, output)
        neg_scribble_overlay = _get_overlay(output, scribbles[1, ...], const_color="red")
        cv2.addWeighted(neg_scribble_overlay, alpha, output, 1 - alpha, 0, output)
    return output

def viz_pred_mask(img, 
                  mask=None, 
                  point_coords=None, 
                  point_labels=None, 
                  bbox_coords=None, 
                  seperate_scribble_masks=None, 
                  binary=True):
    """
    Visualize image with clicks, scribbles, predicted mask overlaid
    """
    assert isinstance(img, np.ndarray), "Image must be numpy array, got type: " + str(type(img))
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
    if binary and mask is not None:
        mask = 1 * (mask > 0.5)
    out = image_overlay(img, mask=mask, scribbles=seperate_scribble_masks)
    H, W = img.shape[:2]
    marker_size = min(H, W) // 100

    if point_coords is not None:
        for i, (col, row) in enumerate(point_coords):
            if point_labels[i] == 1:
                cv2.circle(out, (col, row), marker_size, (0, 255, 0), -1)
            else:
                cv2.circle(out, (col, row), marker_size, (255, 0, 0), -1)
    if bbox_coords is not None:
        for i in range(len(bbox_coords) // 2):
            cv2.rectangle(out, bbox_coords[2 * i], bbox_coords[2 * i + 1], (255, 165, 0), marker_size)
        if len(bbox_coords) % 2 == 1:
            cv2.circle(out, tuple(bbox_coords[-1]), marker_size, (255, 165, 0), -1)
    return out.astype(np.uint8)

# =============================================================================
# Collect scribbles
# =============================================================================
def get_scribbles(seperate_scribble_masks, last_scribble_mask, scribble_img):
    """
    Record scribbles
    """
    if seperate_scribble_masks is None:  # 如果为 None，初始化为全零数组
        seperate_scribble_masks = np.zeros((2, H, W), dtype=np.float32)
    # assert isinstance(seperate_scribble_masks, np.ndarray), "seperate_scribble_masks must be numpy array, got type: " + str(type(seperate_scribble_masks))
    if scribble_img is not None:
       
        # Only use first layer
        color_mask = scribble_img.get('layers')[0]
        
        positive_scribbles = 1.0*(color_mask[...,1] > 128)
        negative_scribbles = 1.0*(color_mask[...,0] > 128)
        
        seperate_scribble_masks = np.stack([positive_scribbles, negative_scribbles], axis=0)
        last_scribble_mask = None
        return seperate_scribble_masks, last_scribble_mask

def get_predictions(predictor, input_img, click_coords, click_labels, bbox_coords, seperate_scribble_masks, 
                    low_res_mask, img_features, multimask_mode):
    box = None
    if len(bbox_coords) == 1:
        gr.Error("Please click a second time to define the bounding box")
        box = None
    elif len(bbox_coords) == 2:
        box = torch.Tensor(bbox_coords).flatten()[None, None, ...].int().to(device)  # B x n x 4
    if seperate_scribble_masks is not None:
        scribble = torch.from_numpy(seperate_scribble_masks)[None, ...].to(device)
    else:
        scribble = None
    prompts = dict(
        img=torch.from_numpy(input_img)[None, None, ...].to(device) / 255,
        point_coords=torch.Tensor([click_coords]).int().to(device) if len(click_coords) > 0 else None,
        point_labels=torch.Tensor([click_labels]).int().to(device) if len(click_labels) > 0 else None,
        scribble=scribble,
        mask_input=low_res_mask.to(device) if low_res_mask is not None else None,
        box=box,
    )
    mask, img_features, low_res_mask = predictor.predict(prompts, img_features, multimask_mode=multimask_mode)
    return mask, img_features, low_res_mask

def refresh_predictions(predictor, input_img, output_img, click_coords, click_labels, bbox_coords, brush_label,
                        scribble_img, seperate_scribble_masks, last_scribble_mask, 
                        best_mask, low_res_mask, img_features, binary_checkbox, multimask_mode,
                        volume_file, current_slice):
    
    # Ensure bbox_coords is initialized (if it's None, initialize it as an empty list)
    if bbox_coords is None:
        bbox_coords = []

    # Check if volume_file and current_slice are valid
    if volume_file is not None and current_slice is not None:
        input_img = current_slice
        print(f"[DEBUG] 3D输入原始尺寸: {current_slice.shape}")  # 新增打印
    # Input image processing after loading
    print(f"[DEBUG] 输入图像最终尺寸: {input_img.shape}")  # 新增打印
    
    # Ensure separate_scribble_masks is initialized and not None
    if seperate_scribble_masks is None:
        seperate_scribble_masks = np.zeros((2, input_img.shape[0], input_img.shape[1]), dtype=np.float32)
    
    # Record any new scribbles
    seperate_scribble_masks, last_scribble_mask = get_scribbles(
        seperate_scribble_masks, last_scribble_mask, scribble_img
    )
    
    # Make prediction
    best_mask, img_features, low_res_mask = get_predictions(
        predictor, input_img, click_coords, click_labels, bbox_coords, seperate_scribble_masks, low_res_mask, img_features, multimask_mode
    )
    
    # Update input visualizations
    mask_to_viz = best_mask.numpy()
    click_input_viz = viz_pred_mask(input_img, mask_to_viz, click_coords, click_labels, bbox_coords, seperate_scribble_masks, binary_checkbox)
    
    empty_channel = np.zeros(input_img.shape[:2]).astype(np.uint8)
    full_channel = 255 * np.ones(input_img.shape[:2]).astype(np.uint8)
    gray_mask = (255 * mask_to_viz).astype(np.uint8) 

    bg = viz_pred_mask(input_img, mask_to_viz, click_coords, click_labels, bbox_coords, None, binary_checkbox)
    
    # Ensure scribble_img exists and is not None
    if scribble_img is not None:
        old_scribbles = scribble_img.get('layers')[0] 
    else:
        old_scribbles = np.zeros(input_img.shape)  # Empty scribbles if none exist

    scribble_mask = 255 * (old_scribbles > 0).any(-1)

    scribble_input_viz = {
        "background": np.stack([bg[..., i] for i in range(3)] + [full_channel], axis=-1),
        "layers": [np.stack([
            (255 * seperate_scribble_masks[1]).astype(np.uint8), 
            (255 * seperate_scribble_masks[0]).astype(np.uint8), 
            empty_channel, 
            scribble_mask
        ], axis=-1)],
        "composite": np.stack([click_input_viz[..., i] for i in range(3)] + [empty_channel], axis=-1),
    }

    # For binary checkbox
    mask_img = 255 * (mask_to_viz[..., None].repeat(axis=2, repeats=3) > 0.5) if binary_checkbox else mask_to_viz[..., None].repeat(axis=2, repeats=3)

    out_viz = [
        viz_pred_mask(input_img, mask_to_viz, None, None, None, None, binary_checkbox),
        input_img,
        mask_img,
    ]
    
    return click_input_viz, scribble_input_viz, out_viz, best_mask, low_res_mask, img_features, seperate_scribble_masks, last_scribble_mask




def get_select_coords(predictor, input_img, brush_label, bbox_label, best_mask, low_res_mask, 
                      click_coords, click_labels, bbox_coords,
                      seperate_scribble_masks, last_scribble_mask, scribble_img, img_features,
                      output_img, binary_checkbox, multimask_mode, autopredict_checkbox, evt: gr.SelectData):
    """
    Record user click and update the prediction
    """
    # Record click coordinates
    if bbox_label:
        bbox_coords.append(evt.index)
    elif brush_label in ['Positive (green)', 'Negative (red)']:
        click_coords.append(evt.index)
        click_labels.append(1 if brush_label == 'Positive (green)' else 0)
    else:
        raise TypeError("Invalid brush label: {brush_label}")
    
    # Only make new prediction if not waiting for additional bounding box click
    if (len(bbox_coords) % 2 == 0) and autopredict_checkbox:

        click_input_viz, scribble_input_viz, output_viz, best_mask, low_res_mask, img_features, seperate_scribble_masks, last_scribble_mask = refresh_predictions(
            predictor, input_img, output_img, click_coords, click_labels, bbox_coords, brush_label,
            scribble_img, seperate_scribble_masks, last_scribble_mask, 
            best_mask, low_res_mask, img_features, binary_checkbox, multimask_mode,
            None, None 
        )
        return click_input_viz, scribble_input_viz, output_viz, best_mask, low_res_mask, img_features, click_coords, click_labels, bbox_coords, seperate_scribble_masks, last_scribble_mask
    
    else:
        click_input_viz = viz_pred_mask(
            input_img, best_mask, click_coords, click_labels, bbox_coords, seperate_scribble_masks, binary_checkbox
        ) 
        scribble_input_viz = viz_pred_mask(
            input_img, best_mask, click_coords, click_labels, bbox_coords, None, binary_checkbox
        )  
        # Don't update output image if waiting for additional bounding box click
        return click_input_viz, scribble_input_viz, output_img, best_mask, low_res_mask, img_features, click_coords, click_labels, bbox_coords, seperate_scribble_masks, last_scribble_mask   

def undo_click(predictor, input_img, brush_label, bbox_label, best_mask, low_res_mask, click_coords, click_labels, bbox_coords,
               seperate_scribble_masks, last_scribble_mask, scribble_img, img_features,
               output_img, binary_checkbox, multimask_mode, autopredict_checkbox):
    """
    Remove last click and then update the prediction
    """
    if bbox_label:
        if len(bbox_coords) > 0:
            bbox_coords.pop()
    elif brush_label in ['Positive (green)', 'Negative (red)']:
        if len(click_coords) > 0:
            click_coords.pop()
            click_labels.pop()
    else:
        raise TypeError("Invalid brush label: {brush_label}")
   # Only make new prediction if not waiting for additional bounding box click
    if (len(bbox_coords)==0 or len(bbox_coords)==2) and autopredict_checkbox:

        click_input_viz, scribble_input_viz, output_viz, best_mask, low_res_mask, img_features, seperate_scribble_masks, last_scribble_mask = refresh_predictions(
            predictor, input_img, output_img, click_coords, click_labels, bbox_coords, brush_label,
            scribble_img, seperate_scribble_masks, last_scribble_mask, 
            best_mask, low_res_mask, img_features, binary_checkbox, multimask_mode
        )
        return click_input_viz, scribble_input_viz, output_viz, best_mask, low_res_mask, img_features, click_coords, click_labels, bbox_coords, seperate_scribble_masks, last_scribble_mask
    
    else:
        click_input_viz = viz_pred_mask(
            input_img, best_mask, click_coords, click_labels, bbox_coords, seperate_scribble_masks, binary_checkbox
        ) 
        scribble_input_viz = viz_pred_mask(
            input_img, best_mask, click_coords, click_labels, bbox_coords, None, binary_checkbox
        )  

        # Don't update output image if waiting for additional bounding box click
        return click_input_viz, scribble_input_viz, output_img, best_mask, low_res_mask, img_features, click_coords, click_labels, bbox_coords, seperate_scribble_masks, last_scribble_mask 

# =============================================================================
# 3D 图像和视频的处理函数
# =============================================================================

def load_volume_slice(file_path, slice_index,target_size=(1000, 1000)):
    """
    加载 3D 图像（如 NIfTI 格式），返回指定切片
    """
    volume = nib.load(file_path).get_fdata()  # 得到 numpy 数组
    num_slices = volume.shape[2]
    if slice_index < 0 or slice_index >= num_slices:
        raise ValueError(f"切片索引 {slice_index} 超出范围 0 - {num_slices - 1}")
    slice_img = volume[:, :, slice_index]
    # 归一化到 0-255
    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8) * 255
    slice_img = slice_img.astype(np.uint8)
    # 调整尺寸到 target_size
    slice_img_resized = cv2.resize(slice_img, target_size)
    return slice_img_resized

def load_npy_slice(file_path, slice_index):
    """
    加载 .npy 文件中的 3D 数组，并返回第三维的指定切片
    假设数组形状为 (H, W, D)
    """
    volume = np.load(file_path)
    if volume.ndim != 3:
        raise ValueError(f"期待 3 维数组 (H, W, D)，但得到形状: {volume.shape}")
    num_slices = volume.shape[2]
    if slice_index < 0 or slice_index >= num_slices:
        raise ValueError(f"切片索引 {slice_index} 超出范围 0 - {num_slices - 1}")
    slice_img = volume[:, :, slice_index]
    # 归一化到 0-255
    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8) * 255
    slice_img = slice_img.astype(np.uint8)
    return slice_img

import tempfile

def get_temp_file(file_obj):
    # 如果 file_obj 是字典，尝试读取 file_obj["data"]
    if isinstance(file_obj, dict) and "data" in file_obj:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj["name"])[1])
        with open(temp.name, "wb") as f:
            f.write(file_obj["data"])
        return temp.name
    # 如果直接传入文件路径或 file 对象
    elif isinstance(file_obj, str):
        return file_obj
    else:
        raise ValueError("无法解析上传的文件")

def load_video_frame(file_path, frame_index,target_size=(1000, 1000)):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {file_path}")
        raise IOError(f"无法打开视频文件: {file_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}, 当前索引: {frame_index}")
    if frame_index < 0 or frame_index >= total_frames:
        raise ValueError(f"帧索引 {frame_index} 超出范围 0 - {total_frames - 1}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError(f"无法读取第 {frame_index} 帧")
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(frame_gray, target_size)
    return frame_resized

def update_current_slice(volume_file, slider_value):
    if volume_file is None:
        return None
    # 如果 volume_file 是字典，尝试提取文件路径或写入临时文件
    if isinstance(volume_file, dict):
        file_path = get_temp_file(volume_file)
    else:
        file_path = volume_file
    file_ext = os.path.splitext(file_path)[1].lower()
    idx = int(slider_value) - 1

    if file_ext in ['.nii', '.nii.gz']:
        try:
            img = load_volume_slice(file_path, idx,target_size=(1000, 1000))
        except Exception as e:
            print("加载 3D 图像时出错：", e)
            return None
    elif file_ext in ['.mp4', '.avi']:
        try:
            img = load_video_frame(file_path, idx)
        except Exception as e:
            print("加载视频时出错：", e)
            return None
    elif file_ext == '.npy':  # <-- 处理 .npy
        try:
            img = load_npy_slice(file_path, idx)
        except Exception as e:
            print("加载 .npy 时出错：", e)
            return None
    else:
        print("不支持的文件格式: ", file_ext)
        return None
    
    ##img = cv2.resize(img, (256, 256))
    print(f"[DEBUG] 加载后的切片尺寸: {img.shape}")  # 新增打印
    return img.astype(np.uint8)

# --------------------------------------------------

with gr.Blocks(theme=gr.themes.Default(text_size=gr.themes.sizes.text_lg)) as demo:

    # State variables
    seperate_scribble_masks = gr.State(np.zeros((2, H, W), dtype=np.float32))
    last_scribble_mask = gr.State(np.zeros((H, W), dtype=np.float32))

    click_coords = gr.State([])
    click_labels = gr.State([])
    bbox_coords = gr.State([])

    # Load default model
    predictor = gr.State(load_model()[0])
    img_features = gr.State(None)  # For SAM models
    best_mask = gr.State(None)
    low_res_mask = gr.State(None)

    gr.HTML("""\
    <h1 style="text-align: center; font-size: 28pt;">ScribblePrompt: Fast and Flexible Interactive Segmentation for Any Biomedical Image</h1>
    <p style="text-align: center; font-size: large;">
            <b>ScribblePrompt</b> is an interactive segmentation tool designed to help users segment <b>new</b> structures in medical images using scribbles, clicks <b>and</b> bounding boxes.
        [<a href="https://arxiv.org/abs/2312.07381">paper</a> | <a href="https://scribbleprompt.csail.mit.edu">website</a> | <a href="https://github.com/halleewong/ScribblePrompt">code</a>]    
    </p>
    """)

    with gr.Accordion("Open for instructions!", open=False):
        gr.Markdown(
        """
            * Select an input image from the examples below or upload your own image through the <b>'Input Image'</b> tab.
            * Use the <b>'Scribbles'</b> tab to draw <span style='color:green'>positive</span> or <span style='color:red'>negative</span> scribbles.
                - Use the buttons in the top right hand corner of the canvas to undo or adjust the brush size
                - Note: the app cannot detect new scribbles drawn on top of previous scribbles in a different color. Please undo/erase the scribble before drawing on the same pixel in a different color.
            * Use the <b>'Clicks/Boxes'</b> tab to draw <span style='color:green'>positive</span> or <span style='color:red'>negative</span> clicks and <span style='color:orange'>bounding boxes</span> by placing two clicks.
            * The <b>'Output'</b> tab will show the model's prediction based on your current inputs and the previous prediction.
            * The <b>'Clear Input Mask'</b> button will clear the latest prediction (which is used as an input to the model).
            * The <b>'Clear All Inputs'</b> button will clear all inputs (including scribbles, clicks, bounding boxes, and the last prediction). 
        """
        )
    
    # Interface ------------------------------------

    with gr.Row():
        model_dropdown = gr.Dropdown(
            label="Model", 
            choices=list(model_dict.keys()), 
            value=default_model, 
            multiselect=False,
            interactive=False,
            visible=False
        )

    with gr.Row():
        with gr.Column(scale=1):
            brush_label = gr.Radio(["Positive (green)", "Negative (red)"], 
                           value="Positive (green)", label="Scribble/Click Label") 
            bbox_label = gr.Checkbox(value=False, label="Bounding Box (2 clicks)")
        with gr.Column(scale=1):
            
            binary_checkbox = gr.Checkbox(value=True, label="Show binary masks", visible=False)
            autopredict_checkbox = gr.Checkbox(value=True, label="Auto-update prediction on clicks")
            with gr.Accordion("Troubleshooting tips", open=False): 
                gr.Markdown("<span style='color:orange'>If you encounter an <span style='color:orange'>error</span> try clicking 'Clear All Inputs'.")
            multimask_mode = gr.Checkbox(value=True, label="Multi-mask mode", visible=False)

    with gr.Row():

        green_brush = gr.Brush(colors=["#00FF00"], color_mode="fixed", default_size=3)
        red_brush = gr.Brush(colors=["#FF0000"], color_mode="fixed", default_size=3)

        with gr.Column(scale=1):
            with gr.Tab("Scribbles"):
                scribble_img = gr.ImageEditor(
                    label="Input",
                    image_mode="RGB",
                    brush=green_brush,
                    type='numpy',
                    value=default_example,
                    transforms=(),
                    sources=(),
                    container=True,
                    show_download_button=True,
                    height=display_height + 60
                )
            with gr.Tab("Clicks/Boxes") as click_tab:
                click_img = gr.Image(
                    label="Input",
                    type='numpy',
                    value=default_example,
                    show_download_button=True,
                    container=True,
                    height=display_height
                )
                with gr.Row():
                    undo_click_button = gr.Button("Undo Last Click")
                    clear_click_button = gr.Button("Clear Clicks/Boxes", variant="stop")

            with gr.Tab("Input Image"):
                input_img = gr.Image(
                    label="Input",
                    image_mode="L",
                    value=default_example,
                    show_download_button=True,
                    container=True,
                    height=display_height
                )
                gr.Markdown("To upload your own image: click the `x` in the top right corner to clear the current image, then drag & drop")
                    
            with gr.Tab("3D/Video Input"):
                volume_input = gr.File(label="Upload 3D Volume/Video", 
                               file_types=[".nii", ".nii.gz", ".dcm", ".mp4", ".avi", ".npy"])
                frame_slider = gr.Slider(1, 110, 1, step=1, label="Slice/Frame Number", interactive=True)
                current_slice = gr.Image(label="Current Slice/Frame", image_mode="L", height=display_height)
       
        with gr.Column(scale=1):
            with gr.Tab("Output"):
                output_img = gr.Gallery(
                    label='Output',
                    columns=1,
                    elem_id="gallery",
                    preview=True,
                    object_fit="scale-down",
                    height=display_height + 60,
                    container=True
                )
    submit_button = gr.Button("Refresh Prediction", variant='primary')
    clear_all_button = gr.ClearButton([scribble_img], value="Clear All Inputs", variant="stop") 
    clear_mask_button = gr.Button("Clear Input Mask")            

    # ---------------------------
    # 3D/Video 输入部分
    # ---------------------------
    #with gr.Tab("3D/Video Input"):
        
    #    volume_input = gr.File(label="Upload 3D Volume/Video", 
    #                           file_types=[".nii", ".nii.gz", ".dcm", ".mp4", ".avi", ".npy"])
    #    frame_slider = gr.Slider(1,110, 1, step=1, label="Slice/Frame Number", interactive=True)
    #    current_slice = gr.Image(label="Current Slice/Frame", image_mode="L", height=display_height)


    # ----------------------------------------------
    # Loading Examples
    # ----------------------------------------------
    model_dropdown.change(fn=load_model, 
                          inputs=[model_dropdown], 
                          outputs=[predictor, img_features])
    
    gr.Examples(examples=test_examples,
                inputs=[input_img],
                examples_per_page=12,
                label='Examples from datasets unseen during training')

    # When clear clicks button is clicked
    def clear_click_history(input_img):
        return input_img, input_img, [], [], [], None, None
    
    clear_click_button.click(clear_click_history,
                             inputs=[input_img], 
                             outputs=[click_img, scribble_img, click_coords, click_labels, bbox_coords, best_mask, low_res_mask])

    # When clear all button is clicked
    def clear_all_history(input_img):
        # 不清除 input_img，确保上传的图像不会被清除
        input_shape = input_img.shape[:2] if input_img is not None else (H, W)
        
        # 清除与用户交互相关的状态
        click_coords = []  # 清空点击坐标
        click_labels = []  # 清空点击标签
        bbox_coords = []  # 清空框的坐标
        seperate_scribble_masks = np.zeros((2, H, W), dtype=np.float32)  # 清空scribbles
        last_scribble_mask = np.zeros((H, W), dtype=np.float32)  # 清空scribbles

        # 清除与3D图像相关的状态
        current_slice = None  # 清空当前切片
        volume_file = None    # 清空3D图像文件

        # 清除相关模型预测结果
        best_mask = None
        low_res_mask = None
        img_features = None

        # 返回图像本身，并重置交互状态
        return input_img, input_img, [], [], [], None, None, seperate_scribble_masks, last_scribble_mask, best_mask, low_res_mask, img_features, current_slice, volume_file

    def upload_new_image(input_img, volume_file):
        # 每次上传新图像时，重置模型的预测结果
        best_mask = None
        low_res_mask = None
        img_features = None

      
        predictor = load_model()[0]  # 加载模型
        mask, img_features, low_res_mask = predictor.predict(input_img)

        return mask, img_features, low_res_mask


    def update_input_image(input_img, volume_file):
        # 每次上传新图像时，清空之前的预测结果
        best_mask, low_res_mask, img_features = upload_new_image(input_img, volume_file)

        # 更新 UI 显示新的预测结果
        return best_mask, low_res_mask, img_features




    # def clear_history_and_pad_input(input_img):
    #     if input_img is not None:
    #         h,w = input_img.shape[:2]
    #         if h != w:
    #             # Pad to square
    #             pad = abs(h-w)
    #             if h > w:
    #                 padding = [(0,0), (math.ceil(pad/2),math.floor(pad/2))]
    #             else:
    #                 padding = [(math.ceil(pad/2),math.floor(pad/2)), (0,0)]

    #             input_img = np.pad(input_img, padding, mode='constant', constant_values=0)

    #     return clear_all_history(input_img)

    
    input_img.change(
        fn=clear_all_history,
        inputs=[input_img], 
        outputs=[click_img, scribble_img, 
                output_img, click_coords, click_labels, bbox_coords, 
                seperate_scribble_masks, last_scribble_mask, 
                best_mask, low_res_mask, img_features,current_slice
        ]
    )

    
    clear_all_button.click(
        fn=clear_all_history,
        inputs=[input_img], 
        outputs=[click_img, scribble_img, 
                output_img, click_coords, click_labels, bbox_coords, 
                seperate_scribble_masks, last_scribble_mask, 
                best_mask, low_res_mask, img_features,current_slice
        ]
    )

    
    # -------------------------------------------
    # 清除当前预测 mask
    def clear_best_mask(input_img, click_coords, click_labels, bbox_coords, seperate_scribble_masks):
        click_input_viz = viz_pred_mask(input_img, None, click_coords, click_labels, bbox_coords, seperate_scribble_masks)
        scribble_input_viz = viz_pred_mask(input_img, None, click_coords, click_labels, bbox_coords, None)
        return None, None, click_input_viz, scribble_input_viz

    clear_mask_button.click(
    clear_best_mask,
    inputs=[input_img, click_coords, click_labels, bbox_coords, seperate_scribble_masks],
    outputs=[best_mask, low_res_mask, click_img, scribble_img]
    )


    # -------------------------------------------
    # slider 变化时更新当前切片/帧显示
    frame_slider.change(
        fn=update_current_slice,
        inputs=[volume_input, frame_slider],
        outputs=current_slice
    )

    # ----------------------------------------------
    # Clicks
    # ----------------------------------------------
    click_img.select(get_select_coords,
                     inputs=[
                        predictor,
                        input_img, brush_label, bbox_label, best_mask, low_res_mask, click_coords, click_labels, bbox_coords,
                        seperate_scribble_masks, last_scribble_mask, scribble_img, img_features,
                        output_img, binary_checkbox, multimask_mode, autopredict_checkbox
                      ],
                     outputs=[click_img, scribble_img, output_img, best_mask, low_res_mask, img_features,
                              click_coords, click_labels, bbox_coords, seperate_scribble_masks, last_scribble_mask],
                     api_name="get_select_coords"
                    )

    # 修改Refresh Prediction按钮的回调，额外传入 volume_input 与 current_slice
    submit_button.click(
        fn=refresh_predictions,
        inputs=[
            predictor, input_img, output_img, click_coords, click_labels, bbox_coords, brush_label,
            scribble_img, seperate_scribble_masks, last_scribble_mask,
            best_mask, low_res_mask, img_features, binary_checkbox, multimask_mode,
            volume_input, current_slice
        ],
        outputs=[
            click_img, scribble_img, output_img, best_mask, low_res_mask, img_features, 
            seperate_scribble_masks, last_scribble_mask
        ],
        api_name="refresh_predictions"
    )

    undo_click_button.click(
        fn=undo_click,
        inputs=[
            predictor,
            input_img, brush_label, bbox_label, best_mask, low_res_mask, click_coords, 
            click_labels, bbox_coords,
            seperate_scribble_masks, last_scribble_mask, scribble_img, img_features,
            output_img, binary_checkbox, multimask_mode, autopredict_checkbox
        ],
        outputs=[
            click_img, scribble_img, output_img, best_mask, low_res_mask, img_features,
            click_coords, click_labels, bbox_coords, seperate_scribble_masks, last_scribble_mask
        ],
        api_name="undo_click"
    )

    def update_click_img(input_img, click_coords, click_labels, bbox_coords, seperate_scribble_masks, binary_checkbox, 
                         last_scribble_mask, scribble_img, brush_label, best_mask):
        seperate_scribble_masks, last_scribble_mask = get_scribbles(seperate_scribble_masks, last_scribble_mask, scribble_img)
        click_input_viz = viz_pred_mask(input_img, best_mask, click_coords, click_labels, bbox_coords, seperate_scribble_masks, binary_checkbox)
        return click_input_viz, seperate_scribble_masks, last_scribble_mask

    click_tab.select(fn=update_click_img,
        inputs=[input_img, click_coords, click_labels, bbox_coords, seperate_scribble_masks, 
                binary_checkbox, last_scribble_mask, scribble_img, brush_label, best_mask],
        outputs=[click_img, seperate_scribble_masks, last_scribble_mask],
        api_name="update_click_img"
    )

    # ----------------------------------------------
    # Scribbles
    # ----------------------------------------------

    def change_brush_color(seperate_scribble_masks, last_scribble_mask, scribble_img, label):
        if label == "Negative (red)":
            brush_update = gr.update(brush=red_brush)
        elif label == "Positive (green)":
            brush_update = gr.update(brush=green_brush)
        else:
            raise TypeError("Invalid brush color")
        return seperate_scribble_masks, last_scribble_mask, brush_update

    brush_label.change(fn=change_brush_color,
        inputs=[seperate_scribble_masks, last_scribble_mask, scribble_img, brush_label],
        outputs=[seperate_scribble_masks, last_scribble_mask, scribble_img],
        api_name="change_brush_color"
    )

if __name__ == "__main__":
    demo.queue(api_open=False).launch(show_api=False, share=True)
