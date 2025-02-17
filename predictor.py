import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import network

class Predictor:
    """
    Wrapper for ScribblePrompt Unet model
    """
    def __init__(self, path: str, verbose: bool = False):
        
        self.verbose = verbose

        assert path.exists(), f"Checkpoint {path} does not exist"
        self.path = path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.load()
        self.model.eval()
        self.to_device()

    def build_model(self):
        """
        Build the model
        """
        self.model = network.UNet(
            in_channels = 5,
            out_channels = 1,
            features = [192, 192, 192, 192],
        )

    def load(self):
        """
        Load the state of the model from a checkpoint file.
        """
        with (self.path).open("rb") as f:
            state = torch.load(f, map_location=self.device)
            self.model.load_state_dict(state, strict=True)
            if self.verbose:
                print(
                    f"Loaded checkpoint from {self.path} to {self.device}"
                )
        
    def to_device(self):
        """
        Move the model to cpu or gpu
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def predict(self, prompts: Dict[str, any], img_features: Optional[torch.Tensor] = None, multimask_mode: bool = False):
        """
        Make predictions for a specific slice or frame.

        Args:
            prompts (Dict[str, any]): Dictionary containing model input data.
            img_features (Optional[torch.Tensor]): Precomputed image features (for SAM models).
            multimask_mode (bool): Enable multi-mask output (if supported by the model).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
                - mask (torch.Tensor): H x W prediction mask.
                - img_features (Optional[torch.Tensor]): Image features (if applicable).
                - low_res_mask (torch.Tensor): Low-resolution logits.
        """
        if self.verbose:
            print("point_coords", prompts.get("point_coords", None))
            print("point_labels", prompts.get("point_labels", None))
            print("box", prompts.get("box", None))
            print("img shape", prompts.get("img").shape)
            if prompts.get("scribble") is not None:
                print("scribble", prompts.get("scribble").shape)

        # Extract selected slice/frame if 3D input
        img = prompts.get("img")
        if img.ndim == 3:  # 3D single-channel input (e.g., CT/MRI)
            slice_idx = prompts.get("slice_idx", 0)
            img = img[..., slice_idx]
        elif img.ndim == 4:  # 3D multi-channel input (e.g., RGB video)
            frame_idx = prompts.get("frame_idx", 0)
            img = img[..., frame_idx, :]
        prompts["img"] = img

        # Store original image shape
        original_shape = img.shape[-2:]

        # Rescale inputs to match model input size (e.g., 128x128)
        prompts = rescale_inputs(prompts)

        # Prepare model inputs (e.g., concatenate channels for scribbles, bounding boxes, etc.)
        x = prepare_inputs(prompts).float()

        # Run the model
        with torch.no_grad():
            yhat = self.model(x.to(self.device)).cpu()

        # Apply sigmoid activation to obtain binary mask probabilities
        mask = torch.sigmoid(yhat)

        # Resize mask back to original resolution
        mask = F.interpolate(mask, size=original_shape, mode="bilinear").squeeze()
        return mask, None, yhat

        

# -----------------------------------------------------------------------------
# Prepare inputs
# -----------------------------------------------------------------------------

def rescale_inputs(inputs: Dict[str,any], res=128):
    """
    Rescale the inputs 
    """ 
    h,w = inputs['img'].shape[-2:]

    if h != res or w != res:
        
        inputs.update(dict(
            img = F.interpolate(inputs['img'], size=(res,res), mode='bilinear')
        ))

        if inputs.get('scribble') is not None:
            inputs.update({
                'scribble': F.interpolate(inputs['scribble'], size=(res,res), mode='bilinear') 
            })
        
        if inputs.get("box") is not None:
            boxes = inputs.get("box").clone()
            coords = boxes.reshape(-1, 2, 2)
            coords[..., 0] = coords[..., 0] * (res / w)
            coords[..., 1] = coords[..., 1] * (res / h)
            inputs.update({'box': coords.reshape(1, -1, 4).int()})
        
        if inputs.get("point_coords") is not None:
            coords = inputs.get("point_coords").clone()
            coords[..., 0] = coords[..., 0] * (res / w)
            coords[..., 1] = coords[..., 1] * (res / h)
            inputs.update({'point_coords': coords.int()})

    return inputs

def prepare_inputs(inputs: Dict[str,torch.Tensor], device = None) -> torch.Tensor:
    """
    Prepare inputs for ScribblePrompt Unet

    Returns: 
        x (torch.Tensor): B x 5 x H x W
    """
    img = inputs['img']
    if device is None:
        device = img.device

    img = img.to(device)
    shape = tuple(img.shape[-2:])
    
    if inputs.get("box") is not None:
        # Embed bounding box
        # Input: B x 1 x 4 
        # Output: B x 1 x H x W
        box_embed = bbox_shaded(inputs['box'], shape=shape, device=device)
    else:
        box_embed = torch.zeros(img.shape, device=device)

    if inputs.get("point_coords") is not None:
        # Encode points
        # B x 2 x H x W
        scribble_click_embed = click_onehot(inputs['point_coords'], inputs['point_labels'], shape=shape)
    else:
        scribble_click_embed = torch.zeros((img.shape[0], 2) + shape, device=device)

    if inputs.get("scribble") is not None:
        # Combine scribbles with click encoding
        # B x 2 x H x W
        scribble_click_embed = torch.clamp(scribble_click_embed + inputs.get('scribble'), min=0.0, max=1.0)

    if inputs.get('mask_input') is not None:
        # Previous prediction
        mask_input = inputs['mask_input']
    else:
        # Initialize empty channel for mask input
        mask_input = torch.zeros(img.shape, device=img.device)

    x = torch.cat((img, box_embed, scribble_click_embed, mask_input), dim=-3)
    # B x 5 x H x W

    return x
    
# -----------------------------------------------------------------------------
# Encode clicks and bounding boxes
# -----------------------------------------------------------------------------

def click_onehot(point_coords, point_labels, shape: Tuple[int,int] = (128,128), indexing='xy'):
    """
    Represent clicks as two HxW binary masks (one for positive clicks and one for negative) 
    with 1 at the click locations and 0 otherwise

    Args:
        point_coords (torch.Tensor): BxNx2 tensor of xy coordinates
        point_labels (torch.Tensor): BxN tensor of labels (0 or 1)
        shape (tuple): output shape     
    Returns:
        embed (torch.Tensor): Bx2xHxW tensor 
    """
    assert indexing in ['xy','uv'], f"Invalid indexing: {indexing}"
    assert len(point_coords.shape) == 3, "point_coords must be BxNx2"
    assert point_coords.shape[-1] == 2, "point_coords must be BxNx2"
    assert point_labels.shape[-1] == point_coords.shape[1], "point_labels must be BxN"
    assert len(shape)==2, f"shape must be 2D: {shape}"

    device = point_coords.device
    batch_size = point_coords.shape[0]
    n_points = point_coords.shape[1]

    embed = torch.zeros((batch_size,2)+shape, device=device)
    labels = point_labels.flatten().float()

    idx_coords = torch.cat((
        torch.arange(batch_size, device=device).reshape(-1,1).repeat(1,n_points)[...,None], 
        point_coords
    ), axis=2).reshape(-1,3)

    if indexing=='xy':
        embed[ idx_coords[:,0], 0, idx_coords[:,2], idx_coords[:,1] ] = labels
        embed[ idx_coords[:,0], 1, idx_coords[:,2], idx_coords[:,1] ] = 1.0-labels
    else:
        embed[ idx_coords[:,0], 0, idx_coords[:,1], idx_coords[:,2] ] = labels
        embed[ idx_coords[:,0], 1, idx_coords[:,1], idx_coords[:,2] ] = 1.0-labels

    return embed


def bbox_shaded(boxes, shape: Tuple[int,int] = (128,128), device='cpu'):
    """
    Represent bounding boxes as a binary mask with 1 inside boxes and 0 otherwise

    Args:
        boxes (torch.Tensor): Bx1x4 [x1, y1, x2, y2]
    Returns:
        bbox_embed (torch.Tesor): Bx1xHxW according to shape
    """
    assert len(shape)==2, "shape must be 2D"
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.int().cpu().numpy()

    batch_size = boxes.shape[0]
    n_boxes = boxes.shape[1]
    bbox_embed = torch.zeros((batch_size,1)+tuple(shape), device=device, dtype=torch.float32)

    if boxes is not None:
        for i in range(batch_size):
            for j in range(n_boxes):
                x1, y1, x2, y2 = boxes[i,j,:]
                x_min = min(x1,x2)
                x_max = max(x1,x2)
                y_min = min(y1,y2)
                y_max = max(y1,y2)
                bbox_embed[ i, 0, y_min:y_max, x_min:x_max ] = 1.0

    return bbox_embed