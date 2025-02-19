import nibabel as nib
from skimage.measure import marching_cubes
import numpy as np
class VolumeProcessor:
    def __init__(self):
        self.volume_data = None
        
    def load(self, filepath):
        if filepath.endswith(('.nii', '.nii.gz')):
            self.volume_data = nib.load(filepath).get_fdata()
        return self.normalize_data()
    
    def normalize_data(self):
        """标准化数据到0-255范围"""
        data_min = self.volume_data.min()
        data_max = self.volume_data.max()
        self.volume_data = (self.volume_data - data_min) / (data_max - data_min) * 255
        return self.volume_data.shape[-1]  # 返回切片数
    
    def get_slice(self, index):
        return self.volume_data[..., int(index)].astype(np.uint8)
    
    def generate_mesh(self, pred_volume):
        verts, faces, _, _ = marching_cubes(pred_volume, 0.5)
        return {"vertices": verts, "faces": faces}