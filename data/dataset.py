import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from .util.sampling import (uniform_mask, twoside_mask, nonuniform_mask)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('L')

######### RadioMapSeer #########

class RadioMap(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader, samples_per_map=80):

        self.data_root = data_root
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.sampling_mode = self.mask_config['sampling_mode']
        self.sampling_num = self.mask_config['sampling_num']
        self.image_size = image_size

        # Setting paths relative to the base data_root
        building_map_root = os.path.join(data_root, 'png', 'buildings_complete')
        self.gain_root = os.path.join(data_root, 'gain', 'DPM')
        self.tx_location_root = os.path.join(data_root, 'png', 'antennas')
        self.samples_per_map = samples_per_map
        self.building_maps = make_dataset(building_map_root)
        
        self.total_imgs = len(self.building_maps) * samples_per_map
        if data_len > 0:
            self.total_imgs = min(self.total_imgs, data_len)

    def __getitem__(self, index):
        map_index = index // self.samples_per_map
        sample_index = index % self.samples_per_map

        building_map_path = self.building_maps[map_index]
        base_file_name = os.path.splitext(os.path.basename(building_map_path))[0]
        gain_file_name = f"{base_file_name}_{sample_index}.png"
        tx_location_file_name = gain_file_name

        gain_path = os.path.join(self.gain_root, gain_file_name)
        tx_location_path = os.path.join(self.tx_location_root, tx_location_file_name)

        building_map = self.tfs(self.loader(building_map_path))
        img_gain = self.tfs(self.loader(gain_path))
        tx_location = self.tfs(self.loader(tx_location_path))

        mask = self.get_sampling()
        img_sample = img_gain*mask

        mask_bool = mask == 1
        building_bool = building_map == 1
        combined_mask = ~(building_bool | mask_bool).int()
        
        mask_image = img_gain * (1.-combined_mask)

        cond_image = torch.cat([building_map, tx_location, img_sample], dim=0)
        mask_image = torch.tensor(mask_image, dtype=torch.float)
        cond_img = torch.tensor(cond_image, dtype=torch.float) 
        mask = torch.tensor(combined_mask, dtype=float)
        
        ret = {
            'building_map': building_map,
            'img_gain': img_gain, # gt
            'tx_location': tx_location,
            'mask_image': mask_image, #mask_image
            'cond_image': cond_img,   #cond_image
            'mask': mask.float(), 
            'path': gain_file_name  #path
        }
        return ret

    def get_sampling(self):
        if self.sampling_mode == "uniform":
            samples = uniform_mask(self.image_size, self.sampling_num)
        elif self.sampling_mode == "twoside":
            samples = twoside_mask(self.image_size)
        elif self.sampling_mode == "nonuniform":
            samples = nonuniform_mask(self.image_size)
        else:
            raise NotImplementedError(
                f'Mask mode {self.sampling_mode} has not been implemented.')
        

        return torch.from_numpy(samples).permute(2,0,1)

    def __len__(self):
        return self.total_imgs