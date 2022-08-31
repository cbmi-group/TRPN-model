
from .few_model_oneshot import FS2_One_Shot_V1, FS2_One_Shot_V2
from .few_model_fiveshot import FS2_Five_Shot
from .few_model_ranet import FS2_One_Shot_RANet
from .unet import UNet

model_dict_segmentation = {"one_model": FS2_One_Shot_V1,
                           "five_model": FS2_Five_Shot,
                           "ranet_model": FS2_One_Shot_RANet,
                           "unet": UNet}
