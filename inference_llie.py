import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob

import cv2
import argparse

from basicsr.archs.lied_arch import LIED

parser = argparse.ArgumentParser(description='Demo Low-light Image Enhancement')
parser.add_argument('--test_path', default='./datasets/LOLv1/test/low/', type=str, help='Input images')
parser.add_argument('--result_path', default='./results/LOLv1/', type=str, help='Directory for results')
parser.add_argument('--ckpt',
                    default='./checkpoints/LOLv1/models/model_bestPSNR.pth', type=str,
                    help='Path to weights')

args = parser.parse_args()


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

inp_dir = args.test_path
out_dir = args.result_path

os.makedirs(out_dir, exist_ok=True)

files = natsorted(#glob(os.path.join(inp_dir, '*.jpg')) +
                  glob(os.path.join(inp_dir, '*.JPG'))
                  + glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.PNG')))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

# Load corresponding models architecture and weights
model = LIED()
model.cuda()

load_checkpoint(model, args.ckpt)
model.eval()

print('restoring images......')

mul = 16
index = 0
psnr_val_rgb = []
for file_ in files:
    img = Image.open(file_).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()

    # Pad the input if not_multiple_of 16
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    with torch.no_grad():
        restored = model(input_)

    restored = torch.clamp(restored, 0, 1)
    restored = restored[:, :, :h, :w]
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])

    f = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img((os.path.join(out_dir, f + '.png')), restored)
    index += 1
    print('%d/%d' % (index, len(files)))

print(f"Files saved at {out_dir}")
print('finish !')
