import os
from glob import glob
from os.path import join
import random
import string
import numpy as np
import argparse
import torch
import torchvision
from PIL import Image
from tqdm.notebook import trange
from functools import partial
import jax
import jax.numpy as jnp
from flax.jax_utils import replicate
import matplotlib.pyplot as plt
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel
from flax.training.common_utils import shard, shard_prng_key


parser = argparse.ArgumentParser(description="This script generates image based on the prompt you give using DALL-E mini")
parser.add_argument(
    '--model', 
    type=str,
    default="mega",
    choices=["mega", "mini"],
    help="Prompt to generate an image of"
)
parser.add_argument(
    '--prompt', 
    type=str, 
    default="a city skyline on a cloudy night",
    help="Prompt to generate an image of"
)
parser.add_argument(
    '--output', 
    type=str, 
    default="./output",
    help="Path to save images to"
)
parser.add_argument(
    '--show', 
    action='store_true',
    help="Show saved images in a window"
)
args = parser.parse_args()

def show(pil_images, nrow=4, size=14, save_dir=None, show=True):
    """
    :param pil_images: list of images in PIL
    :param nrow: number of rows
    :param size: size of the images
    :param save_dir: dir for separately saving of images, example: save_dir='./output'
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        count = len(glob(join(save_dir, 'img_*.png')))
        for i, pil_image in enumerate(pil_images):
            pil_image.save(join(save_dir, f'img_{count+i}.png'))

    pil_images = [pil_image.convert('RGB') for pil_image in pil_images]
    imgs = torchvision.utils.make_grid(pil_list_to_torch_tensors(pil_images), nrow=nrow)
    if not isinstance(imgs, list):
        imgs = [imgs.cpu()]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(size, size))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        if save_dir is not None:
            count = len(glob(join(save_dir, 'group_*.png')))
            img.save(join(save_dir, f'group_{count+i}.png'))
        if show:
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if show:
        fix.show()
        plt.show()

def pil_list_to_torch_tensors(pil_images):
    result = []
    for pil_image in pil_images:
        image = np.array(pil_image, dtype=np.uint8)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).unsqueeze(0)
        result.append(image)
    return torch.cat(result, dim=0)

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

# model inference
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )

# decode image
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)

# Model references
if args.model == "mega":
    DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"
else:
    DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"

DALLE_COMMIT_ID = None

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

# check how many devices are available
jax.local_device_count()

# Load dalle-mini
model, params = DalleBart.from_pretrained(
    DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
)

# Load VQGAN
vqgan, vqgan_params = VQModel.from_pretrained(
    VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
)

# Model parameters are replicated on each device for faster inference.
params = replicate(params)
vqgan_params = replicate(vqgan_params)

# create a random key
seed = random.randint(0, 2**32 - 1)
key = jax.random.PRNGKey(seed)

processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)
tokenized_prompt = processor([args.prompt])
tokenized_prompt = replicate(tokenized_prompt)

# number of predictions per prompt
n_predictions = 8

# We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
gen_top_k = None
gen_top_p = None
temperature = None
cond_scale = 10.0

# generate images
images = []
for i in trange(max(n_predictions // jax.device_count(), 1)):
    # get a new key
    key, subkey = jax.random.split(key)
    # generate images
    encoded_images = p_generate(
        tokenized_prompt,
        shard_prng_key(subkey),
        params,
        gen_top_k,
        gen_top_p,
        temperature,
        cond_scale,
    )
    # remove BOS
    encoded_images = encoded_images.sequences[..., 1:]
    # decode images
    decoded_images = p_decode(encoded_images, vqgan_params)
    decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
    for decoded_img in decoded_images:
        img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
        # TODO: implement super resolution model
        img = img.resize((512,512))
        images.append(img)

if args.show:
    show(images, 3, save_dir=args.output)
else:
    for i in images:
        random_str = get_random_string(6)
        print(images)
        print(i)
        i.save("{}/image-{}.jpg".format(args.output, random_str))
