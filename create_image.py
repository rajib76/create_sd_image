import random
import cv2
import numpy as np
from diffusers import LMSDiscreteScheduler
from PIL import Image

from stable_diffusion_engine import StableDiffusionEngine
import streamlit as st



def main(prompt):
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        tensor_format="np"
    )
    seed = random.randint(0, 2 ** 30)
    np.random.seed(seed)
    engine = StableDiffusionEngine(
        model="bes-dev/stable-diffusion-v1-4-openvino",
        scheduler=scheduler,
        tokenizer="openai/clip-vit-large-patch14"
    )
    image = engine(
        prompt=prompt,
        init_image=None,
        mask=None,
        strength=0.5,
        num_inference_steps=32,
        guidance_scale=7.5,
        eta=0.0
    )
    return image


if __name__ == "__main__":
    prompt = st.text_input("What is your image prompt")
    if prompt >"":
        if st.button("Create my image"):
            with st.spinner("Your image is cooking in the kitchen..."):
                image = main(prompt)
                cv2.imwrite("output.png", image)
                img = Image.open("output.png")
                st.image(img, caption = prompt)

    # prompt = "hyperrealistic image , featuring rain forest of amazon, stunning octane comprehensive render, istvan sandorfi greg rutkowski, unreal engine, symmetrical, dim volumetric cinematic lighting, hyper detailed, intricate, masterpiece, trending on cgsociety"
    # main(prompt)
