from __future__ import annotations
import math
import random
import gradio as gr
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline

example_instructions = [
    "move the lemon to the right of the table"
]

def main():
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("McGill-NLP/AURORA", safety_checker=None).to("cuda")
    example_image = Image.open("example.jpg").convert("RGB")

    def load_example(
        steps: int,
        seed: int,
        text_cfg_scale: float,
        image_cfg_scale: float,
    ):
        example_instruction = random.choice(example_instructions)
        return [example_image, example_instruction] + generate(
            example_image,
            example_instruction,
            steps,
            seed,
            text_cfg_scale,
            image_cfg_scale,
        )

    def generate(
        input_image: Image.Image,
        instruction: str,
        steps: int,
        seed: int,
        text_cfg_scale: float,
        image_cfg_scale: float,
    ):
        width, height = input_image.size
        factor = 512 / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

        if instruction == "":
            return [input_image, seed]

        generator = torch.manual_seed(seed)
        edited_image = pipe(
            instruction, image=input_image,
            guidance_scale=text_cfg_scale, image_guidance_scale=image_cfg_scale,
            num_inference_steps=steps, generator=generator,
        ).images[0]
        return [seed, text_cfg_scale, image_cfg_scale, edited_image]

    def reset():
        return [50, 42, 7.5, 1.5, None]

    with gr.Blocks() as demo:
        gr.HTML("""<h1 style="font-weight: 900; margin-bottom: 10px;">
            AURORA: Learning Action and Reasoning-Centric Image Editing from Videos and Simulations
        </h1>
        <p>
            AURORA (Action Reasoning Object Attribute) enables training an instruction-guided image editing model that can perform action and reasoning-centric edits, in addition to "simpler" established object, attribute or global edits. <b> To illustrate this, please click "Load example" </b>.
        </p>""")
        
        with gr.Row():
            with gr.Column(scale=3):
                instruction = gr.Textbox(lines=1, label="Edit instruction", interactive=True)
            with gr.Column(scale=1, min_width=100):
                generate_button = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1, min_width=100):
                reset_button = gr.Button("Reset", variant="stop")
            with gr.Column(scale=1, min_width=100):
                load_button = gr.Button("Load example")

        with gr.Row():
            input_image = gr.Image(label="Input image", type="pil", interactive=True)
            edited_image = gr.Image(label=f"Edited image", type="pil", interactive=False)

        with gr.Row():
            steps = gr.Number(value=50, precision=0, label="Steps", interactive=True)
            seed = gr.Number(value=42, precision=0, label="Seed", interactive=True)
            text_cfg_scale = gr.Number(value=7.5, label=f"Text CFG", interactive=True)
            image_cfg_scale = gr.Number(value=1.5, label=f"Image CFG", interactive=True)

        load_button.click(
            fn=load_example,
            inputs=[
                steps,
                seed,
                text_cfg_scale,
                image_cfg_scale,
            ],
            outputs=[input_image, instruction, seed, text_cfg_scale, image_cfg_scale, edited_image],
        )
        generate_button.click(
            fn=generate,
            inputs=[
                input_image,
                instruction,
                steps,
                seed,
                text_cfg_scale,
                image_cfg_scale,
            ],
            outputs=[seed, text_cfg_scale, image_cfg_scale, edited_image],
        )
        reset_button.click(
            fn=reset,
            inputs=[],
            outputs=[steps, seed, text_cfg_scale, image_cfg_scale, edited_image],
        )

    demo.queue()
    demo.launch()
    # demo.launch(share=True)

if __name__ == "__main__":
    main()
