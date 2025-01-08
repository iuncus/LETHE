import os
import random
import torch
import argparse
from diffusers import StableDiffusionPipeline, PNDMScheduler
from transformers import CLIPTextModel

model = "runwayml/stable-diffusion-v1-5"
device = "cuda:0"

# python infer_batch.py "red apple" R_maximize_BG_no_Norm

def main(args):

    # pipe = StableDiffusionPipeline.from_pretrained(model, safety_checker=None, revision = "fp16", torch_dtype = torch.float16,).to(device)
    pipe = StableDiffusionPipeline.from_pretrained(model, safety_checker=None).to(device)
    # pipe.scheduler = PNDMScheduler(
    #     beta_start=0.00085,
    #     beta_end=0.012,
    #     beta_schedule="scaled_linear",
    #     num_train_timesteps=1000,
    #     skip_prk_steps=True,
    #     steps_offset=1
    # )
    Load_and_Infer(pipe)

def Load_and_Infer(pipe):
    encoder_folder = args.text_encoder_path
    encoder_path = "updated_CLIP/" + encoder_folder
    # seed = 6979249
    for folder in os.listdir(encoder_path):
        path = f"{encoder_path}/{folder}"
        if os.path.isdir(path):
            text_encoder = CLIPTextModel.from_pretrained(path)

            pipe.text_encoder = text_encoder.to(device)

            steps = 50
            seed = random.randint(0, 9999999)
            
            generator = torch.Generator("cpu").manual_seed(seed)
            images = pipe(
                args.prompt,
                num_inference_steps=steps,
                num_images_per_prompt=args.num_images_per_prompt,
                generator=generator
            ).images

            Save(images, seed, encoder_folder, folder)

            del text_encoder
            del images
            torch.cuda.empty_cache()

def Save(images, seed,encoder_folder, folder):
    # path = args.text_encoder_path.split("/")
    
    os.makedirs(f"{args.save_dir}/{encoder_folder}/{args.prompt}", exist_ok=True)
    # os.makedirs(f"{args.save_dir}/{encoder_folder}/PNDS/{args.prompt}", exist_ok=True)


    for i in range(len(images)):
        images[i].save(f"{args.save_dir}/{encoder_folder}/{args.prompt}/{folder}-{seed}_{i:02}.png")
        # images[i].save(f"{args.save_dir}/{encoder_folder}/PNDS/{args.prompt}/{folder}-{seed}_{i:02}.png")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('text_encoder_path', type=str)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--num_images_per_prompt', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default="exp/batch")
    args = parser.parse_args()
    
    main(args)