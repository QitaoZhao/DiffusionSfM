import os
import time
import shutil
import argparse
import functools
import torch
import torchvision
from PIL import Image
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import trimesh

from diffusionsfm.dataset.custom import CustomDataset
from diffusionsfm.dataset.co3d_v2 import unnormalize_image
from diffusionsfm.inference.load_model import load_model
from diffusionsfm.inference.predict import predict_cameras
from diffusionsfm.utils.visualization import add_scene_cam


def info_fn():
    gr.Info("Data preprocessing completed!")


def get_select_index(evt: gr.SelectData):
    selected = evt.index
    return examples_full[selected][0], selected


def check_img_input(control_image):
    if control_image is None:
        raise gr.Error("Please select or upload an input image.")


def preprocess(args, image_block, selected):
    cate_name = time.strftime("%m%d_%H%M%S") if selected is None else examples_list[selected]

    demo_dir = os.path.join(args.output_dir, f'demo/{cate_name}')
    shutil.rmtree(demo_dir, ignore_errors=True)

    os.makedirs(os.path.join(demo_dir, 'source'), exist_ok=True)
    os.makedirs(os.path.join(demo_dir, 'processed'), exist_ok=True)

    dataset = CustomDataset(image_block)
    batch = dataset.get_data()
    batch['cate_name'] = cate_name

    processed_image_block = []
    for i, file_path in enumerate(image_block):
        file_name = os.path.basename(file_path)
        raw_img = Image.open(file_path)
        try:
            raw_img.save(os.path.join(demo_dir, 'source', file_name))
        except OSError:
            raw_img.convert('RGB').save(os.path.join(demo_dir, 'source', file_name))

        batch['image_for_vis'][i].save(os.path.join(demo_dir, 'processed', file_name))
        processed_image_block.append(os.path.join(demo_dir, 'processed', file_name))

    return processed_image_block, batch


def transform_cameras(pred_cameras):
    num_cameras = pred_cameras.R.shape[0]
    Rs = pred_cameras.R.transpose(1, 2).detach()
    ts = pred_cameras.T.unsqueeze(-1).detach()
    c2ws = torch.zeros(num_cameras, 4, 4)
    c2ws[:, :3, :3] = Rs
    c2ws[:, :3, -1:] = ts
    c2ws[:, 3, 3] = 1
    c2ws[:, :2] *= -1  # PyTorch3D to OpenCV
    c2ws = torch.linalg.inv(c2ws).numpy()

    return c2ws


def run_inference(args, cfg, model, batch):
    device = args.device
    images = batch["image"].to(device)
    crop_parameters = batch["crop_parameters"].to(device)

    _, additional_cams = predict_cameras(
        model=model,
        images=images,
        device=device,
        crop_parameters=crop_parameters,
        num_patches_x=cfg.training.full_num_patches_x,
        num_patches_y=cfg.training.full_num_patches_y,
        additional_timesteps=list(range(11)),
        calculate_intrinsics=True,
        max_num_images=8,
        mode="segment",
        return_rays=True,
        use_homogeneous=True,
        seed=0,
    )
    pred_cameras, pred_rays = additional_cams[10]

    # Unnormalize and resize input images
    images = unnormalize_image(images, return_numpy=False, return_int=False)
    images = torchvision.transforms.Resize(256)(images)
    rgbs = images.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    xyzs = pred_rays.get_segments().view(-1, 3).cpu()

    # Create point cloud and scene
    scene = trimesh.Scene()
    point_cloud = trimesh.points.PointCloud(xyzs, colors=rgbs)
    scene.add_geometry(point_cloud)

    # Add predicted cameras to the scene
    num_images = images.shape[0]
    c2ws = transform_cameras(pred_cameras)
    cmap = plt.get_cmap("hsv")

    for i, c2w in enumerate(c2ws):
        color_rgb = (np.array(cmap(i / num_images))[:3] * 255).astype(int)
        add_scene_cam(
            scene=scene,
            c2w=c2w,
            edge_color=color_rgb,
            image=None,
            focal=None,
            imsize=(256, 256),
            screen_width=0.1
        )

    # Export GLB
    cate_name = batch['cate_name']
    output_path = os.path.join(args.output_dir, f'demo/{cate_name}/{cate_name}.glb')
    scene.export(output_path)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output/multi_diffusionsfm_dense', type=str, help='Output directory')
    parser.add_argument('--device', default='cuda', type=str, help='Device to run inference on')
    args = parser.parse_args()

    _TITLE = "DiffusionSfM: Predicting Structure and Motion via Ray Origin and Endpoint Diffusion"
    _DESCRIPTION = """
    <div>
    <a style="display:inline-block" href="https://qitaozhao.github.io/DiffusionSfM"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
    <a style="display:inline-block; margin-left: .5em" href='https://github.com/QitaoZhao/SparseAGS'><img src='https://img.shields.io/github/stars/QitaoZhao/SparseAGS?style=social'/></a>
    </div>
    DiffusionSfM learns to predict scene geometry and camera poses as pixel-wise ray origins and endpoints using a denoising diffusion model.
    """

    # Load demo examples
    examples_list = ["kew_gardens_ruined_arch", "jellycat", "kotor_cathedral", "jordan"]
    examples_full = []
    for example in examples_list:
        folder = os.path.join(os.path.dirname(__file__), "data/demo", example)
        examples = sorted(os.path.join(folder, x) for x in os.listdir(folder))
        examples_full.append([examples])

    model, cfg = load_model(args.output_dir, device=args.device)
    print("Loaded DiffusionSfM model!")

    preprocess = functools.partial(preprocess, args)
    run_inference = functools.partial(run_inference, args, cfg, model)

    with gr.Blocks(title=_TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {_TITLE}")
        gr.Markdown(_DESCRIPTION)

        with gr.Row(variant='panel'):
            with gr.Column(scale=2):
                image_block = gr.Files(file_count="multiple", label="Upload Images")

                gr.Markdown(
                    "You can run our model by either: (1) **Uploading images** above "
                    "or (2) selecting a **pre-collected example** below."
                )

                gallery = gr.Gallery(
                    value=[example[0][0] for example in examples_full],
                    label="Examples",
                    show_label=True,
                    columns=[4],
                    rows=[1],
                    object_fit="contain",
                    height="256",
                )

                selected = gr.State()
                batch = gr.State()

                preprocessed_data = gr.Gallery(
                    label="Preprocessed Images",
                    show_label=True,
                    columns=[4],
                    rows=[1],
                    object_fit="contain",
                    height="256",
                )

                with gr.Row(variant='panel'):
                    run_inference_btn = gr.Button("Run Inference")

            with gr.Column(scale=4):
                output_3D = gr.Model3D(
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    height=520,
                    zoom_speed=0.5,
                    pan_speed=0.5,
                    label="3D Point Cloud and Cameras"
                )

        # Link image gallery selection
        gallery.select(
            fn=get_select_index,
            inputs=None,
            outputs=[image_block, selected]
        ).success(
            fn=preprocess,
            inputs=[image_block, selected],
            outputs=[preprocessed_data, batch],
            queue=False,
            show_progress="full"
        )

        # Handle user uploads
        image_block.upload(
            preprocess,
            inputs=[image_block],
            outputs=[preprocessed_data, batch],
            queue=False,
            show_progress="full"
        ).success(info_fn, None, None)

        # Run 3D reconstruction
        run_inference_btn.click(
            check_img_input,
            inputs=[image_block],
            queue=False
        ).success(
            run_inference,
            inputs=[batch],
            outputs=[output_3D]
        )

    demo.queue().launch(share=True)