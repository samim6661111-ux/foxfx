"""
foxfx v2: teacher -> student pipeline.
Usage (Colab GPU, or GitHub Actions CPU via .github/workflows/train.yml):
  !pip -q install diffusers transformers accelerate safetensors onnx
  1) Put YOUR game screenshots in /content/shots  (300-1500 jpg/png, varied scenes)
  2) !python train_foxfx_v2.py teacher     # makes /content/pairs (slow, one time)
  3) !python train_foxfx_v2.py student     # trains CNN, exports /content/out/fox_net.onnx
  4) !python train_foxfx_v2.py preview     # saves /content/out/preview.png (input | output)
Use only your own screenshots as training data.
"""
import os, sys, glob, random
import numpy as np
from PIL import Image

SHOTS = os.environ.get("SHOTS", "shots")
PAIRS = os.environ.get("PAIRS", "pairs")
OUT = os.environ.get("OUT", "out")
W, H = 576, 256  # multiples of 8, ~2.25:1 (matches phone landscape recordings)
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "0"))      # 0 = all
SD_STEPS = int(os.environ.get("SD_STEPS", "25"))
STUDENT_STEPS = int(os.environ.get("STUDENT_STEPS", "6000"))

PROMPT = ("photorealistic city street, wet asphalt road with sharp reflections of sky and palm trees, "
          "rich saturated deep blue sky, crisp detailed textures, high dynamic range, natural sunlight, "
          "realistic car paint and glass, 8k photo")
NEG = "cartoon, blurry, low quality, deformed, text, watermark, oversaturated"


def teacher():
    import torch
    from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
    from transformers import pipeline as hf_pipeline

    os.makedirs(PAIRS, exist_ok=True)
    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    dt = torch.float16 if cuda else torch.float32
    depth_est = hf_pipeline("depth-estimation", model="LiheYoung/depth-anything-small-hf",
                            device=0 if cuda else -1)
    cn = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=dt)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn, torch_dtype=dt, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if cuda:
        pipe.enable_model_cpu_offload()

    if os.path.exists("shots.zip") and not glob.glob(f"{SHOTS}/*.jpg"):
        import zipfile
        os.makedirs(SHOTS, exist_ok=True)
        zipfile.ZipFile("shots.zip").extractall(SHOTS)
    files = sorted(sum([glob.glob(f"{SHOTS}/*.{e}") for e in ("jpg", "jpeg", "png")], []))
    if MAX_IMAGES and len(files) > MAX_IMAGES:   # spread evenly across all videos
        files = [files[int(k * len(files) / MAX_IMAGES)] for k in range(MAX_IMAGES)]
    print("screenshots:", len(files), "device:", dev)
    for i, f in enumerate(files):
        dst_in, dst_out = f"{PAIRS}/{i:05d}_in.png", f"{PAIRS}/{i:05d}_out.png"
        if os.path.exists(dst_out):
            continue
        img = Image.open(f).convert("RGB").resize((W, H), Image.LANCZOS)
        d = depth_est(img)["depth"].convert("RGB").resize((W, H))
        g = torch.Generator(dev).manual_seed(1234)  # fixed seed = consistent look
        res = pipe(PROMPT, negative_prompt=NEG, image=img, control_image=d,
                   strength=0.4,                        # low = keeps game geometry
                   controlnet_conditioning_scale=1.0,
                   num_inference_steps=SD_STEPS, guidance_scale=7.0, generator=g).images[0]
        img.save(dst_in); res.save(dst_out)
        print(i + 1, "/", len(files), flush=True)


def student():
    import torch, torch.nn as nn, torch.nn.functional as F
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def conv(i, o, s=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.ReLU(inplace=True))

    class FoxNet(nn.Module):
        """Small U-Net. Downsample x4 gives big receptive field so it can learn
        'dark flat road -> wet reflective', 'sky -> deeper blue'. Predicts a residual."""
        def __init__(self, c=16):
            super().__init__()
            self.e1 = nn.Sequential(conv(3, c), conv(c, c))
            self.e2 = nn.Sequential(conv(c, c * 2, 2), conv(c * 2, c * 2))
            self.e3 = nn.Sequential(conv(c * 2, c * 4, 2), conv(c * 4, c * 4), conv(c * 4, c * 4))
            self.d2 = nn.Sequential(conv(c * 4 + c * 2, c * 2), conv(c * 2, c * 2))
            self.d1 = nn.Sequential(conv(c * 2 + c, c), conv(c, c))
            self.out = nn.Conv2d(c, 3, 3, 1, 1)

        def forward(self, x):
            a = self.e1(x); b = self.e2(a); c_ = self.e3(b)
            u2 = F.interpolate(c_, scale_factor=2, mode="bilinear", align_corners=False)
            d2 = self.d2(torch.cat([u2, b], 1))
            u1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
            d1 = self.d1(torch.cat([u1, a], 1))
            return torch.clamp(x + 0.5 * torch.tanh(self.out(d1)), 0, 1)

    ins = sorted(glob.glob(f"{PAIRS}/*_in.png"))
    assert ins, "run teacher first"
    load = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)
    X = torch.stack([load(p) for p in ins]); Y = torch.stack([load(p.replace("_in", "_out")) for p in ins])
    n_val = max(4, len(X) // 20)
    Xv, Yv, X, Y = X[:n_val], Y[:n_val], X[n_val:], Y[n_val:]

    net = FoxNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3)
    steps, bs, crop = STUDENT_STEPS, 8, 192
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)

    def grad(t):
        return t[..., :, 1:] - t[..., :, :-1], t[..., 1:, :] - t[..., :-1, :]

    for s in range(steps):
        idx = torch.randint(0, len(X), (bs,))
        xb, yb = [], []
        for i in idx:
            y0 = random.randint(0, H - crop); x0 = random.randint(0, W - crop)
            a = X[i, :, y0:y0 + crop, x0:x0 + crop]; b = Y[i, :, y0:y0 + crop, x0:x0 + crop]
            if random.random() < 0.5:
                a, b = a.flip(-1), b.flip(-1)
            xb.append(a); yb.append(b)
        xb, yb = torch.stack(xb).to(dev), torch.stack(yb).to(dev)
        p = net(xb)
        gx, gy = grad(p); tx, ty = grad(yb)
        loss = F.l1_loss(p, yb) + 0.5 * (F.l1_loss(gx, tx) + F.l1_loss(gy, ty))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 250 == 0:
            with torch.no_grad():
                v = F.l1_loss(net(Xv.to(dev)), Yv.to(dev)).item()
            print(f"step {s} loss {loss.item():.4f} val_l1 {v:.4f}")

    net.eval().cpu()
    torch.save(net.state_dict(), f"{OUT}/fox_net.pt")
    dummy = torch.zeros(1, 3, H, W)
    torch.onnx.export(net, dummy, f"{OUT}/fox_net.onnx", opset_version=13,
                      input_names=["x"], output_names=["y"])
    print("params:", sum(p.numel() for p in net.parameters()))
    print("saved", OUT)


def preview():
    import onnxruntime as ort
    sess = ort.InferenceSession(f"{OUT}/fox_net.onnx")
    ins = sorted(glob.glob(f"{PAIRS}/*_in.png"))[:3]
    rows = []
    for p in ins:
        x = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)[None].transpose(0, 3, 1, 2) / 255
        y = sess.run(None, {"x": x})[0][0].transpose(1, 2, 0)
        rows.append(np.concatenate([x[0].transpose(1, 2, 0), y], 1))
    Image.fromarray((np.concatenate(rows, 0) * 255).clip(0, 255).astype(np.uint8)).save(f"{OUT}/preview.png")
    print("saved preview.png")


if __name__ == "__main__":
    {"teacher": teacher, "student": student, "preview": preview}[sys.argv[1]]()
