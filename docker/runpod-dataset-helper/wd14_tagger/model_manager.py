import os
import aiohttp
import asyncio

HF_MODELS = {
    "wd-v1-4-moat-tagger-v2":
        "https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2"
}

async def download_model(model_name, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    base = HF_MODELS[model_name] + "/resolve/main/"

    async with aiohttp.ClientSession() as session:
        for src, dst in [
            ("model.onnx", model_name + ".onnx"),
            ("selected_tags.csv", model_name + ".csv"),
        ]:
            out = os.path.join(model_dir, dst)
            if os.path.exists(out):
                continue

            async with session.get(base + src) as r:
                r.raise_for_status()
                with open(out, "wb") as f:
                    f.write(await r.read())