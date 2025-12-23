import os
import csv
import numpy as np
from PIL import Image
import onnxruntime as ort
from collections import defaultdict

class WD14Tagger:
    def __init__(self, model_dir, providers=None):
        self.model_dir = model_dir
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sessions = {}

    def _load_model(self, model_name):
        if model_name in self.sessions:
            return self.sessions[model_name]

        onnx_path = os.path.join(self.model_dir, model_name + ".onnx")
        csv_path = os.path.join(self.model_dir, model_name + ".csv")

        session = ort.InferenceSession(onnx_path, providers=self.providers)

        tags = []
        general_index = None
        character_index = None

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if general_index is None and row[2] == "0":
                    general_index = reader.line_num - 2
                elif character_index is None and row[2] == "4":
                    character_index = reader.line_num - 2
                tags.append(row[1])

        self.sessions[model_name] = (session, tags, general_index, character_index)
        return self.sessions[model_name]

    def _run_model(
        self,
        image: Image.Image,
        model_name: str,
        threshold: float,
        character_threshold: float,
    ):
        session, tags, g_idx, c_idx = self._load_model(model_name)

        input_tensor = session.get_inputs()[0]
        size = input_tensor.shape[1]

        ratio = size / max(image.size)
        new_size = tuple(int(x * ratio) for x in image.size)
        image = image.resize(new_size, Image.LANCZOS)

        square = Image.new("RGB", (size, size), (255, 255, 255))
        square.paste(image, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))

        img = np.array(square).astype(np.float32)
        img = img[:, :, ::-1]  # RGB → BGR
        img = np.expand_dims(img, 0)

        probs = session.run(
            [session.get_outputs()[0].name],
            {input_tensor.name: img}
        )[0][0]

        results = list(zip(tags, probs))

        general = [t for t in results[g_idx:c_idx] if t[1] >= threshold]
        character = [t for t in results[c_idx:] if t[1] >= character_threshold]

        return character + general

    def tag_single(
        self,
        image: Image.Image,
        model_name: str,
        threshold: float = 0.35,
        character_threshold: float = 0.35,
        replace_underscore: bool = True,
    ):
        """
        Tag a single image with a single model.
        Returns a list of (tag, score) tuples.
        """
        tags = self._run_model(
            image,
            model_name,
            threshold=threshold,
            character_threshold=character_threshold,
        )

        if replace_underscore:
            tags = [(t.replace("_", " "), s) for t, s in tags]

        return tags

    def tag_multi(self, image, models, replace_underscore=True):
        all_tags = defaultdict(float)

        for model_name, cfg in models.items():
            tags = self.tag_single(
                image,
                model_name,
                threshold=cfg["threshold"],
                character_threshold=cfg["character_threshold"],
                replace_underscore=replace_underscore,
            )

            for tag, score in tags:
                all_tags[tag] = max(all_tags[tag], score)

        return sorted(all_tags.keys())