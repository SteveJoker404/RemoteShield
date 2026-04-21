"""
RemoteShield model wrapper built on top of Qwen3-VL Transformers inference.

This script provides a lightweight inference class for the three task families
used in RemoteShield: scene classification, VQA, and visual grounding.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


_NUM_RE = r"-?\d+(?:\.\d+)?"
_BBOX_2D_LIST_RE = re.compile(rf"\[\s*(\[\s*{_NUM_RE}\s*,\s*{_NUM_RE}\s*,\s*{_NUM_RE}\s*,\s*{_NUM_RE}\s*\]\s*(?:,\s*\[\s*{_NUM_RE}\s*,\s*{_NUM_RE}\s*,\s*{_NUM_RE}\s*,\s*{_NUM_RE}\s*\]\s*)*)\]")
_BBOX_1D_SINGLE_RE = re.compile(rf"\[\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*\]")
_BBOX_PAREN_RE = re.compile(rf"\(\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*(?:,\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*)?\)")


SCENE_CLASSIFICATION_PROMPT = """You are RemoteShield, a vision-language assistant for remote sensing scene classification.

Read the image and the query carefully.
Output ONLY the selected scene category name.
Do not output any explanation, prefix, or extra text.

Query:
"""

VQA_PROMPT = """You are RemoteShield, a vision-language assistant for remote sensing visual question answering.

Read the image and the query carefully.
Output ONLY a short final answer.
Do not output any explanation, prefix, or extra text.

Query:
"""

VISUAL_GROUNDING_PROMPT = """You are RemoteShield, a vision-language assistant for remote sensing visual grounding.

Read the image and the query carefully.
Localize the referenced target and output ONLY bounding boxes in the format:
[[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
Do not output any explanation, prefix, or extra text.

Query:
"""

TASK_PROMPTS = {
    "scene_classification": SCENE_CLASSIFICATION_PROMPT,
    "vqa": VQA_PROMPT,
    "visual_grounding": VISUAL_GROUNDING_PROMPT,
}


def ensure_parent(path: str) -> None:
    """Create the parent directory for a file path when needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_rgb_image(image_path: str) -> Image.Image:
    """Load one image file and convert it to RGB mode."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def get_image_size(image_path: str) -> List[int]:
    """Return image width and height for bbox denormalization."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(path) as image:
        width, height = image.size
    return [width, height]


def extract_bboxes_from_text(text: str) -> List[List[float]]:
    """Extract bounding boxes from several common serialized output formats."""
    bboxes: List[List[float]] = []
    if not text:
        return bboxes

    match_2d = _BBOX_2D_LIST_RE.search(text)
    if match_2d:
        try:
            parsed = json.loads(match_2d.group(0))
            if isinstance(parsed, list):
                for bbox in parsed:
                    if isinstance(bbox, list) and len(bbox) == 4:
                        bboxes.append([float(value) for value in bbox])
            if bboxes:
                return bboxes
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    for match in _BBOX_1D_SINGLE_RE.finditer(text):
        try:
            bboxes.append([float(match.group(i)) for i in range(1, 5)])
        except (TypeError, ValueError):
            continue
    if bboxes:
        return bboxes

    paren_matches = _BBOX_PAREN_RE.findall(text)
    if paren_matches:
        temp_coords: List[List[float]] = []
        for match in paren_matches:
            if match[2] and match[3]:
                try:
                    bboxes.append([float(match[0]), float(match[1]), float(match[2]), float(match[3])])
                except (TypeError, ValueError):
                    continue
            else:
                try:
                    temp_coords.append([float(match[0]), float(match[1])])
                except (TypeError, ValueError):
                    continue
        if temp_coords and not bboxes:
            for idx in range(0, len(temp_coords) - 1, 2):
                x1, y1 = temp_coords[idx]
                x2, y2 = temp_coords[idx + 1]
                bboxes.append([x1, y1, x2, y2])
        if bboxes:
            return bboxes

    numbers = re.findall(_NUM_RE, text)
    if len(numbers) >= 4 and len(numbers) % 4 == 0:
        try:
            for idx in range(0, len(numbers), 4):
                bboxes.append([float(numbers[idx]), float(numbers[idx + 1]), float(numbers[idx + 2]), float(numbers[idx + 3])])
        except (TypeError, ValueError, IndexError):
            return []
    return bboxes


def normalize_bbox_format(bboxes: List[List[float]]) -> List[List[int]]:
    """Canonicalize predicted boxes to sorted integer 0-1000 coordinates."""
    normalized: List[List[int]] = []
    for bbox in bboxes:
        if len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        x1 = max(0.0, min(1000.0, x1))
        y1 = max(0.0, min(1000.0, y1))
        x2 = max(0.0, min(1000.0, x2))
        y2 = max(0.0, min(1000.0, y2))

        xi1, yi1, xi2, yi2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        if xi2 <= xi1 or yi2 <= yi1:
            continue
        normalized.append([xi1, yi1, xi2, yi2])

    normalized.sort(key=lambda box: (box[0], box[1], box[2], box[3]))
    return normalized


def denorm1000_bboxes(bboxes: List[List[int]], image_width: int, image_height: int) -> List[List[int]]:
    """Convert 0-1000 normalized boxes to absolute pixel coordinates."""
    denormalized: List[List[int]] = []
    for x1, y1, x2, y2 in bboxes:
        denormalized.append(
            [
                round(x1 / 1000.0 * image_width),
                round(y1 / 1000.0 * image_height),
                round(x2 / 1000.0 * image_width),
                round(y2 / 1000.0 * image_height),
            ]
        )
    return denormalized


class RemoteShield:
    """RemoteShield inference wrapper for scene classification, VQA, and visual grounding."""

    def __init__(
        self,
        model_path: str,
        gpu_id: int = 0,
        torch_dtype: Union[str, torch.dtype] = "auto",
        attn_implementation: str = "sdpa",
    ) -> None:
        """Load the RemoteShield checkpoint and the matching processor."""
        self.model_path = model_path
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map={"": self.device},
            attn_implementation=attn_implementation,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def _build_messages(self, task_type: str, image_path: str, query: str) -> List[Dict[str, Any]]:
        """Build one multimodal user message following the Qwen3-VL chat format."""
        if task_type not in TASK_PROMPTS:
            raise ValueError(
                f"Unsupported task_type '{task_type}'. Expected scene_classification, vqa, or visual_grounding."
            )

        image = load_rgb_image(image_path)
        full_query = TASK_PROMPTS[task_type] + query
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": full_query},
                ],
            }
        ]

    def _generate_text(
        self,
        task_type: str,
        image_path: str,
        query: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
        enable_thinking: bool = False,
    ) -> str:
        """Run one generation pass and return only the assistant continuation."""
        messages = self._build_messages(task_type, image_path, query)

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=enable_thinking,
            )
        except TypeError:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

        inputs = {key: value.to(self.model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.inference_mode():
            generated_ids = self.model.generate(**generation_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        trimmed_ids = generated_ids[:, prompt_len:]
        decoded = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""

    def classify_scene(
        self,
        image_path: str,
        query: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> str:
        """Run RemoteShield on a scene classification sample."""
        return self._generate_text(
            "scene_classification",
            image_path,
            query,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def answer_vqa(
        self,
        image_path: str,
        query: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> str:
        """Run RemoteShield on a VQA sample."""
        return self._generate_text(
            "vqa",
            image_path,
            query,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def ground(
        self,
        image_path: str,
        query: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Run RemoteShield on a visual grounding sample and parse the bbox output."""
        raw_output = self._generate_text(
            "visual_grounding",
            image_path,
            query,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        image_width, image_height = get_image_size(image_path)
        boxes = normalize_bbox_format(extract_bboxes_from_text(raw_output))
        boxes_abs = denorm1000_bboxes(boxes, image_width, image_height)
        return {
            "raw_output": raw_output,
            "bboxes_norm1000": boxes,
            "bboxes_abs": boxes_abs,
            "image_size": [image_width, image_height],
        }

    def infer(
        self,
        task_type: str,
        image_path: str,
        query: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> Union[str, Dict[str, Any]]:
        """Unified task interface for RemoteShield inference."""
        if task_type == "scene_classification":
            return self.classify_scene(
                image_path,
                query,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        if task_type == "vqa":
            return self.answer_vqa(
                image_path,
                query,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        if task_type == "visual_grounding":
            return self.ground(
                image_path,
                query,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        raise ValueError(
            f"Unsupported task_type '{task_type}'. Expected scene_classification, vqa, or visual_grounding."
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single-sample RemoteShield inference."""
    parser = argparse.ArgumentParser(description="Run RemoteShield on one image-query sample.")
    parser.add_argument("--model-path", type=str, default="/path/to/your/file", help="RemoteShield checkpoint path.")
    parser.add_argument(
        "--task-type",
        type=str,
        required=True,
        choices=["scene_classification", "vqa", "visual_grounding"],
        help="Task family to run.",
    )
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--query", type=str, required=True, help="Task query or instruction.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id used for inference.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use 0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p used when sampling is enabled.")
    parser.add_argument("--output-file", type=str, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for RemoteShield single-sample inference."""
    args = parse_args()
    model = RemoteShield(model_path=args.model_path, gpu_id=args.gpu_id)
    result = model.infer(
        task_type=args.task_type,
        image_path=args.image_path,
        query=args.query,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.output_file:
        ensure_parent(args.output_file)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    if isinstance(result, dict):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()
