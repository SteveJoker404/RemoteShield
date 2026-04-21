"""Build DPO preference data from aligned clean and perturbed samples."""

import os
import re
import json
import copy
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from tqdm import tqdm
import torch
import numpy as np
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor
)
import multiprocessing as mp
import time
from PIL import Image


from score import score_from_dict


# ---------------------------------------------------------------------------
# Visual-grounding parsing utilities
# ---------------------------------------------------------------------------
# These helpers are shared by scoring and export code. They normalize model
# outputs into a stable bbox representation and optionally convert normalized
# coordinates back into pixel space when needed.
_NUM_RE = r'-?\d+(?:\.\d+)?'


_BBOX_2D_LIST_RE = re.compile(
    rf'\[\s*\[\s*{_NUM_RE}\s*,\s*{_NUM_RE}\s*,\s*{_NUM_RE}\s*,\s*{_NUM_RE}\s*\]'
    rf'(?:\s*,\s*\[\s*{_NUM_RE}\s*,\s*{_NUM_RE}\s*,\s*{_NUM_RE}\s*,\s*{_NUM_RE}\s*\])*\s*\]'
)


_BBOX_1D_SINGLE_RE = re.compile(
    rf'\[\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*\]'
)


_BBOX_PAREN_RE = re.compile(
    rf'\(\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*(?:,\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*)?\)'
)

_BBOX_TASK_TYPES = {'visual_grounding'}

_WH_CACHE: Dict[str, Tuple[int, int]] = {}


def _get_image_wh(image_path: str) -> Optional[Tuple[int, int]]:
    """Return the cached image width and height tuple when available."""
    if image_path in _WH_CACHE:
        return _WH_CACHE[image_path]
    try:
        from PIL import Image as PilImage
        with PilImage.open(image_path) as img:
            wh = img.size
        _WH_CACHE[image_path] = wh
        return wh
    except Exception as e:
        print(f"Warning: Cannot get image size for '{image_path}': {e}")
        return None


def extract_bboxes_from_text(text: str) -> List[List[float]]:
    """Extract bbox candidates from several common text serialization formats."""
    bboxes = []

    if not text:
        return bboxes


    # Normalize a few full-width punctuation variants before regex parsing.
    text = (
        text.replace('【', '[')
            .replace('】', ']')
            .replace('（', '(')
            .replace('）', ')')
            .replace('，', ',')
            .replace('：', ':')
            .replace('；', ';')
    )


    # Strategy 1: try the canonical nested-list format first.
    match_2d = _BBOX_2D_LIST_RE.search(text)
    if match_2d:
        try:
            parsed = json.loads(match_2d.group(0))
            if isinstance(parsed, list):
                for bbox in parsed:
                    if isinstance(bbox, list) and len(bbox) == 4:
                        bboxes.append([float(x) for x in bbox])
            if bboxes:
                return bboxes
        except (json.JSONDecodeError, ValueError, TypeError):
            pass


    # Strategy 2: recover repeated single-box list patterns.
    for match in _BBOX_1D_SINGLE_RE.finditer(text):
        try:
            bbox = [float(match.group(i)) for i in range(1, 5)]
            bboxes.append(bbox)
        except (ValueError, TypeError):
            continue

    if bboxes:
        return bboxes


    # Strategy 3: support parenthesized coordinate expressions.
    paren_matches = _BBOX_PAREN_RE.findall(text)
    if paren_matches:
        temp_coords = []
        for match in paren_matches:

            if match[2] and match[3]:
                try:
                    bbox = [float(match[0]), float(match[1]), float(match[2]), float(match[3])]
                    bboxes.append(bbox)
                except (ValueError, TypeError):
                    continue
            else:
                try:
                    temp_coords.append((float(match[0]), float(match[1])))
                except (ValueError, TypeError):
                    continue


        if temp_coords and not bboxes:
            for i in range(0, len(temp_coords) - 1, 2):
                x1, y1 = temp_coords[i]
                x2, y2 = temp_coords[i + 1]
                bboxes.append([x1, y1, x2, y2])

        if bboxes:
            return bboxes


    # Strategy 4: final fallback for loosely formatted number-only outputs.
    numbers = re.findall(_NUM_RE, text)
    if len(numbers) >= 4 and len(numbers) % 4 == 0:
        try:
            for i in range(0, len(numbers), 4):
                bbox = [float(numbers[i]), float(numbers[i+1]),
                       float(numbers[i+2]), float(numbers[i+3])]
                bboxes.append(bbox)
            if bboxes:
                print(f"Warning: Using fallback number extraction for bbox parsing")
                return bboxes
        except (ValueError, TypeError, IndexError):
            pass

    return bboxes


def normalize_bbox_format(bboxes: List[List[float]]) -> str:
    """Convert parsed boxes into a canonical 0-1000 integer list string."""
    if not bboxes:
        return "[]"

    def _canonicalize_one_bbox(bbox: List[float]) -> Optional[List[int]]:
        if not (isinstance(bbox, list) and len(bbox) == 4):
            return None
        try:
            x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (ValueError, TypeError):
            return None


        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1


        def clamp(v: float) -> float:
            return 0.0 if v < 0.0 else 1000.0 if v > 1000.0 else v

        x1, y1, x2, y2 = clamp(x1), clamp(y1), clamp(x2), clamp(y2)


        xi1, yi1, xi2, yi2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


        if xi2 <= xi1 or yi2 <= yi1:
            return None
        return [xi1, yi1, xi2, yi2]


    # Canonicalization keeps the serialized format stable across reruns and
    # removes obviously invalid or degenerate predictions.
    valid_bboxes: List[List[int]] = []
    for bbox in bboxes:
        canon = _canonicalize_one_bbox(bbox)
        if canon is not None:
            valid_bboxes.append(canon)

    if not valid_bboxes:
        return "[]"


    valid_bboxes.sort(key=lambda bb: (bb[0], bb[1], bb[2], bb[3]))


    return json.dumps(valid_bboxes)


def parse_visual_grounding_output(answer: str, image_path: str) -> Dict[str, Any]:
    """Parse a grounding answer and optionally convert normalized boxes to pixels."""
    result = {
        'format': 'plain',
        'raw_answer': answer,
        'bboxes_norm1000': [],
        'bboxes_abs': [],
        'labels': [],
        'formatted_content': "[]"
    }


    # The model is expected to emit 0-1000 normalized boxes. We keep that
    # normalized form for exported training data but also derive pixel-space
    # boxes for inspection and scoring.
    bboxes_norm1000 = extract_bboxes_from_text(answer)

    if bboxes_norm1000:

        wh = _get_image_wh(image_path)
        if wh:
            img_w, img_h = wh
            bboxes_abs = [
                [
                    round(x1 / 1000.0 * img_w),
                    round(y1 / 1000.0 * img_h),
                    round(x2 / 1000.0 * img_w),
                    round(y2 / 1000.0 * img_h)
                ]
                for x1, y1, x2, y2 in bboxes_norm1000
            ]
        else:
            bboxes_abs = [[int(round(x)) for x in bbox] for bbox in bboxes_norm1000]


        formatted_content = normalize_bbox_format(bboxes_norm1000)

        result.update({
            'format': 'plain',
            'bboxes_norm1000': bboxes_norm1000,
            'bboxes_abs': bboxes_abs,
            'formatted_content': formatted_content
        })
        return result


    print(f"Warning: No valid bbox found in answer, fallback to [] for bbox-only training")
    return result


def denorm1000_bboxes_in_text(text: str, img_w: int, img_h: int) -> str:
    """Replace normalized 0-1000 boxes in a text string with pixel-space boxes."""

    bboxes_norm = extract_bboxes_from_text(text)

    if not bboxes_norm:
        return text


    bboxes_abs = [
        [
            round(x1 / 1000.0 * img_w),
            round(y1 / 1000.0 * img_h),
            round(x2 / 1000.0 * img_w),
            round(y2 / 1000.0 * img_h),
        ]
        for x1, y1, x2, y2 in bboxes_norm
    ]


    normalized_bbox_str = normalize_bbox_format(bboxes_abs)


    match_2d = _BBOX_2D_LIST_RE.search(text)
    if match_2d:
        return text[:match_2d.start()] + normalized_bbox_str + text[match_2d.end():]


    matches = list(_BBOX_1D_SINGLE_RE.finditer(text))
    if matches:
        first_match = matches[0]
        last_match = matches[-1]
        return text[:first_match.start()] + normalized_bbox_str + text[last_match.end():]


    matches = list(_BBOX_PAREN_RE.finditer(text))
    if matches:
        first_match = matches[0]
        last_match = matches[-1]
        return text[:first_match.start()] + normalized_bbox_str + text[last_match.end():]


    return text + " " + normalized_bbox_str


# Output file names used by export and resume bookkeeping.
FILE_CLEAN = "dpo_clean.jsonl"
FILE_PERT = "dpo_pert.jsonl"
LEGACY_FILE_PERT_ALIAS = "dpo_attack.jsonl"
FILE_SKIPPED_IDS = "skipped_sample_ids.log"

_SKIP_WARNING_RE = re.compile(
    r"Warning: All outputs have same score \([^)]+\) for sample (.+?), skipping"
)


def load_completed_ids(output_dir: str) -> Set[str]:
    """Collect sample ids already written to the output JSONL files."""
    def read_ids_from_jsonl(path: str) -> Set[str]:
        ids = set()
        if not os.path.exists(path):
            return ids
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if 'id' in obj:
                            ids.add(str(obj['id']))
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"  Warning: Failed to read {path}: {e}")
        return ids

    clean_ids    = read_ids_from_jsonl(os.path.join(output_dir, FILE_CLEAN))
    pert_ids   = read_ids_from_jsonl(os.path.join(output_dir, FILE_PERT))
    if not pert_ids:
        pert_ids = read_ids_from_jsonl(os.path.join(output_dir, LEGACY_FILE_PERT_ALIAS))

    # Only ids present in both single-turn outputs are considered safely completed.
    completed_ids = clean_ids & pert_ids

    if completed_ids:
        print(f"  Loaded {len(completed_ids)} completed samples from output JSONL files")

        union_size = len(clean_ids | pert_ids)
        if union_size != len(completed_ids):
            diff = union_size - len(completed_ids)
            print(f"  Warning: {diff} ids appear in only one of the single-turn files")
            print(f"  These will be discarded and regenerated (they won't be re-appended)")

    return completed_ids


def load_skipped_ids(skip_log_path: str) -> Set[str]:
    """Load sample ids that were skipped in previous runs."""
    skipped_ids = set()
    if not os.path.exists(skip_log_path):
        return skipped_ids

    try:
        with open(skip_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample_id = line.strip()
                if sample_id:
                    skipped_ids.add(sample_id)
    except Exception as e:
        print(f"  Warning: Failed to read skip id log {skip_log_path}: {e}")

    return skipped_ids


def _extract_skipped_ids_from_progress_log(log_path: str) -> Set[str]:
    """Parse skipped ids from a human-readable progress log."""
    ids = set()
    if not os.path.exists(log_path):
        return ids

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                m = _SKIP_WARNING_RE.search(line)
                if m:
                    ids.add(m.group(1).strip())
    except Exception as e:
        print(f"  Warning: Failed to parse progress log {log_path}: {e}")

    return ids


def refresh_skipped_ids_from_progress_logs(output_dir: str) -> Set[str]:
    """Rebuild the skipped-id set from progress logs in the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    skip_log_path = os.path.join(output_dir, FILE_SKIPPED_IDS)

    if not os.path.exists(skip_log_path):
        open(skip_log_path, 'a', encoding='utf-8').close()

    skip_ids = load_skipped_ids(skip_log_path)

    progress_logs = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith('gpu_') and f.endswith('_progress.log')
    ]

    extracted = set()
    for log_path in progress_logs:
        extracted |= _extract_skipped_ids_from_progress_log(log_path)

    new_ids = extracted - skip_ids
    merged_ids = skip_ids | new_ids

    if new_ids:

        with open(skip_log_path, 'a', encoding='utf-8') as f:
            for sample_id in sorted(new_ids):
                f.write(sample_id + '\n')
        print(
            f"  Updated skip id log: +{len(new_ids)} new ids "
            f"(total {len(merged_ids)})"
        )
    else:
        print(f"  Skip id log is up-to-date: {len(merged_ids)} ids")

    return merged_ids


def normalize_record_id(record: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure each output record stores a string id field."""
    normalized = copy.deepcopy(record)
    if 'id' in normalized and normalized['id'] is not None:
        normalized['id'] = str(normalized['id'])
    return normalized


def append_triple(
    clean_sample: Dict,
    pert_sample: Dict,
    output_dir: str,
    lock
):
    """Append clean and perturbed records to their JSONL outputs."""
    clean_sample = normalize_record_id(clean_sample)
    pert_sample = normalize_record_id(pert_sample)

    line_clean    = json.dumps(clean_sample,    ensure_ascii=False)
    line_pert     = json.dumps(pert_sample,     ensure_ascii=False)

    with lock:
        with open(os.path.join(output_dir, FILE_CLEAN),    'a', encoding='utf-8') as f:
            f.write(line_clean    + '\n')
        with open(os.path.join(output_dir, FILE_PERT),     'a', encoding='utf-8') as f:
            f.write(line_pert     + '\n')


class Config:
    """Default runtime configuration for generation and export."""

    # Model and decoding defaults.
    MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    NUM_SAMPLES = 4
    TEMPERATURE = 0.7


    MAX_NEW_TOKENS = 1536


    ENABLE_THINKING = False


    PRED_BBOX_NORM1000 = True


    SYSTEM_PROMPT = """You are a vision-language assistant for remote sensing image analysis.

Determine the task type internally from the query, but never reveal the task type in the output.

Task definitions:
- scene_classification: The query provides candidate scene categories and asks to choose one.
- vqa: The query asks a question requiring a short answer.
- visual_grounding: The query explicitly asks for localization, or only contains an object description/reference without a question.

Strict output rules:
- Output ONLY the final answer.
- NEVER output the task type.
- NEVER output any prefix, label, or explanation.
- NEVER output text such as "scene_classification:", "vqa:", or "visual_grounding:".
- Bounding boxes may be output ONLY for visual_grounding.
- For visual_grounding, output ONLY:
  [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
- For scene_classification, output ONLY the selected category name.
- For vqa, output ONLY a short answer.

\nQuery:\n"""

    # Export defaults. These fields can be overridden from the CLI.
    CLEAN_JSON = ""
    PERT_JSON = ""
    IMAGE_BASE_DIR = ""


    OUTPUT_DIR = "dpo_dataset"


class Qwen3VLInferencer:
    """Thin Qwen3-VL wrapper used by each worker process."""

    def __init__(self, model_name: str, gpu_id: int = 0):
        """Load the model and processor on the specified GPU."""
        print(f"[Model] Loading {model_name} on GPU {gpu_id}...")


        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            attn_implementation="sdpa",
            device_map={"": f"cuda:{gpu_id}"},
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        print(f"[Model] Loaded successfully")
        print(f"  - Device: {self.model.device}")
        print(f"  - Dtype: {self.model.dtype}")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        num_samples: int = 1,
        enable_thinking: bool = False
    ) -> List[str]:
        """Generate one or more responses from a chat-style multimodal prompt."""

        try:
            # Newer processors expose the enable_thinking flag explicitly.
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Fall back to the older signature when the installed processor does
            # not yet support enable_thinking.
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        inputs = inputs.to(self.model.device)


        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                num_return_sequences=num_samples,
            )


        # Strip the prompt tokens so only newly generated assistant text remains.
        prompt_len = inputs.input_ids.shape[1]
        new_token_ids = generated_ids[:, prompt_len:]
        decoded_outputs = self.processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


        final_answers = [text.strip() for text in decoded_outputs]


        del inputs
        del generated_ids
        torch.cuda.empty_cache()

        return final_answers


def load_json_data(file_path: str) -> List[Dict]:
    """Load either a list-style JSON file or a dict with a `data` field."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'data' in data:
        return data['data']
    else:
        raise ValueError(f"Unknown JSON format in {file_path}")


def build_sample_dict(samples: List[Dict]) -> Dict[str, Dict]:
    """Map sample ids to records and warn about duplicates."""
    sample_dict = {}
    duplicate_ids = []

    for sample in samples:
        sample_id = sample.get('id')
        if sample_id is None:
            print(f"Warning: Sample without ID found, skipping: {sample}")
            continue

        if sample_id in sample_dict:
            duplicate_ids.append(sample_id)
        else:
            sample_dict[sample_id] = sample

    if duplicate_ids:
        print(f"Warning: Found {len(duplicate_ids)} duplicate IDs: {duplicate_ids[:5]}...")

    return sample_dict


def construct_messages(
    question: str,
    image_path: str,
    system_prompt: str = ""
) -> List[Dict[str, Any]]:
    """Build the single-turn multimodal user message for generation."""
    content = []

    # The model always receives the image first, followed by the text query.
    content.append({"type": "image", "image": image_path})


    raw_text = question.replace("<image>", "").strip()
    if system_prompt:
        full_text = system_prompt.strip() + "\n" + raw_text
    else:
        full_text = raw_text

    if full_text:
        content.append({"type": "text", "text": full_text})

    return [{"role": "user", "content": content}]


def process_sample_pair(
    clean_sample: Dict,
    pert_sample: Dict,
    inferencer: Qwen3VLInferencer,
    image_base_dir: str,
    config: Config
) -> Optional[Tuple[str, str, List[str], List[float]]]:
    """Generate, score, and select preferred/rejected outputs for one sample pair."""
    try:

        def resolve_image_path(image_field: str, base_dir: str) -> str:
            if os.path.isabs(image_field):
                return image_field
            else:
                return os.path.join(base_dir, image_field)

        clean_image = resolve_image_path(clean_sample['image'], image_base_dir)
        pert_image = resolve_image_path(pert_sample['image'], image_base_dir)


        if not os.path.exists(clean_image):
            print(f"\n  Warning: Clean image not found: {clean_image}")
            return None
        if not os.path.exists(pert_image):
            print(f"\n  Warning: Perturbed image not found: {pert_image}")
            return None


        # Evaluate the four clean/perturbed text-image combinations jointly.
        combinations = [

            {
                'name': 'clean_text_clean_image',
                'question': clean_sample['question'],
                'image': clean_image,
                'sample': clean_sample
            },

            {
                'name': 'clean_text_pert_image',
                'question': clean_sample['question'],
                'image': pert_image,
                'sample': clean_sample
            },

            {
                'name': 'pert_text_clean_image',
                'question': pert_sample['question'],
                'image': clean_image,
                'sample': pert_sample
            },

            {
                'name': 'pert_text_pert_image',
                'question': pert_sample['question'],
                'image': pert_image,
                'sample': pert_sample
            }
        ]


        # Collect every sampled response across all four conditions into one
        # pooled candidate set so preference selection is global.
        all_outputs = []
        all_scores = []


        # Group by image path so repeated image loads stay local to one batch of calls.
        image_groups = {}
        for idx, comb in enumerate(combinations):
            img = comb['image']
            if img not in image_groups:
                image_groups[img] = []
            image_groups[img].append((idx, comb))


        comb_outputs = [None] * len(combinations)

        for image_path, group in image_groups.items():


            for comb_idx, comb in group:
                messages = construct_messages(
                    comb['question'],
                    image_path,
                    config.SYSTEM_PROMPT
                )

                outputs = inferencer.generate(
                    messages,
                    temperature=config.TEMPERATURE,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    num_samples=config.NUM_SAMPLES,
                    enable_thinking=config.ENABLE_THINKING
                )
                comb_outputs[comb_idx] = outputs


                torch.cuda.empty_cache()


        # Score every sampled output in the pooled candidate set.
        for comb_idx, comb in enumerate(combinations):
            outputs = comb_outputs[comb_idx]


            # Route scoring through the corresponding sample record while
            # swapping in the actual image path used for this condition.
            sample_for_scoring = comb['sample'].copy()
            sample_for_scoring['image'] = comb['image']

            for output in outputs:

                try:
                    score = score_from_dict(
                        output,
                        sample_for_scoring,
                        pred_bbox_norm1000=config.PRED_BBOX_NORM1000
                    )

                    if score != score or not math.isfinite(score):
                        print(f"  Warning: Invalid score (NaN/Inf), setting to 0.0")
                        score = 0.0

                    score = max(0.0, min(1.0, float(score)))
                except Exception as e:
                    print(f"  Warning: Scoring failed with error: {e}, setting score to 0.0")
                    score = 0.0

                all_outputs.append(output)
                all_scores.append(score)


        if not all_outputs or not all_scores:
            print(f"\n  Error: No valid outputs generated for sample {clean_sample.get('id', 'unknown')}")
            return None

        if len(all_outputs) != len(all_scores):
            print(f"\n  Error: Output/score count mismatch for sample {clean_sample.get('id', 'unknown')}")
            return None


        fallback_answer = "Sorry, I don't know."
        all_fallback = all(output.strip() == fallback_answer for output in all_outputs)
        if all_fallback:
            print(f"\n  Warning: All outputs are fallback answers for sample {clean_sample.get('id', 'unknown')}")


        try:
            max_score = max(all_scores)
            min_score = min(all_scores)
            max_idx = all_scores.index(max_score)
            min_idx = all_scores.index(min_score)
        except (ValueError, IndexError) as e:
            print(f"\n  Error: Failed to find max/min scores for sample {clean_sample.get('id', 'unknown')}: {e}")
            return None

        # Preferred/rejected are selected globally from the pooled candidate set.
        preferred_output = all_outputs[max_idx]
        rejected_output = all_outputs[min_idx]


        task_type = clean_sample.get('task_type', '')
        preferred_parsed = None
        rejected_parsed = None

        # Convert selected grounding outputs to pixel coordinates for easier inspection.
        if config.PRED_BBOX_NORM1000 and task_type in _BBOX_TASK_TYPES:


            preferred_parsed = parse_visual_grounding_output(preferred_output, clean_image)
            rejected_parsed = parse_visual_grounding_output(rejected_output, clean_image)


            preferred_output = preferred_parsed['formatted_content']
            rejected_output = rejected_parsed['formatted_content']

        return (preferred_output, rejected_output, all_outputs, all_scores,
                preferred_parsed, rejected_parsed, clean_image, pert_image)

    except RuntimeError as e:

        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"\n  CUDA Error for sample {clean_sample.get('id', 'unknown')}: {e}")
            print(f"  Consider reducing batch size or max_new_tokens")
        else:
            print(f"\n  Runtime Error processing sample {clean_sample.get('id', 'unknown')}: {e}")
        return None
    except Exception as e:
        print(f"\n  Error processing sample {clean_sample.get('id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_dpo_clean(
    sample: Dict,
    preferred: str,
    rejected: str,
    preferred_parsed: Optional[Dict] = None,
    rejected_parsed: Optional[Dict] = None,
    image_path: Optional[str] = None
) -> Dict:
    """Create the single-turn clean-condition DPO sample."""
    # The exported sample keeps the original clean prompt and attaches the
    # selected preferred/rejected responses as a standard DPO pair.
    dpo_sample = {
        "id": str(sample['id']),
        "task_type": sample['task_type'],
        "messages": [
            {
                "role": "user",
                "content": f"<image>{sample['question']}"
            },
            {
                "role": "assistant",
                "content": preferred
            }
        ],
        "images": [sample['image']],
        "rejected_messages": [
            {
                "role": "user",
                "content": f"<image>{sample['question']}"
            },
            {
                "role": "assistant",
                "content": rejected
            }
        ],
        "rejected_images": [sample['image']]
    }

    return dpo_sample


def create_dpo_pert(
    sample: Dict,
    preferred: str,
    rejected: str,
    preferred_parsed: Optional[Dict] = None,
    rejected_parsed: Optional[Dict] = None,
    image_path: Optional[str] = None
) -> Dict:
    """Create the single-turn perturbed-condition DPO sample."""
    # The perturbed sample mirrors the clean format but uses the perturbed image
    # or question context that was actually scored.
    dpo_sample = {
        "id": str(sample['id']),
        "task_type": sample['task_type'],
        "messages": [
            {
                "role": "user",
                "content": f"<image>{sample['question']}"
            },
            {
                "role": "assistant",
                "content": preferred
            }
        ],
        "images": [image_path if image_path else sample['image']],
        "rejected_messages": [
            {
                "role": "user",
                "content": f"<image>{sample['question']}"
            },
            {
                "role": "assistant",
                "content": rejected
            }
        ],
        "rejected_images": [image_path if image_path else sample['image']]
    }


    if 'text_type' in sample:
        dpo_sample['text_type'] = sample['text_type']

    return dpo_sample


def worker_process(
    gpu_id: int,
    clean_samples: List[Dict],
    pert_dict: Dict[str, Dict],
    image_base_dir: str,
    model_name: str,
    system_prompt: str,
    num_samples: int,
    temperature: float,
    max_new_tokens: int,
    output_queue: mp.Queue,
    completed_ids: Set,
    output_dir: str,
    write_lock,
    pred_bbox_norm1000: bool = False
):
    """Run one shard of sample pairs on a dedicated GPU worker."""
    new_count = 0
    failed_count = 0

    try:
        import multiprocessing
        multiprocessing.current_process().name = f"GPU-{gpu_id}-Worker"


        log_file = os.path.join(output_dir, f"gpu_{gpu_id}_progress.log")

        def log(msg):
            print(msg)
            with open(log_file, 'a', encoding='utf-8') as f:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {msg}\n")
                f.flush()


        # Resume support is implemented per worker by filtering out ids that are
        # already fully materialized in the output shards.
        pending_samples = [
            s for s in clean_samples
            if str(s['id']) not in completed_ids
        ]
        log(f"[GPU {gpu_id}] Starting worker process")
        log(f"[GPU {gpu_id}] Total assigned: {len(clean_samples)} samples")
        log(f"[GPU {gpu_id}] Already completed: {len(clean_samples) - len(pending_samples)} samples")
        log(f"[GPU {gpu_id}] Pending to process: {len(pending_samples)} samples")


        config = Config()
        config.SYSTEM_PROMPT = system_prompt
        config.NUM_SAMPLES = num_samples
        config.TEMPERATURE = temperature
        config.MAX_NEW_TOKENS = max_new_tokens
        config.PRED_BBOX_NORM1000 = pred_bbox_norm1000


        log(f"[GPU {gpu_id}] Loading model: {model_name}")
        inferencer = Qwen3VLInferencer(model_name, gpu_id=gpu_id)
        log(f"[GPU {gpu_id}] Model loaded successfully")


        # Low-score samples are not filtered out automatically, but they are
        # logged for later inspection because they tend to be noisy preference pairs.
        low_score_log_path = os.path.join(output_dir, f"low_score_samples_gpu{gpu_id}.jsonl")
        low_score_threshold = 0.2

        log(f"[GPU {gpu_id}] Starting inference loop...")
        log(f"[GPU {gpu_id}] Config: num_samples={num_samples}, temperature={temperature}, max_new_tokens={max_new_tokens}")

        already_done = len(completed_ids & {str(s['id']) for s in clean_samples})
        pbar = tqdm(
            total=len(clean_samples),
            desc=f"GPU {gpu_id}",
            position=gpu_id,
            initial=already_done,
            leave=True
        )

        log(f"[GPU {gpu_id}] Progress: {already_done}/{len(clean_samples)} (starting)")

        for clean_sample in pending_samples:
            sample_id = clean_sample['id']

            if sample_id not in pert_dict and str(sample_id) not in pert_dict:
                log(f"[GPU {gpu_id}] Warning: No perturbed sample for {sample_id}")
                pbar.update(1)
                failed_count += 1


                total_processed = new_count + failed_count
                total_assigned = len(clean_samples)
                progress_pct = (already_done + total_processed) / total_assigned * 100
                log(f"[GPU {gpu_id}] Progress: {already_done + total_processed}/{total_assigned} ({progress_pct:.1f}%) | New: {new_count} | Failed: {failed_count}")
                continue

            pert_sample = pert_dict.get(sample_id) or pert_dict.get(str(sample_id))


            result = process_sample_pair(
                clean_sample, pert_sample, inferencer, image_base_dir, config
            )

            if result is None:
                pbar.update(1)
                failed_count += 1


                total_processed = new_count + failed_count
                total_assigned = len(clean_samples)
                progress_pct = (already_done + total_processed) / total_assigned * 100
                log(f"[GPU {gpu_id}] Progress: {already_done + total_processed}/{total_assigned} ({progress_pct:.1f}%) | New: {new_count} | Failed: {failed_count}")
                continue


            (preferred, rejected, all_outputs, all_scores,
             preferred_parsed, rejected_parsed,
             clean_image_abs, pert_image_abs) = result


            max_score = max(all_scores)
            min_score = min(all_scores)

            # If every candidate receives the same score, no useful preference
            # ordering can be induced from this sample pair.
            if abs(max_score - min_score) < 1e-6:
                log(f"[GPU {gpu_id}] Warning: All outputs have same score ({max_score:.3f}) for sample {sample_id}, skipping")
                pbar.update(1)
                failed_count += 1


                total_processed = new_count + failed_count
                total_assigned = len(clean_samples)
                progress_pct = (already_done + total_processed) / total_assigned * 100
                log(f"[GPU {gpu_id}] Progress: {already_done + total_processed}/{total_assigned} ({progress_pct:.1f}%) | New: {new_count} | Failed: {failed_count}")
                continue


            if max_score < low_score_threshold:
                log(f"[GPU {gpu_id}] Low score sample {sample_id}: max_score={max_score:.3f}")
                low_score_entry = {
                    'id': str(sample_id),
                    'task_type': clean_sample.get('task_type', ''),
                    'max_score': float(max_score),
                    'min_score': float(min_score),
                    'all_scores': [float(s) for s in all_scores],
                    'preferred': preferred[:200],
                    'rejected': rejected[:200],
                    'ground_truth': clean_sample.get('answer', '')[:200],
                    'question': clean_sample.get('question', '')[:200]
                }
                with open(low_score_log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(low_score_entry, ensure_ascii=False) + '\n')


            # Materialize the selected pair into the exported single-turn formats.
            dpo_clean    = create_dpo_clean(clean_sample, preferred, rejected,
                                           preferred_parsed, rejected_parsed, clean_image_abs)
            dpo_pert     = create_dpo_pert(pert_sample, preferred, rejected,
                                          preferred_parsed, rejected_parsed, pert_image_abs)


            append_triple(dpo_clean, dpo_pert, output_dir, write_lock)

            pbar.update(1)
            new_count += 1


            total_processed = new_count + failed_count
            total_assigned = len(clean_samples)
            progress_pct = (already_done + total_processed) / total_assigned * 100
            log(f"[GPU {gpu_id}] Progress: {already_done + total_processed}/{total_assigned} ({progress_pct:.1f}%) | New: {new_count} | Failed: {failed_count} | ID: {sample_id}")

        pbar.close()
        log(f"[GPU {gpu_id}] ========== Worker Completed ==========")
        log(f"[GPU {gpu_id}] New samples generated: {new_count}")
        log(f"[GPU {gpu_id}] Failed/Skipped: {failed_count}")
        log(f"[GPU {gpu_id}] Total processed: {new_count + failed_count}")
        log(f"[GPU {gpu_id}] ======================================")

        output_queue.put({'gpu_id': gpu_id, 'new': new_count,
                          'failed': failed_count, 'error': None})

    except Exception as e:
        error_msg = f"Fatal error: {e}"
        log(f"[GPU {gpu_id}] {error_msg}")
        import traceback
        error_trace = traceback.format_exc()
        log(f"[GPU {gpu_id}] Traceback:\n{error_trace}")

        log(f"[GPU {gpu_id}] {new_count} samples already written to JSONL files")
        output_queue.put({'gpu_id': gpu_id, 'new': new_count,
                          'failed': failed_count, 'error': str(e)})


def build_preference_data(
    clean_json_path: str,
    pert_json_path: str,
    image_base_dir: str,
    output_dir: str,
    model_name: str,
    system_prompt: str = "",
    num_samples: int = 4,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    gpu_ids: List[int] = None,
    resume: bool = True,
    pred_bbox_norm1000: bool = False
):
    """Launch multi-GPU generation and export DPO samples from paired data."""
    if gpu_ids is None:
        gpu_ids = [0]
    num_gpus = len(gpu_ids)
    total_inferences_per_sample = 4 * num_samples
    print("="*80)
    print(f"Preference data construction (4 combinations x {num_samples} samples = {total_inferences_per_sample} inferences per pair)")
    print("="*80)
    print(f"Clean JSON      : {clean_json_path}")
    print(f"Pert JSON       : {pert_json_path}")
    print(f"Image base dir  : {image_base_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Model           : {model_name}")
    print(f"Samples/combination : {num_samples}")
    print(f"Total inferences    : {total_inferences_per_sample}")
    print(f"Temperature     : {temperature}")
    print(f"GPU IDs         : {gpu_ids} ({num_gpus} GPU(s))")
    print(f"Resume          : {resume}")
    print(f"Qwen3-VL mode   : {pred_bbox_norm1000}")
    print(f"Output JSONL    :")
    print(f"  {os.path.join(output_dir, FILE_CLEAN)}")
    print(f"  {os.path.join(output_dir, FILE_PERT)}")
    print("="*80)


    if not os.path.exists(clean_json_path):
        raise FileNotFoundError(f"Clean JSON file not found: {clean_json_path}")
    if not os.path.exists(pert_json_path):
        raise FileNotFoundError(f"Pert JSON file not found: {pert_json_path}")
    if image_base_dir and not os.path.isdir(image_base_dir):
        print(f"\nWarning: Image base directory not found or not a directory: {image_base_dir}")
        print("Continuing anyway - images will be checked during processing...")


    os.makedirs(output_dir, exist_ok=True)


    print("\n[1] Loading data...")
    clean_samples = load_json_data(clean_json_path)
    pert_samples = load_json_data(pert_json_path)

    print(f"  Clean samples: {len(clean_samples)}")
    print(f"  Pert samples : {len(pert_samples)}")

    if not clean_samples:
        print("\nError: No clean samples found!")
        return
    if not pert_samples:
        print("\nError: No perturbed samples found!")
        return

    pert_dict = build_sample_dict(pert_samples)


    print("\n[1.5] Refreshing skipped sample id log from GPU progress logs...")
    skipped_from_logs = refresh_skipped_ids_from_progress_logs(output_dir)

    if resume:
        print("\n[1.5] Scanning checkpoint files for completed/skipped sample ids...")
        completed_from_jsonl = load_completed_ids(output_dir)
        completed_ids = completed_from_jsonl | skipped_from_logs
        print(f"  Completed from JSONL: {len(completed_from_jsonl)} samples")
        print(f"  Skipped from logs   : {len(skipped_from_logs)} samples")
        print(f"  Total resume-filter : {len(completed_ids)} samples")
    else:
        print("\n[1.5] Starting from scratch (--no_resume), clearing existing JSONL files...")
        print(f"  Skipped id log synced: {len(skipped_from_logs)} ids (not used in --no_resume)")
        completed_ids = set()
        for fname in [FILE_CLEAN, FILE_PERT, LEGACY_FILE_PERT_ALIAS]:
            path = os.path.join(output_dir, fname)
            if os.path.exists(path):
                os.remove(path)
                print(f"  Removed: {path}")


    pending_samples = [s for s in clean_samples if str(s['id']) not in completed_ids]
    print(f"\n[2] Pending samples: {len(pending_samples)} / {len(clean_samples)}")

    if len(pending_samples) == 0:
        print("  All samples already completed. Nothing to do.")
        _print_final_stats(output_dir, len(clean_samples), len(completed_ids), 0, 0)
        return


    manager = mp.Manager()
    write_lock = manager.Lock()
    output_queue = mp.Queue()

    # Run either a single worker or one worker per visible GPU.
    if num_gpus == 1:
        print(f"[3] Single GPU mode - GPU {gpu_ids[0]}...")

        p = mp.Process(
            target=worker_process,
            args=(
                gpu_ids[0],
                pending_samples,
                pert_dict,
                image_base_dir,
                model_name,
                system_prompt,
                num_samples,
                temperature,
                max_new_tokens,
                output_queue,
                completed_ids,
                output_dir,
                write_lock,
                pred_bbox_norm1000
            )
        )
        p.start()
        p.join()
    else:
        print(f"[3] Multi-GPU mode - splitting {len(pending_samples)} samples across {num_gpus} GPUs {gpu_ids}...")
        # Split pending samples by contiguous shards to keep scheduling simple.
        samples_per_gpu = len(pending_samples) // num_gpus
        processes = []
        for i, gpu_id in enumerate(gpu_ids):
            start_idx = i * samples_per_gpu
            end_idx = len(pending_samples) if i == num_gpus - 1 else (i + 1) * samples_per_gpu
            split = pending_samples[start_idx:end_idx]
            print(f"  GPU {gpu_id}: {len(split)} samples")
            p = mp.Process(
                target=worker_process,
                args=(
                    gpu_id,
                    split,
                    pert_dict,
                    image_base_dir,
                    model_name,
                    system_prompt,
                    num_samples,
                    temperature,
                    max_new_tokens,
                    output_queue,
                    completed_ids,
                    output_dir,
                    write_lock,
                    pred_bbox_norm1000
                )
            )
            p.start()
            processes.append(p)

        print("\n" + "="*80)
        print("All workers started. Progress bars will appear below:")
        print("="*80 + "\n")
        for p in processes:
            p.join()


    print("\n[4] Collecting worker stats...")
    total_new = 0
    total_failed = 0
    n_workers = num_gpus
    for i in range(n_workers):
        try:
            r = output_queue.get(timeout=30)
            total_new    += r.get('new', 0)
            total_failed += r.get('failed', 0)
            status = f"error: {r['error']}" if r.get('error') else f"new={r.get('new',0)}"
            print(f"  GPU {r['gpu_id']}: {status}")
        except Exception as e:
            print(f"  Warning: Failed to get stats from worker {i}: {e}")

    manager.shutdown()
    _print_final_stats(output_dir, len(clean_samples), len(completed_ids), total_new, total_failed)


def _print_final_stats(
    output_dir: str,
    total: int,
    skipped: int,
    new_count: int,
    failed: int
):
    """Print a compact summary of the generated dataset shards."""
    def count_lines(path):
        if not os.path.exists(path):
            return 0
        with open(path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())

    n_clean = count_lines(os.path.join(output_dir, FILE_CLEAN))
    n_pert = count_lines(os.path.join(output_dir, FILE_PERT))


    low_score_logs = [f for f in os.listdir(output_dir) if f.startswith('low_score_samples_gpu') and f.endswith('.jsonl')]
    total_low_score = sum(count_lines(os.path.join(output_dir, f)) for f in low_score_logs)

    print(f"\n{'='*80}")
    print("Generation Statistics:")
    print(f"{'='*80}")
    print(f"  Total input samples: {total}")
    print(f"  Already completed:   {skipped}")
    print(f"  Newly generated:     {new_count}")
    print(f"  Failed/Skipped:      {failed}")
    print(f"\nOutput JSONL files (lines = samples):")
    print(f"  - {FILE_CLEAN}: {n_clean} samples -> {os.path.join(output_dir, FILE_CLEAN)}")
    print(f"  - {FILE_PERT}: {n_pert} samples -> {os.path.join(output_dir, FILE_PERT)}")
    if n_clean == n_pert:
        print(f"  ✓ Both single-turn files are consistent ({n_clean} samples each)")
    else:
        print(f"  ⚠ Inconsistency detected! clean={n_clean}, pert={n_pert}")

    if total_low_score > 0:
        print(f"\n⚠ Low-score samples detected:")
        print(f"  - {total_low_score} samples with preferred score < 0.2")
        print(f"  - Check low_score_samples_gpu*.jsonl in {output_dir}")
        print(f"  - These samples may indicate data quality issues or model confusion")

    print("="*80)


if __name__ == "__main__":


    mp.set_start_method("spawn", force=True)

    import argparse

    parser = argparse.ArgumentParser(
        description="Build DPO preference data from clean and perturbed samples"
    )
    parser.add_argument(
        '--clean_json',
        type=str,
        required=True,
        help='Path to clean samples JSON file'
    )
    parser.add_argument(
        '--pert_json',
        type=str,
        required=False,
        help='Path to perturbed samples JSON file'
    )
    parser.add_argument(
        '--attack_json',
        dest='pert_json_legacy',
        type=str,
        default=None,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--image_base_dir',
        type=str,
        required=True,
        help=(
            'Required image root directory used to resolve relative paths from the JSON files. '
            'For example, with --image_base_dir /path/to/your/file, '
            'origin/a.jpg becomes /path/to/your/file/origin/a.jpg.'
        )
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='dpo_dataset',
        help='Output directory for DPO datasets (default: dpo_dataset)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen/Qwen2.5-VL-7B-Instruct',
        help='Model name or path (default: Qwen/Qwen2.5-VL-7B-Instruct)'
    )
    parser.add_argument(
        '--system_prompt',
        type=str,
        default=None,
        help='System prompt for the model (default: use Config.SYSTEM_PROMPT)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=4,
        help='Number of samples per input (default: 4)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=1536,
        help='Maximum new tokens (default: 1536, increase if seeing "Thinking was truncated" warnings)'
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        default='0',
        help=(
            'Comma-separated list of relative GPU IDs to use, e.g. "0,1" (default: "0"). '
            'These are relative IDs and should be used together with CUDA_VISIBLE_DEVICES. '
            'For example, CUDA_VISIBLE_DEVICES=2,3 and --gpu_ids "0,1" use physical GPUs 2 and 3.'
        )
    )
    parser.add_argument(
        '--no_resume',
        action='store_true',
        help='Start from scratch: clear existing JSONL files and regenerate all samples'
    )
    parser.add_argument(
        '--pred_bbox_norm1000',
        action='store_true',
        help=(
            'Enable when the model outputs 0-1000 normalized bbox coordinates (e.g. Qwen3-VL). '
            'Scores and DPO samples will use denormalized absolute pixel coordinates. '
            'Default: False (model outputs absolute pixel coords, e.g. Qwen2.5-VL).'
        )
    )

    args = parser.parse_args()


    system_prompt = args.system_prompt if args.system_prompt is not None else Config.SYSTEM_PROMPT


    try:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        pert_json = args.pert_json or args.pert_json_legacy
        if not pert_json:
            raise ValueError("Please provide --pert_json.")
        build_preference_data(
            clean_json_path=args.clean_json,
            pert_json_path=pert_json,
            image_base_dir=args.image_base_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            system_prompt=system_prompt,
            num_samples=args.num_samples,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            gpu_ids=gpu_ids,
            resume=not args.no_resume,
            pred_bbox_norm1000=args.pred_bbox_norm1000
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Checkpoints have been saved.")
        print("You can resume by running the same command again.")
        import sys
        sys.exit(0)
    except Exception as e:
        print("\n" + "="*80)
        print("FATAL ERROR")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("="*80)
        print("\nCheckpoints may have been saved. You can try to resume.")
        import sys
        sys.exit(1)
