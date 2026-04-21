"""Consistency-oriented scoring utilities for preference construction."""

import re
import math
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from scipy.optimize import linear_sum_assignment


# Cache image sizes so repeated grounding samples do not reopen the same file.
_IMAGE_SIZE_CACHE: Dict[str, Tuple[int, int]] = {}


class ConsistencyOrientedScorer:
    """Score predictions for scene classification, VQA, and visual grounding."""

    def __init__(self, iou_threshold: float = 0.5, numeric_tau: float = 3.0,
                 pred_bbox_norm1000: bool = False):
        """Initialize scorer parameters and task-family routing."""

        if not 0 <= iou_threshold <= 1:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
        if numeric_tau <= 0:
            raise ValueError(f"numeric_tau must be positive, got {numeric_tau}")

        self.iou_threshold = iou_threshold
        self.numeric_tau = min(numeric_tau, 100.0)
        self.pred_bbox_norm1000 = pred_bbox_norm1000


        self.task_handlers = {
            'scene_classification': self._score_classification,
            'vqa_text': self._score_vqa_text,
            'vqa_counting': self._score_numeric,
            'visual_grounding': self._score_bbox,
        }

    def _resolve_task_mode(self, task_type: str, ground_truth: str) -> str:
        """Resolve the three task families and split VQA into text-output or counting-style."""
        task_type = (task_type or "").strip().lower()
        gt_clean = self._normalize_text(ground_truth)

        if task_type == 'scene_classification':
            return 'scene_classification'
        if task_type == 'visual_grounding':
            return 'visual_grounding'
        if task_type == 'vqa':
            return 'vqa_counting' if self._looks_like_counting_answer(gt_clean) else 'vqa_text'

        raise ValueError(
            f"Unsupported task_type '{task_type}'. Expected scene_classification, vqa, or visual_grounding."
        )

    def _looks_like_counting_answer(self, text: str) -> bool:
        """Decide whether a VQA target should be scored as counting-style VQA."""
        if not text:
            return False

        normalized = text.strip().lower().rstrip('.')
        if normalized in self._WORD_TO_NUM:
            return True
        return re.fullmatch(r'-?\d+(\.\d+)?', normalized) is not None


    def score(self,
              prediction: str,
              ground_truth: str,
              task_type: str,
              question: Optional[str] = None,
              image_path: Optional[str] = None) -> float:
        """Score a single prediction against its reference for a given task type."""
        try:

            # Normalize non-string inputs before dispatch.
            if not isinstance(prediction, str):
                print(f"Warning: prediction is not a string (type: {type(prediction).__name__}), converting...")
                prediction = str(prediction) if prediction is not None else ""
            if not isinstance(ground_truth, str):
                print(f"Warning: ground_truth is not a string (type: {type(ground_truth).__name__}), converting...")
                ground_truth = str(ground_truth) if ground_truth is not None else ""
            if not isinstance(task_type, str):
                print(f"Warning: task_type is not a string (type: {type(task_type).__name__}), converting...")
                task_type = str(task_type) if task_type is not None else ""


            task_mode = self._resolve_task_mode(task_type, ground_truth)
            handler = self.task_handlers[task_mode]


            # Clean task outputs before task-specific scoring.
            pred_clean = self._extract_answer(prediction, task_mode)
            gt_clean = self._extract_answer(ground_truth, task_mode)


            if task_mode == 'visual_grounding':
                score = handler(pred_clean, gt_clean, image_path=image_path)
            elif task_mode == 'vqa_counting':
                score = handler(pred_clean, gt_clean, task_type='vqa_counting')
            else:
                score = handler(pred_clean, gt_clean)


            score = max(0.0, min(1.0, float(score)))


            if not math.isfinite(score):
                print(f"Warning: Score is not finite ({score}), returning 0.0")
                return 0.0

            return score

        except Exception as e:

            print(f"Warning: Scoring error for task_type='{task_type}', pred='{prediction[:50]}...', gt='{ground_truth[:50]}...': {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def batch_score(self,
                    predictions: List[str],
                    ground_truths: List[str],
                    task_types: List[str],
                    questions: Optional[List[str]] = None) -> List[float]:
        """Score a batch of prediction/reference pairs."""

        n = len(predictions)
        if len(ground_truths) != n:
            raise ValueError(f"Length mismatch: predictions ({n}) vs ground_truths ({len(ground_truths)})")
        if len(task_types) != n:
            raise ValueError(f"Length mismatch: predictions ({n}) vs task_types ({len(task_types)})")

        if questions is None:
            questions = [None] * n
        elif len(questions) != n:
            raise ValueError(f"Length mismatch: predictions ({n}) vs questions ({len(questions)})")

        scores = []
        for pred, gt, task_type, question in zip(predictions, ground_truths, task_types, questions):
            score = self.score(pred, gt, task_type, question)
            scores.append(score)

        return scores


    def _extract_answer(self, text: str, task_type: str) -> str:
        """Apply lightweight task-aware cleanup before scoring."""
        if not text:
            return ""


        cleaned = text.strip()


        # Remove chat-template artifacts that should not affect task scoring.
        cleaned = re.sub(r'<image>', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()


        text_tasks = {'scene_classification', 'vqa_text'}
        if task_type in text_tasks:
            cleaned = cleaned.rstrip('.')

        return cleaned


    def _score_classification(self, pred: str, gt: str) -> float:
        """Score closed-set scene classification by normalized exact match."""
        pred_norm = self._normalize_text(pred)
        gt_norm = self._normalize_text(gt)

        return 1.0 if pred_norm == gt_norm else 0.0

    def _score_vqa_text(self, pred: str, gt: str) -> float:
        """Score text-output VQA by normalized exact match."""
        pred_norm = self._normalize_text(pred)
        gt_norm = self._normalize_text(gt)

        return 1.0 if pred_norm == gt_norm else 0.0

    def _score_numeric(self, pred: str, gt: str, task_type: Optional[str] = None) -> float:
        """Score numeric answers with a relative-error-based continuous reward."""

        # Counting-style VQA occasionally collapses into box-like text; those should
        # never be rewarded as valid counts.
        if task_type == 'vqa_counting' and ('[' in pred or ']' in pred):
            return 0.0


        pred_num = self._extract_first_number(pred)
        gt_num = self._extract_first_number(gt)

        if pred_num is None or gt_num is None:
            return 0.0


        if not math.isfinite(pred_num) or not math.isfinite(gt_num):
            return 0.0


        if abs(pred_num - gt_num) < 1e-6:
            return 1.0


        if abs(gt_num) < 1e-6:
            return 0.0


        relative_error = abs(pred_num - gt_num) / abs(gt_num)


        # Large numeric misses are treated as complete failures.
        if relative_error > 0.5:
            return 0.0


        # Within the tolerance region, use an exponential decay so DPO gets a
        # smoother ranking signal than a hard correct/incorrect label.
        exponent = -self.numeric_tau * relative_error

        exponent = max(exponent, -100.0)
        return float(np.exp(exponent))

    def _score_bbox(self, pred: str, gt: str,
                    image_path: Optional[str] = None) -> float:
        """Score grounding outputs by Hungarian-matched box IoU."""

        pred_boxes = self._parse_bboxes(pred)
        gt_boxes = self._parse_bboxes(gt)


        if self.pred_bbox_norm1000 and pred_boxes:
            if image_path:
                size = self._get_image_size(image_path)
                if size:
                    pred_boxes = self._denorm1000_boxes(pred_boxes, size[0], size[1])
                else:
                    print(f"Warning: Cannot get image size for {image_path}, bbox score = 0")
                    return 0.0
            else:
                print("Warning: pred_bbox_norm1000=True but no image_path provided, bbox score = 0")
                return 0.0


        if not gt_boxes:

            return 1.0 if not pred_boxes else 0.0

        if not pred_boxes:

            return 0.0


        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=float)
        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou_matrix[i, j] = self._compute_bbox_iou(gt_box, pred_box)


        # Hungarian matching gives a one-to-one alignment between predicted and
        # ground-truth boxes before averaging IoU over GT boxes.
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)


        total_iou = iou_matrix[row_ind, col_ind].sum()
        avg_iou = total_iou / len(gt_boxes)

        return float(avg_iou)


    def _get_image_size(self, image_path: str) -> Optional[Tuple[int, int]]:
        """Load and cache image size for bbox denormalization."""
        if image_path in _IMAGE_SIZE_CACHE:
            return _IMAGE_SIZE_CACHE[image_path]
        try:
            from PIL import Image as PilImage
            with PilImage.open(image_path) as img:
                size = img.size
            _IMAGE_SIZE_CACHE[image_path] = size
            return size
        except Exception as e:
            print(f"Warning: Cannot read image size from '{image_path}': {e}")
            return None

    def _denorm1000_boxes(self, boxes: List[List[float]],
                          img_w: int, img_h: int) -> List[List[float]]:
        """Convert 0-1000 normalized boxes to absolute image coordinates."""
        result = []
        for box in boxes:
            x1, y1, x2, y2 = box
            result.append([
                round(x1 / 1000.0 * img_w),
                round(y1 / 1000.0 * img_h),
                round(x2 / 1000.0 * img_w),
                round(y2 / 1000.0 * img_h),
            ])
        return result

    def _normalize_text(self, text: str) -> str:
        """Lowercase and strip lightweight formatting for exact-match scoring."""
        text = text.lower().strip()
        text = text.rstrip('.')
        text = re.sub(r'\s+', ' ', text)
        return text

    _WORD_TO_NUM = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    }

    def _extract_first_number(self, text: str) -> Optional[float]:
        """Extract the first numeric mention, with support for simple number words."""
        if not text:
            return None

        text_lower = text.lower()
        candidates = []


        for m in re.finditer(r'-?\d+\.?\d*', text_lower):
            try:
                num = float(m.group())
                if math.isfinite(num):
                    candidates.append((m.start(), num))
            except (ValueError, OverflowError):
                pass


        pattern = r'\b(' + '|'.join(self._WORD_TO_NUM.keys()) + r')\b'
        for m in re.finditer(pattern, text_lower):
            candidates.append((m.start(), float(self._WORD_TO_NUM[m.group()])))

        if not candidates:
            return None


        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _parse_bboxes(self, text: str) -> List[List[float]]:
        """Parse bounding boxes from several supported serialized formats."""
        if not text:
            return []

        text = text.strip()

        if not text or text == '[]':
            return []

        bboxes = []


        # First try list-like formats, which are the most common model outputs.
        if '[' in text and ']' in text:
            try:

                import json
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:


                    if re.match(r'^[\[\]\d\s,.-]+$', text):
                        parsed = eval(text)
                    else:
                        raise ValueError("Invalid bbox format")

                if isinstance(parsed, list):

                    if len(parsed) >= 4 and all(isinstance(x, (int, float)) for x in parsed[:4]):
                        try:
                            bbox = [float(x) for x in parsed[:4]]
                            if all(math.isfinite(x) for x in bbox):
                                bboxes.append(bbox)
                        except (ValueError, TypeError, OverflowError):
                            pass


                    for item in parsed:
                        if isinstance(item, (list, tuple)) and len(item) >= 4:

                            try:
                                bbox = [float(x) for x in item[:4]]

                                if all(math.isfinite(x) for x in bbox):
                                    bboxes.append(bbox)
                            except (ValueError, TypeError, OverflowError):
                                continue
                return bboxes
            except (SyntaxError, ValueError, TypeError) as e:

                pass


        # Fall back to brace-wrapped coordinate groups if list parsing fails.
        bbox_strs = re.findall(r'\{([^}]+)\}', text)
        for bbox_str in bbox_strs:

            nums = re.findall(r'-?\d+\.?\d*', bbox_str)
            if len(nums) >= 4:
                try:

                    bbox = [float(x) for x in nums[:4]]

                    if all(math.isfinite(x) for x in bbox):
                        bboxes.append(bbox)
                except (ValueError, OverflowError):
                    continue

        return bboxes

    def _compute_bbox_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two axis-aligned bounding boxes."""
        if len(box1) < 4 or len(box2) < 4:
            return 0.0

        try:
            x1_min, y1_min, x1_max, y1_max = box1[:4]
            x2_min, y2_min, x2_max, y2_max = box2[:4]


            coords = [x1_min, y1_min, x1_max, y1_max, x2_min, y2_min, x2_max, y2_max]
            if not all(math.isfinite(x) for x in coords):
                return 0.0


            x1_min, x1_max = min(x1_min, x1_max), max(x1_min, x1_max)
            y1_min, y1_max = min(y1_min, y1_max), max(y1_min, y1_max)
            x2_min, x2_max = min(x2_min, x2_max), max(x2_min, x2_max)
            y2_min, y2_max = min(y2_min, y2_max), max(y2_min, y2_max)


            if x1_min >= x1_max or y1_min >= y1_max:
                return 0.0
            if x2_min >= x2_max or y2_min >= y2_max:
                return 0.0


            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)


            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0


            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)


            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)


            union_area = area1 + area2 - inter_area


            if union_area <= 1e-10:
                return 0.0

            iou = inter_area / union_area


            return max(0.0, min(1.0, float(iou)))

        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            return 0.0


def score_sample(prediction: str,
                 ground_truth: str,
                 task_type: str,
                 question: Optional[str] = None) -> float:
    """Convenience wrapper for scoring a single prediction."""
    scorer = ConsistencyOrientedScorer()
    return scorer.score(prediction, ground_truth, task_type, question)


def score_batch(predictions: List[str],
                ground_truths: List[str],
                task_types: List[str],
                questions: Optional[List[str]] = None) -> List[float]:
    """Convenience wrapper for scoring a batch of predictions."""
    scorer = ConsistencyOrientedScorer()
    return scorer.batch_score(predictions, ground_truths, task_types, questions)


def score_from_dict(prediction: str, sample: Dict,
                    pred_bbox_norm1000: bool = False) -> float:
    """Score one prediction using the fields stored in a sample dict."""
    if not isinstance(sample, dict):
        raise TypeError(f"sample must be a dict, got {type(sample).__name__}")

    ground_truth = sample.get('answer', '')
    task_type    = sample.get('task_type', '')
    question     = sample.get('question', None)
    image_path   = sample.get('image', None)

    if not task_type:
        raise ValueError("sample must contain 'task_type' field")

    scorer = ConsistencyOrientedScorer(pred_bbox_norm1000=pred_bbox_norm1000)
    return scorer.score(
        prediction=prediction,
        ground_truth=ground_truth,
        task_type=task_type,
        question=question,
        image_path=image_path
    )


if __name__ == '__main__':
    print("=" * 70)
    print("Consistency-oriented scorer - self test")
    print("=" * 70)

    scorer = ConsistencyOrientedScorer()


    test_cases = [

        ('scene_classification', 'church', 'church', (1.0, 1.0)),
        ('scene_classification', 'Church.', 'church', (1.0, 1.0)),
        ('scene_classification', 'residential', 'church', (0.0, 0.0)),

        ('vqa', 'yes', 'yes', (1.0, 1.0)),
        ('vqa', 'Yes.', 'yes', (1.0, 1.0)),
        ('vqa', 'red', 'red', (1.0, 1.0)),
        ('vqa', 'Red car', 'red', (0.0, 1.0)),
        ('vqa', '10', '10', (1.0, 1.0)),
        ('vqa', '11', '10', (0.5, 1.0)),
        ('vqa', 'There are 5 cars', '5', (1.0, 1.0)),
        ('vqa', 'one', 'There is only one airplane', (1.0, 1.0)),

        ('visual_grounding', '[[10, 20, 30, 40]]', '[[10, 20, 30, 40]]', (1.0, 1.0)),
        ('visual_grounding', '[[10, 20, 30, 40]]', '[[15, 25, 35, 45]]', (0.3, 0.7)),

        ('visual_grounding', '[[10, 20, 30, 40], [50, 60, 70, 80]]',
         '[[10, 20, 30, 40], [50, 60, 70, 80]]', (1.0, 1.0)),
    ]

    print("\nTest results:")
    print("-" * 70)

    for task_type, pred, gt, (min_score, max_score) in test_cases:
        score = scorer.score(pred, gt, task_type)
        status = "PASS" if min_score <= score <= max_score else "FAIL"
        print(f"{status} {task_type:20s} | Score: {score:.3f} | Pred: {pred[:30]:30s} | GT: {gt[:20]}")

    print("=" * 70)
