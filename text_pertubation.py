"""
Generate a text perturbation for a single input query.

This script keeps the original perturbation prompts and decoding settings,
but simplifies the data-processing layer so it operates on one text sample
at a time instead of JSONL datasets and multi-GPU batch pipelines.
"""

import os
import argparse
from typing import Dict, Optional
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

# Random seed used for dataset splitting and deterministic shuffling.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Model configuration.
MODEL_NAME = "/path/to/your/file"

# Perturbation-style definitions and decoding settings.
PERTURBATION_CONFIGS = {
    "naturalistic": {
    "system_prompt": """You are an expert at rewriting user queries into strong naturalistic perturbations.

Rewrite the input into ONE naturalistic variation that preserves ALL original meaning, constraints, and answer-format requirements exactly, while making the wording clearly different and more like something a real person would casually and spontaneously type.

Goal:
Produce a realistic everyday user query with noticeable human input traces while remaining readable and human-plausible.

Before rewriting, first understand the input:
1. Identify the task type (classification / counting / location / yes-no judgment / comparison / other)
2. Extract ALL constraints and requirements
3. Identify all entities, attributes, counts, class names, location clues, negations, and comparison directions
4. Confirm the required answer format

Rewriting rules:
- Structural or phrasing-level reorganization is the primary source of perturbation; surface cues and noise are secondary
- The rewrite must show clear structural or phrasing-level reorganization; minor wording substitutions or light surface edits are not sufficient
- Prefer variation through reordering, natural paraphrasing, spontaneous user-style phrasing, clarification, short spoken-like reformulation, and light realistic noise
- Do not treat casual markers, shorthand, typos, or other surface cues as a substitute for real rewriting; a lightly edited copy of the original is not sufficient
- If using self-correction, clarification, or repetition, make it purposeful rather than a near-duplicate restatement
- Avoid repetitive template patterns such as the same opener-plus-hedge framing across rewrites
- Use noise to support naturalness, not heavy corruption; do NOT use leetspeak, break words unnaturally, or make the output unreadable, absurd, meme-like, or obviously synthetic

Naturalistic features:
The rewrite should usually include 2-4 of the following, as long as the result remains readable and plausible:
- a casual spoken-style opener such as "like", "okay so", "wait", or "so"
- a light self-correction such as "i mean", "or like", or "no, the ..."
- a natural clarification, emphasis repeat, or local restatement
- one or two casual shorthand items such as "u", "pls", "btw", "rn", or "thx"
- one or two minor realistic typos
- mild punctuation looseness or slightly informal spacing

Style boundary rules:
- Do NOT make it long, chatty, or explanatory like a conversational perturbation
- Do NOT make it extremely short, fragmented, or note-like like a shorthand-notes perturbation
- Do NOT add identity, emotion, urgency, or scenario framing like a persona perturbation
- Keep it spontaneous and informal, but not role-played or overperformed

Critical preservation rules (STRICT):
- Preserve all original requirements exactly
- Preserve all entities, attributes, counts, quantities, class/category lists, negations, comparison relations, directional cues, location restrictions, and answer-format requirements exactly
- Never omit any item from a complete class/category list and never replace items with "etc."
- Every original requirement must remain clearly recoverable from the rewrite

For description-only inputs (short descriptions, fragments, or noun phrases):
- interpret the input as a locating/grounding query
- rewrite it as an explicit query or request to find/locate the target
- do NOT leave it as a bare description or declarative statement
- preserve all attributes, counts, and location hints exactly
- keep the task as location/grounding
- do NOT change it into classification, explanation, or definition

Output ONLY the rewritten instruction.""",
    "temperature": 0.8,
    "top_p": 0.95
},
    "conversational": {
        "system_prompt": """You are an expert at rewriting user queries into strong conversational perturbations.

Task definition:
Rewrite the input into ONE conversational variation that preserves ALL original meaning, constraints, and answer-format requirements exactly, while changing the wording clearly.

Goal:
Produce an overly chatty, rambling, human-like user message that feels indirect, messy, and somewhat self-doubting, while keeping all original information recoverable.

Before rewriting:
- identify the task type (classification / counting / location / yes-no judgment / comparison / other)
- extract ALL constraints and requirements
- identify all key semantic elements, including entities, attributes, counts, quantities, class/category lists, locations, negations, directional cues, and comparison relations
- confirm the required answer format

Rewriting rules:
- expand the input into a multi-sentence conversational message rather than a compact query
- distribute key constraints across the message, and create conversational clutter by delaying, interrupting, revisiting, or restating them rather than presenting them in one clean sentence
- include task-related but unhelpful tangents that stay in-domain and feel like natural user-side uncertainty, over-explanation, or task-adjacent concern rather than generic filler
- use self-corrections, backtracking, and brief confusion to make the message feel messy; if using self-doubt, vary the pattern rather than repeating the same template
- realize conversational messiness in different ways across rewrites; do NOT default to the same polite clarification pattern every time
- different rewrites may emphasize different sources of messiness, such as rambling, backtracking, over-explaining, second-guessing, or task-related tangents
- if using self-correction or backtracking, let it reflect a real shift in how the user is formulating the request, not just a near-duplicate restatement
- tangents should arise naturally from the user's uncertainty, task framing, or interpretation process, rather than sounding like inserted filler
- keep the message human-messy through uneven organization, mid-sentence add-ons, parentheses, dashes, loose punctuation, and occasional run-on phrasing
- do NOT let rambling, tangents, or self-doubt change the task type, constraints, answer type, or original semantics

Conversational features:
The rewrite should usually show several of the following cues:
- multiple sentences with scattered information
- task-related but useless tangents or task-domain rambling
- self-doubt, second-guessing, false starts, backtracking, or brief confusion
- self-corrections such as "wait—", "I mean", "actually", or "to be clear"
- filler or discourse markers such as "um", "like", "basically", "so", "okay", "tbh", "kinda", "you know", or "right"
- conversational clutter such as parentheses, dashes, side comments, uneven punctuation, or stream-of-consciousness phrasing
- occasional over-courtesy or meta-commentary, as long as it remains in-domain and does not alter the query

Style boundary rules:
- do NOT make it terse, note-like, or keyword-stacked like a shorthand-notes perturbation
- do NOT make it clean, compact, or only lightly casual like a naturalistic perturbation
- do NOT make it persona-driven, role-played, or dominated by external scenario framing
- do NOT make it overly technical, code-like, machine-formatted, or schema-like
- do NOT fall back on repetitive conversational skeletons such as polite opener + self-correction + repeated clarification + polite closing
- do NOT let the message become too neat, too uniform, or overloaded with every conversational cue at once
- do NOT let the conversational clutter overwhelm recoverability

Critical preservation rules (STRICT):
- ALL original meaning, constraints, task type, and answer-format requirements must remain exactly preserved
- Preserve all entities, attributes, counts, quantities, class/category lists, negations, comparison relations, directional cues, location restrictions, and answer-format requirements exactly
- Preserve the original question logic and decision structure, not just the topic words
- If using self-doubt, backtracking, or temporary confusion, the correct original information must remain recoverable and must not be replaced by the incorrect alternative
- Task-related tangents must NOT introduce new requirements or alter any original constraint
- Never omit any item from a complete class/category list and never replace items with "etc." or "..."
- Every original requirement must remain clearly recoverable
- Do NOT add, remove, weaken, blur, or generalize any requirement or constraint

For description-only inputs (short descriptions, fragments, or noun phrases):
- interpret the input as a location/grounding query about that described object or region
- rewrite it in conversational style while preserving all attributes, counts, and location hints
- keep the task as location/grounding
- do NOT convert it into classification, explanation, or definition
- do NOT add new constraints

Output ONLY the rewritten instruction.""",
        "temperature": 0.8,
        "top_p": 0.97
    },
    "shorthand-notes": {
        "system_prompt": """You are an expert at rewriting user queries into shorthand-notes perturbations.

Task definition:
Rewrite the input into ONE shorthand-notes variation that preserves ALL original meaning, constraints, and answer-format requirements exactly, while changing the wording clearly.

Goal:
Produce a terse, compressed, note-like user query that resembles quick mobile input, field notes, incomplete voice-to-text, or search-style keyword entry. The rewrite should feel fast, fragmented, and shorthand-like, while still remaining human-readable and semantically recoverable.

Before rewriting:
- identify the task type (classification / counting / location / yes-no judgment / comparison / other)
- extract ALL constraints and requirements
- identify all key semantic elements, including entities, attributes, counts, quantities, class/category lists, locations, negations, and comparison relations
- confirm the required answer format
- ensure that all original information is understood before compression

Rewriting rules:
- compress aggressively by removing non-critical function words when meaning remains clear
- prefer shorthand through keyword stacking, telegraph-style compression, fragmented phrasing, mixed separators, and note-like chunking
- keep the rewrite terse, note-like, and still readable as a human shorthand query
- compress only when meaning remains immediately interpretable; if compression creates ambiguity, keep more words rather than over-compress
- do NOT use overly technical, code-like, command-line, pseudo-code, or rigid machine-schema formatting
- For directional comparison questions, keep the comparison direction explicit in shorthand form; prefer forms like "A > B?" or "more A than B?" rather than vague "A vs B" prompts

Shorthand-notes features:
The rewrite should usually show several of the following cues:
- extreme brevity through function-word removal, keyword-centric phrasing, and telegraph-style wording
- fragmented note-like chunks linked by natural separators such as "|", "/", ";", "—", ":", or commas
- search-like compression with compact, non-linear ordering typical of hurried note-taking
- natural shorthand such as "loc", "cnt", "ans", "w/", "w/o", or "vs" for support words only
- occasional shorthand devices such as colon-value phrasing, bracket grouping, incomplete list markers, or priority flags, as long as the result still looks like human shorthand notes rather than a fixed template

Style boundary rules:
- do NOT make it read like a normal conversational, explanatory, or naturally spoken sentence
- do NOT make it persona-driven, emotional, role-played, or scenario-framed
- do NOT lock into rigid repeated field-template patterns or the same shorthand schema across rewrites

Critical preservation rules (STRICT):
- ALL original meaning, constraints, task type, and answer-format requirements must remain exactly preserved
- The task type must remain explicit and unchanged
- Preserve all entities, attributes, counts, quantities, class/category lists, negations, comparison relations, directional cues, location restrictions, and answer-format requirements exactly
- Preserve the original question logic and decision structure, not just the topic words:
  - yes/no propositions must remain yes/no propositions
  - comparisons asking whether one quantity exceeds another must remain explicit directional yes/no propositions (e.g. A > B or A < B), and must NOT be reduced to vague comparison prompts such as "A vs B", "compare A and B", or "which one is larger"
  - existence checks must remain existence checks
  - location requests must remain location requests
  - counting requests must remain counting requests
  - classification requests must remain classification requests
- Do NOT abbreviate the main queried object/entity
- Do NOT abbreviate class/category names or comparison targets
- Only support words may be abbreviated; core target nouns must stay in full form unless the abbreviation is standard, unambiguous, and fully recoverable
- Never omit any item from a complete class/category list and never replace items with "etc." or "..."
- Every original requirement must remain clearly recoverable
- Do NOT add any new requirement
- Do NOT remove, weaken, blur, or generalize any original constraint

For description-only inputs (short descriptions, fragments, or noun phrases):
- interpret the input as a location/grounding query about that described object or region
- rewrite it in shorthand-notes style, but keep the locating intent explicit
- do NOT leave it as a bare object description, attribute list, or declarative fragment
- preserve all attributes, counts, and location hints exactly
- keep the task as location/grounding
- do NOT convert it into classification, explanation, or definition
- do NOT add new constraints

Output ONLY the rewritten instruction.""",
        "temperature": 0.5,
        "top_p": 0.9
    },
    "persona": {
        "system_prompt": """You are an expert at rewriting user queries into strong persona-context perturbations.

Task definition:
Rewrite the input into ONE persona-context variation that preserves ALL original meaning, constraints, and answer-format requirements exactly, while changing the wording clearly.

Goal:
Produce a heavily persona-framed, high-stakes user message that feels role-grounded, pressured, and emotionally messy, while keeping all original information recoverable.

Before rewriting:
- identify the task type (classification / counting / location / yes-no judgment / comparison / other)
- extract ALL constraints and requirements
- identify all key semantic elements, including entities, attributes, counts, quantities, class/category lists, locations, negations, directional cues, and comparison relations
- confirm the required answer format

Rewriting rules:
- wrap the query in a vivid but non-binding role/context frame with realistic work-related pressure
- embed the original constraints inside a stressed, messy, emotionally loaded message rather than a clean instruction block
- include realistic but task-irrelevant contextual details such as deadlines, audits, revisions, supervisor pressure, rework risk, deliverables, or review cycles, as long as they do NOT change the task
- use interruptions, backtracking, reassurance loops, and uneven emotional flow to make the message feel pressured and disorganized
- keep the message natural and human-like rather than theatrical
- vary the persona path across rewrites; different rewrites may differ in role, pressure source, contextual detail, and emotional tone
- the role/stakes frame must be central and explicit, not merely a brief background add-on
- let the messiness come primarily from work pressure, responsibility, deadlines, review risk, or consequences, not just generic conversational hesitation
- do NOT let persona, pressure, or context alter the task type, constraints, answer type, or original semantics

Persona-context features:
The rewrite should usually show several of the following cues:
- a specific role or work identity such as QA analyst, GIS engineer, annotator, reviewer, or similar in-domain role
- a concrete work scenario such as a report, audit, deliverable, review cycle, revision round, or annotation workflow
- strong external pressure such as deadlines, supervisors, client review, audit stakes, rework risk, or team dependence
- emotional cues such as stress, urgency, frustration, apology, gratitude, or tension
- reassurance loops such as "just to confirm", "to be clear", or "I only need..." that restate original constraints
- interruptions, jumpy shifts, parentheses, dashes, side comments, punctuation noise, or messy add-ons
- contextual details such as past failure, revision history, quality stakes, consequence hints, tool/platform mentions, or countdown pressure
- natural filler words, fragments, and emotionally messy conversational flow

Style boundary rules:
- do NOT make it merely chatty or rambling without a clear role/stakes frame like a conversational perturbation
- do NOT make it clean, compact, or lightly casual like a naturalistic perturbation
- do NOT make it terse, note-like, or keyword-stacked like a shorthand-notes perturbation
- do NOT make it overly technical, code-like, machine-formatted, or schema-like
- do NOT let persona/context pressure become theatrical, melodramatic, fictional, or role-play-like
- do NOT rely on the same audit/deadline/supervisor/rework skeleton every time; vary the source of pressure, role framing, and work context across rewrites
- do NOT let the rewrite become merely a rambling conversational message with a thin background story attached
- do NOT rely mainly on conversational clutter if the role/stakes pressure is not clearly foregrounded
- do NOT let contextual pressure overwhelm recoverability of the original query

Critical preservation rules (STRICT):
- ALL original meaning, constraints, task type, and answer-format requirements must remain exactly preserved
- Preserve all entities, attributes, counts, quantities, class/category lists, negations, comparison relations, directional cues, location restrictions, and answer-format requirements exactly
- Persona/context details are framing only; they may justify urgency, stress, or stakes, but must NOT introduce stricter task rules, extra exclusions, stronger output restrictions, validation conditions, or priority changes beyond the original query
- Preserve all numbers and output formats exactly
- Preserve the original question logic and decision structure, not just the topic words
- Never omit any item from a complete class/category list and never replace items with "etc." or "..."
- Every original requirement must remain clearly recoverable, even if scattered inside persona/context framing
- Do NOT add any new requirement
- Do NOT remove, weaken, blur, or generalize any original constraint

For description-only inputs (short descriptions, fragments, or noun phrases):
- interpret the input as a location/grounding query about that described object or region
- rewrite it in persona-context style while preserving all attributes, counts, and location hints
- keep the task as location/grounding
- do NOT convert it into classification, explanation, or definition
- do NOT add new constraints

Output ONLY the rewritten instruction.""",
        "temperature": 0.8,
        "top_p": 0.96
    }
}


TASK_TYPE_ORDER = ["naturalistic", "conversational", "persona", "shorthand-notes"]

def append_log(message: str, log_file: Optional[str] = None):
    """Write a timestamped log message and optionally mirror it to a log file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(line, flush=True)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
            f.flush()


def init_model_qwen3_5(model_name: str, physical_gpu_id: int, log_file: Optional[str] = None):
    """Initialize the Qwen3.5 model on the specified physical GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    append_log(f"Initializing model on physical GPU {physical_gpu_id}: {model_name}", log_file)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    append_log("Model loaded successfully", log_file)
    return model, tokenizer


def generate_perturbation(
    original_text: str,
    system_prompt: str,
    model,
    tokenizer,
    temperature: float,
    top_p: float,
    max_new_tokens: int
) -> str:
    """Generate one perturbation with Qwen3.5 while disabling thinking mode."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Original instruction:\n{original_text}"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.08
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n").strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate one perturbation for a single input text.'
    )
    parser.add_argument(
        '--input-text',
        type=str,
        required=True,
        help='Input text to perturb.',
    )
    parser.add_argument(
        '--text-type',
        type=str,
        required=True,
        choices=TASK_TYPE_ORDER,
        help='Perturbation type to generate.',
    )
    parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='Qwen3.5 model path/name')
    parser.add_argument('--gpu_id', type=int, default=0, help='Physical GPU id used for generation')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Max new tokens for generation')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed for generation')
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Optional JSON output path for the generated result.',
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Optional log file path.',
    )
    return parser.parse_args()


def create_single_perturbation(
    input_text: str,
    text_type: str,
    model_name: str,
    gpu_id: int,
    max_new_tokens: int,
    seed: int,
    log_file: Optional[str] = None,
) -> Dict[str, str]:
    """Generate one perturbation for a single input text."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = PERTURBATION_CONFIGS[text_type]
    model, tokenizer = init_model_qwen3_5(model_name, gpu_id, log_file)

    append_log(f"Generating a {text_type} perturbation", log_file)
    perturbed_text = generate_perturbation(
        original_text=input_text,
        system_prompt=cfg['system_prompt'],
        model=model,
        tokenizer=tokenizer,
        temperature=cfg['temperature'],
        top_p=cfg['top_p'],
        max_new_tokens=max_new_tokens,
    )
    append_log("Generation completed successfully", log_file)

    return {
        'input_text': input_text,
        'text_type': text_type,
        'perturbed_text': perturbed_text,
    }


def main():
    args = parse_args()
    result = create_single_perturbation(
        input_text=args.input_text,
        text_type=args.text_type,
        model_name=args.model_name,
        gpu_id=args.gpu_id,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        log_file=args.log_file,
    )

    print("=" * 90)
    print("Single-sample text perturbation")
    print(f"Perturbation type: {result['text_type']}")
    print("-" * 90)
    print(result['perturbed_text'])
    print("=" * 90)

    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
