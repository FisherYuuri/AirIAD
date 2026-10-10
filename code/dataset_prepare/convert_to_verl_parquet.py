import os
import json
import argparse

from datasets import Dataset
from typing import List, Dict, Any

# AiRIAD system prompt with inspect tool and spatial/semantic CoT
SYSTEM_PROMPT = (
    "You are an industrial anomaly detection expert equipped with a visual inspection tool.\n\n"
    "## Role\n"
    "Your task is to localize and identify defects in industrial product images. You reason iteratively: each turn you propose a spatial crop and a semantic query, then verify whether the retrieved evidence is sufficient to conclude.\n\n"
    "## Tool\n"
    "You have access to one tool:\n\n"
    "  inspect(crop, query)\n\n"
    "  Parameters:\n"
    "    crop  — [x1, y1, x2, y2], normalized floats in [0,1000], top-left to bottom-right\n"
    "    query — the LETTER (A, B, C, D, etc.) of the anomaly type from the candidate list provided in the user message; choose the letter that best matches your current visual hypothesis\n\n"
    "  Returns:\n"
    "    - A zoomed image of the cropped region\n"
    "    - Retrieved knowledge: text descriptions of defects matching the query\n\n"
    "  Call this tool when you need more visual or semantic evidence. Do NOT call it if you have already reached a conclusion.\n\n"
    "## Output Format\n"
    "At every turn, output two brief analysis lines, then either a tool call OR a final answer — never both.\n\n"
    "Analysis format (required before every action):\n"
    "<spatial>[Evaluate the current visual evidence. Where is the suspicious region? Does the crop need adjustment?]</spatial>\n"
    "<semantic>[Evaluate the retrieved knowledge against the visual. Does the selected anomaly type match what you see? Should it be refined?]</semantic>\n\n"
    "Tool call format (when more evidence needed):\n"
    "<tool_call\n"
    "  <name>inspect</name>\n"
    "  <crop>[x1, y1, x2, y2]</crop>\n"
    "  <query>A/B/C/D/...</query>\n"
    "</tool_call\n\n"
    "Final answer format (when conclusion is reached):\n"
    "<answer>\n"
    "  <has_anomaly>yes/no</has_anomaly>\n"
    "  <anomaly_type>A/B/C/D/... or \"none\"</anomaly_type>\n"
    "  <bbox>[x1, y1, x2, y2] or null</bbox>\n"
    "</answer>\n\n"
    "IMPORTANT:\n"
    "- Use ONLY the LETTER (A, B, C, etc.) to refer to anomaly types, NOT the full name\n"
    "- If has_anomaly is \"no\", set anomaly_type to \"none\" and bbox to null"
)


def make_map_fn(split: str, data_source: str, mode: str = 'zero_shot'):
    """Build a verl-format mapping function. mode: zero_shot / one_shot."""

    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        question = example.get('question', '')
        image_path = example.get('image', '')
        class_name = example.get('class_name', '')
        gt_answer = example.get('gt_answer', False)
        anomaly_type = example.get('anomaly_type', 'none')
        anomaly_list = example.get('anomaly_list', [])
        bbox = example.get('bbox', [])
        qid = example.get('qid', f'{split}_{idx}')
        correct_answer = example.get('correct_answer', '')
        reference_image = example.get('similar_templates', [''])[0]

        prompt_content = f"{question}<image>" if '<image>' not in question else question
        verl_images = [{"image": image_path}] if image_path else []

        if mode == 'one_shot' and reference_image:
            hint = "A normal reference image is provided for comparison. Compare the target image with the reference to identify any visual differences. "
            prompt_content = prompt_content.replace("Determine whether", hint + "Determine whether")
            prompt_content += "\n\nHere is the normal reference image:<image>"
            verl_images.append({"image": reference_image})

        gt_dict = {
            'qid': qid, 'gt_answer': gt_answer, 'bbox': bbox,
            'anomaly_type': anomaly_type, 'correct_answer': correct_answer,
        }

        return {
            "agent_name": "airiad_tool_agent",
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_content},
            ],
            "images": verl_images,
            "ability": "anomaly_detection",
            "reward_model": {"style": "rule", "ground_truth": gt_dict},
            "extra_info": {
                'split': split, 'index': idx,
                'qid': qid, 'gt_answer': gt_answer, 'bbox': bbox,
                'correct_answer': correct_answer, 'anomaly_type': anomaly_type,
                'class_name': class_name,
                'ground_truth': {**gt_dict, 'is_anomaly': bool(gt_answer)},
                'tools_kwargs': {
                    'inspect': {'create_kwargs': {
                        'image': image_path, 'class_name': class_name, 'anomaly_list': anomaly_list,
                    }}
                },
            },
        }

    return process_fn


def main():
    parser = argparse.ArgumentParser(description='Convert AiRIAD JSON data to verl parquet format')
    parser.add_argument('--train_file', default='dataset/train-grpo366.json')
    parser.add_argument('--test_file', default='dataset/test-grpo.json')
    parser.add_argument('--output_dir', default='dataset')
    parser.add_argument('--data_source', default='agentiad/grpo')
    parser.add_argument('--mode', choices=['zero_shot', 'one_shot', 'mix'], default='mix',
                        help='zero_shot / one_shot / mix (alternating zs+os)')
    parser.add_argument('--test_mode', choices=['zero_shot', 'one_shot'], default='zero_shot')
    args = parser.parse_args()

    if args.test_mode is None:
        args.test_mode = 'one_shot' if args.mode == 'mix' else args.mode
    print(f"Train mode: {args.mode}, Test mode: {args.test_mode}")

    with open(args.train_file, 'r', encoding='utf-8', errors='ignore') as f:
        train_data = json.load(f)
    with open(args.test_file, 'r', encoding='utf-8', errors='ignore') as f:
        test_data = json.load(f)

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    if args.mode == 'mix':
        n = len(train_dataset)
        train_zs = train_dataset.select(range(0, n, 2)).map(
            make_map_fn('train', args.data_source, 'zero_shot'), with_indices=True, remove_columns=train_dataset.column_names)
        train_os = train_dataset.select(range(1, n, 2)).map(
            make_map_fn('train', args.data_source, 'one_shot'), with_indices=True, remove_columns=train_dataset.column_names)
        mixed = []
        for i in range(min(len(train_zs), len(train_os))):
            mixed.extend([train_zs[i], train_os[i]])
        if len(train_zs) > len(train_os):
            mixed.append(train_zs[len(train_os)])
        train_dataset = Dataset.from_list(mixed)
        print(f"Mix: {(n+1)//2} zero_shot + {n//2} one_shot")
    else:
        train_dataset = train_dataset.map(
            make_map_fn('train', args.data_source, args.mode), with_indices=True, remove_columns=train_dataset.column_names)

    test_dataset = test_dataset.map(
        make_map_fn('test', args.data_source, args.test_mode), with_indices=True, remove_columns=test_dataset.column_names)

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = '_mix' if args.mode == 'mix' else ('_os' if args.mode == 'one_shot' else '_zs')
    test_suffix = '_os' if args.test_mode == 'one_shot' else '_zs'
    train_path = os.path.join(args.output_dir, f'train{suffix}.parquet')
    test_path = os.path.join(args.output_dir, f'test{test_suffix}.parquet')
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    print(f"Saved: {train_path}, {test_path}")


if __name__ == '__main__':
    main()
