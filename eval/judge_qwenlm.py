import argparse
import json
import os
import re
import sys
import threading
import time

from tqdm import tqdm

MCQ_BENCHMARKS = [
    "hrbench-4k",
    "hrbench-8k",
    "vstar",
    "zoombench",
    "mme-realworld",
    "mme-realworld-cn",
]

PROMPT_TEMPLATE = (
    "Your task is to judge whether the response expresses the same meaning "
    "as the answer of a question.\n"
    "The question is: {question}\n"
    "The answer is: {gt}\n"
    "The response is: {response}\n"
    "For multiple-choice questions, treat the response as correct if its final "
    "selected option letter or option text matches the ground-truth answer. "
    "Ignore reasoning that mentions other options before the final answer. "
    "Please check and compare them and then judge. "
    "If the response is correct, your output should be Yes. "
    "Otherwise, your output should be No. Directly give me your output."
)


def extract_answer(model_answer_raw):
    if not isinstance(model_answer_raw, str):
        return ""
    text = model_answer_raw
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    blocks = re.findall(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    marker = "Answer:"
    if marker in text:
        return text[text.rfind(marker) :].strip()
    return text.strip()


def extract_gt_option(gt):
    if not isinstance(gt, str):
        return ""
    match = re.search(r"([A-F])", gt.strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def extract_strict_final_option(answer):
    if not isinstance(answer, str) or not answer.strip():
        return ""
    text = answer.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    blocks = re.findall(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if blocks:
        text = blocks[-1].strip()

    strong_patterns = [
        r"(?:final\s+answer|correct\s+answer|answer|final\s+choice|correct\s+option)"
        r"\s*(?:is)?\s*:?\s*(?:\*\*)?\s*\(?([A-F])\)?\s*(?:\*\*)?(?=$|[\s\.\),;:])",
        r"(?:therefore|thus|so)\s*,?\s*(?:the\s+)?(?:final\s+)?(?:correct\s+)?answer\s*(?:is)?"
        r"\s*:?\s*(?:\*\*)?\s*\(?([A-F])\)?\s*(?:\*\*)?(?=$|[\s\.\),;:])",
    ]
    candidates = []
    for pattern in strong_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidates.append((match.start(), match.group(1).upper()))
    if candidates:
        return sorted(candidates)[-1][1]

    short = text.strip()
    match = re.search(r"^\s*(?:\*\*)?\s*\(?([A-F])\)?\s*(?:\*\*)?\s*[\.)]?\s*$", short, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    for line in reversed([line.strip() for line in short.splitlines() if line.strip()]):
        match = re.search(
            r"^\s*(?:\*\*)?\s*\(?([A-F])\)?\s*(?:\*\*)?\s*[\.)]?\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()
        if len(line) > 80:
            break
    return ""


def normalize_judge(text):
    value = (text or "").strip()
    has_yes = re.search(r"\byes\b", value, flags=re.IGNORECASE)
    has_no = re.search(r"\bno\b", value, flags=re.IGNORECASE)
    if has_yes and not has_no:
        return "Yes"
    if has_no and not has_yes:
        return "No"
    if value.lower().startswith("yes"):
        return "Yes"
    if value.lower().startswith("no"):
        return "No"
    return value


def judge_via_api(prompts, api_base, api_key, judge_model, judge_max_tokens, parallel_workers=32, judge_rpm=0):
    from openai import OpenAI

    thread_local = threading.local()

    def get_client():
        c = getattr(thread_local, "client", None)
        if c is None:
            c = OpenAI(api_key=api_key, base_url=api_base, timeout=600)
            thread_local.client = c
        return c

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [""] * len(prompts)
    errors = [""] * len(prompts)

    min_interval = 60.0 / judge_rpm if judge_rpm and judge_rpm > 0 else 0.0
    rate_lock = threading.Lock()
    next_allowed_time = [0.0]

    def wait_for_rate_limit_slot():
        if min_interval <= 0:
            return
        with rate_lock:
            now = time.monotonic()
            wait_time = max(0.0, next_allowed_time[0] - now)
            next_allowed_time[0] = max(now, next_allowed_time[0]) + min_interval
        if wait_time > 0:
            time.sleep(wait_time)

    def call_one(idx, prompt):
        client = get_client()
        last_error = ""
        for attempt in range(6):
            try:
                wait_for_rate_limit_slot()
                resp = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=judge_max_tokens,
                )
                return idx, (resp.choices[0].message.content or "").strip(), ""
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                sleep_time = 5.0 * (attempt + 1)
                if "429" in last_error or "Rate limit" in last_error:
                    sleep_time = max(sleep_time, 65.0)
                time.sleep(sleep_time)
        return idx, "NeedsJudge", last_error or "unknown error"

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [executor.submit(call_one, i, p) for i, p in enumerate(prompts)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM Judge"):
            idx, text, error = future.result()
            results[idx] = text
            errors[idx] = error

    return results, errors


def judge_via_vllm(prompts, judge_model_path, judge_max_tokens):
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(max_tokens=judge_max_tokens, temperature=0)
    llm = LLM(model=judge_model_path, tensor_parallel_size=1, gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(judge_model_path)

    chat_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_prompts.append(text)

    outputs = llm.generate(chat_prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs], [""] * len(outputs)


def main():
    parser = argparse.ArgumentParser(description="LLM-based judge for benchmark evaluation")
    parser.add_argument("--benchmark", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--judge_model_path", default=None, type=str, help="Local model path for vLLM-based judging")
    parser.add_argument("--api_base", default=None, type=str, help="OpenAI-compatible API base URL for judging")
    parser.add_argument("--api_key", default="EMPTY", type=str)
    parser.add_argument("--judge_model", default=None, type=str, help="Model name for API-based judging")
    parser.add_argument("--judge_max_tokens", default=2048, type=int)
    parser.add_argument(
        "--judge_parallel_workers",
        default=int(os.environ.get("JUDGE_PARALLEL_WORKERS", "32")),
        type=int,
        help="Parallel workers for API-based judging.",
    )
    parser.add_argument(
        "--judge_rpm",
        default=float(os.environ.get("JUDGE_RPM", "0")),
        type=float,
        help="Optional API request rate limit per minute. Use this for rate-limited judge APIs.",
    )
    parser.add_argument(
        "--answer_dir",
        default=os.environ.get("OUT_DIR", "model_answer"),
        type=str,
        help="Directory containing model answers, relative to EVAL_DATA_DIR/current directory.",
    )
    parser.add_argument(
        "--judge_dir",
        default=os.environ.get("JUDGE_DIR", "judge"),
        type=str,
        help="Directory for judge outputs, relative to EVAL_DATA_DIR/current directory.",
    )
    args = parser.parse_args()

    if not args.api_base and not args.judge_model_path:
        print(
            "ERROR: Either --api_base (for API-based judging) or --judge_model_path "
            "(for local vLLM judging) must be provided.",
            file=sys.stderr,
        )
        sys.exit(1)

    answer_path = os.path.join(args.answer_dir, args.benchmark, f"{args.model}_answer.jsonl")
    save_path = os.path.join(args.judge_dir, args.benchmark, f"{args.model}_answer.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    is_mcq = args.benchmark in MCQ_BENCHMARKS

    data_list = []
    with open(answer_path, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))

    try:
        from mathruler.grader import grade_answer

        has_mathruler = True
    except ImportError:
        has_mathruler = False

    to_llm_indices = []
    prompt_lists = []

    for i, item in enumerate(tqdm(data_list, desc="Rule-based grading")):
        question = item["query"].replace("<image>", "")
        model_answer_raw = item["model_answer"]
        extracted_answer = extract_answer(model_answer_raw)
        gt = item["response"]
        item["extracted_answer"] = extracted_answer

        is_correct = False
        if has_mathruler:
            try:
                is_correct = grade_answer(gt, extracted_answer)
            except Exception:
                is_correct = False

        gt_option = ""
        pred_option = ""
        if not is_correct and is_mcq:
            try:
                gt_option = extract_gt_option(gt)
                pred_option = extract_strict_final_option(extracted_answer)
            except Exception:
                gt_option = ""
                pred_option = ""

        if is_correct:
            item["judge"] = "Yes"
            item["judge_source"] = "mathruler"
        elif gt_option and pred_option:
            item["judge"] = "Yes" if gt_option == pred_option else "No"
            item["judge_source"] = "strict_final_option"
            item["strict_final_option"] = pred_option
        else:
            prompt = PROMPT_TEMPLATE.format(gt=gt, response=extracted_answer, question=question)
            to_llm_indices.append(i)
            prompt_lists.append(prompt)

    if prompt_lists:
        print(f"Calling LLM judge for {len(prompt_lists)} remaining cases...")
        if args.api_base:
            judge_model_name = args.judge_model or "default"
            results, errors = judge_via_api(
                prompt_lists,
                args.api_base,
                args.api_key,
                judge_model_name,
                args.judge_max_tokens,
                parallel_workers=args.judge_parallel_workers,
                judge_rpm=args.judge_rpm,
            )
        else:
            results, errors = judge_via_vllm(prompt_lists, args.judge_model_path, args.judge_max_tokens)

        for idx_in_llm, (response_text, error) in enumerate(zip(results, errors, strict=True)):
            original_idx = to_llm_indices[idx_in_llm]
            data_list[original_idx]["judge"] = normalize_judge(response_text)
            data_list[original_idx]["judge_raw"] = response_text
            data_list[original_idx]["judge_source"] = "api_error" if error else "llm"
            if error:
                data_list[original_idx]["judge_error"] = error

    print(f"Total: {len(data_list)}, LLM used: {len(prompt_lists)}")

    with open(save_path, "w", encoding="utf-8") as out_file:
        json.dump(data_list, out_file, ensure_ascii=False, indent=4)
    print(f"Saved judge results to: {save_path}")


if __name__ == "__main__":
    main()
