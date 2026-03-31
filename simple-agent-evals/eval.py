"""
Braintrust Evaluations for the Simple Agent.

This module runs offline evaluations against the agent using:
- Braintrust Eval() framework
- Autoevals built-in scorers (Factuality, ClosedQA) using Claude Sonnet 4.6 as judge
- Custom scorers (tool selection, response completeness, latency, scope awareness)

Usage:
    uv run python eval.py
    uv run python eval.py --dataset dataset2.json --output eval_metrics2.json
    uv run python eval.py --no-send-logs  # Run locally without sending to Braintrust
    uv run python eval.py --debug
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import (
    Any,
    Optional,
)

from autoevals.llm import (
    ClosedQA,
    Factuality,
)
from braintrust import Eval
from dotenv import load_dotenv
from openai import OpenAI


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()


# Constants
DEFAULT_DATASET_PATH = "dataset.json"
DEFAULT_OUTPUT_PATH = "eval_metrics.json"
BRAINTRUST_PROJECT_NAME = "simple-agent-evals"
EVAL_JUDGE_MODEL = "claude-sonnet-4-6"
ANTHROPIC_OPENAI_BASE_URL = "https://api.anthropic.com/v1/"


def _create_judge_client() -> OpenAI:
    """
    Create an OpenAI-compatible client pointing at Anthropic's API.

    Autoevals scorers use the OpenAI SDK interface. Anthropic provides
    an OpenAI-compatible endpoint so we can use Claude as the judge model.

    Returns:
        OpenAI client configured for Anthropic
    """
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")

    return OpenAI(
        api_key=anthropic_api_key,
        base_url=ANTHROPIC_OPENAI_BASE_URL,
    )


def _load_dataset(
    dataset_path: str
) -> list[dict]:
    """
    Load evaluation dataset from a JSON file.

    Args:
        dataset_path: Path to the dataset JSON file

    Returns:
        List of test case dictionaries
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(path, "r") as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} test cases from {dataset_path}")
    return dataset


def _run_agent_on_input(
    input_text: str
) -> dict:
    """
    Run the agent on a single input and capture output and tool usage.

    Args:
        input_text: The user's question to send to the agent

    Returns:
        Dictionary with 'output' text and 'tools_used' list
    """
    from agent import create_agent_for_eval

    agent = create_agent_for_eval()

    logger.info(f"Running agent on: {input_text[:80]}...")
    start_time = time.time()

    response = agent(input_text)

    elapsed = time.time() - start_time
    output_text = str(response)

    # Extract tool names from the agent's message history
    tools_used = _extract_tools_used(agent)

    logger.info(f"Agent responded in {elapsed:.1f}s, tools used: {tools_used}")

    return {
        "output": output_text,
        "tools_used": tools_used,
        "latency_seconds": elapsed,
    }


def _extract_tools_used(
    agent: Any
) -> list[str]:
    """
    Extract the list of tool names used by the agent from its message history.

    Args:
        agent: The Strands Agent instance after invocation

    Returns:
        List of tool name strings
    """
    tools_used = []

    messages = getattr(agent, "messages", [])
    for message in messages:
        if not isinstance(message, dict):
            continue

        content = message.get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue

            # Strands uses camelCase "toolUse" key with nested "name"
            tool_use = block.get("toolUse")
            if tool_use and isinstance(tool_use, dict):
                tool_name = tool_use.get("name", "")
                if tool_name and tool_name not in tools_used:
                    tools_used.append(tool_name)

    return tools_used


# ---------------------------------------------------------------------------
# Custom Scorers
# ---------------------------------------------------------------------------


def tool_selection_scorer(
    input: str,
    output: str,
    expected: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Check if the agent used the expected tools for this test case.

    Compares the tools the agent actually used against the expected_tools
    defined in the test case metadata. Scores based on overlap.

    Args:
        input: The user's question
        output: The agent's response
        expected: The expected output text
        metadata: Test case metadata containing expected_tools and tools_used

    Returns:
        Score dict with name, score (0-1), and metadata
    """
    if not metadata:
        return None

    expected_tools = metadata.get("expected_tools", [])
    tools_used = metadata.get("tools_used", [])

    if not expected_tools:
        return None

    # Calculate overlap score
    expected_set = set(expected_tools)
    used_set = set(tools_used)

    if not expected_set:
        return None

    # Intersection over expected (did we use the tools we should have?)
    correct_tools = expected_set.intersection(used_set)
    recall = len(correct_tools) / len(expected_set)

    # Penalty for using extra unexpected tools (mild penalty)
    extra_tools = used_set - expected_set
    precision_penalty = len(extra_tools) * 0.1

    score = max(0.0, recall - precision_penalty)

    return {
        "name": "ToolSelection",
        "score": score,
        "metadata": {
            "expected_tools": list(expected_set),
            "tools_used": list(used_set),
            "correct_tools": list(correct_tools),
            "extra_tools": list(extra_tools),
        },
    }


def response_completeness_scorer(
    input: str,
    output: str,
    expected: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Check if the response contains specific data points expected for each tool type.

    For weather queries: checks for temperature numbers.
    For directions queries: checks for distance and duration.
    For search queries: checks for substantive content length.

    Args:
        input: The user's question
        output: The agent's response
        expected: The expected output text
        metadata: Test case metadata containing category

    Returns:
        Score dict with name, score (0-1), and metadata
    """
    if not metadata or not output:
        return None

    category = metadata.get("category", "")
    checks_passed = 0
    checks_total = 0
    details = {}

    # Check for weather-related content
    if category in ("weather", "multi_tool"):
        checks_total += 1
        has_temperature = bool(re.search(r'\d+\.?\d*\s*[°FfCc]|degrees|\d+F|\d+C', output))
        if has_temperature:
            checks_passed += 1
        details["has_temperature"] = has_temperature

    # Check for directions-related content
    if category in ("directions", "multi_tool"):
        checks_total += 2
        has_distance = bool(re.search(r'\d+\.?\d*\s*(miles|mi|km|kilometers)', output, re.IGNORECASE))
        has_duration = bool(re.search(r'\d+\s*(hour|minute|hr|min)', output, re.IGNORECASE))
        if has_distance:
            checks_passed += 1
        if has_duration:
            checks_passed += 1
        details["has_distance"] = has_distance
        details["has_duration"] = has_duration

    # Check for search-related content (minimum substantive length)
    if category in ("search", "multi_tool"):
        checks_total += 1
        has_substance = len(output.split()) > 30
        if has_substance:
            checks_passed += 1
        details["has_substance"] = has_substance

    if checks_total == 0:
        return None

    score = checks_passed / checks_total

    return {
        "name": "ResponseCompleteness",
        "score": score,
        "metadata": details,
    }


def latency_scorer(
    input: str,
    output: str,
    expected: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Score the agent's response time. Faster is better.

    Scoring:
        < 10 seconds: 1.0
        10-20 seconds: 0.75
        20-30 seconds: 0.5
        30-60 seconds: 0.25
        > 60 seconds: 0.0

    Args:
        input: The user's question
        output: The agent's response
        expected: The expected output text
        metadata: Test case metadata containing latency_seconds

    Returns:
        Score dict with name, score (0-1), and metadata
    """
    if not metadata:
        return None

    latency = metadata.get("latency_seconds")
    if latency is None:
        return None

    if latency < 10:
        score = 1.0
    elif latency < 20:
        score = 0.75
    elif latency < 30:
        score = 0.5
    elif latency < 60:
        score = 0.25
    else:
        score = 0.0

    return {
        "name": "Latency",
        "score": score,
        "metadata": {"latency_seconds": round(latency, 2)},
    }


def no_error_scorer(
    input: str,
    output: str,
    expected: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Check that the agent response does not contain error messages.

    Args:
        input: The user's question
        output: The agent's response
        expected: The expected output text
        metadata: Test case metadata

    Returns:
        Score dict with name, score (0 or 1), and metadata
    """
    if not output:
        return {"name": "NoError", "score": 0, "metadata": {"reason": "empty output"}}

    error_patterns = [
        r"error",
        r"failed",
        r"could not",
        r"unable to",
        r"exception",
        r"timed out",
        r"I don't have access",
        r"I cannot",
    ]

    output_lower = output.lower()
    found_errors = []
    for pattern in error_patterns:
        if re.search(pattern, output_lower):
            found_errors.append(pattern)

    # Only flag if the error patterns seem to indicate actual failures,
    # not just the agent discussing errors in general
    is_error = len(found_errors) >= 2 or "exception" in found_errors

    return {
        "name": "NoError",
        "score": 0 if is_error else 1,
        "metadata": {"error_patterns_found": found_errors},
    }


def scope_awareness_scorer(
    input: str,
    output: str,
    expected: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[dict]:
    """
    Check if the agent correctly handles out-of-scope requests.

    For out_of_scope cases: the agent should decline and explain its limitations.
    For in-scope cases: the agent should NOT decline.

    Args:
        input: The user's question
        output: The agent's response
        expected: The expected output text
        metadata: Test case metadata containing category

    Returns:
        Score dict with name, score (0 or 1), and metadata
    """
    if not metadata or not output:
        return None

    category = metadata.get("category", "")
    output_lower = output.lower()

    # Phrases that indicate the agent is declining or acknowledging limitations
    decline_phrases = [
        "i can't",
        "i cannot",
        "i'm not able to",
        "i don't have the ability",
        "i don't have access",
        "i'm unable to",
        "outside my capabilities",
        "beyond my capabilities",
        "don't have a tool",
        "no tool",
        "not able to book",
        "not able to send",
        "not able to order",
        "not able to place",
        "unable to book",
        "unable to send",
        "unable to order",
        "unable to place",
        "unfortunately",
    ]

    is_declining = any(phrase in output_lower for phrase in decline_phrases)

    if category == "out_of_scope":
        # Agent SHOULD decline
        score = 1 if is_declining else 0
        return {
            "name": "ScopeAwareness",
            "score": score,
            "metadata": {
                "expected_behavior": "decline",
                "agent_declined": is_declining,
            },
        }

    # For in-scope categories, agent should NOT decline
    if is_declining:
        score = 0
        return {
            "name": "ScopeAwareness",
            "score": score,
            "metadata": {
                "expected_behavior": "answer",
                "agent_declined": is_declining,
            },
        }

    return {
        "name": "ScopeAwareness",
        "score": 1,
        "metadata": {
            "expected_behavior": "answer",
            "agent_declined": False,
        },
    }


# ---------------------------------------------------------------------------
# Task Function and Data Loader
# ---------------------------------------------------------------------------



def _create_wrapped_task(
    dataset: list[dict],
):
    """
    Create a task function that also injects runtime metadata (tools_used, latency)
    into the test case metadata for custom scorers to consume.

    Args:
        dataset: List of test case dictionaries

    Returns:
        Tuple of (task_function, enriched_data_function)
    """
    # We need to pass runtime info (tools_used, latency) to scorers via metadata.
    # Braintrust passes metadata from data(), but tools_used is only known at runtime.
    # Solution: run agent in data() and cache results, then task() returns cached output.

    results_cache = {}

    def data():
        cases = []
        for case in dataset:
            input_text = case["input"]

            # Run agent and cache result
            logger.info(f"Running agent for test case: {input_text[:60]}...")
            result = _run_agent_on_input(input_text)
            results_cache[input_text] = result

            cases.append({
                "input": input_text,
                "expected": case.get("expected_output", ""),
                "metadata": {
                    "expected_tools": case.get("expected_tools", []),
                    "category": case.get("category", ""),
                    "difficulty": case.get("difficulty", ""),
                    "tools_used": result["tools_used"],
                    "latency_seconds": result["latency_seconds"],
                },
            })

        return cases

    def task(input: str) -> str:
        # Return cached output (agent already ran in data())
        if input in results_cache:
            return results_cache[input]["output"]
        # Fallback: run agent if not cached
        result = _run_agent_on_input(input)
        return result["output"]

    return task, data


def _print_eval_summary(
    eval_result: Any,
    dataset: list[dict],
) -> None:
    """
    Print a detailed summary of evaluation results to the console and log.

    Shows per-scorer averages, per-category breakdowns, and any failed cases.

    Args:
        eval_result: The EvalResultWithSummary returned by Braintrust Eval()
        dataset: The original dataset for category info
    """
    results = eval_result.results

    if not results:
        logger.warning("No evaluation results to summarize")
        return

    # Build category lookup from dataset
    category_lookup = {}
    for case in dataset:
        category_lookup[case["input"]] = case.get("category", "unknown")

    # Collect scores per scorer and per category
    scorer_scores = {}
    category_scores = {}
    error_cases = []

    for r in results:
        input_text = str(r.input) if r.input else ""
        category = category_lookup.get(input_text, "unknown")

        if r.error:
            error_cases.append({"input": input_text[:80], "error": str(r.error)})
            continue

        for scorer_name, score_val in r.scores.items():
            if score_val is None:
                continue

            # Per-scorer aggregate
            if scorer_name not in scorer_scores:
                scorer_scores[scorer_name] = []
            scorer_scores[scorer_name].append(score_val)

            # Per-category aggregate
            cat_key = f"{category}/{scorer_name}"
            if cat_key not in category_scores:
                category_scores[cat_key] = []
            category_scores[cat_key].append(score_val)

    # Print overall summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total test cases: {len(results)}")
    print(f"Errors: {len(error_cases)}")
    print()

    # Per-scorer averages
    print("-" * 80)
    print(f"{'Scorer':<30} {'Avg Score':>10} {'Min':>8} {'Max':>8} {'Count':>8}")
    print("-" * 80)

    for scorer_name in sorted(scorer_scores.keys()):
        scores = scorer_scores[scorer_name]
        avg = sum(scores) / len(scores)
        min_s = min(scores)
        max_s = max(scores)
        print(f"{scorer_name:<30} {avg:>10.2%} {min_s:>8.2f} {max_s:>8.2f} {len(scores):>8}")

    print()

    # Per-category breakdown
    categories = sorted(set(category_lookup.values()))
    print("-" * 80)
    print("PER-CATEGORY BREAKDOWN")
    print("-" * 80)

    for category in categories:
        print(f"\n  [{category}]")
        for scorer_name in sorted(scorer_scores.keys()):
            cat_key = f"{category}/{scorer_name}"
            if cat_key in category_scores:
                scores = category_scores[cat_key]
                avg = sum(scores) / len(scores)
                print(f"    {scorer_name:<28} {avg:>8.2%}  ({len(scores)} cases)")

    print()

    # Error cases
    if error_cases:
        print("-" * 80)
        print("FAILED CASES")
        print("-" * 80)
        for case in error_cases:
            print(f"  Input: {case['input']}")
            print(f"  Error: {case['error']}")
            print()

    print("=" * 80 + "\n")

    # Also log the summary
    logger.info(
        f"Eval summary: {len(results)} cases, {len(error_cases)} errors, "
        f"scorers: {', '.join(f'{k}={sum(v)/len(v):.2%}' for k, v in sorted(scorer_scores.items()))}"
    )


def _export_eval_metrics(
    eval_result: Any,
    dataset: list[dict],
    output_path: str = "eval_metrics.json",
) -> None:
    """
    Export evaluation metrics to a JSON file for submission and review.

    The JSON file contains overall averages, per-scorer details, per-category
    breakdowns, and per-case scores so students can submit it as evidence
    of their evaluation run.

    Args:
        eval_result: The EvalResultWithSummary returned by Braintrust Eval()
        dataset: The original dataset for category info
        output_path: Path to write the JSON file
    """
    results = eval_result.results

    if not results:
        logger.warning("No results to export")
        return

    # Build category lookup
    category_lookup = {}
    for case in dataset:
        category_lookup[case["input"]] = case.get("category", "unknown")

    # Collect per-scorer and per-category aggregates
    scorer_scores = {}
    category_scores = {}
    per_case_results = []
    error_count = 0

    for r in results:
        input_text = str(r.input) if r.input else ""
        category = category_lookup.get(input_text, "unknown")

        case_entry = {
            "input": input_text[:120],
            "category": category,
            "scores": {},
            "error": None,
        }

        if r.error:
            error_count += 1
            case_entry["error"] = str(r.error)
            per_case_results.append(case_entry)
            continue

        for scorer_name, score_val in r.scores.items():
            if score_val is None:
                continue

            case_entry["scores"][scorer_name] = round(score_val, 4)

            if scorer_name not in scorer_scores:
                scorer_scores[scorer_name] = []
            scorer_scores[scorer_name].append(score_val)

            cat_key = f"{category}/{scorer_name}"
            if cat_key not in category_scores:
                category_scores[cat_key] = []
            category_scores[cat_key].append(score_val)

        per_case_results.append(case_entry)

    # Build overall summary
    overall = {}
    for scorer_name, scores in sorted(scorer_scores.items()):
        overall[scorer_name] = {
            "average": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "count": len(scores),
        }

    # Build per-category summary
    categories = sorted(set(category_lookup.values()))
    per_category = {}
    for category in categories:
        per_category[category] = {}
        for scorer_name in sorted(scorer_scores.keys()):
            cat_key = f"{category}/{scorer_name}"
            if cat_key in category_scores:
                scores = category_scores[cat_key]
                per_category[category][scorer_name] = {
                    "average": round(sum(scores) / len(scores), 4),
                    "count": len(scores),
                }

    # Assemble final output
    metrics = {
        "total_cases": len(results),
        "errors": error_count,
        "overall_scores": overall,
        "per_category": per_category,
        "per_case": per_case_results,
    }

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Evaluation metrics exported to {output_path}")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Braintrust evaluations on the Simple Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Run evals with default dataset and output
    uv run python eval.py

    # Run with custom dataset and output file
    uv run python eval.py --dataset dataset2.json --output eval_metrics2.json

    # Run locally without sending to Braintrust
    uv run python eval.py --no-send-logs

    # Run with debug logging
    uv run python eval.py --debug
""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to evaluation dataset JSON (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for the output eval metrics JSON (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--no-send-logs",
        action="store_true",
        help="Run evaluations locally without sending results to Braintrust",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this evaluation experiment (default: auto-generated)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def main() -> None:
    """Main function to run evaluations."""
    args = _parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting Simple Agent Evaluations")
    start_time = time.time()

    # Load dataset
    dataset = _load_dataset(args.dataset)
    logger.info(f"Loaded {len(dataset)} test cases")

    # Create task and data functions
    # The wrapped task runs the agent in data() and caches results
    # so that runtime metadata (tools_used, latency) is available to scorers
    task_fn, data_fn = _create_wrapped_task(dataset)

    # Create OpenAI-compatible client pointing at Anthropic API
    # so autoevals LLM-as-judge scorers use Claude Sonnet 4.6 instead of GPT-4o
    judge_client = _create_judge_client()

    # Define all scorers
    # Built-in autoevals (LLM-as-judge via Claude Sonnet 4.6)
    #   Factuality: Is the output factually consistent with expected?
    #   ClosedQA: Given input + expected, is the agent's answer correct?
    # Custom scorers (heuristic)
    #   ToolSelection: Did the agent call the right tools?
    #   ResponseCompleteness: Does the output contain expected data points?
    #   Latency: How fast did the agent respond?
    #   NoError: Does the response avoid error/failure indicators?
    #   ScopeAwareness: Does the agent decline out-of-scope requests?
    all_scorers = [
        # Factuality(model=EVAL_JUDGE_MODEL, client=judge_client),
        # ClosedQA(model=EVAL_JUDGE_MODEL, client=judge_client),
        tool_selection_scorer,
        response_completeness_scorer,
        latency_scorer,
        no_error_scorer,
        scope_awareness_scorer,
    ]

    # Build eval kwargs
    eval_kwargs = {
        "data": data_fn,
        "task": task_fn,
        "scores": all_scorers,
    }

    if args.experiment_name:
        eval_kwargs["experiment_name"] = args.experiment_name

    if args.no_send_logs:
        eval_kwargs["no_send_logs"] = True
        logger.info("Running in local mode (no logs sent to Braintrust)")

    # Run evaluation
    logger.info("Running Braintrust evaluation...")
    eval_result = Eval(
        BRAINTRUST_PROJECT_NAME,
        **eval_kwargs,
    )

    # Print detailed summary and export metrics
    _print_eval_summary(eval_result, dataset)
    _export_eval_metrics(eval_result, dataset, output_path=args.output)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    if minutes > 0:
        logger.info(f"Evaluation completed in {minutes} minutes and {seconds:.1f} seconds")
    else:
        logger.info(f"Evaluation completed in {seconds:.1f} seconds")


if __name__ == "__main__":
    main()
