import copy
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Template
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
ACE_APPWORLD_ROOT = REPO_ROOT / "libs" / "ace-appworld"
PROMPT_FILE = (
    ACE_APPWORLD_ROOT / "experiments" / "prompts" / "appworld_react_code_agent_playbook_generator_prompt.txt"
)

os.environ.setdefault("APPWORLD_ROOT", str(ACE_APPWORLD_ROOT))
if str(ACE_APPWORLD_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ACE_APPWORLD_ROOT / "src"))
if str(REPO_ROOT / "libs" / "ace") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "libs" / "ace"))


def ensure_localhost_no_proxy() -> None:
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        entries = [item.strip() for item in current.split(",") if item.strip()]
        for host in ("127.0.0.1", "localhost"):
            if host not in entries:
                entries.append(host)
        os.environ[key] = ",".join(entries)


def configure_noninteractive_environment() -> None:
    os.environ.setdefault("TERM", "dumb")
    os.environ.setdefault("PROMPT_TOOLKIT_NO_CPR", "1")
    try:
        from prompt_toolkit.input import defaults as pt_defaults
        from prompt_toolkit.input.base import DummyInput
    except Exception:
        return

    if getattr(pt_defaults, "_appworld_patched_noninteractive_input", False):
        return

    original_create_input = pt_defaults.create_input

    def create_input_no_tty(stdin=None, always_prefer_tty: bool = False):
        target = stdin if stdin is not None else sys.stdin
        if target is not None:
            try:
                if not target.isatty():
                    return DummyInput()
            except Exception:
                return DummyInput()
        return original_create_input(stdin=stdin, always_prefer_tty=always_prefer_tty)

    pt_defaults.create_input = create_input_no_tty
    pt_defaults._appworld_patched_noninteractive_input = True


ensure_localhost_no_proxy()
configure_noninteractive_environment()


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def configure_experiment_outputs_dir(experiment_outputs_dir: str | None) -> None:
    if experiment_outputs_dir:
        output_dir = os.path.abspath(experiment_outputs_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.environ["APPWORLD_EXPERIMENT_OUTPUTS"] = output_dir


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_reasoning_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def extract_json_payload(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Empty model response")
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(candidate.strip() for candidate in fenced if candidate.strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    start_positions = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not start_positions:
        raise ValueError(f"Unable to parse JSON from response: {text[:500]}")
    start = min(start_positions)
    for end in range(len(text), start, -1):
        snippet = text[start:end].strip()
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Unable to parse JSON from response: {text[:500]}")


def render_trace(trace: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for entry in trace:
        if "turn" in entry:
            blocks.append(
                "\n".join(
                    [
                        f"Turn {entry['turn']} code:",
                        entry["code"],
                        "",
                        f"Turn {entry['turn']} execution result:",
                        entry["execution_result"],
                    ]
                )
            )
        elif "test_report" in entry:
            blocks.append("Test report:\n" + entry["test_report"])
    return "\n\n".join(blocks).strip()


def render_conversation_history(trace: list[dict[str, Any]]) -> str:
    blocks = ["=== FULL CONVERSATION HISTORY ==="]
    for entry in trace:
        turn = entry.get("turn", "?")
        code = strip_reasoning_blocks(entry.get("code", ""))
        execution_result = strip_reasoning_blocks(entry.get("execution_result", ""))
        blocks.append(f"[turn {turn}] ASSISTANT:\n```python\n{code}\n```")
        blocks.append(f"[turn {turn}] USER:\nOutput:\n```\n{execution_result}\n```")
    return "\n\n".join(blocks)


def truncate_text(text: str, limit: int = 20000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[TRUNCATED]"


@dataclass
class ExecutionIO:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        use_max_completion_tokens: bool = False,
        temperature: float = 0.7,
        top_p: float | None = 0.8,
        presence_penalty: float | None = 1.5,
        max_tokens: int = 4096,
        max_retries: int = 5,
        retry_wait: int = 5,
        log_file_path: str | None = None,
        enable_thinking: bool = True,
    ):
        self.model = model_name
        client_kwargs: dict[str, Any] = {}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        self.client = OpenAI(**client_kwargs)
        self.use_max_completion_tokens = use_max_completion_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.log_file_path = log_file_path
        self.enable_thinking = enable_thinking

    def _log_call(
        self,
        messages: list[dict[str, str]],
        response_content: str | None = None,
        reasoning_content: str | None = None,
        error: str | None = None,
        attempt: int = 1,
    ) -> None:
        if not self.log_file_path:
            return
        payload = {
            "timestamp": int(time.time()),
            "model_name": self.model,
            "attempt": attempt,
            "messages": messages,
            "response": response_content,
            "reasoning_content": reasoning_content,
            "error": error,
        }
        with open(self.log_file_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                request_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                }
                if self.top_p is not None:
                    request_kwargs["top_p"] = self.top_p
                if self.presence_penalty is not None:
                    request_kwargs["presence_penalty"] = self.presence_penalty
                if self.use_max_completion_tokens:
                    request_kwargs["max_completion_tokens"] = self.max_tokens
                else:
                    request_kwargs["max_tokens"] = self.max_tokens
                if not self.enable_thinking:
                    request_kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                response = self.client.chat.completions.create(
                    **request_kwargs,
                )
                content = response.choices[0].message.content or ""
                reasoning_content = getattr(response.choices[0].message, "reasoning_content", None) or None
                self._log_call(messages=messages, response_content=content, reasoning_content=reasoning_content, attempt=attempt + 1)
                return {"content": content, "reasoning_content": reasoning_content}
            except Exception as exc:
                print(f"[LLM] Attempt {attempt + 1} failed: {str(exc)[:200]}")
                self._log_call(
                    messages=messages,
                    error=str(exc),
                    attempt=attempt + 1,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_wait)
        return {"content": ""}


def launch_sglang_server(
    model_name: str,
    tp: int = 1,
    dp: int = 1,
    host: str = "127.0.0.1",
    extra_args: list[str] | None = None,
    timeout: int = 3600,
) -> tuple[subprocess.Popen, str]:
    from sglang.utils import wait_for_server

    sglang_bin = shutil.which("sglang")
    cmd_parts = (
        [sglang_bin, "serve"]
        if sglang_bin
        else [sys.executable, "-m", "sglang.launch_server"]
    )
    cmd_parts.extend(["--model-path", model_name, "--host", host, "--tp", str(tp), "--dp", str(dp)])
    if "qwen3" in model_name.lower():
        cmd_parts.extend(["--reasoning-parser", "qwen3"])
    if extra_args:
        cmd_parts.extend(extra_args)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        port = sock.getsockname()[1]
    cmd_parts.extend(["--port", str(port)])

    log_dir = SCRIPT_DIR / "log" / "sglang"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sglang_{int(time.time())}_{os.getpid()}.log"
    log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    command = " ".join(cmd_parts)
    log_handle.write(f"$ {command}\n")
    proc = subprocess.Popen(
        cmd_parts,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    proc.sglang_log_handle = log_handle  # type: ignore[attr-defined]
    wait_for_server(f"http://{host}:{port}", timeout=timeout, process=proc)
    return proc, f"http://{host}:{port}/v1"


def shutdown_sglang_server(proc: subprocess.Popen) -> None:
    try:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                proc.terminate()

            deadline = time.time() + 10
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.2)

            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception:
                    proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
    finally:
        handle = getattr(proc, "sglang_log_handle", None)
        if handle:
            handle.flush()
            handle.close()


class PlaybookReActAgent:
    def __init__(
        self,
        prompt_template: str,
        playbook: str,
        llm_client: LLMClient,
        max_steps: int = 40,
        ignore_multiple_calls: bool = True,
        max_output_length: int = 400_000,
    ):
        self.prompt_template = prompt_template
        self.playbook = playbook
        self.llm_client = llm_client
        self.max_steps = max_steps
        self.ignore_multiple_calls = ignore_multiple_calls
        self.max_output_length = max_output_length
        self.partial_code_regex = r".*```python\n(.*)"
        self.full_code_regex = r"```python\n(.*?)```"
        self.messages: list[dict[str, Any]] = []
        self.num_instruction_messages = 0
        self.initial_prompt = ""

    def initialize(self, world) -> None:
        template = Template(self.prompt_template)
        app_descriptions = json.dumps(
            [{"name": k, "description": v} for k, v in world.task.app_descriptions.items()],
            indent=1,
        )
        template_params = {
            "input_str": world.task.instruction,
            "main_user": world.task.supervisor,
            "app_descriptions": app_descriptions,
            "relevant_apis": str(world.task.ground_truth.required_apis),
            "playbook": self.playbook,
        }
        output_str = template.render(template_params).lstrip() + "\n\n"
        self.initial_prompt = output_str
        self.messages = self._text_to_messages(output_str)
        self.num_instruction_messages = len(self.messages)

    def step(self, last_execution_outputs: list[ExecutionIO]) -> list[ExecutionIO]:
        if last_execution_outputs:
            content = truncate_text(last_execution_outputs[0].content)
            self.messages.append({"role": "user", "content": f"Output:\n```\n{content}```\n\n"})
        output = self.llm_client.generate(messages=self._trimmed_messages())
        code, fixed_content = self._extract_code_and_fix_content(output["content"])
        self.messages.append({"role": "assistant", "content": fixed_content + "\n\n"})
        return [ExecutionIO(content=code)]

    def _extract_code_and_fix_content(self, text: str) -> tuple[str, str]:
        if text is None:
            return "", ""
        original_text = text
        output_code = ""
        match_end = 0
        for re_match in re.finditer(self.full_code_regex, original_text, flags=re.DOTALL):
            code = re_match.group(1).strip()
            if self.ignore_multiple_calls:
                return code, original_text[: re_match.end()]
            output_code += code + "\n"
            match_end = re_match.end()
        partial_match = re.match(self.partial_code_regex, original_text[match_end:], flags=re.DOTALL)
        if partial_match:
            output_code += partial_match.group(1).strip()
            if not text.endswith("\n"):
                text += "\n"
            text += "```"
        return output_code, text

    def _text_to_messages(self, input_str: str) -> list[dict[str, Any]]:
        messages_json: list[dict[str, Any]] = []
        last_start = 0
        for match in re.finditer(r"(USER|ASSISTANT|SYSTEM):\n", input_str, flags=re.IGNORECASE):
            last_end = match.span()[0]
            if messages_json:
                messages_json[-1]["content"] = input_str[last_start:last_end]
            role = match.group(1).lower()
            messages_json.append({"role": role, "content": None})
            last_start = match.span()[1]
        messages_json[-1]["content"] = input_str[last_start:]
        return messages_json

    def _messages_to_text(self, messages: list[dict[str, Any]]) -> str:
        output = []
        for message in messages:
            output.append(f"{message['role'].upper()}:\n{message['content']}")
        return "".join(output)

    def _trimmed_messages(self) -> list[dict[str, Any]]:
        if self.max_output_length is None:
            return self.messages
        messages = copy.deepcopy(self.messages)
        pre_messages = messages[: self.num_instruction_messages - 1]
        post_messages = messages[self.num_instruction_messages - 1 :]
        output_str = self._messages_to_text(post_messages)
        remove_prefix = ""
        if "Task: " in output_str:
            remove_prefix = output_str[: output_str.index("Task: ") + 6]
            output_str = output_str.removeprefix(remove_prefix)
        observation_index = 0
        while len(output_str) > self.max_output_length:
            found_block = False
            if observation_index < len(post_messages) - 5:
                for message_index, message in enumerate(post_messages[observation_index:]):
                    if message["role"] == "user" and message["content"].startswith("Output:"):
                        message["content"] = "Output:\n```\n[NOT SHOWN FOR BREVITY]```\n\n"
                        found_block = True
                        observation_index += message_index + 1
                        break
            if not found_block and len(post_messages):
                first_post_message = copy.deepcopy(post_messages[0])
                if not first_post_message["content"].endswith("[TRIMMED HISTORY]\n\n"):
                    first_post_message["content"] += "[TRIMMED HISTORY]\n\n"
                post_messages = [first_post_message] + post_messages[2:]
                found_block = True
            output_str = self._messages_to_text(post_messages)
            if remove_prefix:
                output_str = output_str.removeprefix(remove_prefix)
        return pre_messages + post_messages


def extract_trace_from_messages(messages: list[dict[str, Any]], start_index: int) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    turn = 0
    for index in range(start_index, len(messages) - 1):
        assistant = messages[index]
        user = messages[index + 1]
        if assistant["role"] != "assistant" or user["role"] != "user":
            continue
        match = re.search(r"```python\n(.*?)```", assistant["content"], flags=re.DOTALL)
        if not match:
            continue
        output_match = re.search(r"Output:\n```\n(.*?)```", user["content"], flags=re.DOTALL)
        turn += 1
        trace.append(
            {
                "turn": turn,
                "code": strip_reasoning_blocks(match.group(1).strip()),
                "execution_result": strip_reasoning_blocks((output_match.group(1) if output_match else "").strip()),
            }
        )
    return trace


def solve_task(
    task_id: str,
    experiment_name: str,
    prompt_template: str,
    playbook: str,
    llm_client: LLMClient,
    max_steps: int,
    appworld_config: dict[str, Any],
) -> dict[str, Any]:
    from appworld import AppWorld
    from appworld.evaluator import evaluate_task
    from appworld.task import Task

    result = {"task_id": task_id, "success": False, "error": None, "num_steps": 0, "score": 0.0}
    agent: PlaybookReActAgent | None = None
    try:
        with AppWorld(task_id=task_id, experiment_name=experiment_name, **appworld_config) as world:
            agent = PlaybookReActAgent(
                prompt_template=prompt_template,
                playbook=playbook,
                llm_client=llm_client,
                max_steps=max_steps,
            )
            agent.initialize(world)
            execution_outputs: list[ExecutionIO] = []
            for step in range(max_steps):
                if world.task_completed():
                    break
                execution_inputs = agent.step(execution_outputs)
                result["num_steps"] = step + 1
                if not execution_inputs or not execution_inputs[0].content.strip():
                    break
                execution_outputs = [ExecutionIO(content=world.execute(execution_inputs[0].content))]
                if world.task_completed():
                    break
        test_tracker, report = evaluate_task(task_id, experiment_name)
        task = Task.load(task_id, ground_truth_mode="full")
        trace = (
            extract_trace_from_messages(agent.messages, agent.num_instruction_messages)
            if agent is not None
            else []
        )
        if report is None:
            report_text = ""
        elif isinstance(report, str):
            report_text = report
        else:
            report_text = report.export_text(clear=False, styles=False)
        result.update(
            {
                "instruction": task.instruction,
                "ground_truth_code": task.ground_truth.compiled_solution_code if task.ground_truth else "",
                "trace": trace,
                "test_report": report_text,
                "success": test_tracker.success,
                "score": test_tracker.pass_percentage / 100.0,
            }
        )
    except Exception:
        result["error"] = traceback.format_exc()
        if agent is not None:
            result["trace"] = extract_trace_from_messages(agent.messages, agent.num_instruction_messages)
        else:
            result["trace"] = []
        result["test_report"] = result["error"]
        task = Task.load(task_id, ground_truth_mode="full")
        result["instruction"] = task.instruction
        result["ground_truth_code"] = task.ground_truth.compiled_solution_code if task.ground_truth else ""
    return result


def normalize_section_name(section: str) -> str:
    normalized = section.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized.endswith("s") and len(normalized) > 1:
        normalized = normalized[:-1]
    return normalized


def normalize_tag_prefix(section: str) -> str:
    normalized = normalize_section_name(section)
    return normalized or "item"


def infer_tag_prefix_from_tag(tag: str) -> str:
    match = re.match(r"(?P<prefix>.+)-\d+$", tag.strip())
    if match:
        return normalize_tag_prefix(match.group("prefix"))
    return "item"


def parse_tagged_context(context: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    current_section = ""
    for raw_line in context.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            current_section = normalize_section_name(line[3:])
            continue
        match = re.match(r"\[(?P<tag>[^\]]+)\]\s*(?P<content>.*)", line)
        if match:
            tag = match.group("tag")
            section = current_section or normalize_section_name(infer_tag_prefix_from_tag(tag))
            items.append(
                {"tag": tag, "content": match.group("content"), "section": section}
            )
    return items


def next_tag(context: str, prefix: str = "item") -> str:
    normalized_prefix = normalize_tag_prefix(prefix)
    highest = 0
    for item in parse_tagged_context(context):
        if infer_tag_prefix_from_tag(item["tag"]) != normalized_prefix:
            continue
        match = re.search(r"(\d+)$", item["tag"])
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{normalized_prefix}-{highest + 1:05d}"


def render_flat_items(items: list[dict[str, str]]) -> str:
    return "\n".join(f"[{item['tag']}] {item['content']}" for item in items).strip()


def render_notebook(items: list[dict[str, str]]) -> str:
    observations = [item for item in items if normalize_section_name(item.get("section", "")) == "observation"]
    questions = [
        item for item in items if normalize_section_name(item.get("section", "")) == "question"
    ]
    blocks = ["## OBSERVATIONS"]
    blocks.extend(f"[{item['tag']}] {item['content']}" for item in observations)
    blocks.extend(["", "## OPEN QUESTIONS"])
    blocks.extend(f"[{item['tag']}] {item['content']}" for item in questions)
    return "\n".join(blocks).strip()


def apply_tagged_operations(
    context: str,
    operations: list[dict[str, Any]],
    include_sections: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    items = parse_tagged_context(context)
    index_by_tag = {item["tag"]: idx for idx, item in enumerate(items)}
    normalized_ops: list[dict[str, Any]] = []
    next_indices: dict[str, int] = {}
    for item in items:
        prefix = infer_tag_prefix_from_tag(item["tag"])
        match = re.search(r"(\d+)$", item["tag"])
        if match:
            next_indices[prefix] = max(next_indices.get(prefix, 1), int(match.group(1)) + 1)
    for op in operations:
        action = op["action"].lower()
        if action == "edit":
            tag = op["tag"]
            if tag in index_by_tag:
                items[index_by_tag[tag]]["content"] = op["content"].strip()
                if include_sections and op.get("section"):
                    items[index_by_tag[tag]]["section"] = normalize_section_name(op["section"])
                normalized_ops.append(op)
        elif action == "delete":
            tag = op["tag"]
            if tag in index_by_tag:
                items.pop(index_by_tag[tag])
                index_by_tag = {item["tag"]: idx for idx, item in enumerate(items)}
                normalized_ops.append(op)
        elif action == "add":
            section = normalize_tag_prefix(op.get("section", ""))
            next_index = next_indices.get(section, 1)
            tag = f"{section}-{next_index:05d}"
            next_indices[section] = next_index + 1
            item = {
                "tag": tag,
                "content": op["content"].strip(),
                "section": section,
            }
            after_tag = op.get("after_tag")
            if after_tag and after_tag in index_by_tag:
                insert_at = index_by_tag[after_tag] + 1
                items.insert(insert_at, item)
            else:
                items.append(item)
            index_by_tag = {entry["tag"]: idx for idx, entry in enumerate(items)}
            normalized = dict(op)
            normalized["tag"] = tag
            normalized_ops.append(normalized)
    if include_sections:
        return render_notebook(items), normalized_ops
    return render_flat_items(items), normalized_ops


class BaseModel:
    name = "base"
    prompt_file: Path | None = None  # override to use a custom agent prompt

    def initialize_context(self) -> str:
        return ""

    def update_context(
        self,
        llm_client: LLMClient,
        current_context: str,
        task_instruction: str,
        full_trace: list[dict[str, Any]],
        test_report: str,
        success: bool,
        ground_truth_code: str = "",
    ) -> tuple[str, Any]:
        raise NotImplementedError


def run_experiment(method: BaseModel, args: Any) -> None:
    from appworld.task import load_task_ids

    appworld_outputs_base = Path(args.experiment_outputs_dir or SCRIPT_DIR / "outputs")
    appworld_outputs_base.mkdir(parents=True, exist_ok=True)
    appworld_experiment_name = args.experiment_name or method.name
    configure_experiment_outputs_dir(str(appworld_outputs_base))
    method_outputs_dir = appworld_outputs_base / appworld_experiment_name
    method_outputs_dir.mkdir(parents=True, exist_ok=True)
    traces_jsonl = method_outputs_dir / "traces.jsonl"
    latest_context_file = method_outputs_dir / "latest_context.txt"
    agent_api_log = method_outputs_dir / "api_calls_agent.jsonl"
    context_api_log = method_outputs_dir / "api_calls_context.jsonl"
    prompt_template_str = getattr(method, "prompt_template", None)
    if prompt_template_str is not None:
        prompt_template = prompt_template_str.lstrip()
    else:
        prompt_file = getattr(method, "prompt_file", None) or PROMPT_FILE
        prompt_template = load_text(prompt_file).lstrip()
    task_ids = list(args.task_ids or load_task_ids(args.dataset))
    if args.task_seed is not None:
        random.Random(args.task_seed).shuffle(task_ids)
    if args.task_offset:
        task_ids = task_ids[args.task_offset :]
    if args.task_limit is not None:
        task_ids = task_ids[: args.task_limit]

    server_proc = None
    base_url = normalize_base_url(args.sglang_base_url or "http://127.0.0.1:30000/v1")
    if not args.skip_server_launch:
        server_proc, base_url = launch_sglang_server(
            model_name=args.model_name,
            tp=args.tp,
            dp=args.dp,
            host=args.sglang_host,
            extra_args=args.sglang_extra_args,
        )
    enable_thinking = not getattr(args, "disable_thinking", False)
    top_p = getattr(args, "top_p", 0.8)
    presence_penalty = getattr(args, "presence_penalty", 1.5)
    llm_client = LLMClient(
        model_name=args.model_name,
        base_url=base_url,
        api_key="None",
        use_max_completion_tokens=False,
        temperature=args.temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        max_tokens=args.max_tokens,
        log_file_path=str(agent_api_log),
        enable_thinking=enable_thinking,
    )
    context_model = args.context_model_name or args.model_name
    if context_model == args.model_name:
        context_client = llm_client
    else:
        context_client = LLMClient(
            model_name=context_model,
            base_url=None,
            api_key=None,
            use_max_completion_tokens=True,
            temperature=args.temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            max_tokens=args.max_tokens,
            log_file_path=str(context_api_log),
            enable_thinking=enable_thinking,
        )
    context_save_every = max(0, int(getattr(args, "context_save_every", 0)))
    current_context = method.initialize_context()
    latest_context_file.write_text(current_context, encoding="utf-8")
    try:
        for index, task_id in enumerate(task_ids, start=1):
            print(f"[{method.name}] {index}/{len(task_ids)} {task_id}")
            solve_result = solve_task(
                task_id=task_id,
                experiment_name=appworld_experiment_name,
                prompt_template=prompt_template,
                playbook=current_context,
                llm_client=llm_client,
                max_steps=args.max_steps,
                appworld_config={"random_seed": args.random_seed},
            )
            context_after, context_delta = method.update_context(
                llm_client=context_client,
                current_context=current_context,
                task_instruction=solve_result["instruction"],
                full_trace=solve_result["trace"],
                test_report=solve_result["test_report"],
                success=solve_result["success"],
                ground_truth_code=solve_result.get("ground_truth_code", ""),
            )
            record = {
                "task_id": task_id,
                "context_before": current_context,
                "full_trace": solve_result["trace"],
                "test_report": solve_result["test_report"],
                "context_after": context_after,
                "context_delta": context_delta,
                "metadata": {
                    "task_instruction": solve_result["instruction"],
                    "ground_truth_code": solve_result.get("ground_truth_code", ""),
                    "success": solve_result["success"],
                    "num_steps": solve_result["num_steps"],
                    "score": solve_result["score"],
                },
            }
            with traces_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            current_context = context_after
            latest_context_file.write_text(current_context, encoding="utf-8")
            if context_save_every and index % context_save_every == 0:
                context_snapshot_file = method_outputs_dir / f"context_step{index}.txt"
                context_snapshot_file.write_text(current_context, encoding="utf-8")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if server_proc is not None:
            shutdown_sglang_server(server_proc)
