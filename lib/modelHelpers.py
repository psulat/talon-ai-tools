import base64
import json
import logging
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Literal, NotRequired, Optional, TypedDict

import requests
from talon import actions, app, clip, resource, settings

from ..lib.pureHelpers import strip_markdown
from .anthropicBackend import (
    AnthropicError,
    create_message,
    extract_text,
    to_anthropic_content,
)
from .modelState import GPTState
from .modelTypes import GPTMessage, GPTMessageItem

""""
All functions in this this file have impure dependencies on either the model or the talon APIs
"""


# TypedDict definition for model configuration
class ModelConfig(TypedDict):
    name: str
    model_id: NotRequired[str]
    system_prompt: NotRequired[str]
    llm_options: NotRequired[dict[str, Any]]
    api_options: NotRequired[dict[str, Any]]


# Path to the models.json file
MODELS_PATH = Path(__file__).parent.parent / "models.json"

# Store loaded model configurations
model_configs: dict[str, ModelConfig] = {}

# --- Local claude-backend additions (2026-11-20, personal/docs/ai_plan.md) ---

# T1.6: conversation state for "model and ..." follow-ups on the Claude path.
# Append-only on purpose. Opus/Sonnet 5.5 bind each thinking block to the
# exact system prompt and every earlier message, and reject a replay whose
# history was edited (400 "bound to a different conversation"). So the
# system prompt is frozen at the thread's first turn (talon-ai-tools rebuilds
# it per request from the active window), old turns are never trimmed, and
# past the caps a new thread starts instead.
CLAUDE_THREAD_MAX_TURNS = 10
CLAUDE_THREAD_TTL_S = 600
_claude_thread: dict[str, Any] = {"system": "", "messages": [], "updated": 0.0}

# T1.7: local JSONL log of every model request, outside every git repo
# (%APPDATA%\talon\ai_logs on Windows, ~/.talon/ai_logs on Mac). It holds the
# text sent and received, so it never leaves this machine.
LOG_PATH = Path(__file__).resolve().parents[3] / "ai_logs" / "text.jsonl"


def load_model_config(f: IO) -> None:
    """
    Load model configurations from models.json
    """
    global model_configs
    try:
        content = f.read()
        configs = json.loads(content)
        # Convert list to dictionary with name as key
        model_configs = {config["name"]: config for config in configs}
    except Exception as e:
        notify(f"Failed to load models.json: {e!r}")
        model_configs = {}


def ensure_models_file_exists():
    if not MODELS_PATH.exists():
        with open(MODELS_PATH, "w") as f:
            f.write("[]")


ensure_models_file_exists()


# Set up file watcher to reload configuration when models.json changes
@resource.watch(str(MODELS_PATH))
def on_update(f: IO):
    load_model_config(f)


def resolve_model_name(model: str) -> str:
    """
    Get the actual model name from the model list value.
    """
    if model == "model":
        # Check for deprecated setting first for backward compatibility
        openai_model: str = settings.get("user.openai_model")  # type: ignore
        if openai_model != "do_not_use":
            logging.warning(
                "The setting 'user.openai_model' is deprecated. Please use 'user.model_default' instead."
            )
            model = openai_model
        else:
            model = settings.get("user.model_default")  # type: ignore
    return model


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """
    Get the configuration for a specific model from the loaded configs
    """
    return model_configs.get(model_name)


def messages_to_string(messages: list[GPTMessageItem]) -> str:
    """Format messages as a string"""
    formatted_messages = []
    for message in messages:
        if message.get("type") == "image_url":
            formatted_messages.append("image")
        else:
            formatted_messages.append(message.get("text", ""))
    return "\n\n".join(formatted_messages)


def notify(message: str):
    """Send a notification to the user. Defaults the Andreas' notification system if you have it installed"""
    try:
        actions.user.notify(message)
    except Exception:
        app.notify(message)
    # Log in case notifications are disabled
    print(message)


def get_token() -> str:
    """Get the OpenAI API key from the environment"""
    try:
        return os.environ["OPENAI_API_KEY"]
    except KeyError:
        message = "GPT Failure: env var OPENAI_API_KEY is not set."
        notify(message)
        raise Exception(message)


def format_messages(
    role: Literal["user", "system", "assistant"], messages: list[GPTMessageItem]
) -> GPTMessage:
    return {
        "role": role,
        "content": messages,
    }


def format_message(content: str) -> GPTMessageItem:
    return {"type": "text", "text": content}


def extract_message(content: GPTMessageItem) -> str:
    return content.get("text", "")


def format_clipboard() -> GPTMessageItem:
    clipped_image = clip.image()
    if clipped_image:
        data = clipped_image.encode().data()
        base64_image = base64.b64encode(data).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/;base64,{base64_image}"},
        }
    else:
        if not clip.text():
            raise RuntimeError(
                "User requested info from the clipboard but there is nothing in it"
            )

        return format_message(clip.text())  # type: ignore Unclear why this is not narrowing the type


def send_request(
    prompt: GPTMessageItem,
    content_to_process: Optional[GPTMessageItem],
    model: str,
    thread: str,
    destination: str = "",
) -> GPTMessageItem:
    """Generate run a GPT request and return the response"""
    model = resolve_model_name(model)

    continue_thread = thread == "continueLast"
    # For the request log (local addition); prompt["text"] is mutated below.
    original_prompt_text = prompt.get("text", "")

    notification = "GPT Task Started"
    if len(GPTState.context) > 0:
        notification += ": Reusing Stored Context"

    # Use specified model if provided
    if model:
        notification += f", Using model: {model}"

    if settings.get("user.model_verbose_notifications"):
        notify(notification)

    # Get model configuration if available
    config = get_model_config(model)

    language = actions.code.language()
    language_context = (
        f"The user is currently in a code editor for the programming language: {language}."
        if language != ""
        else None
    )
    application_context = f"The following describes the currently focused application:\n\n{actions.user.talon_get_active_context()}"
    snippet_context = (
        "\n\nPlease return the response as a snippet with placeholders. A snippet can control cursors and text insertion using constructs like tabstops ($1, $2, etc., with $0 as the final position). Linked tabstops update together. Placeholders, such as ${1:foo}, allow easy changes and can be nested (${1:another ${2:}}). Choices, using ${1|one,two,three|}, prompt user selection."
        if destination == "snip"
        else None
    )

    system_message = "\n\n".join(
        [
            item
            for item in [
                (
                    config["system_prompt"]
                    if config and "system_prompt" in config
                    else settings.get("user.model_system_prompt")
                ),
                language_context,
                application_context,
                snippet_context,
            ]
            + actions.user.gpt_additional_user_context()
            + [context.get("text") for context in GPTState.context]
            if item
        ]
    )

    content: list[GPTMessageItem] = [prompt]
    if content_to_process is not None:
        if content_to_process["type"] == "image_url":
            image = content_to_process
            # If we are processing an image, we have
            # to add it as a second message
            content = [prompt, image]
        elif content_to_process["type"] == "text":
            # If we are processing text content, just
            # add the text on to the same message instead
            # of splitting it into multiple messages
            prompt["text"] = (
                prompt["text"] + '\n\n"""' + content_to_process["text"] + '"""'  # type: ignore a Prompt has to be of type text
            )
            content = [prompt]

    request = GPTMessage(
        role="user",
        content=content,
    )

    model_endpoint: str = settings.get("user.model_endpoint")  # type: ignore
    started = time.perf_counter()
    response: Optional[GPTMessageItem] = None
    error: Optional[Exception] = None
    try:
        if model_endpoint == "llm":
            response = send_request_to_llm_cli(
                prompt, content_to_process, system_message, model, continue_thread
            )
        elif model_endpoint == "anthropic":
            # Local claude-backend branch addition (2026-11-20).
            response = send_request_to_anthropic(
                request, system_message, model, continue_thread
            )
        else:
            if continue_thread:
                notify(
                    "Warning: Thread continuation is only supported when using setting user.model_endpoint = 'llm'"
                )
            response = send_request_to_api(request, system_message, model)
        return response
    except Exception as e:
        error = e
        raise
    finally:
        log_request(
            endpoint=model_endpoint,
            model=model,
            prompt_text=original_prompt_text,
            content=content_to_process,
            response=response,
            continue_thread=continue_thread,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error=error,
        )


def log_request(
    *,
    endpoint: str,
    model: str,
    prompt_text: str,
    content: Optional[GPTMessageItem],
    response: Optional[GPTMessageItem],
    continue_thread: bool,
    elapsed_ms: float,
    error: Optional[Exception],
) -> None:
    """Append one request to the local JSONL log (local addition, personal/docs/ai_plan.md T1.7).

    Logging must never break a model command, so every failure here is swallowed.
    """
    try:
        if not settings.get("user.model_log_enabled"):
            return
        try:
            app_name = actions.app.name()
        except Exception:
            app_name = ""
        if content is None:
            input_text = ""
        elif content.get("type") == "text":
            input_text = content.get("text", "")
        else:
            input_text = "[image]"
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "app": app_name,
            "endpoint": endpoint if endpoint in ("anthropic", "llm") else "openai-style",
            "model": model,
            "thread": "continue" if continue_thread else "new",
            "ms": round(elapsed_ms),
            "ok": error is None,
            "error": str(error) if error else "",
            "prompt": prompt_text,
            "input": input_text,
            "output": response.get("text", "") if response else "",
        }
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"talon-ai-tools: could not write request log: {e!r}")


def send_request_to_api(
    request: GPTMessage, system_message: str, model: str
) -> GPTMessageItem:
    """Send a request to the model API endpoint and return the response"""
    # Get model configuration if available
    config = get_model_config(model)

    # Use model_id from configuration if available
    model_id = config["model_id"] if config and "model_id" in config else model

    data = {
        "messages": (
            [
                format_messages(
                    "system",
                    [GPTMessageItem(type="text", text=system_message)],
                ),
            ]
            if system_message
            else []
        )
        + [request],
        "max_tokens": 2024,
        "n": 1,
        "model": model_id,
    }

    # Check for deprecated temperature setting
    temperature: float = settings.get("user.model_temperature")  # type: ignore
    if temperature != -1.0:
        logging.warning(
            "The setting 'user.model_temperature' is deprecated. Please configure temperature in models.json instead."
        )
        data["temperature"] = temperature

    # Apply API options from configuration if available
    if config and "api_options" in config:
        data.update(config["api_options"])

    if GPTState.debug_enabled:
        print(data)

    url: str = settings.get("user.model_endpoint")  # type: ignore
    headers = {"Content-Type": "application/json"}
    token = get_token()
    # If the model endpoint is Azure, we need to use a different header
    if "azure.com" in url:
        headers["api-key"] = token
    else:
        headers["Authorization"] = f"Bearer {token}"

    # Timeout added on the local claude-backend branch (2026-11-20): upstream
    # waits forever, which freezes Talon if the endpoint hangs.
    timeout_s: int = settings.get("user.model_request_timeout")  # type: ignore
    raw_response = requests.post(
        url, headers=headers, data=json.dumps(data), timeout=(5, timeout_s)
    )

    match raw_response.status_code:
        case 200:
            if settings.get("user.model_verbose_notifications"):
                notify("GPT Task Completed")
            resp = raw_response.json()["choices"][0]["message"]["content"].strip()
            formatted_resp = strip_markdown(resp)
            return format_message(formatted_resp)
        case _:
            notify("GPT Failure: Check the Talon Log")
            raise Exception(raw_response.json())


def send_request_to_anthropic(
    request: GPTMessage,
    system_message: str,
    model: str,
    continue_thread: bool = False,
) -> GPTMessageItem:
    """Send a request to Claude (Anthropic Messages API) and return the response.

    Local addition on the claude-backend branch (2026-11-20); the HTTP client
    and the reason it isn't the official SDK are in lib/anthropicBackend.py.
    With continue_thread, the request is appended to the current thread
    (see _claude_thread above).
    """
    config = get_model_config(model)
    model_id = config["model_id"] if config and "model_id" in config else model
    new_turn = {"role": "user", "content": to_anthropic_content(request["content"])}  # type: ignore
    effort: str = settings.get("user.model_anthropic_effort")  # type: ignore
    timeout_s: int = settings.get("user.model_request_timeout")  # type: ignore

    history: list[dict] = _claude_thread["messages"]
    thread_usable = (
        bool(history)
        and time.monotonic() - _claude_thread["updated"] < CLAUDE_THREAD_TTL_S
        and len(history) // 2 < CLAUDE_THREAD_MAX_TURNS
    )
    if continue_thread and thread_usable:
        system = _claude_thread["system"]
        messages = history + [new_turn]
    else:
        if continue_thread:
            notify("No recent Claude conversation to continue - starting a new one")
        system = system_message
        messages = [new_turn]

    if GPTState.debug_enabled:
        print({"model": model_id, "effort": effort, "system": system, "messages": messages})

    def call(msgs: list[dict]) -> dict:
        return create_message(
            model=model_id,
            system=system,
            messages=msgs,
            effort=effort,
            timeout_s=timeout_s,
        )

    try:
        try:
            message = call(messages)
        except AnthropicError as e:
            # One-time recovery if the API rejects replayed thinking blocks
            # (should not happen with an append-only thread): strip them and
            # retry once, keeping the text of every turn.
            if not (e.status == 400 and "signature" in str(e) and len(messages) > 1):
                raise
            messages = _strip_thinking_blocks(messages)
            message = call(messages)
        text = extract_text(message)
    except AnthropicError as e:
        notify(f"Claude failed: {e}")
        raise

    # Replies are stored exactly as returned (thinking blocks included).
    _claude_thread.update(
        system=system,
        messages=messages + [{"role": "assistant", "content": message["content"]}],
        updated=time.monotonic(),
    )

    if message.get("stop_reason") == "max_tokens":
        notify("Claude's response was cut off (hit max_tokens)")
    elif settings.get("user.model_verbose_notifications"):
        notify("GPT Task Completed")
    return format_message(strip_markdown(text))


def _strip_thinking_blocks(messages: list[dict]) -> list[dict]:
    """Drop thinking/redacted_thinking blocks from assistant turns (local addition)."""
    stripped = []
    for msg in messages:
        if msg["role"] == "assistant" and isinstance(msg["content"], list):
            msg = {
                **msg,
                "content": [
                    block
                    for block in msg["content"]
                    if block.get("type") not in ("thinking", "redacted_thinking")
                ],
            }
        stripped.append(msg)
    return stripped


def send_request_to_llm_cli(
    prompt: GPTMessageItem,
    content_to_process: Optional[GPTMessageItem],
    system_message: str,
    model: str,
    continue_thread: bool,
) -> GPTMessageItem:
    """Send a request to the LLM CLI tool and return the response"""
    # Get model configuration if available
    config = get_model_config(model)

    # Use model_id from configuration if available
    model_id = config["model_id"] if config and "model_id" in config else model

    # Build command
    command: list[str] = [settings.get("user.model_llm_path")]  # type: ignore
    if continue_thread:
        command.append("-c")
    command.append(prompt["text"])  # type: ignore
    cmd_input: bytes | None = None
    if content_to_process and content_to_process["type"] == "image_url":
        img_url: str = content_to_process["image_url"]["url"]  # type: ignore
        if img_url.startswith("data:"):
            command.extend(["-a", "-"])
            base64_data: str = img_url.split(",", 1)[1]
            cmd_input = base64.b64decode(base64_data)
        else:
            command.extend(["-a", img_url])

    # Add model option
    command.extend(["-m", model_id])

    # Check for deprecated temperature setting
    temperature: float = settings.get("user.model_temperature")  # type: ignore
    if temperature != -1.0:
        logging.warning(
            "The setting 'user.model_temperature' is deprecated. Please configure temperature in models.json instead."
        )
        command.extend(["-o", "temperature", str(temperature)])

    # Apply llm_options from configuration if available
    if config and "llm_options" in config:
        for key, value in config["llm_options"].items():
            if isinstance(value, bool):
                if value:
                    command.extend(["-o", key, "true"])
                else:
                    command.extend(["-o", key, "false"])
            else:
                command.extend(["-o", key, str(value)])

    # Add system message if available
    if system_message:
        command.extend(["-s", system_message])

    if GPTState.debug_enabled:
        print(command)

    # Configure output encoding
    process_env = os.environ.copy()
    if platform.system() == "Windows":
        process_env["PYTHONUTF8"] = "1"  # For Python 3.7+ to enable UTF-8 mode
    # On other platforms, UTF-8 is also the common/expected encoding.
    output_encoding = "utf-8"

    # Execute command and capture output.
    try:
        result = subprocess.run(
            command,
            input=cmd_input,
            capture_output=True,
            check=True,
            creationflags=(
                subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0  # type: ignore
            ),
            env=process_env if platform.system() == "Windows" else None,
        )
        if settings.get("user.model_verbose_notifications"):
            notify("GPT Task Completed")
        resp = result.stdout.decode(output_encoding).strip()
        formatted_resp = strip_markdown(resp)
        return format_message(formatted_resp)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode(output_encoding).strip() if e.stderr else str(e)
        notify(f"GPT Failure: {error_msg}")
        raise e
    except Exception as e:
        notify("GPT Failure: Check the Talon Log")
        raise e
