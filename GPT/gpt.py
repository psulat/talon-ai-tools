import os
import threading
import time
from typing import Any, Optional

from talon import Module, actions, clip, cron, settings, ui

from ..lib.HTMLBuilder import Builder
from ..lib.modelConfirmationGUI import confirmation_gui
from ..lib.modelHelpers import (
    extract_message,
    format_clipboard,
    format_message,
    messages_to_string,
    notify,
    send_request,
)
from ..lib.modelState import GPTState
from ..lib.modelTypes import GPTMessageItem

mod = Module()
mod.tag(
    "model_window_open",
    desc="Tag for enabling the model window commands when the window is open",
)


# Destinations that type into the focused window (local claude-backend
# addition, 2026-11-20): if the window changed while waiting for the model,
# these are redirected to the clipboard instead.
_PASTE_DESTINATIONS = ("", "paste", "above", "below", "chain", "snip")


def _active_window_id():
    try:
        return ui.active_window().id
    except Exception:
        return None


# --- Local claude-backend additions (2026-11-20, personal/ai_plan.md) ---
# gpt_apply_prompt runs its request in a background thread so Talon keeps
# listening (it used to block speech for the whole request, ~1-2 s, tripping
# Talon's 2 s watchdog). Only one request at a time; "jarvis cancel" bumps the
# id so a late result is ignored.
_job: dict[str, Any] = {"id": 0, "running": False, "started": 0.0}
_JOB_STALE_S = 60


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _text_pattern():
    """UI Automation text pattern of the focused control, or None if it has none."""
    try:
        element = ui.focused_element()
        return element.text_pattern if "Text" in element.patterns else None
    except Exception:
        return None


# Characters editors substitute as you type. Word's AutoCorrect turned the
# dictated "what's" into "what’s" on 2026-11-20, so exact matching never found
# the last dictated phrase. Comparisons fold these (and case) first.
_FOLD = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "–": "-",
    "—": "-",
    " ": " ",
    "…": "...",
    "\r": "\n",
}


def _fold_with_map(text: str) -> tuple[str, list[int]]:
    """Fold `text` for tolerant comparison.

    Returns (folded, index_map), where index_map[j] is the index in `text` of
    the character that produced folded[j], so a match in folded text can be
    mapped back to real character positions.
    """
    folded: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        if ch == "\r" and text[i + 1 : i + 2] == "\n":
            continue  # CRLF counts once, as its "\n"
        for f in _FOLD.get(ch, ch).lower():
            folded.append(f)
            index_map.append(i)
    return "".join(folded), index_map


def _fold(text: str) -> str:
    return _fold_with_map(text)[0]


def _selection_matches(expected: str) -> bool:
    """True if the current selection is `expected`, compared tolerantly (see _FOLD).

    Asks UI Automation first (instant, leaves the clipboard alone) and falls
    back to copying the selection.
    """
    want = _fold(expected).strip()
    pattern = _text_pattern()
    if pattern is not None:
        try:
            ranges = pattern.selection
            if ranges and _fold(ranges[0].text).strip() == want:
                return True
        except Exception:
            pass
    return _fold(actions.edit.selected_text()).strip() == want


def _uia_select_before_cursor(text: str, pattern=None, caret=None) -> bool:
    """Select `text`, which should end just before the cursor, via UI Automation.

    Reads the characters before the cursor, checks (tolerantly, see _FOLD)
    that they end with `text` (only whitespace may follow it), and selects
    exactly those characters - leading/trailing whitespace excluded. Instant:
    about 10-15 ms in Notepad, vs over a second of Shift+Left presses for
    ~400 characters. (An earlier version used TextRange.find_text, which
    can't tolerate AutoCorrect's curly quotes; note Talon's find_text raises
    OSError("...completed successfully") when nothing matches.)
    `pattern`/`caret` default to the focused control and its cursor (they are
    parameters so the logic can be tested against a background window).
    """
    pattern = pattern or _text_pattern()
    if pattern is None:
        return False
    target = _fold(text).strip()
    if not target:
        return False
    try:
        if caret is None:
            ranges = pattern.selection
            if not ranges:
                return False
            caret = ranges[0]
        # Collapsed range at the cursor. (Word reports a range covering the
        # paragraph mark after the cursor, so collapse to its end.)
        end = caret.clone()
        end.move_endpoint_by_range("Start", "End", target=caret)
        head = end.clone()
        head.move_endpoint_by_unit("Start", "Character", -(len(text) + 200))
        head_text = head.text
        folded, index_map = _fold_with_map(head_text)
        trimmed = folded.rstrip()
        if not trimmed.endswith(target):
            return False
        first = index_map[len(trimmed) - len(target)]
        last = index_map[len(trimmed) - 1] + 1
        selection = end.clone()
        selection.move_endpoint_by_unit("Start", "Character", -(len(head_text) - first))
        selection.move_endpoint_by_unit("End", "Character", -(len(head_text) - last))
        selection.select()
        return True
    except Exception as e:
        print(f"GPT: UI Automation select failed: {e!r}")
    return False


def _keys_select_before_cursor(text: str) -> bool:
    """Fallback for controls without UI Automation text: Shift+Left by character count.

    Moving by characters is immune to word wrap (upstream's lines-up +
    characters selected too little on 2026-11-20). One fast burst first; if
    the check fails (Notepad dropped about 7 of ~50 rapid Backspaces that
    day), collapse and retry once in slow chunks.
    """
    # Leading whitespace is not selected (same as the UIA path), so the
    # result replaces just the words.
    count = len(_normalize(text).strip())
    for chunk, pause in ((count, "0ms"), (10, "20ms")):
        remaining = count
        while remaining > 0:
            step = min(chunk, remaining)
            actions.key(f"shift-left:{step}")
            if pause != "0ms":
                actions.sleep(pause)
            remaining -= step
        if _selection_matches(text):
            return True
        actions.key("right")  # collapse to the selection's end, where the cursor was
    return False


def _select_before_cursor(text: str) -> bool:
    """Select `text`, which should end at the cursor, and verify the selection."""
    if not _normalize(text).strip():
        return False
    if _uia_select_before_cursor(text):
        if _selection_matches(text):
            return True
        actions.key("right")
    if _keys_select_before_cursor(text):
        return True
    _log_select_failure(text)
    return False


def _log_select_failure(text: str) -> None:
    """Write what was expected vs. what is before the cursor to talon.log (local only)."""
    before = "(no UI Automation text)"
    pattern = _text_pattern()
    if pattern is not None:
        try:
            head = pattern.document_range.clone()
            head.move_endpoint_by_range("End", "End", target=pattern.selection[0])
            before = head.text[-80:]
        except Exception as e:
            before = f"(error reading: {e!r})"
    print(
        f"GPT: couldn't select {len(text)} chars ending {text[-60:]!r}; "
        f"text before the cursor: {before!r}"
    )


def _recent_paste_here(window_id) -> bool:
    """True if the last response was pasted in this window within the last 10 minutes."""
    return (
        GPTState.last_was_pasted
        and bool(GPTState.last_response)
        and _job.get("paste_window") == window_id
        and time.monotonic() - _job.get("paste_time", 0.0) < 600
    )


def _clean_result(result: str, original: str) -> str:
    """Undo small model habits when replacing `original` in place."""
    # Wrapping quotes the original didn't have (Haiku did this once).
    for open_q, close_q in (('"', '"'), ("“", "”"), ("'", "'")):
        if (
            len(result) >= 2
            and result.startswith(open_q)
            and result.endswith(close_q)
            and not original.strip().startswith(open_q)
        ):
            result = result[1:-1]
            break
    # Leading/trailing whitespace of the original is never selected, so the
    # result shouldn't carry its own either.
    return result.strip()


def gpt_query(
    prompt: GPTMessageItem,
    text_to_process: Optional[GPTMessageItem],
    model: str,
    thread: str,
    destination: str = "",
):
    """Send a prompt to the GPT API and return the response"""

    # Reset state before pasting
    GPTState.last_was_pasted = False

    response = send_request(prompt, text_to_process, model, thread, destination)
    GPTState.last_response = extract_message(response)
    return response


@mod.action_class
class UserActions:
    def gpt_generate_shell(text_to_process: str, model: str, thread: str) -> str:
        """Generate a shell command from a spoken instruction"""
        shell_name = settings.get("user.model_shell_default")
        if shell_name is None:
            raise Exception("GPT Error: Shell name is not set. Set it in the settings.")

        prompt = f"""
        Generate a {shell_name} shell command that will perform the given task.
        Only include the code. Do not include any comments, backticks, or natural language explanations. Do not output the shell name, only the code that is valid {shell_name}.
        Condense the code into a single line such that it can be ran in the terminal.
        """

        result = gpt_query(
            format_message(prompt), format_message(text_to_process), model, thread
        )
        return extract_message(result)

    def gpt_generate_sql(text_to_process: str, model: str, thread: str) -> str:
        """Generate a SQL query from a spoken instruction"""

        prompt = """
       Generate SQL to complete a given request.
       Output only the SQL in one line without newlines.
       Do not output comments, backticks, or natural language explanations.
       Prioritize SQL queries that are database agnostic.
        """
        return gpt_query(
            format_message(prompt), format_message(text_to_process), model, thread
        ).get("text", "")

    def gpt_start_debug():
        """Enable debug logging"""
        GPTState.start_debug()

    def gpt_stop_debug():
        """Disable debug logging"""
        GPTState.stop_debug()

    def gpt_clear_context():
        """Reset the stored context"""
        GPTState.clear_context()

    def gpt_push_context(context: str | list[str]):
        """Add the selected text to the stored context"""
        if isinstance(context, list):
            context = "\n".join(context)
        GPTState.push_context(format_message(context))

    def gpt_additional_user_context() -> list[str]:
        """This is an override function that can be used to add additional context to the prompt"""
        return []

    def gpt_select_last() -> None:
        """select all the text in the last GPT output"""
        if not GPTState.last_was_pasted:
            notify("Tried to select GPT output, but it was not pasted in an editor")
            return

        # Local change (2026-11-20): select by character count and verify,
        # instead of upstream's lines-up + characters (broken by word wrap).
        if not _select_before_cursor(GPTState.last_response):
            notify("Couldn't select the last response - is the cursor still at its end?")

    def gpt_apply_prompt(
        prompt: str,
        model: str,
        thread: str,
        source: str = "",
        destination: str = "",
    ):
        """Apply an arbitrary prompt to arbitrary text"""

        # Local claude-backend changes (2026-11-20, personal/ai_plan.md
        # T1.5/T1.6): the request runs in a background thread (see _job
        # above), with guards before sending and again before pasting.
        if _job["running"] and time.monotonic() - _job["started"] < _JOB_STALE_S:
            notify("Jarvis is still working on the last request (say 'jarvis cancel' to drop it)")
            return
        start_window_id = _active_window_id()
        continuing = thread == "continueLast"

        try:
            text_to_process: GPTMessageItem = actions.user.gpt_get_source_text(source)
        except Exception as e:
            # gpt_get_source_text already told the user what was wrong.
            print(f"GPT: no source text: {e}")
            return
        if not text_to_process.get("text") and not text_to_process.get("image_url"):
            text_to_process = None  # type: ignore

        # A selection-based prompt with nothing selected would send the bare
        # prompt and paste whatever comes back at the cursor. If a result was
        # just pasted here, apply the prompt to it instead ("jarvis make it
        # friendlier" right after "jarvis make it professional" works without
        # "and" - on 2026-11-20 the "and" was missed twice). Otherwise refuse.
        # Free-form "please"/"ask" prompts and "jarvis and ..." follow-ups
        # legitimately run without a selection.
        if (
            text_to_process is None
            and source in ("", "this")
            and not continuing
            and not prompt.startswith(("ask", "please"))
        ):
            if _recent_paste_here(start_window_id) and _select_before_cursor(
                GPTState.last_response
            ):
                text_to_process = format_message(GPTState.last_response)
            else:
                notify("Nothing selected - select some text first (nothing was sent)")
                return

        # Text that must still be selected when the result arrives, so the
        # result replaces exactly what was sent.
        replace_text = ""
        if text_to_process is not None and source in ("", "this", "lastTalonDictation"):
            replace_text = text_to_process.get("text", "")

        # Follow-up with nothing selected ("jarvis and make it shorter"):
        # select the previous pasted response so the new one replaces it
        # instead of being pasted next to it.
        if (
            continuing
            and text_to_process is None
            and _recent_paste_here(start_window_id)
            and destination in ("", "paste")
        ):
            if not _select_before_cursor(GPTState.last_response):
                notify(
                    "Couldn't select the previous result - select it, then say 'jarvis and ...' again"
                )
                return
            replace_text = GPTState.last_response

        # Handle special cases in the prompt
        ### Ask is a special case, where the text to process is the prompted question, not selected text
        if prompt.startswith("ask"):
            text_to_process = format_message(prompt.removeprefix("ask"))
            prompt = "Generate text that satisfies the question or request given in the input."

        _job["id"] += 1
        job_id = _job["id"]
        _job.update(running=True, started=time.monotonic(), working_shown=False)
        model_label = settings.get("user.model_default") if model == "model" else model
        started = time.perf_counter()

        # Silent on success (Pete, 2026-11-20: a message on every request, and
        # then even only on 2 s requests, was too much). "working" appears
        # only if a request is unusually slow, and there is no "done".
        def show_working():
            if _job["id"] == job_id and _job["running"]:
                notify(f"Jarvis still working ({model_label})... say 'jarvis cancel' to drop it")

        cron.after("5s", show_working)

        def finish(response: Optional[GPTMessageItem], error: Optional[Exception]):
            if _job["id"] != job_id:
                return  # cancelled; a newer request owns the state now
            _job["running"] = False
            if error is not None:
                print(f"GPT: request failed: {error!r}")  # the backend already notified
                return
            text = extract_message(response)  # type: ignore
            if replace_text:
                text = _clean_result(text, replace_text)
            GPTState.last_response = text
            insert_to = destination
            if insert_to in _PASTE_DESTINATIONS:
                if _active_window_id() != start_window_id:
                    notify("Window changed while waiting - result copied to clipboard instead")
                    insert_to = "clipboard"
                elif replace_text and not _selection_matches(replace_text):
                    notify("Selection changed while waiting - result copied to clipboard instead")
                    insert_to = "clipboard"
            actions.user.gpt_insert_response(format_message(text), insert_to)
            if insert_to in _PASTE_DESTINATIONS:
                _job.update(paste_window=start_window_id, paste_time=time.monotonic())
            if source == "lastTalonDictation" and insert_to != "clipboard":
                # So "jarvis fix last" again targets the corrected text.
                actions.user.add_phrase_to_history(text)

        def worker():
            try:
                result = gpt_query(
                    format_message(prompt), text_to_process, model, thread, destination
                )
                cron.after("0ms", lambda: finish(result, None))
            except Exception as e:
                # Bind now: Python unbinds `e` when the except block ends.
                cron.after("0ms", lambda err=e: finish(None, err))

        threading.Thread(target=worker, name="jarvis-request", daemon=True).start()

    def gpt_cancel():
        """Ignore the result of the request still running (local addition, 2026-11-20)"""
        if not _job["running"]:
            notify("Jarvis: nothing to cancel")
            return
        _job["id"] += 1
        _job["running"] = False
        notify("Jarvis: cancelled - the result will be ignored")

    def gpt_apply_prompt_for_cursorless(
        prompt: str,
        model: str,
        thread: str,
        source: list[str],
    ) -> str:
        """Apply a prompt to text from Cursorless and return a string result.
        This function is specifically designed for Cursorless integration
        and does not trigger insertion actions."""

        # Join the list into a single string
        source_text = "\n".join(source)
        text_to_process = format_message(source_text)

        # Send the request but don't insert the response (Cursorless will handle insertion)
        response = gpt_query(format_message(prompt), text_to_process, model, thread, "")

        # Return just the text string
        return extract_message(response)

    def gpt_pass(source: str = "", destination: str = "") -> None:
        """Passes a response from source to destination"""
        actions.user.gpt_insert_response(
            actions.user.gpt_get_source_text(source), destination
        )

    def gpt_help() -> None:
        """Open the GPT help file in the web browser"""
        # get the text from the file and open it in the web browser
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, "lists", "staticPrompt.talon-list")
        with open(file_path, "r") as f:
            lines = f.readlines()[2:]

        builder = Builder()
        builder.h1("Talon GPT Prompt List")
        for line in lines:
            if "##" in line:
                builder.h2(line)
            else:
                builder.p(line)

        builder.render()

    def gpt_reformat_last(how_to_reformat: str, model: str, thread: str) -> str:
        """Reformat the last model output"""
        PROMPT = f"""The last phrase was written using voice dictation. It has an error with spelling, grammar, or just general misrecognition due to a lack of context. Please reformat the following text to correct the error with the context that it was {how_to_reformat}."""
        last_output = actions.user.get_last_phrase()
        if last_output:
            # Local change (2026-11-20): select and verify instead of one
            # Backspace per character; the caller's paste replaces it.
            if not _select_before_cursor(last_output):
                notify("Couldn't find your last dictation just before the cursor - nothing was sent")
                raise Exception("GPT: last dictation is not just before the cursor")
            return extract_message(
                gpt_query(
                    format_message(PROMPT), format_message(last_output), model, thread
                )
            )
        else:
            notify("No text to reformat")
            raise Exception("No text to reformat")

    def gpt_insert_response(
        gpt_message: GPTMessageItem,
        method: str = "",
        cursorless_destination: Any = None,
    ) -> None:
        """Insert a GPT result in a specified way"""
        # Use a custom default if nothing is provided and the user has set
        # a different default destination
        if method == "":
            method = settings.get("user.model_default_destination")

        if gpt_message.get("type") != "text":
            actions.app.notify(
                f"Tried to insert an image to {method}, but that is not currently supported. To insert an image to this destination use a prompt to convert it to text."
            )
            return

        message_text_no_images = extract_message(gpt_message)
        match method:
            case "above":
                actions.key("left")
                actions.edit.line_insert_up()
                GPTState.last_was_pasted = True
                actions.user.paste(message_text_no_images)
            case "below":
                actions.key("right")
                actions.edit.line_insert_down()
                GPTState.last_was_pasted = True
                actions.user.paste(message_text_no_images)
            case "clipboard":
                clip.set_text(message_text_no_images)
            case "snip":
                actions.user.insert_snippet(message_text_no_images)
            case "context":
                GPTState.push_context(gpt_message)
            case "newContext":
                GPTState.clear_context()
                GPTState.push_context(gpt_message)
            case "appendClipboard":
                if clip.text() is not None:
                    clip.set_text(clip.text() + "\n" + message_text_no_images)  # type: ignore Unclear why this is throwing a type error in pylance
                else:
                    clip.set_text(message_text_no_images)
            case "browser":
                builder = Builder()
                builder.h1("Talon GPT Result")
                for line in message_text_no_images.split("\n"):
                    builder.p(line)
                builder.render()
            case "textToSpeech":
                try:
                    actions.user.tts(message_text_no_images)
                except KeyError:
                    notify("GPT Failure: text to speech is not installed")

            # Although we can insert to a cursorless destination, the cursorless_target capture
            # Greatly increases DFA compliation times and should be avoided if possible
            case "cursorless":
                actions.user.cursorless_insert(
                    cursorless_destination, message_text_no_images
                )
            # Don't add to the window twice if the thread is enabled
            case "window":
                # If there was prior text in the confirmation GUI and the user
                # explicitly passed new text to the gui, clear the old result
                GPTState.text_to_confirm = message_text_no_images
                actions.user.confirmation_gui_append(message_text_no_images)
            case "chain":
                GPTState.last_was_pasted = True
                actions.user.paste(message_text_no_images)
                actions.user.gpt_select_last()

            case "paste":
                GPTState.last_was_pasted = True
                actions.user.paste(message_text_no_images)
            # If the user doesn't specify a method assume they want to paste.
            # However if they didn't specify a method when the confirmation gui
            # is showing, assume they don't want anything to be inserted
            case _ if not confirmation_gui.showing:
                GPTState.last_was_pasted = True
                actions.user.paste(message_text_no_images)
            # Don't do anything if none of the previous conditions were valid
            case _:
                pass

    def gpt_get_source_text(spoken_text: str) -> GPTMessageItem:
        """Get the source text that is will have the prompt applied to it"""
        match spoken_text:
            case "clipboard":
                return format_clipboard()
            case "context":
                if GPTState.context == []:
                    notify("GPT Failure: Context is empty")
                    raise Exception(
                        "GPT Failure: User applied a prompt to the phrase context, but there was no context stored"
                    )
                return format_message(messages_to_string(GPTState.context))
            case "gptResponse":
                if GPTState.last_response == "":
                    raise Exception(
                        "GPT Failure: User applied a prompt to the phrase GPT response, but there was no GPT response stored"
                    )
                return format_message(GPTState.last_response)

            case "lastTalonDictation":
                last_output = actions.user.get_last_phrase()
                if last_output:
                    # Local change (2026-11-20): select the phrase and verify
                    # it, instead of deleting it with one Backspace per
                    # character (Notepad dropped some, leaving "I talk"
                    # behind). The result then replaces the selection.
                    if not _select_before_cursor(last_output):
                        notify(
                            "Couldn't find your last dictation just before the cursor - nothing was sent"
                        )
                        raise Exception("GPT: last dictation is not just before the cursor")
                    return format_message(last_output)
                else:
                    notify("GPT Failure: No last dictation to reformat")
                    raise Exception(
                        "GPT Failure: User applied a prompt to the phrase last Talon Dictation, but there was no text to reformat"
                    )
            case "this" | _:
                return format_message(actions.edit.selected_text())
