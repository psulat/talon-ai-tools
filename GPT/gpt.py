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


# --- Local claude-backend additions (2026-11-20, personal/docs/ai_plan.md) ---
# gpt_apply_prompt runs its request in a background thread so Talon keeps
# listening (it used to block speech for the whole request, ~1-2 s, tripping
# Talon's 2 s watchdog). Only one request at a time; "jarvis cancel" bumps the
# id so a late result is ignored.
_job: dict[str, Any] = {"id": 0, "running": False, "started": 0.0}
# Subject-line flow (local, 2026-11-20, Pete's request): true while the last
# pasted result is a list of subject-line options, so picking one ("jarvis
# please keep only the second one") sends it to the email's Subject box.
# Kept on GPTState (lib/modelState.py), not in this module, so they survive
# Talon reloading gpt.py: every code edit reloaded it and wiped "what was
# just pasted where", and the next "jarvis please keep only the second one"
# found nothing to work on (2026-11-20).
if not hasattr(GPTState, "subject_flow"):
    GPTState.subject_flow = {"options_pasted": False}  # type: ignore[attr-defined]
if not hasattr(GPTState, "paste_info"):
    GPTState.paste_info = {"window": None, "time": 0.0}  # type: ignore[attr-defined]
_subject_flow: dict[str, bool] = GPTState.subject_flow  # type: ignore[attr-defined]
_paste_info: dict = GPTState.paste_info  # type: ignore[attr-defined]


_PASTE_RESTORE_DELAY = "600ms"


def _paste(text: str) -> None:
    """Paste `text`, put the old clipboard back, and check the text landed (local, 2026-11-20).

    Like community user.paste, but waits 600 ms instead of 150 ms before
    restoring the clipboard, so a slow app can't paste the old contents
    instead. On 2026-11-20 a stale clipboard entry (an earlier jarvis result)
    turned up in a Gmail draft, and no paste command had been spoken. Then
    reads the text before the cursor via UI Automation and warns if it
    doesn't end with what was pasted.
    """
    with clip.revert():
        clip.set_text(text)
        actions.edit.paste()
        actions.sleep(_PASTE_RESTORE_DELAY)
    cron.after("200ms", lambda: _check_pasted(text))


def _squash(text: str) -> str:
    """Folded text with all whitespace runs collapsed (apps reformat line breaks on paste)."""
    return " ".join(_fold(text).split())


def _check_pasted(text: str) -> None:
    before = _text_before_cursor(len(text) + 60)
    if not before:
        return  # no UI Automation text in this app: nothing to check against
    expected = _squash(text)[-80:]
    if expected and not _squash(before).endswith(expected):
        print(f"GPT: paste check failed; expected end {expected!r}, found {_squash(before)[-80:]!r}")
        notify("The paste didn't come out as expected - check the text (Ctrl+Z undoes it)")


def _put_in_email_subject(text: str, remove_selection: bool) -> str:
    """Put a subject line into the active email compose window.

    Uses user.email_put_subject from Pete's personal repo (ai_actions.py),
    which writes Chrome's Subject box via UI Automation and, with
    remove_selection, first deletes the selected text from the body.
    Returns "set", "failed" (a Subject box exists but didn't take it - never
    paste then: writing it can move the cursor into the Subject, which is how
    the line got pasted there twice on 2026-11-20), or "no_box" (no compose
    window, or the action doesn't exist on this machine) so callers paste.
    """
    try:
        return str(actions.user.email_put_subject(text, remove_selection))
    except Exception as e:
        print(f"GPT: no email subject action: {e!r}")
        return "no_box"
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


def _uia_selection_text() -> Optional[str]:
    """The selected text via UI Automation, "" if nothing is selected, or None if the
    focused control can't report its selection (then the caller copies instead).

    Preferred over copying (2026-11-20): reading the selection with Ctrl+C once
    typed a bare "c" over a selected paragraph in Word - Ctrl apparently didn't
    register, the same symptom as October's "copy that" typing "c" - and copying
    costs a clipboard round trip of up to 250 ms. Word reports an empty selection
    as its paragraph mark, which folds to "".
    """
    pattern = _text_pattern()
    if pattern is None:
        return None
    try:
        ranges = pattern.selection
        if not ranges:
            return None if pattern.supported_text_selection == "None" else ""
        text = ranges[0].text
        return text if _fold(text).strip() else ""
    except Exception:
        return None


def _read_selection() -> str:
    """Selected text: UI Automation when available, otherwise copy (community edit.selected_text)."""
    text = _uia_selection_text()
    return actions.edit.selected_text() if text is None else text


def _selection_matches(expected: str) -> bool:
    """True if the current selection is `expected`, compared tolerantly (see _FOLD)."""
    return _fold(_read_selection()).strip() == _fold(expected).strip()


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
        # Offsets, not TextRange.find_text: in Gmail, Chrome's find_text with
        # a multi-line string returned a range starting at line 2 (verified
        # 2026-11-20), while these offsets select the list exactly.
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
        if _wait_for_selection(text):
            return True
        actions.key("right")  # collapse to the selection's end, where the cursor was
    return False


def _wait_for_selection(text: str, timeout_s: float = 0.8) -> bool:
    """True once the selection is `text`, re-checking for up to `timeout_s`.

    Chrome applies a new selection a moment later: in Gmail, an immediate
    read-back still reported the old (empty) selection, so a correct
    selection was judged wrong and "couldn't select the previous result"
    (2026-11-20). Notepad and Word apply it at once. Where there's no UI
    Automation text (the check would copy with Ctrl+C), check once after a
    short pause instead of polling.
    """
    if _uia_selection_text() is None:
        actions.sleep("150ms")
        return _selection_matches(text)
    deadline = time.monotonic() + timeout_s
    while True:
        if _selection_matches(text):
            return True
        if time.monotonic() >= deadline:
            return False
        actions.sleep("80ms")


def _select_before_cursor(text: str) -> bool:
    """Select `text`, which should end at the cursor, and verify the selection."""
    if not _normalize(text).strip():
        return False
    if _uia_select_before_cursor(text):
        if _wait_for_selection(text):
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
    try:
        element = ui.focused_element()
        focused = f"{element.control_type} {element.name[:30]!r}"
    except Exception:
        focused = "?"
    selected = _uia_selection_text()
    print(
        f"GPT: couldn't select {len(text)} chars ending {text[-60:]!r}; "
        f"text before the cursor: {before!r}; focused: {focused}; "
        f"selection now: {(selected or '')[:80]!r}"
    )


def _recent_paste_here(window_id) -> bool:
    """True if the last response was pasted in this window within the last 10 minutes."""
    return (
        GPTState.last_was_pasted
        and bool(GPTState.last_response)
        and _paste_info["window"] == window_id
        and time.monotonic() - _paste_info["time"] < 600
    )


def _text_before_cursor(max_chars: int) -> str:
    """Up to `max_chars` of text before the cursor, via UI Automation ("" if unavailable)."""
    pattern = _text_pattern()
    if pattern is None:
        return ""
    try:
        ranges = pattern.selection
        if not ranges:
            return ""
        head = ranges[0].clone()
        head.move_endpoint_by_range("End", "Start", target=ranges[0])
        head.move_endpoint_by_unit("Start", "Character", -max_chars)
        return head.text
    except Exception as e:
        print(f"GPT: couldn't read text before the cursor: {e!r}")
        return ""


# "jarvis ask ..." (local, 2026-11-20). Upstream's "Generate text that
# satisfies the question" without context gave five generic subject lines with
# [Topic] placeholders; Haiku also prefixed a list with a heading.
_ASK_PROMPT = (
    "Answer the question or fulfil the request below, using the context if given. "
    "If it asks for one thing (a title, a subject line, a sentence), give the single best one "
    "unless it explicitly asks for several; if several, put one per line, numbered. "
    "Return only the answer itself: no heading, no introduction, no quotes."
)


def _edit_prompt(instruction: str, previous_response: bool = False) -> str:
    """Prompt for "jarvis please <instruction>" (local, 2026-11-20).

    The result replaces the selection, so ask for the text exactly as it
    should now read. Two failures shaped the wording: "turn the second half
    into a bulleted list" returned only the bullets (both paragraphs got
    replaced); after "return the complete text" was added, "keep only the
    second one" kept the list's heading and the "2." numbering.
    """
    target = "your previous response" if previous_response else "the text below"
    return (
        f"Edit {target} according to this instruction: {instruction}. "
        "Return the text exactly as it should now read in the document, and nothing else "
        "(no commentary, no quotes around it). Keep everything the instruction doesn't touch. "
        "If the instruction picks or keeps only some of the text, return just that, "
        "without list numbers, bullets or headings left over."
    )


def _drop_lead_in(answer: str) -> str:
    """Drop a heading line like "Three subject lines for this email:" above a list.

    Haiku added one to "ask for three subject lines" (2026-11-20) despite the
    prompt; only removed when it ends with ":" and a list follows.
    """
    lines = answer.split("\n")
    rest = [line for line in lines[1:] if line.strip()]
    if (
        len(lines) > 1
        and lines[0].rstrip().endswith(":")
        and rest
        and all(line.lstrip()[:1].isdigit() or line.lstrip()[:1] in "-*•" for line in rest)
    ):
        return "\n".join(lines[1:]).strip()
    return answer


def _clean_result(result: str, original: str) -> str:
    """Undo small model habits in a result (`original` is the text it replaces, or "")."""
    result = result.strip()
    # The triple quotes talon-ai-tools wraps around the text it sends, echoed
    # back (Haiku, 2026-11-20).
    if result.startswith('"""') and result.endswith('"""') and len(result) >= 6:
        result = result[3:-3].strip()
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

        # Local claude-backend changes (2026-11-20, personal/docs/ai_plan.md
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

        # Local split (2026-11-20): "please ..." always EDITS text, "ask ..."
        # always WRITES/ANSWERS. Upstream's "please" with nothing selected
        # sent the bare instruction, and Claude's reply ("I can't see the
        # document...") got pasted into Word.
        asking = prompt.startswith("ask")

        # Nothing selected. If a result was just pasted here, apply the prompt
        # to it ("jarvis make it friendlier" or "jarvis please keep only the
        # second one" right after a result works without "and" - on
        # 2026-11-20 the "and" was missed twice). Otherwise refuse: a prompt
        # with nothing to work on would paste a reply to a bare prompt.
        # "jarvis and ..." follow-ups are handled below.
        if (
            text_to_process is None
            and source in ("", "this")
            and not continuing
            and not asking
        ):
            if not _recent_paste_here(start_window_id):
                notify("Nothing selected - select some text first (nothing was sent)")
                return
            if not _select_before_cursor(GPTState.last_response):
                notify("Couldn't select the previous result - select the text, then try again")
                return
            text_to_process = format_message(GPTState.last_response)

        # Text that must still be selected when the result arrives, so the
        # result replaces exactly what was sent. ("ask" never replaces.)
        replace_text = ""
        if (
            not asking
            and text_to_process is not None
            and source in ("", "this", "lastTalonDictation")
        ):
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

        # Subject-line request: asking for subject line(s), or editing a list
        # of subject options just pasted. A single-line result then goes to
        # the email's Subject box instead of the body (see finish()).
        about_subject = (asking and "subject" in prompt.lower()) or (
            _subject_flow["options_pasted"]
            and bool(replace_text)
            and replace_text == GPTState.last_response
        )

        # Handle special cases in the prompt
        if asking:
            # Local change (2026-11-20): upstream sent only the question, so
            # "what is a good subject line for this email" got five generic
            # options with [Topic] placeholders. Now the selection - or, with
            # none, the text above the cursor - goes along as context, and
            # one best answer is asked for unless several are requested. With
            # a selection, the answer goes on a new line below it.
            question = prompt.removeprefix("ask").strip()
            context = (text_to_process or {}).get("text", "") or _text_before_cursor(4000)
            if text_to_process is not None and destination == "":
                destination = "below"
            text_to_process = format_message(
                f"Question or request: {question}\n\n"
                + (f"Context - the text the user is working on:\n{context}" if context.strip() else "")
            )
            prompt = _ASK_PROMPT
        elif prompt.startswith("please"):
            # Local change (2026-11-20): see _edit_prompt.
            prompt = _edit_prompt(
                prompt.removeprefix("please").strip(),
                previous_response=text_to_process is None,
            )

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
            text = _clean_result(extract_message(response), replace_text)  # type: ignore
            if asking:
                text = _drop_lead_in(text)
            GPTState.last_response = text
            insert_to = destination
            if insert_to in _PASTE_DESTINATIONS:
                if _active_window_id() != start_window_id:
                    notify("Window changed while waiting - result copied to clipboard instead")
                    insert_to = "clipboard"
                elif replace_text and not _selection_matches(replace_text):
                    notify("Selection changed while waiting - result copied to clipboard instead")
                    insert_to = "clipboard"
            single_line = len([line for line in text.splitlines() if line.strip()]) == 1
            if about_subject and single_line and insert_to in _PASTE_DESTINATIONS:
                # The option list it was picked from (still selected, checked
                # above) is removed from the body first, then the line goes
                # into the Subject box.
                status = _put_in_email_subject(text, remove_selection=bool(replace_text))
                if status != "no_box":
                    GPTState.last_was_pasted = False
                    _subject_flow["options_pasted"] = False
                    if status != "set":
                        restore = "; Ctrl+Z in the body brings the list back" if replace_text else ""
                        notify(f"Couldn't set the subject - it's on the clipboard{restore}")
                    return
            # Options to choose from stay in the body; remember that they are
            # subject lines so the pick goes to the Subject box.
            _subject_flow["options_pasted"] = (
                about_subject and not single_line and insert_to in _PASTE_DESTINATIONS
            )
            to_paste = text
            if asking and insert_to in ("", "paste"):
                # An answer typed at the end of a sentence ran into it ("...the
                # twelfth1. Migration update", Gmail, 2026-11-20): start it on
                # its own line. last_response stays without the break, so
                # follow-ups still find and select exactly the answer.
                before = _text_before_cursor(1)
                if before and not before.isspace():
                    to_paste = "\n" + text
            actions.user.gpt_insert_response(format_message(to_paste), insert_to)
            if insert_to in _PASTE_DESTINATIONS:
                _paste_info.update(window=start_window_id, time=time.monotonic())
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
                _paste(message_text_no_images)
            case "below":
                actions.key("right")
                actions.edit.line_insert_down()
                GPTState.last_was_pasted = True
                _paste(message_text_no_images)
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
                _paste(message_text_no_images)
                actions.user.gpt_select_last()

            case "paste":
                GPTState.last_was_pasted = True
                _paste(message_text_no_images)
            # If the user doesn't specify a method assume they want to paste.
            # However if they didn't specify a method when the confirmation gui
            # is showing, assume they don't want anything to be inserted
            case _ if not confirmation_gui.showing:
                GPTState.last_was_pasted = True
                _paste(message_text_no_images)
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
                # Local change (2026-11-20): UI Automation first, no Ctrl+C
                # where avoidable (see _uia_selection_text).
                return format_message(_read_selection())
