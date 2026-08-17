"""Completion-only label masking.

Kept in its own module, free of unsloth/torch imports, so the masking rule can
be unit tested offline. `arc_solver` calls it from the data collator.
"""

import numpy as np

IGNORE_INDEX = -100


def completion_labels(ids, im_start_id, eos_id, newline_id,
                      ignore_index=IGNORE_INDEX):
    """Supervise only the assistant turns of a chat-formatted sequence.

    Turns alternate user, assistant, user, assistant, ... Each turn looks like

        <|im_start|> [role] \\n  <content...>  <|im_end|>

    and we train on the content plus the closing <|im_end|>, so the model also
    learns where to stop. The role word is optional: on some checkpoints the
    tokenizer drops it, so we locate content by scanning to the newline that
    terminates the header rather than by assuming a fixed offset.

    Returns an int64 array the same length as `ids`.
    """
    ids = np.asarray(ids)
    labels = np.full(ids.shape, ignore_index, dtype=np.int64)

    starts = np.where(ids == im_start_id)[0].tolist()
    ends = np.where(ids == eos_id)[0].tolist()

    for turn, (start, end) in enumerate(zip(starts, ends)):
        if turn % 2 != 1:           # even turns are prompts
            continue
        nl = start + 1
        while nl < end and int(ids[nl]) != newline_id:
            nl += 1
        content = nl + 1
        stop = end + 1              # inclusive of <|im_end|>
        if content < stop:
            labels[content:stop] = ids[content:stop]

    return labels
