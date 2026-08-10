import re


_LESSON_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


def is_valid_lesson_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and _LESSON_ID.fullmatch(value) is not None
    )
