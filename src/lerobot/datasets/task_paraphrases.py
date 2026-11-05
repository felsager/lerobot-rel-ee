# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Optional task paraphrases loaded from a dataset metadata JSON template bank."""

import json
import random
import re
from pathlib import Path
from string import Formatter


def _compile_template(template: str) -> tuple[re.Pattern[str], set[str]]:
    if not isinstance(template, str) or not template.strip():
        raise ValueError("Paraphrase templates must be nonempty strings")
    parts = []
    fields: set[str] = set()
    for literal, name, format_spec, conversion in Formatter().parse(template):
        parts.append(re.escape(literal))
        if name is None:
            continue
        if not name.isidentifier() or format_spec or conversion:
            raise ValueError(f"Use plain named placeholders, such as {{object}}, in {template!r}")
        parts.append(f"(?P={name})" if name in fields else f"(?P<{name}>.+?)")
        fields.add(name)
    return re.compile("".join(parts)), fields


class TaskParaphrases:
    """Sample equivalent task strings while preserving all captured placeholder values.

    The JSON maps source templates to nonempty lists of replacement templates.
    Templates use plain named placeholders (``{object}``); replacements must contain
    exactly the source's placeholder names. Matching is exact and case-sensitive.
    The first matching source wins. Unmatched tasks pass through unchanged.
    """

    def __init__(self, path: str | Path) -> None:
        with Path(path).open() as file:
            bank = json.load(file)
        if not isinstance(bank, dict) or not bank:
            raise ValueError("Paraphrase bank must be a nonempty JSON object mapping templates to lists")
        self._rules: list[tuple[re.Pattern[str], tuple[str, ...]]] = []
        self._cache: dict[str, tuple[str, ...]] = {}
        for source, replacements in bank.items():
            pattern, fields = _compile_template(source)
            if not isinstance(replacements, list) or not replacements:
                raise ValueError(f"Paraphrases for {source!r} must be a nonempty list")
            for replacement in replacements:
                _, replacement_fields = _compile_template(replacement)
                if replacement_fields != fields:
                    raise ValueError(
                        f"Paraphrase {replacement!r} must preserve placeholders {sorted(fields)}"
                    )
            self._rules.append((pattern, tuple(dict.fromkeys(replacements))))

    def __call__(self, task: str) -> str:
        """Sample on each access; cache only rendered candidates, never the chosen wording.

        Python's random generator follows the training/DataLoader worker seed.
        """
        if task not in self._cache:
            for pattern, replacements in self._rules:
                match = pattern.fullmatch(task)
                if match is not None:
                    self._cache[task] = tuple(text.format_map(match.groupdict()) for text in replacements)
                    break
            else:
                self._cache[task] = ()
        variants = self._cache[task]
        return random.choice(variants) if variants else task
