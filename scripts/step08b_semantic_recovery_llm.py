#!/usr/bin/env python3

"""
STEP 08B
========

Targeted LLM Semantic Recovery

Purpose
-------
Resolve semantic information that could not be determined
deterministically from scientific tables.

The LLM may resolve:
- sample/feedstock identity;
- temperature;
- temperature type;
- whether the row represents the paper's own experiment or
  literature/reference data.

IMPORTANT:
The LLM is NOT allowed to modify, reinterpret, or regenerate
C/H/N/O numerical values.

Default behaviour:
- process only 3 recovery packages;
- save raw responses;
- validate strict JSON;
- keep failures separately.

Set RUN_ALL=1 later to process the full recovery queue.
"""

from pathlib import Path

import json
import os
import re
import time
import urllib.error
import urllib.request


# ============================================================
# Paths
# ============================================================

project_dir = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

processed_tables_dir = (
    project_dir
    / "processed_tables"
)

output_dir = (
    processed_tables_dir
    / "semantic_recovery_llm"
)

raw_dir = (
    output_dir
    / "raw_responses"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

raw_dir.mkdir(
    parents=True,
	    exist_ok=True,
)

input_path = (
    processed_tables_dir
    / "semantic_recovery_packages_enriched.jsonl"
)

results_path = (
    output_dir
    / "semantic_recovery_results.jsonl"
)

failures_path = (
    output_dir
    / "semantic_recovery_failures.jsonl"
)

summary_path = (
    output_dir
    / "semantic_recovery_run_summary.json"
)


# ============================================================
# Runtime configuration
# ============================================================

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "qwen2.5:7b",
)

OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
)

RUN_ALL = (
    os.environ.get(
        "RUN_ALL",
        "0",
    )
    .strip()
    == "1"
)

TEST_LIMIT = int(
    os.environ.get(
        "TEST_LIMIT",
        "3",
    )
)

RESUME = (
    os.environ.get(
        "RESUME",
        "1",
    )
    .strip()
    == "1"
)

TARGET_IDS_RAW = os.environ.get(
    "TARGET_IDS",
    "",
).strip()

TARGET_IDS = {
    value.strip()
    for value in TARGET_IDS_RAW.split(",")
    if value.strip()
}

REQUEST_TIMEOUT = float(
    os.environ.get(
        "REQUEST_TIMEOUT",
        "300",
    )
)


# ============================================================
# Helpers
# ============================================================

def load_jsonl(path):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as source:

        for line_number, line in enumerate(
            source,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}, "
                    f"line {line_number}: {error}"
                ) from error

    return records


def append_jsonl(
    path,
    record,
):
    with path.open(
        "a",
        encoding="utf-8",
    ) as output_file:

        output_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def safe_filename(value):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )


def strip_code_fences(text):
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return text.strip()


def extract_json_object(text):
    cleaned = strip_code_fences(
        text
    )

    try:
        return json.loads(
            cleaned
        )

    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")

    if start == -1:
        raise ValueError(
            "No JSON object found."
        )

    depth = 0
    in_string = False
    escaped = False

    for position in range(
        start,
        len(cleaned),
    ):
        character = cleaned[
            position
        ]

        if in_string:

            if escaped:
                escaped = False

            elif character == "\\":
                escaped = True

            elif character == '"':
                in_string = False

        else:

            if character == '"':
                in_string = True

            elif character == "{":
                depth += 1

            elif character == "}":
                depth -= 1

                if depth == 0:
                    return json.loads(
                        cleaned[
                            start:
                            position + 1
                        ]
                    )

    raise ValueError(
        "JSON response appears truncated."
    )


# ============================================================
# Prompt construction
# ============================================================

SYSTEM_PROMPT = """
You are resolving ambiguous metadata from a scientific table.

Your task is ONLY to interpret:
1. the sample/feedstock identity;
2. the processing temperature;
3. the temperature type;
4. the provenance of the row.

IMPORTANT TARGET-ROW RULES:

1. Resolve ONLY the target row supplied in this recovery package.
2. Never copy a sample name, temperature, or numerical value from another row.
3. If temperature_exact_C is already provided, preserve that exact temperature.
4. Use surrounding rows only to understand abbreviations or context.
5. resolved_temperature_C must be ONE numeric value or null.
   Never return a list, range, string, or object.
6. resolved_sample must be one string or null.
   Never return an object.
7. Copy recovery_id EXACTLY as supplied.
8. Map slow pyrolysis and fast pyrolysis to "pyrolysis".
9. If the target row contains a temperature directly, prefer that value
   over temperatures mentioned elsewhere in the table.
10. If a sample code contains a clear temperature suffix such as PHC500
    or PC400, resolve the suffix as 500 or 400 only when the supplied
    paper context confirms that the suffix represents treatment temperature.

CRITICAL RULES:

- Do NOT calculate, modify, replace, infer, correct, or regenerate
  C, H, N, or O numerical values.

- The numerical elemental values are controlled by deterministic
  code and are outside your authority.

- Use only the supplied table, target row, header, caption,
  and nearby context.

- If something cannot be established from the evidence, return null.

- Do not guess.

- A sample code such as CF-200 may encode both sample identity
  and processing temperature.

- You MAY resolve the numeric suffix as temperature when the
  supplied table or targeted paper context establishes that the
  sample series corresponds to processing or pyrolysis
  temperatures.

- Example reasoning:
  if CF/PW are discussed as biochars produced at different
  pyrolysis temperatures, the table contains CF-200, CF-225,
  CF-250, etc., and the surrounding paper context discusses those
  samples as a function of pyrolysis temperature, then CF-200 may
  be resolved as 200 C.

- Never perform this inference from the naming pattern alone.

- Distinguish the paper's own experimental data from values copied
  from another publication or literature-summary table.

- Before resolving metadata, decide whether the target row is truly a
  material-property record relevant to the extraction task.

- Reject equations, regression tables, statistical coefficients,
  combustion-stage temperatures, DTG peak temperatures, and unrelated
  rows rather than forcing them into the extraction schema.

- Processing temperature means the temperature used to produce or
  heat-treat the material, not a later measurement or combustion
temperature.

Return ONLY one valid JSON object.
"""


def build_user_prompt(package):
    expected_schema = {
        "recovery_id": (
            package[
                "recovery_id"
            ]
        ),
        "row_decision": "resolve",
        "resolved_sample": None,
        "resolved_temperature_C": None,
        "resolved_temperature_type": None,
        "provenance": None,
        "confidence": (
            "low"
        ),
        "evidence_summary": "",
    }
    targeted_context = package.get(
        "targeted_paper_context",
        [],
    )

    if not isinstance(
        targeted_context,
        list,
    ):
        targeted_context = []

    targeted_context_text = "\n\n".join(
        f"[Targeted context {index}]\n{context}"
        for index, context in enumerate(
            targeted_context,
            start=1,
        )
    )

    if not targeted_context_text:
        targeted_context_text = (
            "No additional targeted paper context was retrieved."
        )
    return f"""
RECOVERY ID
-----------
{package['recovery_id']}

PAPER
-----
{package['paper_id']}

CURRENT PROBLEM
---------------
{package['semantic_class']}

TARGET SOURCE ROW
-----------------
{package['raw_source_row']}

REPAIRED TABLE HEADER
---------------------
{json.dumps(
    package['header_row'],
    ensure_ascii=False
)}

CURRENT DETERMINISTIC VALUES
----------------------------
Sample:
{package.get('sample_raw')}

Temperature:
{package.get('temperature_original')}

C:
{package.get('C_value')}

H:
{package.get('H_value')}

N:
{package.get('N_value')}

O:
{package.get('O_value')}

Remember:
C/H/N/O are shown only for context.
DO NOT return corrected or alternative elemental values.

TABLE CAPTION
-------------
{package.get('caption', '')}

TEXT BEFORE TABLE
-----------------
{package.get('preceding_context', '')}

TEXT AFTER TABLE
----------------
{package.get('following_context', '')}

TARGETED CONTEXT RETRIEVED FROM THIS PAPER
------------------------------------------
{targeted_context_text}

IMPORTANT INTERPRETATION RULE
-----------------------------
The targeted context may define sample abbreviations, sample-code
suffixes, processing temperatures, and preparation methods.

For example, if a code such as CF-200 appears and the supplied
paper evidence establishes that CF refers to a feedstock and that
the numbered sample series represents pyrolysis temperatures, you
may resolve the temperature as 200 C.

Do this ONLY when the supplied evidence supports the relationship.
Do not infer a number from a sample code merely because it looks
like a temperature.

COMPLETE SOURCE TABLE
---------------------
{package.get('raw_table_text', '')}

ROW DECISION
------------
First decide whether this row is actually a valid material-property
record for the extraction task.

Use exactly one:

"resolve"
    This row represents a material/sample whose metadata can
    legitimately be recovered.

"reject_non_material_row"
    This is an equation, model coefficient, statistical result,
    combustion-analysis row, unrelated property row, separator,
    or another row that should NOT become a feedstock-temperature-
    CHNO record.

"manual_review"
    It may be relevant, but the supplied evidence is too ambiguous
    for a safe automatic decision.

IMPORTANT:
A temperature appearing in the row is NOT automatically the material
processing temperature.

Combustion temperature ranges, DTG peak temperatures, measurement
temperatures, reaction temperatures, and equation parameters must
not be reported as pyrolysis/carbonization temperature.

PROVENANCE LABELS
-----------------
Use exactly one of:

"paper_experiment"
    The row reports experiments/materials produced or measured
    directly in this paper.

"literature_summary"
    The row is copied, summarized, or compiled from another
    publication/reference.

"unclear"
    The supplied evidence cannot establish provenance.

TEMPERATURE TYPE
----------------
Use one of:

"pyrolysis"
"carbonization"
"carbonisation"
"hydrothermal"
"gasification"
"heat_treatment"
"other"
null

CONFIDENCE
----------
Use exactly:

"high"
"medium"
"low"

OUTPUT SCHEMA
-------------
Return ONLY JSON with exactly these fields:

{json.dumps(
    expected_schema,
    indent=2,
    ensure_ascii=False
)}
""".strip()


# ============================================================
# Ollama request
# ============================================================

def call_ollama(
    system_prompt,
    user_prompt,
):
    url = (
        OLLAMA_BASE_URL.rstrip("/")
        + "/api/chat"
    )

    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": (
                    system_prompt
                ),
            },
            {
                "role": "user",
                "content": (
                    user_prompt
                ),
            },
        ],
        "options": {
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": 512,
        },
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(
            payload
        ).encode(
            "utf-8"
        ),
        headers={
            "Content-Type": (
                "application/json"
            )
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT,
        ) as response:

            response_data = json.loads(
                response.read().decode(
                    "utf-8"
                )
            )

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Ollama request failed: "
            f"{error}"
        ) from error

    message = response_data.get(
        "message",
        {},
    )

    content = message.get(
        "content",
        "",
    )

    if not content:
        raise ValueError(
            "Ollama returned empty content."
        )

    return (
        content,
        response_data,
    )


# ============================================================
# Validation
# ============================================================

ALLOWED_PROVENANCE = {
    "paper_experiment",
    "literature_summary",
    "unclear",
}

ALLOWED_CONFIDENCE = {
    "high",
    "medium",
    "low",
}

ALLOWED_TEMPERATURE_TYPES = {
    "pyrolysis",
    "carbonization",
    "carbonisation",
    "hydrothermal",
    "gasification",
    "heat_treatment",
    "other",
    None,
}

ALLOWED_ROW_DECISIONS = {
    "resolve",
    "reject_non_material_row",
    "manual_review",
}

def validate_result(
    result,
    package,
):
    # Work on a copy so the raw parsed response
    # remains unchanged.
    result = dict(result)

    # --------------------------------------------------------
    # Deterministic recovery ID
    # --------------------------------------------------------
    # The recovery ID comes from the package, not the LLM.
    result["recovery_id"] = str(
        package["recovery_id"]
    )
    # --------------------------------------------------------
    # Lock sample identity when the target row itself
    # explicitly provides a textual sample code/name.
    # --------------------------------------------------------
    raw_source_row = str(
        package.get(
            "raw_source_row",
            "",
        )
    ).strip()

    first_cell = (
        raw_source_row
        .split("|", 1)[0]
        .strip()
        .replace("**", "")
        .strip()
    )

    semantic_class = package.get(
        "semantic_class"
    )

    # Only lock an explicit textual first cell.
    # Pure numbers such as "500" or "750" may actually
    # represent temperature and must not become samples.
    if (
        semantic_class
        in {
            "NEEDS_SAMPLE_INTERPRETATION",
            "SEMANTIC_RECOVERY_REQUIRED",
        }
        and first_cell
        and re.search(
            r"[A-Za-z]",
            first_cell,
        )
    ):
        result[
            "resolved_sample"
        ] = first_cell
    # --------------------------------------------------------
    # Lock deterministic exact temperature
    # --------------------------------------------------------
    exact_temperature = package.get(
        "temperature_exact_C"
    )

    if exact_temperature is not None:

        try:
            exact_temperature = float(
                exact_temperature
            )
        except (
            TypeError,
            ValueError,
        ):
            exact_temperature = None

        # NaN is the only normal float that is
        # not equal to itself.
        if (
            exact_temperature is not None
            and exact_temperature
            == exact_temperature
        ):
            result[
                "resolved_temperature_C"
            ] = exact_temperature

    # --------------------------------------------------------
    # Normalize harmless temperature-type wording
    # --------------------------------------------------------
    temperature_type = result.get(
        "resolved_temperature_type"
    )

    if isinstance(
        temperature_type,
        str,
    ):
        normalized_type = (
            temperature_type
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        type_aliases = {
            "slow_pyrolysis": "pyrolysis",
            "fast_pyrolysis": "pyrolysis",
            "flash_pyrolysis": "pyrolysis",
            "htc": "hydrothermal",
            "hydrothermal_carbonization": (
                "hydrothermal"
            ),
            "hydrothermal_carbonisation": (
                "hydrothermal"
            ),
            "flash_carbonization": (
                "carbonization"
            ),
            "flash_carbonisation": (
                "carbonisation"
            ),
        }

        result[
            "resolved_temperature_type"
        ] = type_aliases.get(
            normalized_type,
            normalized_type,
        )

    required_fields = {
        "recovery_id",
        "row_decision",
        "resolved_sample",
        "resolved_temperature_C",
        "resolved_temperature_type",
        "provenance",
        "confidence",
        "evidence_summary",
    }

    missing = (
        required_fields
        - set(
            result.keys()
        )
    )

    if missing:
        raise ValueError(
            f"Missing fields: "
            f"{sorted(missing)}"
        )

    if (
        str(
            result[
                "recovery_id"
            ]
        )
        != str(
            package[
                "recovery_id"
            ]
        )
    ):
        raise ValueError(
            "recovery_id mismatch"
        )
    if (
        result["row_decision"]
        not in ALLOWED_ROW_DECISIONS
    ):
        raise ValueError(
            "Invalid row_decision: "
            f"{result['row_decision']}"
        )

    if (
        result["provenance"]
        not in ALLOWED_PROVENANCE
    ):
        raise ValueError(
            "Invalid provenance"
        )

    if (
        result["confidence"]
        not in ALLOWED_CONFIDENCE
    ):
        raise ValueError(
            "Invalid confidence"
        )

    if (
        result[
            "resolved_temperature_type"
        ]
        not in ALLOWED_TEMPERATURE_TYPES
    ):
        raise ValueError(
            "Invalid temperature type"
        )

    temperature = result[
        "resolved_temperature_C"
    ]

    if temperature is not None:

        if not isinstance(
            temperature,
            (int, float),
        ):
            raise ValueError(
                "Temperature must be numeric or null"
            )

        if not (
            50
            <= float(temperature)
            <= 3000
        ):
            raise ValueError(
                "Resolved temperature outside "
                "plausible range"
            )
    if (
        result["row_decision"]
        != "resolve"
    ):
        result["resolved_sample"] = None
        result["resolved_temperature_C"] = None
        result["resolved_temperature_type"] = None

    return result


# ============================================================
# Load packages
# ============================================================

if not input_path.exists():
    raise FileNotFoundError(
        f"Step 08A packages not found: "
        f"{input_path}"
    )

packages = load_jsonl(
    input_path
)

print(
    "Recovery packages loaded:",
    len(packages),
)

print(
    "Model:",
    MODEL_NAME,
)

print(
    "Ollama:",
    OLLAMA_BASE_URL,
)


# ============================================================
# Resume support
# ============================================================

completed_ids = set()

if (
    RESUME
    and results_path.exists()
):
    for record in load_jsonl(
        results_path
    ):
        recovery_id = record.get(
            "recovery_id"
        )

        if recovery_id:
            completed_ids.add(
                str(recovery_id)
            )

print(
    "Already completed:",
    len(completed_ids),
)


# ============================================================
# Select run
# ============================================================

pending = [
    package
    for package in packages
    if str(
        package[
            "recovery_id"
        ]
    )
    not in completed_ids
]
temperature_range_count = sum(
    1
    for package in pending
    if package.get(
        "semantic_class"
    )
    == "TEMPERATURE_RANGE"
)

pending = [
    package
    for package in pending
    if package.get(
        "semantic_class"
    )
    != "TEMPERATURE_RANGE"
]
if TARGET_IDS:
    pending = [
        package
        for package in pending
        if str(
            package["recovery_id"]
        )
        in TARGET_IDS
    ]

if (
    not RUN_ALL
    and not TARGET_IDS
):
    pending = pending[
        :TEST_LIMIT
    ]

print(
    "Temperature-range packages excluded:",
    temperature_range_count,
)

print(
    "Eligible semantic recovery:",
    len(
        [
            package
            for package in packages
            if package.get(
                "semantic_class"
            )
            != "TEMPERATURE_RANGE"
        ]
    ),
)

print(
    "Rows selected this run:",
    len(pending),
)

print()


# ============================================================
# Run semantic recovery
# ============================================================

success_count = 0
failure_count = 0

run_start = time.time()

for number, package in enumerate(
    pending,
    start=1,
):

    recovery_id = package[
        "recovery_id"
    ]

    print(
        f"[{number}/{len(pending)}] "
        f"{recovery_id} "
        f"{package['table_id']}"
    )

    user_prompt = build_user_prompt(
        package
    )

    start = time.time()

    try:
        (
            raw_content,
            raw_response,
        ) = call_ollama(
            SYSTEM_PROMPT,
            user_prompt,
        )

        elapsed = (
            time.time()
            - start
        )

        raw_path = (
            raw_dir
            / (
                safe_filename(
                    recovery_id
                )
                + ".json"
            )
        )

        raw_path.write_text(
            json.dumps(
                raw_response,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        parsed = extract_json_object(
            raw_content
        )

        validated = validate_result(
            parsed,
            package,
        )

        output_record = {
            "recovery_id": (
                recovery_id
            ),
            "paper_id": (
                package[
                    "paper_id"
                ]
            ),
            "table_id": (
                package[
                    "table_id"
                ]
            ),
            "source_row_index": (
                package[
                    "source_row_index"
                ]
            ),

            # Deterministic values preserved unchanged.
            "C_value": (
                package.get(
                    "C_value"
                )
            ),
            "H_value": (
                package.get(
                    "H_value"
                )
            ),
            "N_value": (
                package.get(
                    "N_value"
                )
            ),
            "O_value": (
                package.get(
                    "O_value"
                )
            ),

            # LLM-resolved metadata only.
            "row_decision": (
                validated[
                    "row_decision"
                ]
            ),
            "resolved_sample": (
                validated[
                    "resolved_sample"
                ]
            ),
            "resolved_temperature_C": (
                validated[
                    "resolved_temperature_C"
                ]
            ),
            "resolved_temperature_type": (
                validated[
                    "resolved_temperature_type"
                ]
            ),
            "provenance": (
                validated[
                    "provenance"
                ]
            ),
            "confidence": (
                validated[
                    "confidence"
                ]
            ),
            "evidence_summary": (
                validated[
                    "evidence_summary"
                ]
            ),

            "model": MODEL_NAME,
            "elapsed_seconds": round(
                elapsed,
                2,
            ),
        }

        append_jsonl(
            results_path,
            output_record,
        )

        success_count += 1

        print(
            "  SUCCESS",
            f"{elapsed:.2f}s",
        )

        print(
            "  decision:",
            validated[
                "row_decision"
            ],
        )

        print(
            "  sample:",
            validated[
                "resolved_sample"
            ],
        )

        print(
            "  temperature:",
            validated[
                "resolved_temperature_C"
            ],
        )

        print(
            "  provenance:",
            validated[
                "provenance"
            ],
        )

    except Exception as error:

        failure_count += 1

        failure_record = {
            "recovery_id": (
                recovery_id
            ),
            "paper_id": (
                package[
                    "paper_id"
                ]
            ),
            "table_id": (
                package[
                    "table_id"
                ]
            ),
            "error_type": (
                type(error).__name__
            ),
            "error": str(
                error
            ),
        }

        append_jsonl(
            failures_path,
            failure_record,
        )

        print(
            "  FAILED:",
            type(error).__name__,
            str(error),
        )

    print()


# ============================================================
# Run summary
# ============================================================

total_elapsed = (
    time.time()
    - run_start
)

summary = {
    "model": MODEL_NAME,
    "run_all": RUN_ALL,
    "test_limit": TEST_LIMIT,
    "selected_rows": len(
        pending
    ),
    "success_count": (
        success_count
    ),
    "failure_count": (
        failure_count
    ),
    "elapsed_seconds": round(
        total_elapsed,
        2,
    ),
}

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)

print("=" * 70)
print("STEP 08B — SEMANTIC RECOVERY TEST")
print("=" * 70)

print(
    "Success:",
    success_count,
)

print(
    "Failures:",
    failure_count,
)

print(
    "Elapsed:",
    round(
        total_elapsed,
        2,
    ),
    "seconds",
)

print()
print(
    "Results:",
    results_path,
)

print(
    "Failures:",
    failures_path,
)


