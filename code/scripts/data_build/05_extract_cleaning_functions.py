"""
Created on Sun Aug 11 16:54:17 2026

@author: gayukimanikkuge
"""

from pathlib import Path
import ast


# LOCATION OF YOUR ORIGINAL CONSTRUCTION SCRIPT

SCRIPT = (
    Path.home()
    / "Downloads"
    / "01_build_channel_analysis.py"
)

if not SCRIPT.exists():
    raise FileNotFoundError(
        f"Could not find:\n{SCRIPT}"
    )


# READ SOURCE CODE

source = SCRIPT.read_text(
    encoding="utf-8"
)

tree = ast.parse(source)


# Keywords for functions we want to inspect
keywords = [
    "sex",
    "coapp",
    "co_app",
    "age",
    "dti",
    "debt",
    "geo",
    "county",
    "state"
]


print("=" * 80)
print("RELEVANT CLEANING FUNCTIONS")
print("=" * 80)


found = 0


for node in tree.body:

    if isinstance(
        node,
        (ast.FunctionDef, ast.AsyncFunctionDef)
    ):

        name = node.name.lower()

        function_source = ast.get_source_segment(
            source,
            node
        )

        searchable = (
            name
            + "\n"
            + (
                function_source.lower()
                if function_source
                else ""
            )
        )

        if any(
            word in searchable
            for word in keywords
        ):

            found += 1

            print("\n")
            print("#" * 80)
            print(
                f"FUNCTION {found}: {node.name}"
            )
            print("#" * 80)

            print(
                function_source
            )


print("\n")
print("=" * 80)
print(
    f"Functions found: {found}"
)
print("=" * 80)