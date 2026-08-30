import re
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILES = [
    "keeper_telco2.txt",
    #"keeper_telco_leed_v1.txt"
]

OUTPUT_DIR = "extracted_results"

# ============================================================
# REGEX
# ============================================================

# Namespace(...)
namespace_pattern = re.compile(
    r"Namespace\((.*?)\)",
    re.DOTALL
)

# training took XXs
training_pattern = re.compile(
    r"train took .*?s",
    re.IGNORECASE
)

# Metrics
metric_patterns = {
    "ACC": re.compile(r"ACC\s*→.*", re.IGNORECASE),
    "AUC": re.compile(r"AUC\s*→.*", re.IGNORECASE),
    "F1": re.compile(r"F1\s*→.*", re.IGNORECASE),
    "PRECISION": re.compile(r"PRECISION\s*→.*", re.IGNORECASE),
    "RECALL": re.compile(r"RECALL\s*→.*", re.IGNORECASE),
}

# ============================================================
# EXTRACTION
# ============================================================

def extract_blocks(text):

    lines = text.splitlines()

    blocks = []

    current_namespace = None
    current_training = None
    current_metrics = {}

    for line in lines:

        # ----------------------------------------------------
        # Namespace
        # ----------------------------------------------------
        namespace_match = namespace_pattern.search(line)

        if namespace_match:

            # Save previous block if complete
            if current_namespace is not None:
                blocks.append({
                    "namespace": current_namespace,
                    "training": current_training,
                    "metrics": current_metrics.copy()
                })

            # Start new block
            current_namespace = (
                "Namespace(" +
                namespace_match.group(1).strip() +
                ")"
            )

            current_training = None
            current_metrics = {}

            continue

        # ----------------------------------------------------
        # Training time
        # ----------------------------------------------------
        training_match = training_pattern.search(line)

        if training_match:
            current_training = training_match.group(0)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------
        for metric_name, pattern in metric_patterns.items():

            metric_match = pattern.search(line)

            if metric_match:
                current_metrics[metric_name] = metric_match.group(0)

    # --------------------------------------------------------
    # Save last block
    # --------------------------------------------------------
    if current_namespace is not None:
        blocks.append({
            "namespace": current_namespace,
            "training": current_training,
            "metrics": current_metrics.copy()
        })

    return blocks

# ============================================================
# WRITE OUTPUT
# ============================================================

def write_output(blocks, output_file):

    with open(output_file, "w", encoding="utf-8") as f:

        for i, block in enumerate(blocks, start=1):

            f.write("=" * 70 + "\n")
            f.write(f"RUN {i}\n")
            f.write("=" * 70 + "\n\n")

            # Namespace
            f.write(block["namespace"] + "\n\n")

            # Training
            if block["training"]:
                f.write(block["training"] + "\n\n")
            else:
                f.write("training time NOT FOUND\n\n")

            # Metrics
            for metric in ["ACC", "AUC", "F1", "PRECISION", "RECALL"]:

                if metric in block["metrics"]:
                    f.write(block["metrics"][metric] + "\n")
                else:
                    f.write(f"{metric} NOT FOUND\n")

            f.write("\n\n")

# ============================================================
# MAIN
# ============================================================

def main():

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    for input_file in INPUT_FILES:

        input_path = Path(input_file)

        if not input_path.exists():
            print(f"[WARNING] File not found: {input_file}")
            continue

        print(f"Processing: {input_file}")

        # Read file
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Extract all runs
        blocks = extract_blocks(text)

        # Output file
        output_file = output_dir / f"{input_path.stem}_short.txt"

        # Write extracted results
        write_output(blocks, output_file)

        print(f"Saved: {output_file}")
        print(f"Extracted {len(blocks)} runs")

    print("\nDone.")

# ============================================================

if __name__ == "__main__":
    main()