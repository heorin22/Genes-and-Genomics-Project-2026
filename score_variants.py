import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import EsmTokenizer, EsmForMaskedLM


MODEL_NAME = "facebook/esm2_t33_650M_UR50D"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score CFTR missense variants using ESM-2."
    )

    parser.add_argument(
        "--variants",
        required=True,
        help="CSV containing mutation and class columns."
    )

    parser.add_argument(
        "--fasta",
        required=True,
        help="FASTA file containing the CFTR protein sequence."
    )

    parser.add_argument(
        "--output",
        default="variant_scores.csv",
        help="Output CSV file."
    )

    parser.add_argument(
        "--window",
        type=int,
        default=1000,
        help="Maximum sequence window around mutation."
    )

    return parser.parse_args()


def load_fasta(fasta_path):
    """
    Load a single protein sequence from a FASTA file.
    """

    sequence = []

    with open(fasta_path, "r") as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith(">"):
                continue

            sequence.append(line)

    return "".join(sequence)


def get_crop(full_sequence, position, window=1000):
    """
    Extract a sequence window around the mutation.

    position is zero-based.
    """

    start = max(0, position - window // 2)
    end = min(len(full_sequence), start + window)

    if end - start < window:
        start = max(0, end - window)

    cropped_sequence = full_sequence[start:end]
    relative_position = position - start

    return cropped_sequence, relative_position


def parse_mutation(mutation):
    """
    Parse mutation notation such as G551D.

    Returns
    -------
    wt_aa : str
    position : int
        One-based protein position.
    mutant_aa : str
    """

    mutation = str(mutation).strip()

    if len(mutation) < 3:
        raise ValueError(f"Invalid mutation format: {mutation}")

    wt_aa = mutation[0]
    mutant_aa = mutation[-1]

    try:
        position = int(mutation[1:-1])
    except ValueError:
        raise ValueError(f"Invalid mutation format: {mutation}")

    return wt_aa, position, mutant_aa


def score_variant(
    sequence,
    mutation,
    tokenizer,
    model,
    device,
    window
):
    """
    Calculate the ESM-2 log-likelihood ratio:

        LLR = log P(mutant | context) - log P(WT | context)
    """

    wt_aa, position_1based, mutant_aa = parse_mutation(mutation)

    position_0based = position_1based - 1

    if position_0based < 0 or position_0based >= len(sequence):
        raise ValueError(
            f"{mutation}: position outside sequence range."
        )

    observed_residue = sequence[position_0based]

    if observed_residue != wt_aa:
        raise ValueError(
            f"{mutation}: expected WT residue {wt_aa}, "
            f"but FASTA contains {observed_residue}."
        )

    cropped_sequence, relative_position = get_crop(
        sequence,
        position_0based,
        window
    )

    sequence_list = list(cropped_sequence)
    sequence_list[relative_position] = tokenizer.mask_token

    masked_sequence = "".join(sequence_list)

    inputs = tokenizer(
        masked_sequence,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    mask_positions = (
        inputs.input_ids == tokenizer.mask_token_id
    ).nonzero(as_tuple=True)

    if len(mask_positions[1]) != 1:
        raise RuntimeError(
            f"{mutation}: expected exactly one mask token."
        )

    mask_index = mask_positions[1].item()

    log_probs = torch.nn.functional.log_softmax(
        logits[0, mask_index],
        dim=-1
    )

    wt_id = tokenizer.convert_tokens_to_ids(wt_aa)
    mutant_id = tokenizer.convert_tokens_to_ids(mutant_aa)

    wt_log_prob = log_probs[wt_id].item()
    mutant_log_prob = log_probs[mutant_id].item()

    llr = mutant_log_prob - wt_log_prob

    return {
        "Mutation": mutation,
        "WT": wt_aa,
        "Position": position_1based,
        "Mutant": mutant_aa,
        "WT_log_probability": wt_log_prob,
        "Mutant_log_probability": mutant_log_prob,
        "LLR": llr,
    }


def main():

    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ----------------------------
    # Load data
    # ----------------------------

    variants = pd.read_csv(args.variants)

    required_columns = {"mutation", "class"}

    if not required_columns.issubset(variants.columns):
        raise ValueError(
            "Variant CSV must contain 'mutation' and 'class' columns."
        )

    variants = variants.dropna(
        subset=["mutation", "class"]
    )

    variants["mutation"] = (
        variants["mutation"]
        .astype(str)
        .str.strip()
    )

    variants = variants.drop_duplicates(
        subset=["mutation"]
    )

    sequence = load_fasta(args.fasta)

    print(f"CFTR length: {len(sequence)} aa")
    print(f"Variants: {len(variants)}")

    # ----------------------------
    # Load ESM-2
    # ----------------------------

    print(f"Loading {MODEL_NAME}...")

    tokenizer = EsmTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = EsmForMaskedLM.from_pretrained(
        MODEL_NAME
    )

    model = model.to(device)
    model.eval()

    # ----------------------------
    # Score variants
    # ----------------------------

    results = []

    for _, row in variants.iterrows():

        mutation = row["mutation"]
        mutation_class = row["class"]

        print(f"Scoring {mutation}...")

        try:

            result = score_variant(
                sequence=sequence,
                mutation=mutation,
                tokenizer=tokenizer,
                model=model,
                device=device,
                window=args.window
            )

            result["Class"] = mutation_class

            results.append(result)

        except Exception as error:

            print(
                f"WARNING: skipped {mutation}: {error}"
            )

    if not results:
        raise RuntimeError(
            "No variants were successfully scored."
        )

    results_df = pd.DataFrame(results)

    column_order = [
        "Mutation",
        "Class",
        "WT",
        "Position",
        "Mutant",
        "WT_log_probability",
        "Mutant_log_probability",
        "LLR",
    ]

    results_df = results_df[column_order]

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved {len(results_df)} scores "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()