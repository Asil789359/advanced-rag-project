"""Human-judged citation accuracy. Fill audit.csv after generating answers."""
import argparse
import csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", help="CSV columns: claim,citation,supported (1 or 0)")
    args = p.parse_args()
    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(row.get("supported") not in ("0", "1") for row in rows):
        raise ValueError("Provide at least one judged row, with supported=0 or 1")
    score = sum(int(row["supported"]) for row in rows) / len(rows)
    print(f"Citation accuracy: {score:.3f} ({len(rows)} judged claim-citation pairs)")


if __name__ == "__main__":
    main()
