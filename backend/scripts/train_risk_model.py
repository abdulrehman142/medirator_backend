"""Train and persist the family disease risk model.

The script expects a tabular dataset with at least:
- relation: family relation for the member
- disease: raw disease string

Optional but recommended:
- family_id: a case/group identifier so multiple relatives can be aggregated
- target_risk: ground-truth risk score (0-100)

If target_risk is missing, the script can generate a pseudo-label using the
current risk formula via --allow-pseudo-labels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

GROUP_DISPLAY_NAMES = {
    "Heart": "Heart Disease",
    "Diabetes": "Diabetes",
    "Cancer": "Cancer",
    "Respiratory": "Respiratory Disease",
    "Neuro": "Neurological",
    "Mental": "Mental Health",
    "Organ": "Organ",
    "Autoimmune": "Autoimmune",
    "Genetic": "Genetic",
    "Hormonal": "Hormonal",
    "Other": "Other",
}

DISEASE_GROUPS = {
    "Heart": {
        "heart disease",
        "heart attack",
        "stroke",
        "hypertension",
        "high blood pressure",
        "coronary artery disease",
        "heart failure",
        "arrhythmia",
        "atherosclerosis",
        "angina",
        "cardiomyopathy",
        "brain stroke",
    },
    "Diabetes": {
        "diabetes",
        "diabetes type 1",
        "diabetes type 2",
        "prediabetes",
        "obesity",
        "insulin resistance",
        "metabolic syndrome",
    },
    "Cancer": {
        "cancer",
        "lung cancer",
        "breast cancer",
        "prostate cancer",
        "colon cancer",
        "skin cancer",
        "leukemia",
        "brain tumor",
        "pancreatic cancer",
        "liver cancer",
        "cervical cancer",
    },
    "Respiratory": {"asthma", "copd", "bronchitis", "pneumonia", "tuberculosis", "lung infection"},
    "Neuro": {"alzheimers", "parkinsons", "epilepsy", "migraine", "dementia", "multiple sclerosis"},
    "Mental": {"depression", "anxiety", "bipolar disorder", "schizophrenia", "ocd", "ptsd"},
    "Organ": {"kidney disease", "kidney failure", "liver disease", "fatty liver", "cirrhosis", "hepatitis"},
    "Autoimmune": {"lupus", "rheumatoid arthritis", "psoriasis", "celiac disease"},
    "Genetic": {"thalassemia", "sickle cell anemia", "hemophilia", "cystic fibrosis"},
    "Hormonal": {"thyroid", "hypothyroidism", "hyperthyroidism", "pcos", "hormonal imbalance"},
    "Other": {"infection", "fever", "flu", "covid"},
}

REL_WEIGHT = {
    "father": 3.0,
    "mother": 3.0,
    "grandfather": 1.8,
    "grandmother": 1.8,
    "brother": 1.2,
    "sister": 1.2,
}

DISEASE_SEVERITY = {
    "Heart": 1.5,
    "Cancer": 1.6,
    "Diabetes": 1.4,
    "Neuro": 1.3,
    "Autoimmune": 1.2,
    "Respiratory": 1.0,
    "Organ": 1.3,
}

NO_DISEASE_VALUES = {
    "",
    "none",
    "no disease",
    "no known disease",
    "healthy",
    "n/a",
    "na",
    "unknown",
    "normal",
}

USER_RELATIONS = {"self", "you", "patient"}

FEATURE_COLUMNS = [
    "father_count",
    "mother_count",
    "grandfather_count",
    "grandmother_count",
    "brother_count",
    "sister_count",
    "affected_count",
    "family_size",
    "distinct_relations",
    "generations_present",
    "father_present",
    "mother_present",
    "father_mother_combo",
    "same_group_count",
    "same_group_multi_member",
    "same_group_three_plus",
    "severity_multiplier",
    "risk_score",
    "amplified_score",
]


@dataclass(frozen=True)
class TrainingConfig:
    input_path: Path
    output_path: Path
    family_id_column: str
    relation_column: str
    disease_column: str
    target_column: str | None
    allow_pseudo_labels: bool
    test_size: float
    random_state: int



def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("_", " ").replace("-", " ").strip().lower()
    return " ".join(text.split())



def relation_bucket(relation: object) -> str:
    normalized = normalize_text(relation)
    if normalized in USER_RELATIONS:
        return normalized
    if "grandfather" in normalized:
        return "grandfather"
    if "grandmother" in normalized:
        return "grandmother"
    if normalized in {"father", "mother", "brother", "sister"}:
        return normalized
    return normalized



def group_for_disease(disease: object) -> str:
    normalized = normalize_text(disease)
    for group_name, diseases in DISEASE_GROUPS.items():
        if normalized in diseases or any(keyword in normalized for keyword in diseases):
            return group_name
    return "Other"



def humanize_relation(relation: str) -> str:
    if relation in {"father", "mother", "brother", "sister"}:
        return relation.title()
    if relation == "grandfather":
        return "Grandfather"
    if relation == "grandmother":
        return "Grandmother"
    return relation.title() if relation else "Family member"



def compute_group_risk(group: str, frame: pd.DataFrame) -> float:
    score = 0.0
    relation_hits: dict[str, int] = {}
    generations: set[int] = set()

    for _, row in frame.iterrows():
        bucket = relation_bucket(row.get("relation"))
        if bucket in USER_RELATIONS:
            continue
        weight = REL_WEIGHT.get(bucket, 0.0)
        if weight <= 0:
            continue
        score += weight
        relation_hits[bucket] = relation_hits.get(bucket, 0) + 1
        if bucket in {"father", "mother", "brother", "sister"}:
            generations.add(1)
        elif bucket in {"grandfather", "grandmother"}:
            generations.add(2)

    severity_multiplier = DISEASE_SEVERITY.get(group, 1.0)
    risk = (score * severity_multiplier) ** 1.6 * 10.0

    if relation_hits.get("father", 0) and relation_hits.get("mother", 0):
        risk += 30.0
    if len(generations) >= 2:
        risk *= 1.2
    member_count = len(frame)
    if member_count >= 2:
        risk *= 1.25
    if member_count >= 3:
        risk *= 1.4

    return round(min(max(risk, 0.0), 100.0), 4)



def build_group_sample(group: str, family_frame: pd.DataFrame) -> dict[str, float]:
    relations = [relation_bucket(value) for value in family_frame["relation"]]
    relations = [relation for relation in relations if relation not in USER_RELATIONS]
    family_size = float(len(family_frame))
    affected_count = float(len(relations))
    father_count = float(sum(1 for relation in relations if relation == "father"))
    mother_count = float(sum(1 for relation in relations if relation == "mother"))
    grandfather_count = float(sum(1 for relation in relations if relation == "grandfather"))
    grandmother_count = float(sum(1 for relation in relations if relation == "grandmother"))
    brother_count = float(sum(1 for relation in relations if relation == "brother"))
    sister_count = float(sum(1 for relation in relations if relation == "sister"))
    distinct_relations = float(len(set(relations)))
    generations_present = float(len({1 if relation in {"father", "mother", "brother", "sister"} else 2 for relation in relations}))
    father_present = 1.0 if father_count > 0 else 0.0
    mother_present = 1.0 if mother_count > 0 else 0.0
    father_mother_combo = 1.0 if father_count > 0 and mother_count > 0 else 0.0
    same_group_count = float(len(family_frame))
    same_group_multi_member = 1.0 if same_group_count >= 2 else 0.0
    same_group_three_plus = 1.0 if same_group_count >= 3 else 0.0
    severity_multiplier = float(DISEASE_SEVERITY.get(group, 1.0))
    risk_score = father_count * REL_WEIGHT["father"] + mother_count * REL_WEIGHT["mother"]
    risk_score += grandfather_count * REL_WEIGHT["grandfather"]
    risk_score += grandmother_count * REL_WEIGHT["grandmother"]
    risk_score += brother_count * REL_WEIGHT["brother"]
    risk_score += sister_count * REL_WEIGHT["sister"]
    amplified_score = (risk_score * severity_multiplier) ** 1.6 * 10.0
    if father_count > 0 and mother_count > 0:
        amplified_score += 30.0
    if same_group_count >= 2:
        amplified_score *= 1.25
    if same_group_count >= 3:
        amplified_score *= 1.4
    amplified_score = min(max(amplified_score, 0.0), 100.0)

    return {
        "father_count": father_count,
        "mother_count": mother_count,
        "grandfather_count": grandfather_count,
        "grandmother_count": grandmother_count,
        "brother_count": brother_count,
        "sister_count": sister_count,
        "affected_count": affected_count,
        "family_size": family_size,
        "distinct_relations": distinct_relations,
        "generations_present": generations_present,
        "father_present": father_present,
        "mother_present": mother_present,
        "father_mother_combo": father_mother_combo,
        "same_group_count": same_group_count,
        "same_group_multi_member": same_group_multi_member,
        "same_group_three_plus": same_group_three_plus,
        "severity_multiplier": severity_multiplier,
        "risk_score": risk_score,
        "amplified_score": amplified_score,
    }



def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Input dataset is empty.")

    required = {"relation", "disease"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    return df



def build_training_table(df: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    working = df.copy()
    working[config.relation_column] = working[config.relation_column].map(normalize_text)
    working[config.disease_column] = working[config.disease_column].map(normalize_text)
    working = working[~working[config.disease_column].isin(NO_DISEASE_VALUES)]
    working = working[~working[config.relation_column].isin(USER_RELATIONS)]

    if working.empty:
        raise ValueError("No trainable rows remain after removing self/none records.")

    working["group_name"] = working[config.disease_column].map(group_for_disease)

    group_key = config.family_id_column if config.family_id_column in working.columns else None
    if group_key is None:
        working = working.reset_index(drop=False).rename(columns={"index": "__row_id__"})
        group_key = "__row_id__"

    rows: list[dict[str, float]] = []

    for _, family_frame in working.groupby(group_key, dropna=False):
        for group_name, group_frame in family_frame.groupby("group_name", dropna=False):
            sample = build_group_sample(group_name, group_frame)
            target = None
            if config.target_column and config.target_column in group_frame.columns:
                target = float(pd.to_numeric(group_frame[config.target_column], errors="coerce").fillna(0).mean())
            if target is None:
                if not config.allow_pseudo_labels:
                    raise ValueError(
                        "No target column found. Provide a target column or enable --allow-pseudo-labels."
                    )
                target = compute_group_risk(group_name, group_frame)
            sample["target_risk"] = float(min(max(target, 0.0), 100.0))
            sample["group_name"] = group_name
            rows.append(sample)

    training_frame = pd.DataFrame(rows)
    if training_frame.empty:
        raise ValueError("No training samples could be constructed from the dataset.")

    return training_frame



def train_model(training_frame: pd.DataFrame, config: TrainingConfig):
    X = training_frame[FEATURE_COLUMNS].astype(float)
    y = training_frame["target_risk"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    model = GradientBoostingRegressor(random_state=config.random_state)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "samples": int(len(training_frame)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "group_display_names": GROUP_DISPLAY_NAMES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    return bundle, metrics



def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train the family disease risk model.")
    parser.add_argument("--input", required=True, help="Path to the training dataset (CSV/JSON/JSONL).")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "storage" / "risk_model.pkl"),
        help="Where to save the trained model bundle.",
    )
    parser.add_argument("--family-id-column", default="family_id", help="Family/group identifier column name.")
    parser.add_argument("--relation-column", default="relation", help="Relation column name.")
    parser.add_argument("--disease-column", default="disease", help="Disease column name.")
    parser.add_argument("--target-column", default="target_risk", help="Target column name if available.")
    parser.add_argument(
        "--allow-pseudo-labels",
        action="store_true",
        help="Generate targets from the current risk formula when no target column is present.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    return TrainingConfig(
        input_path=Path(args.input).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        family_id_column=args.family_id_column,
        relation_column=args.relation_column,
        disease_column=args.disease_column,
        target_column=args.target_column,
        allow_pseudo_labels=bool(args.allow_pseudo_labels),
        test_size=float(args.test_size),
        random_state=int(args.random_state),
    )



def main() -> None:
    config = parse_args()
    dataset = load_dataset(config.input_path)
    training_frame = build_training_table(dataset, config)
    bundle, metrics = train_model(training_frame, config)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.output_path)

    print(json.dumps({"saved_to": str(config.output_path), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
