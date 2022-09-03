import numpy as np
import pandas as pd

DATA_DIR = "data"


def class_decode(pclass: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd",}.get(pclass)


def survived_decode(surv: int) -> str:
    return {0: "Died", 1: "Survived"}.get(surv)


col_transforms = [
    ("survived", "survived_label", survived_decode),
    ("pclass", "class_label", class_decode),
    ("sex", "sex_label", lambda k: k.title()),
    ("age", "age_rounded", lambda k: k if pd.isnull(k) else int(round(k))),
]


def sex_class_label(row):
    sex, _class = row.sex_label, row.class_label
    return f"{sex} - {_class} Class"


row_transforms = [
    sex_class_label,
]


def jitter_age(df):
    """
    artificially jitter age to allow for nonoverlapping points in the distribution plot
    """

    id_level = ["sex_class_label"]  # resolve overlaps within this group
    out_col = "age_jittered"

    df[out_col] = df.age

    for _id, grp in df.groupby([*id_level, "age"]):
        age = _id[-1]
        n = grp.shape[0]
        if n > 1:
            steps = np.linspace(age, age + 1 - 1 / n, n).round(2)
            for index, step in zip(grp.index, steps):
                df.loc[index, out_col] = step

    return df


if __name__ == "__main__":

    d = pd.read_csv(f"{DATA_DIR}/data.csv")
    d.columns = [col.lower() for col in d.columns]

    for col, out_col, f in col_transforms:
        d[out_col] = d[col].apply(f)

    for f in row_transforms:
        d[f.__name__] = d.apply(f, axis=1)

    d = jitter_age(d)

    d.to_csv(f"{DATA_DIR}/prepped.csv", index=False)
