import pandas as pd
import json
import string
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency

# load data

huff_data = []

with open("News_Category_Dataset_v3.json", "r") as f:
    for line in f:
        huff_data.append(json.loads(line))

huff_df = pd.DataFrame(huff_data)

fox_df = pd.read_csv("fox_news_text_data_2025.csv")


def clean_text(text):
    return str(text).lower().translate(
        str.maketrans("", "", string.punctuation)
    )


huff_df["text"] = (
    huff_df["headline"].fillna("") + " " +
    huff_df["short_description"].fillna("")
).apply(clean_text)

fox_df["text"] = (
    fox_df["title"].fillna("") + " " +
    fox_df["description"].fillna("")
).apply(clean_text)


# movement keywords

movements = {
    "BLM": [
        "black lives matter",
        "blm",
        "george floyd",
        "racial justice",
        "police brutality"
    ],

    "Climate": [
        "climate protest",
        "climate activist",
        "fossil fuel",
        "climate change",
        "global warming"
    ],

    "Jan6": [
        "capitol",
        "january 6",
        "maga",
        "trump supporters",
        "insurrection",
        "capitol riot"
    ],

    "Labor": [
        "strike",
        "union",
        "workers",
        "labor rights",
        "walkout"
    ],

    "Abortion": [
        "abortion rights",
        "roe v wade",
        "pro choice",
        "pro life",
        "abortion ban"
    ],

    "GunControl": [
        "gun control",
        "gun reform",
        "mass shooting",
        "assault weapons ban"
    ],

    "Immigration": [
        "immigration reform",
        "border crisis",
        "migrants",
        "border wall",
        "deportation"
    ],

    "LGBTQ": [
        "lgbtq",
        "gay rights",
        "trans rights",
        "gender identity"
    ],

    "Voting": [
        "voting rights",
        "election integrity",
        "voter suppression",
        "election fraud"
    ],

    "IsraelPalestine": [
        "israel",
        "palestine",
        "gaza",
        "hamas",
        "idf"
    ]
}


positive_labels = [
    "activist",
    "advocate",
    "organizer",
    "leader",
    "peaceful",
    "justice",
    "solidarity"
]

neutral_labels = [
    "protester",
    "demonstrator",
    "marcher",
    "crowd",
    "protest",
    "rally"
]

negative_labels = [
    "rioter",
    "looter",
    "thug",
    "extremist",
    "mob",
    "violent",
    "violence",
    "illegal"
]


def run_analysis(df):

    stats = {}

    for movement, keywords in movements.items():

        movement_df = df[
            df["text"].apply(
                lambda x: any(keyword in x for keyword in keywords)
            )
        ]

        positive_count = 0
        neutral_count = 0
        negative_count = 0

        for text in movement_df["text"]:

            if any(word in text for word in negative_labels):
                negative_count += 1

            elif any(word in text for word in positive_labels):
                positive_count += 1

            elif any(word in text for word in neutral_labels):
                neutral_count += 1

        stats[movement] = {
            "Pos": positive_count,
            "Neu": neutral_count,
            "Neg": negative_count,
            "Total": max(len(movement_df), 1)
        }

    return stats


h_res = run_analysis(huff_df)
f_res = run_analysis(fox_df)


# graph comparing framing proportions

movement_list = list(movements.keys())

x = np.arange(len(movement_list))
width = 0.12


def get_proportion(results, movement, category):

    return (
        results[movement][category] /
        results[movement]["Total"]
    )


plt.figure(figsize=(18, 8))

plt.bar(
    x - 2.5 * width,
    [get_proportion(h_res, m, "Pos") for m in movement_list],
    width,
    label="HuffPost Positive"
)

plt.bar(
    x - 1.5 * width,
    [get_proportion(h_res, m, "Neu") for m in movement_list],
    width,
    label="HuffPost Neutral"
)

plt.bar(
    x - 0.5 * width,
    [get_proportion(h_res, m, "Neg") for m in movement_list],
    width,
    label="HuffPost Negative"
)

plt.bar(
    x + 0.5 * width,
    [get_proportion(f_res, m, "Pos") for m in movement_list],
    width,
    label="Fox Positive",
    hatch="//"
)

plt.bar(
    x + 1.5 * width,
    [get_proportion(f_res, m, "Neu") for m in movement_list],
    width,
    label="Fox Neutral",
    hatch="//"
)

plt.bar(
    x + 2.5 * width,
    [get_proportion(f_res, m, "Neg") for m in movement_list],
    width,
    label="Fox Negative",
    hatch="//"
)

plt.xticks(x, movement_list, rotation=45)

plt.ylabel("Proportion of Articles")

plt.title(
    "Comparative Media Framing Across Social Movements"
)

plt.legend()

plt.tight_layout()

plt.show()


# chi square analysis

print("\n================ STATISTICAL ANALYSIS ================\n")

huff_total = {
    "Pos": 0,
    "Neu": 0,
    "Neg": 0
}

fox_total = {
    "Pos": 0,
    "Neu": 0,
    "Neg": 0
}

for movement in movement_list:

    for category in huff_total:

        huff_total[category] += h_res[movement][category]
        fox_total[category] += f_res[movement][category]


overall_table = [
    [
        huff_total["Pos"],
        huff_total["Neu"],
        huff_total["Neg"]
    ],

    [
        fox_total["Pos"],
        fox_total["Neu"],
        fox_total["Neg"]
    ]
]

chi2, p, dof, expected = chi2_contingency(overall_table)

print("OVERALL CHI-SQUARE TEST\n")

print(f"Chi-square statistic: {chi2:.2f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p:.8f}")

n = np.sum(overall_table)

cramers_v = np.sqrt(
    chi2 / (n * (min(np.shape(overall_table)) - 1))
)

print(f"Cramer's V: {cramers_v:.3f}")


# test each movement

print("\n================ PER-MOVEMENT TESTS ================\n")

movement_scores = []

for movement in movement_list:

    table = [
        [
            h_res[movement]["Pos"],
            h_res[movement]["Neu"],
            h_res[movement]["Neg"]
        ],

        [
            f_res[movement]["Pos"],
            f_res[movement]["Neu"],
            f_res[movement]["Neg"]
        ]
    ]

    chi2_m, p_m, dof_m, expected_m = chi2_contingency(table)

    n_m = np.sum(table)

    cramers_v_m = np.sqrt(
        chi2_m / (n_m * (min(np.shape(table)) - 1))
    )

    movement_scores.append(
        (movement, chi2_m, p_m, dof_m, cramers_v_m)
    )

    print(f"{movement}")
    print(f"Chi-square: {chi2_m:.2f}")
    print(f"Degrees of freedom: {dof_m}")
    print(f"P-value: {p_m:.8f}")
    print(f"Cramer's V: {cramers_v_m:.3f}")
    print()


# sort by chi square value

movement_scores.sort(
    key=lambda x: x[1],
    reverse=True
)

print("\n================ TOP 3 MOST POLARIZED MOVEMENTS ================\n")

for movement, chi2_m, p_m, dof_m, cramers_v_m in movement_scores[:3]:

    print(movement)

    print(f"Chi-square: {chi2_m:.2f}")
    print(f"Degrees of freedom: {dof_m}")
    print(f"P-value: {p_m:.8f}")
    print(f"Cramer's V: {cramers_v_m:.3f}")

    print()


# graph chi square values

chi_matrix = []

for movement in movement_list:

    table = [
        [
            h_res[movement]["Pos"],
            h_res[movement]["Neu"],
            h_res[movement]["Neg"]
        ],

        [
            f_res[movement]["Pos"],
            f_res[movement]["Neu"],
            f_res[movement]["Neg"]
        ]
    ]

    chi2_m, p_m, _, _ = chi2_contingency(table)

    chi_matrix.append([movement, chi2_m])


chi_df = pd.DataFrame(
    chi_matrix,
    columns=["Movement", "ChiSquare"]
)

chi_df = chi_df.sort_values(
    "ChiSquare",
    ascending=False
)

plt.figure(figsize=(10, 5))

sns.barplot(
    data=chi_df,
    x="Movement",
    y="ChiSquare"
)

plt.xticks(rotation=45)

plt.title(
    "Framing Divergence by Movement (Chi-Square Values)"
)

plt.ylabel("Chi-Square Value")

plt.tight_layout()

plt.show()


# graph top 3 movements

top3 = chi_df.head(3)

plt.figure(figsize=(6, 4))

plt.bar(
    top3["Movement"],
    top3["ChiSquare"]
)

plt.title(
    "Top 3 Most Polarized Social Movements"
)

plt.ylabel("Chi-Square Value")

plt.xticks(rotation=30)

plt.tight_layout()

plt.show()


# normalized sentiment totals

h_total_labels = sum(huff_total.values())
f_total_labels = sum(fox_total.values())

h_distribution = [
    huff_total["Pos"] / h_total_labels,
    huff_total["Neu"] / h_total_labels,
    huff_total["Neg"] / h_total_labels
]

f_distribution = [
    fox_total["Pos"] / f_total_labels,
    fox_total["Neu"] / f_total_labels,
    fox_total["Neg"] / f_total_labels
]

sentiment_labels = [
    "Positive",
    "Neutral",
    "Negative"
]

x = np.arange(len(sentiment_labels))

width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    h_distribution,
    width,
    label="HuffPost"
)

plt.bar(
    x + width / 2,
    f_distribution,
    width,
    label="Fox News",
    hatch="//"
)

plt.xticks(x, sentiment_labels)

plt.ylabel("Proportion of Framing Labels")

plt.title(
    "Normalized Sentiment Distribution Comparison"
)

plt.ylim(0, 1)

for i, value in enumerate(h_distribution):

    plt.text(
        i - width / 2,
        value + 0.01,
        f"{value:.2f}",
        ha="center"
    )

for i, value in enumerate(f_distribution):

    plt.text(
        i + width / 2,
        value + 0.01,
        f"{value:.2f}",
        ha="center"
    )

plt.legend()

plt.tight_layout()

plt.show()