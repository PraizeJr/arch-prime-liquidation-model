# Arch Prime liquidation stress model

A small, open model that asks one question: **when an Arch Prime account is liquidated, where would money actually be lost, and who would lose it?**

It accompanies the article *Borrowing Against Bitcoin? The Risk Isn't Where You're Looking* ([WIP). You do not need this repository to follow the article. It is here so you can check my work.

## What this is, and isn't

- It is an **independent, illustrative** model built from Arch's **public** documentation.
- It uses **no Arch data** and is **not** Arch's internal model.
- Results are "if these inputs were right", not forecasts.
- Arch Prime was listed as "coming soon" when this was written. Nothing here is financial advice.

## The three kinds of input

Every input is listed in [`data/assumptions.csv`](data/assumptions.csv) with a category:

| Category | Meaning |
|---|---|
| **documented** | Stated in Arch's public docs (source and date given) |
| **assumed** | Arch does not publish it, so I used a labelled placeholder |
| **unverified** | Matters, but not modelled because there is no public number |

The most important assumed input is how much the strategy receipt counts as collateral. Change it and watch the results move.

## What's in the repository

| File or folder | What it is |
|---|---|
| `arch_prime_liquidation_model.ipynb` | The notebook, already run, so results show on GitHub without installing anything |
| `arch_prime_liquidation_model.py` | The same model as a plain Python script |
| `figures/` | The two charts used in the article |
| `data/scenario_table.csv` | The scenario results |
| `data/assumptions.csv` | Every input, its category, its source and the date it was read |
| `sources/` | Dated copies of the Arch pages used |
| `CHANGELOG.md` | Every correction, with date and who raised it |

## Run it yourself

1. Install Python 3.10 or newer.
2. In a terminal, inside this folder:
   ```
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   jupyter notebook
   ```
3. Open `arch_prime_liquidation_model.ipynb` and choose **Run all** (menu: Kernel, then Restart and Run All).

No install? Open it in Google Colab: `https://colab.research.google.com/github/YOUR-USERNAME/YOUR-REPO/blob/main/arch_prime_liquidation_model.ipynb`

## Found an error?

Open an issue or message me. Corrections are logged in `CHANGELOG.md`.
