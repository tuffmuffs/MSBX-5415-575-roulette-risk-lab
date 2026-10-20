# Roulette Risk Lab

Roulette Risk Lab is an interactive probability application built for a class assignment on chance, randomness, and uncertainty. It combines exact roulette mathematics with Monte Carlo simulation to show how betting systems change the distribution of outcomes without removing the casino's expected advantage.

## Research question

How do wheel type, bankroll, session length, table limits, and betting strategy affect expected loss, volatility, the chance of finishing ahead, and the risk of losing an entire bankroll?

## What the app includes

- European and American roulette probabilities
- Flat, Martingale, and Fibonacci betting strategies
- Reproducible simulations controlled by a random seed
- Thousands of independent sessions per experiment
- Expected value, variance, and standard deviation
- Probability of finishing ahead and probability of ruin
- Monte Carlo uncertainty interval for average profit
- Sample bankroll paths and ending-bankroll distributions
- Controlled strategy comparison using identical wheel outcomes
- Long-run convergence toward the theoretical expected value
- Downloadable session-level results
- Methods, assumptions, limitations, and assignment-rubric documentation

## Mathematical foundation

For a one-dollar even-money roulette bet:

```text
Profit = +$1 after a win
Profit = -$1 after a loss
```

The expected value is:

```text
E[X] = P(win)(+$1) + P(loss)(-$1)
```

| Wheel | Probability of winning | Expected profit per $1 | House edge |
|---|---:|---:|---:|
| European | 18/37 = 48.65% | -$0.0270 | 2.70% |
| American | 18/38 = 47.37% | -$0.0526 | 5.26% |

If the amount wagered is chosen using only previous outcomes, the next independent spin still has the same expected value. Therefore:

```text
Expected session profit = -house edge × expected total dollars wagered
```

## Strategy definitions

### Flat

The player wagers the same base amount on every active spin.

### Martingale

The player doubles the planned stake after a loss and returns to the base bet after a win. The actual stake cannot exceed the selected table limit or remaining bankroll.

### Fibonacci

The player follows the sequence 1, 1, 2, 3, 5, 8, and so on. A loss moves one step forward; a win moves two steps backward. The sequence multiplier is applied to the base bet and remains subject to the table limit and bankroll.

## Project files

```text
roulette-risk-lab/
├── app.py                    # Streamlit interface and interactive charts
├── simulation.py             # Probability and simulation calculations
├── requirements.txt          # Packages installed by Streamlit Community Cloud
├── README.md                 # Project documentation
├── .gitignore                # Files excluded from GitHub
├── .streamlit/
│   └── config.toml           # App theme and server configuration
└── tests/
    └── test_simulation.py    # Independent calculation and behavior tests
```

## Run locally

### 1. Open a terminal in the project folder

```bash
cd roulette-risk-lab
```

### 2. Create an optional virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install the packages

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the app

```bash
python -m streamlit run app.py
```

Streamlit will print a local address, normally `http://localhost:8501`.

## Run the tests

The tests use Python's built-in `unittest` package, so no additional testing package is required.

```bash
python -m unittest discover -s tests -v
```

The tests verify:

- Exact American and European roulette expected values
- Exact per-bet variance
- Reproducible random outcomes
- Flat-bet accounting under all wins
- Martingale progression and bankruptcy under consecutive losses
- Fair strategy comparison using common random outcomes
- Expected loss based on total dollars wagered

## Publish through GitHub and Streamlit Community Cloud

### 1. Create the GitHub repository

Create a public repository named `roulette-risk-lab`. Do not initialize it with a separate README because this project already includes one.

### 2. Push this project to GitHub

Run the following commands from inside the project folder. Replace `YOUR-USERNAME` with your GitHub username.

```bash
git init
git add .
git commit -m "Build Roulette Risk Lab"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/roulette-risk-lab.git
git push -u origin main
```

### 3. Deploy with Streamlit Community Cloud

1. Visit <https://share.streamlit.io/>
2. Sign in with GitHub
3. Click **Create app**
4. Select the `roulette-risk-lab` repository
5. Select the `main` branch
6. Enter `app.py` as the entry-point file
7. Choose a public app URL
8. Open **Advanced settings** and select Python 3.12 if available
9. Click **Deploy**

The public address should resemble:

```text
https://roulette-risk-lab.streamlit.app
```

Test the link in a private browser window before submitting it for class.

## Updating the published app

Commit and push changes to the same GitHub repository:

```bash
git add .
git commit -m "Update analysis"
git push
```

Streamlit Community Cloud will detect the new commit and rebuild the application.

## Suggested assignment description

> Roulette Risk Lab combines exact probability calculations with Monte Carlo simulation to examine how roulette wheel design and betting strategy affect player outcomes. Users can vary the wheel, bankroll, base bet, table limit, session length, simulation size, and random seed. The app compares flat, Martingale, and Fibonacci systems using identical random outcomes, then reports expected value, volatility, the probability of finishing ahead, risk of ruin, and the distribution of ending bankrolls. The analysis demonstrates that a betting system can change short-run risk without changing the negative expected value attached to each dollar wagered.

## Limitations

- Spins are modeled as independent and identically distributed.
- The wheel is assumed to be fair and correctly balanced.
- The model excludes taxes, comps, biased wheels, dealer errors, and casino-specific minimum bets.
- Simulation results vary with the random seed and number of sessions.
- The app is intended for education, not gambling advice.

## Technology

- Python
- Streamlit
- NumPy
- pandas
- Plotly

