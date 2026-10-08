# Credit Risk & Lending Strategy
Predicting loan defaults and testing an approval policy that reduces losses while keeping most lending volume.

## The Main Question

A consumer lender is seeing losses from loan defaults. Leadership wants to know:

1. Which borrowers are most likely to default?
2. How much loss could be avoided by declining the riskiest applicants?
3. What approval policy maximizes profit without giving up too much good business?

## Dataset
[Lending Club 2016-2020](https://data.mendeley.com/datasets/mvvd3dnfyz)

Maloney, David (2022), "lendingclub2016Q1-2020Q2", Mendeley Data, V1, doi: 10.17632/mvvd3dnfyz.1

Lending Club peer-to-peer loan data from Q1 2016 through Q2 2020 (about 1.06 million loans). This project uses loans issued in 2017 or later that have a final outcome (Fully Paid or Charged Off), then samples 150,000 of them.

## Approach
1. **Clean and filter:** keep completed loans issued 2017+, convert text fields (interest rate, term, revolving utilization) to numbers, and define the target (default = 1 if Charged Off).
2. **Features:** loan amount, interest rate, installment, annual income, DTI, term, revolving utilization, plus grade, home ownership, purpose, and income verification (one-hot encoded). Missing values are filled with the column median.
3. **Out-of-time split:** train on loans issued before 2019 (127,237 loans) and test on loans issued 2019 or later (22,763 loans), to mimic real deployment and avoid look-ahead bias.
4. **Modeling:** compare a Logistic Regression baseline with Gradient Boosting, evaluated by ROC AUC.
5. **Risk segmentation:** group test borrowers into Low (safest 60%), Medium (next 30%), and High (riskiest 10%) tiers by predicted default risk.
6. **Loss estimate:** apply an assumed loss given default to charged-off principal.
7. **Policy simulation:** test declining the riskiest 0% to 40% of applicants (in 2.5% steps) and calculate losses avoided, interest income given up, and net benefit.
8. **Visualization:** interactive Plotly charts as well as a calibration plot.

## Results 
-Loans analyzed (sample):	150,000 completed loans, 2017+

-Overall default rate:	21.1%

-Logistic Regression AUC:	0.668

-Gradient Boosting AUC (better model):	0.682

-Default rate by tier (test set):	Low 11.8% · Medium 25.6% · High 36.1%

-Observed charged-off principal rate (test set):	20.4%

-Estimated loss at 80% LGD (test set)	$56.1M (16.3% of principal)

-Best simulated policy: Decline the riskiest 2.5% of applicants

-Default rate after policy:	18.4% → 17.7% (184 → 177 defaults per 1,000 loans)

-Loan volume retained:	97.5%

-Estimated net benefit:	$129,153 (losses avoided minus interest income given up)

### What do the results mean?
The model ranks borrowers meaningfully: the High-risk tier defaults at about 3x the rate of the Low-risk tier (36.1% vs. 11.8%).

Predictive power is moderate (AUC 0.68), and Gradient Boosting beats the logistic baseline only slightly.

Because many high-risk borrowers still repay with interest, blanket declines add limited value: the net benefit peaks at declining just 2.5% of applicants and falls after that. A more promising lever is risk-based pricing or tighter terms for the High tier instead of outright declines.

## Default rate by risk tier
-Policy simulation: losses avoided, interest given up, and net benefit vs. % of applicants declined (with the recommended cutoff marked)

-Default rate by loan grade, purpose, and annual income band

-Calibration plot (predicted vs. actual default rate)

-Defaults per 1,000 loans: current vs. new policy

## Assumptions
-Loss given default is assumed at 80% of principal. The dataset flags charged-off loans but does not report recoveries or realized losses. Unsecured consumer loans typically lose a large share of principal after default; this is an adjustable assumption.

-Interest income on a repaid loan is approximated as (monthly installment × term) minus the amount borrowed.

-Defaulted loans are assumed to earn no interest income.

## Limitations
-Only completed loans (Fully Paid or Charged Off) are included, which may bias results. The 2019+ test set has a lower default rate (18.4%) than the full sample (21.1%).

-Loans from 2020 may reflect COVID-19 effects.

-Lending Club grade and interest rate already reflect the platform's own risk assessment.

-Results come from a 150,000-loan sample of historical data; a production model would need monitoring for drift.

-The net benefit depends on the loss-given-default assumption and has not yet been tested at other values.

-A fairness and compliance review would be required before real-world use, including checking for protected attributes and close proxies such as geography.

## Tech Stack

Python, pandas, NumPy, scikit-learn, Plotly, Matplotlib, Jupyter

## Next Steps

-Test sensitivity of the policy to the loss-given-default assumption (e.g., 60% to 90%).

-Model risk-based pricing for the High tier instead of outright declines.

-Add SHAP-based explanations for individual decisions.

-Run a fairness review across borrower segments.
