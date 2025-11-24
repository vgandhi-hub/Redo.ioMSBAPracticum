import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from datetime import datetime

demographics = pd.read_csv('demographics.csv')
current_commits = pd.read_csv('current_commitments.csv')
prior_commits = pd.read_csv('prior_commitments.csv')

demographics['offense begin date'] = pd.to_datetime(demographics['offense begin date'], errors='coerce')
demographics['time served in years'] = pd.to_numeric(demographics['time served in years'], errors='coerce')
demographics['aggregate sentence in months'] = pd.to_numeric(demographics['aggregate sentence in months'], errors='coerce')

print("Processing eligibility checks (vectorized)...")

demographics['passes_r3'] = demographics['time served in years'] >= 10
demographics['passes_r2'] = demographics['aggregate sentence in months'] >= 240

violent_codes = ['PC187', 'PC261', 'PC286', 'PC288', 'PC289', 'PC206', 'PC207', 'PC209']
pattern = '|'.join(violent_codes)

current_commits['offense_clean'] = current_commits['offense'].astype(str).str.strip().str.upper()
current_violent = current_commits[current_commits['offense_clean'].str.contains(pattern, case=False, na=False)]
current_violent_ids = set(current_violent['cdcno'].unique())

demographics['passes_r4'] = ~demographics['cdcno'].isin(current_violent_ids)

prior_commits['offense_clean'] = prior_commits['offense'].astype(str).str.strip().str.upper()
prior_violent = prior_commits[prior_commits['offense_clean'].str.contains(pattern, case=False, na=False)]
prior_violent_ids = set(prior_violent['cdcno'].unique())

demographics['passes_r5'] = ~demographics['cdcno'].isin(prior_violent_ids)

demographics['eligible'] = (
    demographics['passes_r2'] & 
    demographics['passes_r3'] & 
    demographics['passes_r4'] & 
    demographics['passes_r5']
)

prior_conviction_ids = set(prior_commits['cdcno'].unique())
demographics['has_prior_convictions'] = demographics['cdcno'].isin(prior_conviction_ids)

print("\n" + "="*70)
print("ELIGIBILITY SUMMARY")
print("="*70)
print(f"Total individuals: {len(demographics)}")
print(f"Eligible: {demographics['eligible'].sum()}")
print(f"Not eligible: {(~demographics['eligible']).sum()}")
print(f"Eligibility rate: {demographics['eligible'].mean()*100:.2f}%")
print(f"\nIndividuals with prior convictions: {demographics['has_prior_convictions'].sum()}")
print(f"Individuals without prior convictions: {(~demographics['has_prior_convictions']).sum()}")

print("\n" + "="*70)
print("RULE PASS RATES")
print("="*70)
print(f"Passes r_2 (20+ year sentence): {demographics['passes_r2'].sum()} ({demographics['passes_r2'].mean()*100:.2f}%)")
print(f"Passes r_3 (10+ years served): {demographics['passes_r3'].sum()} ({demographics['passes_r3'].mean()*100:.2f}%)")
print(f"Passes r_4 (no current violent): {demographics['passes_r4'].sum()} ({demographics['passes_r4'].mean()*100:.2f}%)")
print(f"Passes r_5 (no prior violent): {demographics['passes_r5'].sum()} ({demographics['passes_r5'].mean()*100:.2f}%)")

contingency_table = pd.crosstab(
    demographics['has_prior_convictions'],
    demographics['eligible'],
    margins=True
)

print("\n" + "="*70)
print("CONTINGENCY TABLE: Prior Convictions vs Eligibility")
print("="*70)
print(contingency_table)

chi2, p_value, dof, expected = chi2_contingency(contingency_table.iloc[:-1, :-1])

print("\n" + "="*70)
print("CHI-SQUARED TEST RESULTS")
print("="*70)
print(f"Chi-squared statistic: {chi2:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Degrees of freedom: {dof}")

if p_value < 0.05:
    print("\n✓ SIGNIFICANT: There IS a significant association between prior convictions")
    print("  and eligibility (p < 0.05)")
else:
    print("\n✗ NOT SIGNIFICANT: There is NO significant association between prior convictions")
    print("  and eligibility (p >= 0.05)")

print("\n" + "="*70)
print("EXPECTED FREQUENCIES")
print("="*70)
expected_df = pd.DataFrame(
    expected, 
    index=['No Prior Convictions', 'Has Prior Convictions'],
    columns=['Not Eligible', 'Eligible']
)
print(expected_df)

eligible_individuals = demographics[demographics['eligible'] == True]
eligible_individuals.to_csv('eligible.csv', index=False)

demographics.to_csv('demographics_with_eligibility.csv', index=False)

print("\n" + "="*70)
print("OUTPUT FILES")
print("="*70)
print(f"All individuals with eligibility flags: demographics_with_eligibility.csv")
print(f"Eligible individuals only: eligible.csv ({len(eligible_individuals)} individuals)")
print("="*70)
