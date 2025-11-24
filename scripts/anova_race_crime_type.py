import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load data
demographics = pd.read_csv('demographics.csv')
current_commits = pd.read_csv('current_commitments.csv')
prior_commits = pd.read_csv('prior_commitments.csv')

# Clean data types
demographics['offense begin date'] = pd.to_datetime(demographics['offense begin date'], errors='coerce')
demographics['aggregate sentence in months'] = pd.to_numeric(demographics['aggregate sentence in months'], errors='coerce')

print("="*70)
print("DATA STRUCTURE CHECK")
print("="*70)
print(f"Demographics columns: {demographics.columns.tolist()}")
print(f"Current commits columns: {current_commits.columns.tolist()}")
print(f"Prior commits columns: {prior_commits.columns.tolist()}")

print("\n" + "="*70)
print("FILTERING FOR CLEAN COMPARISON GROUP")
print("="*70)

# Step 1: Identify people with prior convictions (via cdcno)
prior_conviction_ids = set(prior_commits['cdcno'].unique())
demographics['has_prior_convictions'] = demographics['cdcno'].isin(prior_conviction_ids)

# Step 2: Count current commitments per person (via cdcno)
current_commit_counts = current_commits.groupby('cdcno').size().reset_index(name='num_current_offenses')
demographics = demographics.merge(current_commit_counts, on='cdcno', how='left')
demographics['num_current_offenses'] = demographics['num_current_offenses'].fillna(0)

# Step 3: Filter to first-time offenders with single current offense
first_time_single_offense = demographics[
    (demographics['has_prior_convictions'] == False) & 
    (demographics['num_current_offenses'] == 1)
].copy()

print(f"\nOriginal sample: {len(demographics)}")
print(f"First-time offenders with single offense: {len(first_time_single_offense)}")
print(f"Reduction: {len(demographics) - len(first_time_single_offense)} individuals")

# Step 4: Merge offense information from current_commitments using cdcno
# Get the offense for each person (should be only 1 per person now)
offense_data = current_commits[['cdcno', 'offense']].copy()
first_time_single_offense = first_time_single_offense.merge(offense_data, on='cdcno', how='left')

# Step 5: Clean ethnicity (from demographics) 
# Check what values we have
print("\n" + "="*70)
print("ETHNICITY VALUES IN DEMOGRAPHICS")
print("="*70)
print(first_time_single_offense['ethnicity'].value_counts())

# Filter to main groups (adjust based on what you see above)
first_time_single_offense = first_time_single_offense[
    first_time_single_offense['ethnicity'].isin(['Black', 'White', 'Hispanic'])
].copy()

# Step 6: Categorize offenses
def categorize_offense(offense_code):
    if pd.isna(offense_code):
        return 'Unknown'
    offense_str = str(offense_code).upper()
    
    # Violent crimes (from CRJA paper)
    if any(x in offense_str for x in ['PC187', 'PC261', 'PC286', 'PC288', 'PC289', 
                                       'PC206', 'PC207', 'PC209', 'PC245']):
        return 'Violent'
    # Property crimes
    elif any(x in offense_str for x in ['PC459', 'PC484', 'PC487', 'PC496', 'PC594']):
        return 'Property'
    # Drug crimes
    elif any(x in offense_str for x in ['HS11350', 'HS11351', 'HS11352', 'HS11378', 'HS11379']):
        return 'Drug'
    # Other
    else:
        return 'Other'

first_time_single_offense['offense_category'] = first_time_single_offense['offense'].apply(categorize_offense)

# Remove unknowns
first_time_single_offense = first_time_single_offense[
    first_time_single_offense['offense_category'] != 'Unknown'
].copy()

print("\n" + "="*70)
print("FINAL SAMPLE COMPOSITION")
print("="*70)
print("\nBy Ethnicity:")
print(first_time_single_offense['ethnicity'].value_counts())
print("\nBy Offense Category:")
print(first_time_single_offense['offense_category'].value_counts())
print("\nEthnicity × Offense Category:")
print(pd.crosstab(first_time_single_offense['ethnicity'], first_time_single_offense['offense_category']))

# Step 7: Check if we have enough data in each cell
print("\n" + "="*70)
print("CELL SIZES (need at least 5-10 per cell for ANOVA)")
print("="*70)
cell_counts = pd.crosstab(first_time_single_offense['ethnicity'], first_time_single_offense['offense_category'])
print(cell_counts)

# Check for cells with too few observations
min_cell_size = cell_counts.min().min()
if min_cell_size < 5:
    print(f"\n⚠️  WARNING: Smallest cell has only {min_cell_size} observations")
    print("   Consider combining categories or using different groupings")

# Remove sentence outliers for better visualization
q1 = first_time_single_offense['aggregate sentence in months'].quantile(0.01)
q99 = first_time_single_offense['aggregate sentence in months'].quantile(0.99)
analysis_data = first_time_single_offense[
    (first_time_single_offense['aggregate sentence in months'] >= q1) &
    (first_time_single_offense['aggregate sentence in months'] <= q99) &
    (first_time_single_offense['aggregate sentence in months'].notna())
].copy()

print(f"\nAfter removing extreme outliers and missing values: {len(analysis_data)} individuals")

# Step 8: Run Two-Way ANOVA with Interaction
print("\n" + "="*70)
print("TWO-WAY ANOVA: Ethnicity × Offense Category")
print("="*70)
print("Testing for racial disparities in sentencing for first-time offenders")
print("="*70)

# Create the model
# C() tells statsmodels these are categorical variables
model = ols('Q("aggregate sentence in months") ~ C(ethnicity) + C(offense_category) + C(ethnicity):C(offense_category)', 
            data=analysis_data).fit()

# ANOVA table (Type II)
anova_table = sm.stats.anova_lm(model, typ=2)
print("\n", anova_table)

# Step 9: Interpret results
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

effects = {
    'C(ethnicity)': 'Main effect of ETHNICITY (do sentences differ by race overall?)',
    'C(offense_category)': 'Main effect of OFFENSE TYPE (do sentences differ by crime type?)',
    'C(ethnicity):C(offense_category)': 'INTERACTION (does racial disparity depend on crime type?)'
}

for index, row in anova_table.iterrows():
    if index in effects:
        print(f"\n{effects[index]}")
        print(f"  F-statistic: {row['F']:.4f}")
        print(f"  P-value: {row['PR(>F)']:.6f}")
        
        if row['PR(>F)'] < 0.001:
            sig = "*** HIGHLY SIGNIFICANT - Strong evidence of disparity"
        elif row['PR(>F)'] < 0.01:
            sig = "** SIGNIFICANT - Moderate evidence of disparity"
        elif row['PR(>F)'] < 0.05:
            sig = "* SIGNIFICANT - Some evidence of disparity"
        else:
            sig = "NOT SIGNIFICANT - No statistical evidence of disparity"
        
        print(f"  Result: {sig}")

# Step 10: Descriptive statistics by group
print("\n" + "="*70)
print("MEAN SENTENCE LENGTH BY ETHNICITY AND OFFENSE TYPE")
print("="*70)
summary_stats = analysis_data.groupby(['ethnicity', 'offense_category'])['aggregate sentence in months'].agg([
    ('Mean', 'mean'),
    ('Median', 'median'),
    ('Std Dev', 'std'),
    ('Count', 'count')
]).round(2)
print("\n", summary_stats)

# Step 11: Pairwise comparisons (if main effect is significant)
if anova_table.loc['C(ethnicity)', 'PR(>F)'] < 0.05:
    print("\n" + "="*70)
    print("POST-HOC PAIRWISE COMPARISONS (Tukey HSD)")
    print("="*70)
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    
    tukey = pairwise_tukeyhsd(endog=analysis_data['aggregate sentence in months'],
                              groups=analysis_data['ethnicity'],
                              alpha=0.05)
    print(tukey)

# Step 12: Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
sns.boxplot(data=analysis_data, x='offense_category', y='aggregate sentence in months', 
            hue='ethnicity', ax=axes[0], palette='Set2')
axes[0].set_title('Sentence Length by Offense Type and Ethnicity\n(First-Time Offenders Only)', 
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Offense Category', fontsize=11)
axes[0].set_ylabel('Sentence (months)', fontsize=11)
axes[0].legend(title='Ethnicity')

# Interaction plot
ethnicity_order = ['White', 'Hispanic', 'Black']  # Order for better visualization
colors = {'White': 'blue', 'Hispanic': 'green', 'Black': 'red'}

for ethnicity in ethnicity_order:
    if ethnicity in analysis_data['ethnicity'].unique():
        race_data = analysis_data[analysis_data['ethnicity'] == ethnicity]
        means = race_data.groupby('offense_category')['aggregate sentence in months'].mean()
        axes[1].plot(means.index, means.values, marker='o', label=ethnicity, 
                    linewidth=2, markersize=8, color=colors.get(ethnicity))

axes[1].set_title('Interaction Plot: Ethnicity × Offense Category\n(Parallel lines = no interaction)', 
                 fontsize=12, fontweight='bold')
axes[1].set_xlabel('Offense Category', fontsize=11)
axes[1].set_ylabel('Mean Sentence (months)', fontsize=11)
axes[1].legend(title='Ethnicity')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('anova_ethnicity_offense_interaction.png', dpi=300, bbox_inches='tight')
print("\n" + "="*70)
print("Visualization saved: anova_ethnicity_offense_interaction.png")
print("="*70)

# Save analysis data
analysis_data.to_csv('anova_analysis_data.csv', index=False)
print("Analysis dataset saved: anova_analysis_data.csv")
print("="*70)

plt.show()
