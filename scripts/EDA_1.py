"""
EDA - Demographics Table (demogr)
Using ACTUAL columns from the dataset (excluding 'id')
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Load data
from load_data import load_data

df_demogr, _, _ = load_data()  # Only need demographics

if df_demogr is None:
    print("Failed to load data. Exiting.")
    exit()

# Drop 'id' column if it exists (not needed)
if 'id' in df_demogr.columns:
    df_demogr = df_demogr.drop('id', axis=1)

print("\n" + "="*80)
print("DEMOGRAPHICS TABLE - EXPLORATORY DATA ANALYSIS")
print("="*80)

# ==============================================================================
# BASIC OVERVIEW
# ==============================================================================

print("\n--- DATASET OVERVIEW ---")
print(f"Total records: {len(df_demogr):,}")
print(f"Unique individuals (cdcno): {df_demogr['cdcno'].nunique():,}")
print(f"Total columns: {len(df_demogr.columns)}")
print(f"Memory usage: {df_demogr.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n--- COLUMNS IN DATASET ---")
for i, col in enumerate(df_demogr.columns, 1):
    print(f"{i:2}. {col}")

# ==============================================================================
# MISSING VALUES ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("MISSING VALUES ANALYSIS")
print("="*80)

missing = df_demogr.isnull().sum()
missing_pct = (missing / len(df_demogr) * 100).round(2)

missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Missing_Percent': missing_pct.values
}).sort_values('Missing_Count', ascending=False)

# Only show columns with missing values
missing_with_nulls = missing_df[missing_df['Missing_Count'] > 0]

if len(missing_with_nulls) > 0:
    print(f"\nColumns with missing values:")
    print(missing_with_nulls.to_string(index=False))
else:
    print("\n✓ No missing values found!")

# ==============================================================================
# ETHNICITY ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("ETHNICITY ANALYSIS")
print("="*80)

ethnicity_counts = df_demogr['ethnicity'].value_counts()

print(f"\nEthnicity Distribution:")
for ethnicity, count in ethnicity_counts.items():
    pct = (count / len(df_demogr)) * 100
    print(f"  {ethnicity}: {count:,} ({pct:.1f}%)")

print(f"\nTotal unique ethnicities: {df_demogr['ethnicity'].nunique()}")

# ==============================================================================
# OFFENSE CATEGORY ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("OFFENSE CATEGORY ANALYSIS")
print("="*80)

offense_counts = df_demogr['offense category'].value_counts()

print(f"\nTop 10 Offense Categories:")
for i, (offense, count) in enumerate(offense_counts.head(10).items(), 1):
    pct = (count / len(df_demogr)) * 100
    print(f"  {i:2}. {offense}: {count:,} ({pct:.1f}%)")

print(f"\nTotal unique offense categories: {df_demogr['offense category'].nunique()}")

# ==============================================================================
# SENTENCE LENGTH ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("SENTENCE LENGTH ANALYSIS")
print("="*80)

# Sentence in months
valid_months = df_demogr['aggregate sentence in months'].dropna()
valid_months = valid_months[valid_months > 0]

if len(valid_months) > 0:
    sentence_stats = valid_months.describe()
    
    print(f"\nSentence Length Statistics (months):")
    print(f"  Count: {sentence_stats['count']:,.0f}")
    print(f"  Mean: {sentence_stats['mean']:.1f} months ({sentence_stats['mean']/12:.1f} years)")
    print(f"  Median: {sentence_stats['50%']:.1f} months ({sentence_stats['50%']/12:.1f} years)")
    print(f"  Std Dev: {sentence_stats['std']:.1f} months")
    print(f"  Min: {sentence_stats['min']:.0f} months")
    print(f"  25th percentile: {sentence_stats['25%']:.1f} months")
    print(f"  75th percentile: {sentence_stats['75%']:.1f} months")
    print(f"  Max: {sentence_stats['max']:.0f} months ({sentence_stats['max']/12:.1f} years)")

# Sentence in years
valid_years = df_demogr['aggregate sentence in years'].dropna()
valid_years = valid_years[valid_years > 0]

if len(valid_years) > 0:
    years_stats = valid_years.describe()
    
    print(f"\nSentence Length Statistics (years):")
    print(f"  Count: {years_stats['count']:,.0f}")
    print(f"  Mean: {years_stats['mean']:.2f} years")
    print(f"  Median: {years_stats['50%']:.2f} years")
    print(f"  Min: {years_stats['min']:.2f} years")
    print(f"  Max: {years_stats['max']:.2f} years")

# ==============================================================================
# TIME SERVED ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("TIME SERVED ANALYSIS")
print("="*80)

valid_time = df_demogr['time served in years'].dropna()
valid_time = valid_time[valid_time >= 0]

if len(valid_time) > 0:
    time_stats = valid_time.describe()
    
    print(f"\nTime Served Statistics (years):")
    print(f"  Count: {time_stats['count']:,.0f}")
    print(f"  Mean: {time_stats['mean']:.2f} years")
    print(f"  Median: {time_stats['50%']:.2f} years")
    print(f"  Min: {time_stats['min']:.2f} years")
    print(f"  Max: {time_stats['max']:.2f} years")
    
    # Compare sentence vs time served
    print(f"\n--- Sentence vs Time Served ---")
    
    # Get records with both sentence and time served
    both_available = df_demogr[
        (df_demogr['aggregate sentence in years'].notna()) & 
        (df_demogr['time served in years'].notna()) &
        (df_demogr['aggregate sentence in years'] > 0)
    ].copy()
    
    if len(both_available) > 0:
        both_available['pct_served'] = (both_available['time served in years'] / 
                                         both_available['aggregate sentence in years'] * 100)
        
        print(f"  Records with both values: {len(both_available):,}")
        print(f"  Average % of sentence served: {both_available['pct_served'].mean():.1f}%")
        print(f"  Median % of sentence served: {both_available['pct_served'].median():.1f}%")

# ==============================================================================
# COUNTY ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("SENTENCING COUNTY ANALYSIS")
print("="*80)

county_counts = df_demogr['controlling case sentencing county'].value_counts()

print(f"\nTop 15 Sentencing Counties:")
for i, (county, count) in enumerate(county_counts.head(15).items(), 1):
    pct = (count / len(df_demogr)) * 100
    print(f"  {i:2}. {county}: {count:,} ({pct:.1f}%)")

print(f"\nTotal unique counties: {df_demogr['controlling case sentencing county'].nunique()}")

# ==============================================================================
# CURRENT LOCATION ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("CURRENT LOCATION ANALYSIS")
print("="*80)

location_counts = df_demogr['current location'].value_counts()

print(f"\nTop 15 Current Locations:")
for i, (location, count) in enumerate(location_counts.head(15).items(), 1):
    pct = (count / len(df_demogr)) * 100
    print(f"  {i:2}. {location}: {count:,} ({pct:.1f}%)")

print(f"\nTotal unique locations: {df_demogr['current location'].nunique()}")

# ==============================================================================
# SENTENCE TYPE ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("SENTENCE TYPE ANALYSIS")
print("="*80)

sentence_type_counts = df_demogr['sentence type'].value_counts()

print(f"\nSentence Type Distribution:")
for sent_type, count in sentence_type_counts.items():
    pct = (count / len(df_demogr)) * 100
    print(f"  {sent_type}: {count:,} ({pct:.1f}%)")

# ==============================================================================
# DATE ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("OFFENSE DATE ANALYSIS")
print("="*80)

# Convert to datetime
df_demogr['offense begin date'] = pd.to_datetime(df_demogr['offense begin date'], errors='coerce')
df_demogr['offense end date'] = pd.to_datetime(df_demogr['offense end date'], errors='coerce')
df_demogr['expected release date'] = pd.to_datetime(df_demogr['expected release date'], errors='coerce')

# Extract year from offense begin date
df_demogr['offense_year'] = df_demogr['offense begin date'].dt.year

valid_years = df_demogr['offense_year'].dropna()
if len(valid_years) > 0:
    print(f"\nOffense Date Range:")
    print(f"  Earliest: {valid_years.min():.0f}")
    print(f"  Latest: {valid_years.max():.0f}")
    
    # Top years
    year_counts = df_demogr['offense_year'].value_counts().sort_index(ascending=False)
    print(f"\nTop 10 Most Recent Years:")
    for year, count in year_counts.head(10).items():
        pct = (count / len(df_demogr)) * 100
        print(f"  {year:.0f}: {count:,} ({pct:.1f}%)")

# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 12))
fig.suptitle('Demographics Table - Comprehensive Analysis', fontsize=18, fontweight='bold')

# 1. Ethnicity Distribution
ax1 = plt.subplot(3, 3, 1)
ethnicity_data = df_demogr['ethnicity'].value_counts()
colors = plt.cm.Set3(range(len(ethnicity_data)))
ax1.pie(ethnicity_data.values, labels=ethnicity_data.index, autopct='%1.1f%%', startangle=90, colors=colors)
ax1.set_title('Ethnicity Distribution', fontsize=12, fontweight='bold')

# 2. Top 10 Offense Categories
ax2 = plt.subplot(3, 3, 2)
offense_data = df_demogr['offense category'].value_counts().head(10)
ax2.barh(range(len(offense_data)), offense_data.values, color='lightgreen', alpha=0.7)
ax2.set_yticks(range(len(offense_data)))
ax2.set_yticklabels(offense_data.index, fontsize=9)
ax2.set_title('Top 10 Offense Categories', fontsize=12, fontweight='bold')
ax2.set_xlabel('Count')
ax2.grid(axis='x', alpha=0.3)

# 3. Sentence Length Distribution (months)
ax3 = plt.subplot(3, 3, 3)
plot_months = valid_months[valid_months < 1000]  # Filter extreme outliers for better viz
ax3.hist(plot_months, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax3.axvline(plot_months.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {plot_months.mean():.0f}')
ax3.axvline(plot_months.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {plot_months.median():.0f}')
ax3.set_title('Sentence Length Distribution (Months)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Months')
ax3.set_ylabel('Count')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Sentence Length Distribution (years)
ax4 = plt.subplot(3, 3, 4)
plot_years = valid_years[valid_years < 50]  # Filter extreme outliers
ax4.hist(plot_years, bins=50, color='plum', edgecolor='black', alpha=0.7)
ax4.set_title('Sentence Length Distribution (Years)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Years')
ax4.set_ylabel('Count')
ax4.grid(alpha=0.3)

# 5. Time Served Distribution
ax5 = plt.subplot(3, 3, 5)
plot_time = valid_time[valid_time < 40]  # Filter extreme outliers
ax5.hist(plot_time, bins=50, color='gold', edgecolor='black', alpha=0.7)
ax5.set_title('Time Served Distribution (Years)', fontsize=12, fontweight='bold')
ax5.set_xlabel('Years')
ax5.set_ylabel('Count')
ax5.grid(alpha=0.3)

# 6. Top 10 Counties
ax6 = plt.subplot(3, 3, 6)
county_data = df_demogr['controlling case sentencing county'].value_counts().head(10)
ax6.barh(range(len(county_data)), county_data.values, color='salmon', alpha=0.7)
ax6.set_yticks(range(len(county_data)))
ax6.set_yticklabels(county_data.index, fontsize=9)
ax6.set_title('Top 10 Sentencing Counties', fontsize=12, fontweight='bold')
ax6.set_xlabel('Count')
ax6.grid(axis='x', alpha=0.3)

# 7. Sentence Type Distribution
ax7 = plt.subplot(3, 3, 7)
sent_type_data = df_demogr['sentence type'].value_counts()
colors = plt.cm.Pastel1(range(len(sent_type_data)))
ax7.pie(sent_type_data.values, labels=sent_type_data.index, autopct='%1.1f%%', startangle=90, colors=colors)
ax7.set_title('Sentence Type Distribution', fontsize=12, fontweight='bold')

# 8. Offenses by Year
ax8 = plt.subplot(3, 3, 8)
if 'offense_year' in df_demogr.columns:
    valid_offense_years = df_demogr['offense_year'].dropna()
    valid_offense_years = valid_offense_years[(valid_offense_years >= 1980) & (valid_offense_years <= 2024)]
    
    if len(valid_offense_years) > 0:
        year_counts = valid_offense_years.value_counts().sort_index()
        ax8.plot(year_counts.index, year_counts.values, marker='o', linewidth=2, markersize=3, color='darkblue')
        ax8.set_title('Number of Offenses by Year', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Year')
        ax8.set_ylabel('Count')
        ax8.grid(alpha=0.3)

# 9. Top 10 Current Locations
ax9 = plt.subplot(3, 3, 9)
location_data = df_demogr['current location'].value_counts().head(10)
ax9.barh(range(len(location_data)), location_data.values, color='lightblue', alpha=0.7)
ax9.set_yticks(range(len(location_data)))
ax9.set_yticklabels(location_data.index, fontsize=9)
ax9.set_title('Top 10 Current Locations', fontsize=12, fontweight='bold')
ax9.set_xlabel('Count')
ax9.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("DEMOGRAPHICS EDA COMPLETE!")
print("="*80)
print("\nKey Findings:")
print(f"  • Total individuals: {df_demogr['cdcno'].nunique():,}")
print(f"  • Total records: {len(df_demogr):,}")

top_ethnicity = df_demogr['ethnicity'].value_counts().index[0]
print(f"  • Most common ethnicity: {top_ethnicity}")

top_offense = df_demogr['offense category'].value_counts().index[0]
print(f"  • Most common offense: {top_offense}")

if len(valid_months) > 0:
    print(f"  • Average sentence: {valid_months.mean():.1f} months ({valid_months.mean()/12:.1f} years)")

top_county = df_demogr['controlling case sentencing county'].value_counts().index[0]
print(f"  • Top sentencing county: {top_county}")

print("\nDataFrame available: df_demogr")
