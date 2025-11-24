import pandas as pd


def clean_offense_code(code):
    """Strip PC/VC prefixes and (a)(1) suffixes"""
    if pd.isna(code):
        return code

    code = str(code).strip().upper()

    # Remove prefixes
    prefixes = ['PC', 'VC', 'HS', 'BP', 'WI', 'CC']
    for prefix in prefixes:
        if code.startswith(prefix):
            code = code[len(prefix):]
            break

    # Remove suffixes (parentheses)
    if '(' in code:
        code = code.split('(')[0]

    return code.strip()


# Load data
print("Loading data...")
current = pd.read_csv("C:\\Users\\gandh\\PycharmProjects\\PythonProject\\data\\current_commitments.csv")
prior = pd.read_csv("C:\\Users\\gandh\\PycharmProjects\\PythonProject\\data\\prior_commitments.csv")

# Clean offense codes
print("Cleaning offense codes...")
current['offense_clean'] = current['offense'].apply(clean_offense_code)
prior['offense_clean'] = prior['offense'].apply(clean_offense_code)

# Save cleaned versions
print("Saving cleaned data...")
current.to_csv("C:\\Users\\gandh\\PycharmProjects\\PythonProject\\data\\current_commitments_clean.csv", index=False)
prior.to_csv("C:\\Users\\gandh\\PycharmProjects\\PythonProject\\data\\prior_commitments_clean.csv", index=False)

