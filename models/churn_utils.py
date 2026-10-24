FINAL_COLUMNS = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCreditCard', 'IsActive', 'Salary',
                 'ProductsRisk', 'ZeroBalance', 'AgeInactive', 'TenureAgeRatio', 'BalanceSalaryRatio',
                 'Country_Germany', 'Country_Spain', 'Gender_Male']

def engineer_features(df):
    df = df.copy()
    df['ProductsRisk'] = (df['NumOfProducts'] != 2).astype(int)
    df['ZeroBalance'] = (df['Balance'] == 0).astype(int)
    df['AgeInactive'] = df['Age'] * (1 - df['IsActive'])
    df['TenureAgeRatio'] = df['Tenure'] / df['Age']
    df['BalanceSalaryRatio'] = df['Balance'] / df['Salary']
    df['Country_Germany'] = (df['Country'] == 'Germany').astype(int)
    df['Country_Spain'] = (df['Country'] == 'Spain').astype(int)
    df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)

    df = df.drop(columns=['Country', 'Gender'])
    return df[FINAL_COLUMNS]
