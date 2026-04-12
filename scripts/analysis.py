import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data/cleaned_leads_dataset.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Preview data
print(df.head())

# Conversion Rate by Source
conversion_rate = df.groupby('Source')['Conversion'].apply(lambda x: (x=='Yes').mean())

print("\nConversion Rate by Source:\n", conversion_rate)

# Plot
conversion_rate.plot(kind='bar')
plt.title('Conversion Rate by Source')
plt.ylabel('Conversion Rate')
plt.xticks(rotation=45)

# Save graph
plt.savefig('visuals/source_conversion.png')

plt.show()


# Conversion rate by Country
country_conversion = df.groupby('Country')['Conversion'].apply(lambda x: (x=='Yes').mean())

print("\nConversion Rate by Country:\n", country_conversion)

# Plot
country_conversion.plot(kind='bar', color='orange')
plt.title('Conversion Rate by Country')
plt.ylabel('Conversion Rate')

# Save
plt.savefig('visuals/country_conversion.png')

plt.show()


# Response time vs Conversion
df_filtered = df[df['Conversion'] != 'Unknown']
sns.boxplot(x='Conversion', y='Response_Time', data=df_filtered)
plt.title('Response Time vs Conversion')

# Save
plt.savefig('visuals/response_time.png')

plt.show()