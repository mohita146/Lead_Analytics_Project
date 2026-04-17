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


# Lead Segmentation
df['Lead_Category'] = pd.cut(df['Lead_Score'],
                            bins=[0,40,70,100],
                            labels=['Low','Medium','High'])

segment_analysis = df.groupby('Lead_Category')['Conversion'].apply(lambda x: (x=='Yes').mean())
print("\nLead Segmentation:\n", segment_analysis)


# Funnel Analysis
funnel = df['Lead_Status'].value_counts()
print("\nLead Funnel:\n", funnel)
funnel.plot(kind='bar')
plt.title('Lead Funnel Analysis')
plt.savefig('visuals/funnel.png')
plt.show()


# KPI by Source
kpi_source = df.groupby('Source').agg({
    'Lead_ID':'count',
    'Conversion': lambda x: (x=='Yes').mean()
})
print("\nKPI by Source:\n", kpi_source)


# Correlation Analysis
numeric_df = df[['Lead_Score','Response_Time','Follow_Up_Count']]
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Analysis')
plt.savefig('visuals/correlation.png')
plt.show()


sns.countplot(x='Source', hue='Conversion', data=df)
plt.title('Source vs Conversion Distribution')
plt.xticks(rotation=45)
plt.savefig('visuals/source_distribution.png')
plt.show()


df['Response_Category'] = pd.cut(df['Response_Time'],
                                bins=[0,10,24,50],
                                labels=['Fast','Medium','Slow'])
response_analysis = df.groupby('Response_Category')['Conversion'].apply(lambda x: (x=='Yes').mean())
print("\nResponse Category:\n", response_analysis)


high_quality = df[df['Lead_Score'] > 70]
if len(high_quality) > 0:
    high_conversion = len(high_quality[high_quality['Conversion']=='Yes']) / len(high_quality)
else:
    high_conversion = 0
print("\nHigh Quality Lead Conversion Rate:", high_conversion)


print("\n BUSINESS INSIGHTS ")
best_source = kpi_source['Conversion'].idxmax()
print("Best Performing Source:", best_source)
best_response = response_analysis.idxmax()
print("Best Response Category:", best_response)
top_country = country_conversion.idxmax()
print("Top Performing Country:", top_country)


print("\n PROJECT SUMMARY ")
print(f"Total Leads: {len(df)}")
print(f"Overall Conversion Rate: {(df['Conversion']=='Yes').mean():.2f}")
print("Top Source:", best_source)
print("Best Response Category:", best_response)