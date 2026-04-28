import pandas as pd

df = pd.read_csv("vggsound.csv")


print(df.head())
print(df.columns)
print(df['label'].value_counts().head(20))


selected_classes = [
    "male speech, man speaking",
    "female speech, woman speaking",
    "hammering nails",
    "chopping wood",
    "people clapping",
    "basketball bounce",
    "door slamming",
    "typing on computer keyboard",
    "playing piano",
    "playing acoustic guitar",
    "playing drum kit",
    "playing tennis" 
]

## filtering the dataset to include only the selected classes and 
## sampling 100 examples from each class
filtered_df = (
    df[df['label'].isin(selected_classes)]
    .groupby('label', group_keys=False)
    .apply(lambda x: x.sample(n=100, random_state=42))
)

filtered_df.to_csv("filtered_vggsound.csv", index=False)

## displaying the value counts for each class in the filtered dataset
print(filtered_df['label'].value_counts())
