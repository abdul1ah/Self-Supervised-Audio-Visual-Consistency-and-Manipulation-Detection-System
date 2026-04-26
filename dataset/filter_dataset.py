import pandas as pd

df = pd.read_csv("vggsound.csv")


print(df.head())
print(df.columns)
print(df['label'].value_counts().head(20))

## selecting the classes we want to use for training
selected_classes = [
    "playing electric guitar",
    "playing acoustic guitar",
    "playing piano",
    "playing violin, fiddle",
    "playing cello",
    "playing bass guitar",
    "engine accelerating, revving, vroom",
    "police car (siren)",
    "toilet flushing",
    "bird chirping, tweeting",
    "chicken crowing",
    "male speech, man speaking"
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
