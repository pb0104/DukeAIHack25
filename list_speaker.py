import re
from collections import Counter

# Path to your transcript file
file_path = "speaker.txt"

# Read the file
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Regex to match speaker names inside [ ... :]
pattern = r"\[\s*([^\]]+?)\s*:\]"

# Find all matches
speakers = re.findall(pattern, text)

# Count occurrences
speaker_counts = Counter(speakers)

# Print results nicely
print("Speaker List and Counts:\n")
for speaker, count in speaker_counts.items():
    print(f"{speaker}: {count}")

# Optional: Get just the list of unique speakers
unique_speakers = list(speaker_counts.keys())
print("\nUnique Speakers:\n", unique_speakers)
