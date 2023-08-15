import pandas as pd

data = [
    ["Monday", 0.2],
    ["Tuesday", 0.1],
    ["Wednesday", 0.0],
    ["Thursday", 0.5],
    ["Friday", 1.3],
    ["Saturday", 0.0],
    ["Sunday", 0.7],
]

df_rain = pd.DataFrame(data, columns=["DayOfTheWeek", "Rainfall(inches)"])

# Write a function to return all days on which it did not rain

# write a function to return the median rainfall for days on which it rained
