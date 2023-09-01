# 0:
#   0.0:convert interaction matrix to df of index drug and target and type of interaction:
#==================================================
import pandas as pd
#=========================================================
df_interactions = pd.read_csv('../../Preprocessing/interaction/interactions.csv',sep='\t',header = None)
print(df_interactions)
#==============================================================
# convert:
df_interactions['drug_id'] = df_interactions.index
print(df_interactions)

df_interactions_id = pd.melt(df_interactions, id_vars=["drug_id"])
print(df_interactions_id)
df_interactions_id.rename(columns={'variable': 'target_id', 'value':'interaction'},inplace = True)

print(df_interactions_id)
#=============================================================================
# write to csv:
df_interactions_id.to_csv("df.csv", index=False)