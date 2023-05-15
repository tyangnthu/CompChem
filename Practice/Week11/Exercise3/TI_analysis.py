import pandas as pd
import sys

# Make sure that you change the temp to match that of your system
temp = 300
cf = temp * 8.314 / 4184 # kT to kcal.mol^-1

f = sys.argv[1]
lines = open(f).read().splitlines()
data = [list(filter(None,line.split(' '))) for line in lines[17:]]
df = pd.DataFrame(data, columns=['lambda', 'DG', 's'], dtype=float)
DG_sum = df.DG.sum() * cf
s_sum = df.s.sum() * cf
print('DG: %.4f kcal/mol' %(DG_sum))
print('s: %.4f kcal/mol' %(s_sum))
