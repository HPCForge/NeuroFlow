import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./out.csv")
print(df)

x = df['x']
y = df['y']

anx = x[:14425]
anyy = y[:14425]

bx = x[14425:(14425+7007)]
by = y[14425:(14425+7007)]

ebx = x[(14425+7007):(14425+7007+988)]
eby = y[(14425+7007):(14425+7007+988)]

sx = x[(14425+7007+988):(14425+7007+988+4864)]
sy = y[(14425+7007+988):(14425+7007+988+4864)]


ssx = x[(14425+7007+988+4864):(14425+7007+988+4864+2976)]
ssy = y[(14425+7007+988+4864):(14425+7007+988+4864+2976)]

swx = x[(14425+7007+988+4864+2976):(14425+7007+988+4864+2976+8162)]
swy = y[(14425+7007+988+4864+2976):(14425+7007+988+4864+2976+8162)]

ux = x[(14425+7007+988+4864+2976+8162):(14425+7007+988+4864+2976+8162+1976)]
uy = y[(14425+7007+988+4864+2976+8162):(14425+7007+988+4864+2976+8162+1976)]



print(df[:14425])
print(df[(14425):(14425+7007)])
print(df[(14425+7007):(14425+7007+988)])
print(df[(14425+7007+988):(14425+7007+988+4864)])
print(df[(14425+7007+988+4864):(14425+7007+988+4864+2976)])
print(df[(14425+7007+988+4864+2976):(14425+7007+988+4864+2976+8162)])
print(df[(14425+7007+988+4864+2976):(14425+7007+988+4864+2976+8162+1976)])

plt.figure(figsize= (15,15))
plt.title("X vs Y Signal-To-Noise Ratio for 7 Class Classification", fontsize=20)
plt.xlabel("X Signal-To-Noise Ratio", fontsize=20)
plt.ylabel("Y Signal-To-Noise Ratio", fontsize=20)
plt.scatter(anx,anyy, s=5,alpha=0.5)
plt.scatter(ssx,ssy, s=5,alpha=0.5)
plt.scatter(swx,swy, s=5,alpha=0.5)
plt.scatter(sx,sy, s=5,alpha=0.5)
plt.scatter(ux,uy, s=5,alpha=0.5)
plt.scatter(bx,by, s=5,alpha=0.5)
plt.scatter(ebx,eby, s=5,alpha=0.5)

plt.legend(["Annular", "Stratified Smooth", "Stratified Wavy", "Slug", "Unstable", "Bubbly", "Elongated Bubbly"], markerscale=4.0, prop={'size': 15}, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=7)
plt.show(block=True)
