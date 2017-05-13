import pandas as pd
import matplotlib.pyplot as plt

g100 = pd.DataFrame.from_csv("../Data/band-ratio-G100.dat",header=1)
#g100
g1000 = pd.DataFrame.from_csv("../Data/band-ratio-G1000.dat",header=1, index_col=None)
#g1000
g10000 = pd.DataFrame.from_csv("../Data/band-ratio-G10000.dat",header=1, index_col=None)
#g10000

def plot_ifrac(df,filename,g0):

    f = plt.figure()
    df[df.columns[-5:-1]].plot(linewidth=2.0)
    plt.ylabel("Band Ratio",fontsize=20)
    plt.xlabel("PAH Ionization Fraction",fontsize=20)
    plt.title("$G_{0}$ = "+str(g0),fontsize=20)
    f.savefig("Plots/"+filename)
    plt.show()
    plt.close()


plot_ifrac(g100, "band-ratio-G100.pdf", 100)
plot_ifrac(g1000, "band-ratio-G1000.pdf",1000)
plot_ifrac(g10000, "band-ratio-G10000.pdf",10000)
