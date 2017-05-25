import pandas as pd
import matplotlib.pyplot as plt

g100 = pd.DataFrame.from_csv("Data/band-ratio-G100.dat",header=1)
#g100
g1000 = pd.DataFrame.from_csv("Data/band-ratio-G1000.dat",header=1, index_col=None)
#g1000
g10000 = pd.DataFrame.from_csv("Data/band-ratio-G10000.dat",header=1, index_col=None)
#g10000

def plot_ifrac(df,filename,g0):

    f = plt.figure()
    df[df.columns[-5:-1]].plot(ax=f.gca(),linewidth=2.0)
    plt.ylabel("Band Ratio",fontsize=20)
    plt.xlabel("PAH Ionization Fraction",fontsize=20)
    plt.xlim(0,1)
    plt.title("$G_{0}$ = "+str(g0),fontsize=20)
    f.show()
    f.savefig("Plots/"+filename)
    #plt.close()


plot_ifrac(g100, "band-ratio-G100.pdf", 100)
plot_ifrac(g1000, "band-ratio-G1000.pdf",1000)
plot_ifrac(g10000, "band-ratio-G10000.pdf",10000)

def plot_big_ifrac(df1,df2,df3,filename):

    fig = plt.figure()

    plt.ylabel("Band Ratio")

    ax1 = plt.subplot(311)
    df1[df1.columns[-5:-1]].plot(ax=ax1,linewidth=2.0, )
    plt.setp(ax1.get_xticklabels(), fontsize=6)

    ax2 = plt.subplot(312, sharex=ax1)
    plt.setp(ax2.get_xticklabels(), visible=False)
    df2[df2.columns[-5:-1]].plot(ax=ax2,legend=None, linewidth=2.0)


    ax3 = plt.subplot(313, sharex= ax1)
    df3[df3.columns[-5:-1]].plot(ax=ax3, legend=None, linewidth=2.0)
    plt.xlim(0,1)
    plt.xlabel("PAH Ionization Fraction",fontsize=20)



    fig.show()
    fig.savefig("Plots/"+filename)

plot_big_ifrac(g100, g1000, g10000, "band-ratio-multiple.pdf")
