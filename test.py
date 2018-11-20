import mslice.cli as mc
import matplotlib.pyplot as plt

fig, ax = plt.subplots(subplot_kw={'projection': 'mslice'})
ws = mc.Load("C:/Users/qzp95127/Documents/Work/mslice_test_data/MAR21335_Ei60.00meV.nxs")
cut_ws = mc.Cut(ws)
slice_ws = mc.Slice(ws)
#p1 = ax.plot(cut_ws)
#p2 = ax.pcolormesh(slice_ws)
f1 = mc.PlotCut(cut_ws)
f2 = mc.PlotSlice(slice_ws)
mc.show()
# p3 = ax.plot([0,1,2,3])
# plt.show()
