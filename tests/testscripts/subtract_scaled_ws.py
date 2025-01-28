import mslice.cli as mc
import mslice.plotting.pyplot as plt

MAR21335_Ei60meV = mc.Load(
    Filename=r"tests/testdata/MAR21335_Ei60meV.nxs", OutputWorkspace="MAR21335_Ei60meV"
)
MAR21335_Ei60meV_scaled = mc.Scale(
    InputWorkspace=MAR21335_Ei60meV, OutputWorkspace="MAR21335_Ei60meV_scaled"
)
MAR21335_Ei60meV_scaled_scaled = mc.Scale(
    InputWorkspace=MAR21335_Ei60meV_scaled, store=False
)
ws_MAR21335_Ei60meV_subtracted = mc.Minus(
    LHSWorkspace=MAR21335_Ei60meV,
    RHSWorkspace=MAR21335_Ei60meV_scaled_scaled,
    OutputWorkspace="MAR21335_Ei60meV_subtracted",
)

fig = plt.gcf()
fig.clf()
ax = fig.add_subplot(111, projection="mslice")
slice_ws = mc.Slice(
    ws_MAR21335_Ei60meV_subtracted,
    Axis1="|Q|,0.27356,11.0351,0.04671",
    Axis2="DeltaE,-30.0,58.2,0.3",
    NormToOne=False,
)

mesh = ax.pcolormesh(slice_ws, cmap="viridis")
mesh.set_clim(-0.1, 0.1)
cb = plt.colorbar(mesh, ax=ax)
cb.set_label(
    "Intensity (arb. units)", labelpad=20, rotation=270, picker=5, fontsize=10.0
)
ax.set_title("MAR21335_Ei60meV_subtracted", fontsize=12.0)
