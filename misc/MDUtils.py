import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mantid.simpleapi import ConvertToMD, CutMD, Load, SetUB, mtd

def dim2array(d):
    """
    Create a numpy array containing bin centers along the dimension d
    input: d - IMDDimension
    return: numpy array, from min+st/2 to max-st/2 with step st  
    """
    dmin = d.getMinimum()
    dmax = d.getMaximum()
    dstep = d.getX(1)-d.getX(0)
    return np.arange(dmin+dstep/2,dmax,dstep)
    

def SaveMDToAscii(ws,filename,IgnoreIntegrated=True,NumEvNorm=False,Format='%.6e'):
    """
    Save an MDHistoToWorkspace to an ascii file (column format)
    input: ws - handle to the workspace
    input: filename - path to the output filename
    input: IgnoreIntegrated - if True, the integrated dimensions are ignored (smaller files), but that information is lost
    input: NumEvNorm - must be set to true if data was converted to MD from a histo workspace (like NXSPE) and no MDNorm... algorithms were used
    input: Format - value to pass to numpy.savetxt
    return: nothing
    """
    if ws.id() != 'MDHistoWorkspace':
        raise ValueError("The workspace is not an MDHistoToWorkspace")
    #get dimensions
    if IgnoreIntegrated:
        dims = ws.getNonIntegratedDimensions()
    else:
        dims = [ws.getDimension(i) for i in range(ws.getNumDims())]
    dimarrays = [dim2array(d) for d in dims]
    newdimarrays = np.meshgrid(*dimarrays,indexing='ij')
    #get data
    data = ws.getSignalArray()*1.
    err2 = ws.getErrorSquaredArray()*1.
    if NumEvNorm:
        nev = ws.getNumEventsArray()
        data /= nev
        err2 /= nev
    err = np.sqrt(err2)
    #write file
    header = "Intensity Error "+" ".join([d.getName() for d in dims])
    header += "\n shape: "+"x".join([str(d.getNBins()) for d in dims])
    toPrint = np.c_[data.flatten(),err.flatten()]
    for d in newdimarrays:
        toPrint = np.c_[toPrint,d.flatten()]
    np.savetxt(filename,toPrint,fmt=Format,header=header)
    

def Plot1DMD(ax,ws,PlotErrors=True,NumEvNorm=False,**kwargs):
    """
    Plot a 1D curve from an MDHistoWorkspace (assume all other dimensions are 1)
    input: ax - axis object
    input: ws - handle to the workspace
    input: PlotErrors - if True, will show error bars
    input: NumEvNorm - must be set to true if data was converted to MD from a histo workspace (like NXSPE) and no MDNorm... algorithms were used
    input: kwargs - arguments that are passed to plot, such as plotting symbol, color, label, etc.
    """
    dims = ws.getNonIntegratedDimensions()
    if len(dims) != 1:
        raise ValueError("The workspace dimensionality is not 1")
    dim = dims[0]
    x = dim2array(dim)
    y = ws.getSignalArray()*1.
    err2 = ws.getErrorSquaredArray()*1.
    if NumEvNorm:
        nev = ws.getNumEventsArray()
        y /= nev
        err2 /= (nev*nev)
    err = np.sqrt(err2)
    y = y.flatten()
    err = err.flatten()
    if PlotErrors:
        pp = ax.errorbar(x,y,yerr=err,**kwargs)
    else:
        pp = ax.plot(x,y,**kwargs)
    ax.set_xlabel(dim.getName())
    ax.set_ylabel("Intensity")
    return pp

def Plot2DMD(ax,ws,NumEvNorm=False,**kwargs):
    """
    Plot a 2D slice from an MDHistoWorkspace (assume all other dimensions are 1)
    input: ax - axis object
    input: ws - handle to the workspace
    input: NumEvNorm - must be set to true if data was converted to MD from a histo workspace (like NXSPE) and no MDNorm... algorithms were used
    input: kwargs - arguments that are passed to plot, such as plotting symbol, color, label, etc.
    """
    dims = ws.getNonIntegratedDimensions()
    if len(dims) != 2:
        raise ValueError("The workspace dimensionality is not 2")
    dimx = dims[0]
    xstep = dimx.getX(1)-dimx.getX(0)
    x = np.arange(dimx.getMinimum(),dimx.getMaximum()+xstep/2,xstep)
    dimy = dims[1]
    ystep = dimy.getX(1)-dimy.getX(0)
    y = np.arange(dimy.getMinimum(),dimy.getMaximum()+ystep/2,ystep)
    intensity = ws.getSignalArray()*1.
    if NumEvNorm:
        nev = ws.getNumEventsArray()
        intensity /= nev
    intensity = intensity.squeeze()
    intensity = np.ma.masked_where(np.isnan(intensity),intensity)
    XX,YY = np.meshgrid(x,y,indexing='ij')
    pcm = ax.pcolormesh(XX,YY,intensity,**kwargs)
    ax.set_xlabel(dimx.getName())
    ax.set_ylabel(dimy.getName())
    return pcm



def example_plots():
    #create slices and cuts
    w = Load(Filename='/SNS/HYS/IPTS-14189/shared/autoreduce/4pixel/HYS_102102_4pixel_spe.nxs')
    SetUB(w,4.53,4.53,11.2,90,90,90,"1,0,0","0,0,1")
    mde = ConvertToMD(w, QDimensions='Q3D')       
    sl1d = CutMD(InputWorkspace=mde, P1Bin='-5,5', P2Bin='-5,5', P3Bin='2,4', P4Bin='-10,0.5,15', NoPix=True)
    sl2d = CutMD(InputWorkspace=mde, P1Bin='-5,5', P2Bin='-5,5', P3Bin='-5,0.1,5', P4Bin='-10,1,15', NoPix=True)
    
    #2 subplots per page
    fig,ax = plt.subplots(2,1)
    Plot1DMD(ax[0],sl1d,NumEvNorm=True,fmt='ro')
    ax[0].set_ylabel("Int(a.u.)")
    pcm = Plot2DMD(ax[1],sl2d,NumEvNorm=True)
    fig.colorbar(pcm,ax=ax[1])
    
    #save to png
    plt.tight_layout(1.08)
    fig.savefig('/tmp/test.png')

def example_pdf():    
    #create slices and cuts
    w = Load(Filename='/SNS/HYS/IPTS-14189/shared/autoreduce/4pixel/HYS_102102_4pixel_spe.nxs')
    SetUB(w,4.53,4.53,11.2,90,90,90,"1,0,0","0,0,1")
    mde = ConvertToMD(w, QDimensions='Q3D')       
    with PdfPages('/tmp/multipage_pdf.pdf') as pdf:
        for i in range(4):
            llims = str(i-0.5)+','+str(i+0.5)
            sl1d = CutMD(InputWorkspace=mde, P1Bin='-5,5', P2Bin='-5,5', P3Bin=llims, P4Bin='-10,0.5,15', NoPix=True)
            fig,ax = plt.subplots()
            Plot1DMD(ax,sl1d,NumEvNorm=True,fmt='ko')
            ax.set_title("L=["+llims+"]")
            pdf.savefig(fig)
    plt.close('all')

example_plots()
example_pdf()
SaveMDToAscii(mtd['sl2d'],"/tmp/test.txt",IgnoreIntegrated=True,NumEvNorm=True)
