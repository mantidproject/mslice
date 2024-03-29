Taking Slices
=============

This page provides a more in depth explanation of slicing in *MSlice*.

.. _Slicing_from_the_GUI:

Slicing from the GUI
--------------------

.. image:: images/slicing/calculate_projection.png

Depending on the loaded data type (see :ref:`PSD_and_non-PSD_modes`), the ``Slice`` tab in the ``Workspace Manager``
may be disabled. If this is the case, your data is in **PSD** mode and you have to first convert it to an ``MD Event``
type workspace for MSlice to make slices and cuts. In the ``Powder`` tab shown above, select the desired axes (either
*Momentum Transfer* vs *Energy Transfer* or *Scattering Angle* vs *Energy Transfer*) and click ``Calculate Projection``.
*MSlice* will then switch focus to the ``MD Event`` workspace produced by this step and enable the ``Slice`` and ``Cut``
tabs. The reason for this intemediate step is to reorganise the large volume pixel data into a form which can be quickly and
efficiently rebinned into 2D slices and 1D cuts. *MSlice* uses the Mantid `BinMD
<http://docs.mantidproject.org/nightly/algorithms/BinMD-v1.html>`_ algorithm to rebin the data for slices and cuts.
For smaller volume **Non-PSD** mode data, *MSlice* uses less efficient algorithms (`Rebin2D
<http://docs.mantidproject.org/nightly/algorithms/Rebin2D-v1.html>`_ and `SofQWNormalisedPolygon
<http://docs.mantidproject.org/nightly/algorithms/SofQWNormalisedPolygon-v1.html>`_) to rebin the loaded data directly.
Another reason for this division depending on the data type is that ``BinMD`` only performs a centre-point rebinning,
whereas the other (strictly 2D) algorithms perform the usual fractional weighted binning which is more accurate for
non-pixellated data.

.. image:: images/quickstart/data_tab.png

Once the ``Slice`` tab is enabled (either by selecting an ``MD Event`` type workspace or a **Non-PSD** mode workspace),
you can plot a 2D slice using the default parameters (axes limits and step sizes) obtained from the data by clicking
``Display``. You can also change the binning parameters before displaying the slice, or you can normalise the displayed data
such that the maximum intensity is unity by checking the ``Norm to 1`` checkbox.

.. image:: images/slicing/slice_options.png

Like with cuts (see :ref:`Cutting_from_the_GUI`), double-clicking on the axes or their titles allows you to change the
limits of the plots of the text of the titles. Clicking on the cog icon will bring up an ``Options`` dialog which allows you
to change all these properties simultaneously.


Overplotting recoil and powder lines
------------------------------------

.. image:: images/slicing/recoil.png
   :scale: 60 %

.. image:: images/slicing/powder_lines.png
   :scale: 60 %

To help with a "first look" data analysis, *MSlice* can overplot on the 2D slices the recoil lines of light nuclei
(Hydrogen :sup:`1`\ H, Deuterium :sup:`2`\ H, Helium and :sup:`4`\ He are implemented but a user defined Z is also
available), and the positions of powder reflections from common sample environment materials (Aluminium, Copper,
Niobium and Tantalum). These functionalities may be accessed from the ``Information`` menu option as shown above.

Powder reflections from an arbitrary crystallographic information format (CIF) file can also be
plotted but note that we use the `PyCifRW <https://pypi.python.org/pypi/PyCifRW/4.3>`_ package to read CIF files and that
some files generated by FullProf or GSAS may not be readable. In these cases, please load the files in `Vesta
<http://jp-minerals.org/vesta/en>`_ or `OpenBabel <http://openbabel.org>`_ and resave them.

.. _Converting_Intensity:

Converting intensity information in displayed data in slices
------------------------------------------------------------

.. image:: images/slicing/intensity_chi.png
   :scale: 80 %

In addition to displaying the data as :math:`S(Q, \omega)`, the slice figure window can also display the data with
the Bose factor corrected, as the dynamical susceptibility :math:`\chi''(Q, \omega)` or as the density of states.
For these options, the sample temperature must be given. This can either be specified as a log key, if the sample
temperature is logged in the datafile, or a user inputted value (just type into the dropdown combobox). If a log key is
given, *MSlice* will compute the mean temperature from the log series.

The different options are related to the default :math:`S(Q, \omega)` view by:

* The dynamical susceptibility, :math:`\chi''(Q, \omega) = \pi \frac{S(Q, \omega)}{<n+\frac{1}{2}\pm\frac{1}{2}>}`
* The magnetic dynamical susceptibility
  :math:`\chi''_{\mathrm{mag}}(Q, \omega) = \frac{\pi}{(g_n r_e)^2} \frac{S(Q, \omega)}{<n+\frac{1}{2}\pm\frac{1}{2}>}`
* The cross-section, :math:`\frac{d^2\sigma}{d\Omega dE} = \frac{k_i}{k_f} S(Q, \omega)`
* A "symmetrised" :math:`S(Q, \omega)` where the intensities for the neutron energy gain (negative energy) side is
  multiplied by the :math:`\exp(|E|/k_BT)` factor.
* The general (neutron weighted) density of states defined by :math:`S(Q, \omega) \frac{|E|}{Q^2} (1-\exp(-|E|/k_BT))`

Where :math:`<n+\frac{1}{2}\pm\frac{1}{2}>` is the boson population factor for neutron energy loss (boson creation,
positive sign) or neutron energy gain (boson anihilation, negative sign). In particular the code uses:

* :math:`\frac{1}{<n+1>} = 1 - \exp(-|E|/k_BT)`
* :math:`\frac{1}{<n>} = \exp(|E|/k_BT)-1`

:math:`g_n` is the neutron g-factor, and :math:`r_e` is the classical electron radius, with :math:`(g_n r_e)^2` = 291
milibarn being the total magnetic cross-section for a moment of :math:`1~\mu_B`.

If the intensity information is converted for a slice, any cut taken from the slice via :ref:`Interactive_Cuts` will
be of the converted intensity type.

Saving slices to file
---------------------

The data in a slice may be saved using the floppy icon in the slice figure window to Nexus (``.nxs``), Matlab (``.mat``)
or ASCII text (``.txt`` or ``.xye`` in a four-column ``x`` (x-coordinate), ``y`` (y-coordinate), ``s`` (signal), ``e``
(uncertainty) format). For **PSD** mode, the same information can be saved using the ``Save`` button in the ``MD Event``
tab. For **Non-PSD** mode data the equivalent ``Save`` button in the ``2D`` tab will save the original (loaded) data
in spectrum number versus energy transfer rather than the units of the slice. Thus for **Non-PSD** mode you must use
the floppy disk icon in the slice figure window to save the data. In addition, you can save an image of the slice to
``.png`` or ``.pdf`` formats.

..
   From the  Command Line
   ----------------------

    *<Docstrings from cli.get_slice and cli.plot_slice should go here>*

   Example
   -------

   First lets load some data and get the projections ready.

   .. testcode::

      import cli
      ws = cli.Load('MAR21335_Ei60.00meV.nxs')
      projection = cli.get_projection(ws, '|Q|', 'DelatE')


   Plotting a simple slice that spans all of the data

   ..  testcode::

      cli.plot_slice(projection)

   Specifying the axis

   .. testcode::

      cli.plot_slice(projection, '|Q|', 'DeltaE')

   Specifying the axis with binning parameters,

   .. testcode::

      cli.plot_slice(projection, '|Q|,0,10,.5', 'DeltaE,-20,20,1')

   Specifiying the binning parameters for a single axis

   .. testcode::

      cli.plot_slice(projection, '|Q|', 'DeltaE,-20,20,100')

   Specifying the intensity range

   .. testcode::

      cli.plot_slice(projection, '|Q|,0,10,.1', 'DeltaE,-20,20,.5', intensity_min=.2, intensity_max=1)

   Normalizing the intensity

   .. testcode::

      cli.plot_slice(projection, '|Q|,0,10,.1', 'DeltaE,-20,20,.5', normalize=True)

   Setting the colormap
       Any valid matplotib colormap object or colormap name maybe passed a a value of the ``colormap`` parameter in
       ``plot_slice``

   .. testcode::

       cli.plot_slice(projection, colormap='coolwarm')

