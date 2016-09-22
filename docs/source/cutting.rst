Taking Cuts
===========
    This page provides a more detailed explanation on taking cuts in MSlice

From the GUI
------------
The cut tab will by default be diabled. When you click on a cuttable workspace it will be enabled.

**To plot a Single Cut**

1. Click on a projection workspace in the *Workspace Manager* to select it.

2. Switch to Cut tab

3. Fill in the mandatory values for the integration axis and the integration range

.. image::images/cutting/mandatory_values.png

4. Optionally Specify the range for the intensity to shown and tick the the `Norm to 1` checkbox to normalize the
   cuts to 1.

5. Click on the **Plot** button to plot the cut.

.. image:: images/cutting/single_cut.png

**To Plot Multiple Equal From one workspace**

1. Click on a projection workspace in the *Workspace Manager* to select it.

2. Switch to Cut tab

3. Fill in the mandatory values for the integration axis and the integration range

4. Specify the `width` parameter. When the the width parameter is set then cuts the values in `integration from` and
    `Integration to ` will specify the range of the data to processed. The range specified will be divided into sections of
    `width` and each section plotted individual. For instance setting the integration from `0` to `40` with width `10` will produce
     4 individual cuts, one with the ranges : from 0 to 10, from 10 to 20, from 20 to 30, from 30 to 40.

.. images/cutting/width_parameters_set.png

4. Optionally Specify the range for the intensity to shown and tick the the `Norm to 1` checkbox to normalize the
   cuts to 1.

5. Click on the **Plot** button to plot the cut.

.. images:: image/cutting/width_cuts

**To plot multiple cuts manually to the same plot**




