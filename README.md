Python bindings for the ngspice simulation engine
=================================================

This is a library that allows Python applications to talk to
[Ngspice](http://ngspice.sourceforge.net/), an engine for simulating electronic
circuits. Currently it supports sending commands to the engine and reading the
results into numpy arrays, for plotting and analysis. Future goals include
voltage and current sources defined by Python functions, and the possibility of
stepping through a simulation in order to inspect results and modify the
circuit mid-run.

Examples
--------

[Low-pass filter](examples/lowpass)

[Bipolar transistor output characteristics](examples/npn)

[Operational amplifier oscillator](examples/quadrature_oscillator)

Getting libngspice
------------------

This library requires libngspice.

#### Linux
On Linux, this means you have to [download the source package for
ngspice](http://ngspice.sourceforge.net/download.html) and compile it like this:

    ./configure --with-ngshared
    make
    sudo make install

It is occasionally necessary to adjust LD_LIBRARY_PATH to help ngspyce find libngspice.
First locate your copy of libngspice.so then try updating the environment variable, with for example:

    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

#### OSX
On OSX, libngspice can be installed with brew. Note that the ngspice package does not supply the required shared libraries. 

#### Windows
On Windows, it currently assumes that `ngspice.dll` is installed in
`C:\Spice\bin_dll` (32-bit Python) or `C:\Spice64\bin_dll` (64-bit Python).
Go to [Ngspice Download](http://ngspice.sourceforge.net/download.html) and
choose one of the packages (such as `ngspice-26plus-scope-inpcom-6-64.7z`)
that contains `ngspice.dll`, and extract it to `C:\`.  (To support all features,
this folder structure must also include `spinit`, `spice2poly.cm`, etc.)

Making netlists
---------------

One fast way to produce SPICE netlists is to draw the schematic with
[GSchem](http://www.geda-project.org/) and then export a netlist with

    gnetlist -g spice-sdb schematic.sch -o netlist.net

For details on simulation commands, check out the [Ngspice
manual](http://ngspice.sourceforge.net/docs/ngspice-manual.pdf).
