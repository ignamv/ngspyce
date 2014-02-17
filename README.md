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

This library requires libngspice. This means you have to
[download the source package for
ngspice](http://ngspice.sourceforge.net/download.html) and compile it like this:

    ./configure --with-ngshared
    make
    sudo make install

Making netlists
---------------

One fast way to produce spice netlists is to draw the schematic with
[GSchem](http://www.geda-project.org/) and then export a netlist with

    gnetlist -g spice-sdb schematic.sch -o netlist.net

For details on simulation commands, check out the [Ngspice
manual](http://ngspice.sourceforge.net/docs/ngspice-manual.pdf).
