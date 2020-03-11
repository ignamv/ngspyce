import numpy as np
import logging
from .sharedspice import *

__all__ = [
    'cmd',
    'circ',
    'plots',
    'vector_names',
    'vectors',
    'vector',
    'try_float',
    'model_parameters',
    'device_state',
    'alter_model',
    'ac',
    'dc',
    'operating_point',
    'linear_sweep',
    'save',
    'destroy',
    'decibel',
    'alter',
    'alterparams',
    'source',
    'xspice_enabled'
]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def initialize():
    spice.ngSpice_Init(printfcn, statfcn, controlled_exit, send_data, None, None,
                       None)
    # Prevent paging output of commands (hangs)
    cmd('set nomoremode')


def cmd(command):
    """
    Send a command to the ngspice engine

    Parameters
    ----------
    command : str
        An ngspice command

    Returns
    -------
    list of str
        Lines of the captured output

    Examples
    --------

    Print all default variables

    >>> ns.cmd('print all')
    ['false = 0.000000e+00',
     'true = 1.000000e+00',
     'boltz = 1.380620e-23',
     'c = 2.997925e+08',
     'e = 2.718282e+00',
     'echarge = 1.602190e-19',
     'i = 0.000000e+00,1.000000e+00',
     'kelvin = -2.73150e+02',
     'no = 0.000000e+00',
     'pi = 3.141593e+00',
     'planck = 6.626200e-34',
     'yes = 1.000000e+00']

    """
    max_length = 1023
    if len(command) > max_length:
        raise ValueError('Command length', len(command), 'greater than',
                         max_length)
    del captured_output[:]
    spice.ngSpice_Command(command.encode('ascii'))
    logger.debug('Command %s returned %s', command, captured_output)
    return captured_output


def circ(netlist_lines):
    """
    Load a netlist

    Parameters
    ----------

    netlist_lines : str or list of str
        Netlist, either as a list of lines, or a
        single multi-line string.  Indentation and white
        space don't matter. Unlike a netlist file, the
        first line doesn't need to be a comment, and you
        don't need to provide the `.end`.

    Returns
    -------
    int
        `1` upon error, otherwise `0`.

    Examples
    --------

    Using a sequence of lines:

    >>> ns.circ(['va a 0 dc 1', 'r a 0 2'])
    0

    Using a single string:

    >>> ns.circ('''va a 0 dc 1
    ...         r a 0 2''')
    0

    """
    if issubclass(type(netlist_lines), str):
        netlist_lines = netlist_lines.split('\n')
    netlist_lines = [line.encode('ascii') for line in netlist_lines]
    # First line is ignored by the engine
    netlist_lines.insert(0, b'* ngspyce-created netlist')
    # Add netlist end
    netlist_lines.append(b'.end')
    # Add list terminator
    netlist_lines.append(None)
    array = (c_char_p * len(netlist_lines))(*netlist_lines)
    return spice.ngSpice_Circ(array)


def plots():
    """
    List available plots (result sets)

    Each plot is a collection of vector results

    Returns
    -------
    list of str
        List of existing plot names

    Examples
    --------

    Each analysis creates a new plot

    >>> ns.circ(['v1 a 0 dc 1', 'r1 a 0 1k']); ns.plots()
    ['const']
    >>> ns.operating_point(); ns.plots()
    ['op1', 'const']
    >>> ns.dc('v1', 0, 5, 1); ns.plots()
    ['dc1', 'op1', 'const']

    Get lists of vectors available in different plots:

    >>> ns.vectors(plot='const').keys()
    dict_keys(['echarge', 'e', 'TRUE', 'FALSE', 'no', 'i', ... 'c', 'boltz'])
    >>> ns.vectors(plot='ac1').keys()
    dict_keys(['V(1)', 'vout', 'v1#branch', 'frequency'])
    """
    ret = []
    plotlist = spice.ngSpice_AllPlots()
    ii = 0
    while True:
        if not plotlist[ii]:
            return ret
        ret.append(plotlist[ii].decode('ascii'))
        ii += 1


def vector_names(plot=None):
    """
    Names of vectors present in the specified plot

    Names of the voltages, currents, etc present in the specified plot.
    Defaults to the current plot.

    Parameters
    ----------
    plot : str, optional
        Plot name. Defaults to the current plot.

    Returns
    -------
    list of str
        Names of vectors in the plot

    Examples
    --------

    List built-in constants

    >>> ns.vector_names('const')
    ['planck', 'boltz', 'echarge', 'kelvin', 'i', 'c', 'e', 'pi', 'FALSE', 'no', 'TRUE', 'yes']

    Vectors produced by last analysis

    >>> ns.circ('v1 a 0 dc 2');
    >>> ns.operating_point();
    >>> ns.vector_names()
    ['v1#branch', 'a']

    """
    names = []
    if plot is None:
        plot = spice.ngSpice_CurPlot().decode('ascii')
    veclist = spice.ngSpice_AllVecs(plot.encode('ascii'))
    ii = 0
    while True:
        if not veclist[ii]:
            return names
        names.append(veclist[ii].decode('ascii'))
        ii += 1


def vectors(names=None):
    """
    Dictionary with the specified vectors (defaults to all in current plot)

    Parameters
    ----------
    names : list of str, optional
        Names of vectors to retrieve.  If omitted, return all vectors
        in current plot

    Returns
    -------
    dict from str to ndarray
        Dictionary of vectors.  Keys are vector names and values are Numpy
        arrays containing the data.

    Examples
    --------

    Do an AC sweep and retrieve the frequency axis and output voltage

    >>> nc.ac('dec', 3, 1e3, 10e6);
    >>> nc.ac_results = vectors(['frequency', 'vout'])

    """
    if names is None:
        names = vector_names()
    return dict(zip(names, map(vector, names)))


def vector(name, plot=None):
    """
    Return a numpy.ndarray with the specified vector

    Uses the current plot by default.

    Parameters
    ----------
    name : str
        Name of vector
    plot : str, optional
        Which plot the vector is in. Defaults to current plot.

    Returns
    -------
    ndarray
        Value of the vector

    Examples
    --------

    Run an analysis and retrieve a vector

    >>> ns.circ(['v1 a 0 dc 2', 'r1 a 0 1k']);
    >>> ns.dc('v1', 0, 2, 1);
    >>> ns.vector('v1#branch')
    array([ 0.   , -0.001, -0.002])

    """
    if plot is not None:
        name = plot + '.' + name
    vec = spice.ngGet_Vec_Info(name.encode('ascii'))
    if not vec:
        raise RuntimeError('Vector {} not found'.format(name))
    vec = vec[0]
    if vec.v_length == 0:
        array = np.array([])
    elif vec.v_flags & dvec_flags.vf_real:
        array = np.ctypeslib.as_array(vec.v_realdata, shape=(vec.v_length,))
    elif vec.v_flags & dvec_flags.vf_complex:
        components = np.ctypeslib.as_array(vec.v_compdata,
                                           shape=(vec.v_length, 2))
        array = np.ndarray(shape=(vec.v_length,), dtype=np.complex128,
                           buffer=components)
    else:
        raise RuntimeError('No valid data in vector')
    logger.debug('Fetched vector {} type {}'.format(name, vec.v_type))
    array.setflags(write=False)
    if name == 'frequency':
        return array.real
    return array


def try_float(s):
    """
    Parse `s` as float if possible, otherwise return `s`.
    """
    try:
        return float(s)
    except ValueError:
        try:
            return float(s.replace(',', '.'))
        except ValueError:
            return s


def model_parameters(device=None, model=None):
    """
    Model parameters for device or model

    Parameters
    ----------
    device : str, optional
        Instance name
    model : str, optional
        Model card name

    Returns
    -------
    dict from str to float or str
        Model parameters

    Examples
    --------

    Parameters of a resistor's model

    >>> ns.circ('r1 a 0 2k');
    >>> ns.model_parameters(device='r1')
    {'description': 'Resistor models (Simple linear resistor)', 'model': 'R',
    'rsh': 0.0, 'narrow': 0.0, 'short': 0.0, 'tc1': 0.0, 'tc2': 0.0,
    'tce': 0.0, 'defw': 0.0, 'l': 0.0, 'kf': 0.0, 'af': 0.0, 'r': 0.0,
    'bv_max': 0.0, 'lf': 0.0, 'wf': 0.0, 'ef': 0.0}
    """
    if device is None:
        if model is not None:
            lines = cmd('showmod #' + model.lower())
        else:
            raise ValueError('Either device or model must be specified')
    else:
        if model is None:
            lines = cmd('showmod ' + device.lower())
        else:
            raise ValueError('Only specify one of device, model')
    ret = dict(description=lines.pop(0))
    ret.update({parts[0]: try_float(parts[1])
                for parts in map(str.split, lines)})
    return ret


def device_state(device):
    """
    Dict with device state

    Parameters
    ----------
    device : str
        Instance name

    Returns
    -------
    dict from str to float or str
        Device description, model, operating point, etc.

    Examples
    --------

    Resistor description

    >>> ns.circ(['r1 a 0 4'])
    >>> ns.device_state('r1')
    {'description': 'Resistor: Simple linear resistor', 'device': 'r1',
    'model': 'R', 'resistance': 4.0, 'ac': 4.0, 'dtemp': 0.0, 'bv_max': 0.0,
    'noisy': 0.0}
    """
    lines = cmd('show ' + device.lower())

    ret = dict(description=lines.pop(0))
    ret.update({parts[0]: try_float(parts[1])
                for parts in map(str.split, lines)})
    return ret


def alter_model(model, **params):
    """
    Change parameters of a model card

    Parameters
    ----------
    model : str
        Model card name
    """
    for k, v in params.items():
        cmd('altermod {} {} = {:.6e}'.format(model, k, v))


def ac(mode, npoints, fstart, fstop):
    """
    Small-signal AC analysis

    Parameters
    ----------
    mode : {'lin', 'oct', 'dec'}
        Frequency axis spacing: linear, octave or decade
    npoints : int
        If mode is ``'lin'``, this is the total number of points for the sweep.
        Otherwise, this is the number of points per decade or per octave.
    fstart : float
        Starting frequency
    fstop : float
        Final frequency

    Returns
    -------
    dict from str to ndarray
        Result vectors: voltages, currents and frequency (under key ``'frequency'``).

    Examples
    --------

    Sweep from 1 kHz to 10 MHz with 3 points per decade

    >>> results = nc.ac('dec', 3, 1e3, 10e6)
    >>> len(results['frequency'])
    13

    Sweep from 20 to 20 kHz in 21 linearly spaced points

    >>> results = nc.ac('lin', 21, 20, 20e3)
    >>> len(results['frequency'])
    21

    Bode plot of low-pass filter::

        ns.circ('''
        v1 in 0 dc 0 ac 1
        r1 in out 1k
        c1 out 0 1n''')
        results = ns.ac('dec', 2, 1e0, 1e9)
        plt.semilogx(results['frequency'], 2*ns.decibel(results['out']))

    .. image:: lowpass.png

    """
    modes = ('dec', 'lin', 'oct')
    if mode.lower() not in modes:
        raise ValueError("'{}' is not a valid AC sweep "
                         "mode: {}".format(mode, modes))
    if fstop < fstart:
        raise ValueError('Start frequency', fstart,
                         'greater than stop frequency', fstop)
    cmd('ac {} {} {} {}'.format(mode, npoints, fstart, fstop))
    return vectors()


def group(iterable, grouplength):
    return zip(*(iterable[ii::grouplength]
                 for ii in range(grouplength)))


def dc(*sweeps):
    """
    Analyze DC transfer function, return vectors with one axis per sweep

    Parameters
    ----------
    sweeps:
        One or two sequences of (src, start, stop, increment).
        src can be an independent voltage or current source, a resistor, or ``'TEMP'``.

    Returns
    -------
    dict from str to ndarray
        Voltages and currents. If there is a secondary sweep, the ndarrays will have two axes.

    Examples
    --------

    Sweep a voltage source

    >>> ns.circ('v1 a 0 dc 0');
    >>> ns.dc('v1', 0, 5, 1)
    {'a': array([ 0.,  1.,  2.,  3.,  4.,  5.]),
     'v-sweep': array([ 0.,  1.,  2.,  3.,  4.,  5.]),
     'v1': array([0, 1, 2, 3, 4, 5]),
     'v1#branch': array([ 0.,  0.,  0.,  0.,  0.,  0.])}

    Add a secondary sweep::

        ns.circ(['v1 a 0 dc 0', 'r1 a 0 1k'])
        results = ns.dc('v1', 0, 3, 1, 'r1', 1e3, 10e3, 1e3)
        plt.plot(-results['v1#branch']);

    .. image:: secondary_sweep.png

    """
    # TODO: support more than two sweeps
    # TODO: implement other sweeps
    cmd('dc ' + ' '.join(map(str, sweeps)))
    sweepvalues = [linear_sweep(*sweep[1:])
                   for sweep in group(sweeps, 4)]
    sweeplengths = tuple(map(len, sweepvalues))
    ret = {k: v.reshape(sweeplengths, order='F')
           for k, v in vectors().items()}
    # Add vectors with swept sources/parameters
    for ii, (name, values) in enumerate(zip(sweeps[::4], sweepvalues)):
        shape = [length if ii == jj else 1
                 for jj, length in enumerate(sweeplengths)]
        ret[name] = values.reshape(shape, order='F')
    return ret


def operating_point():
    """
    Analyze DC operating point

    Returns
    -------
    dict from str to ndarray
        Voltages and currents
    """
    cmd('op')
    return vectors()


def save(vector_name):
    """
    Save this vector in the following analyses

    If this command is used, only explicitly saved vectors will be kept in next analysis.

    Parameters
    ----------
    vector_name : str
        Name of the vector
    """
    cmd('save ' + vector_name)


def destroy(plotname='all'):
    """
    Erase plot from memory

    Parameters
    ----------
    plotname : str, optional
        Name of a plot. If omitted, erase all plots.
    """
    cmd('destroy ' + plotname)


def decibel(x):
    '''Calculate 10*log(abs(x))'''
    return 10. * np.log10(np.abs(x))


def alter(device, **parameters):
    """
    Alter device parameters

    Parameters
    ----------
    device : str
        Instance name

    Examples
    --------

    >>> ns.alter('R1', resistance=200)
    >>> ns.alter('vin', ac=2, dc=3)
    """
    for k, v in parameters.items():
        if not isinstance(v, (list, tuple)):
            v = str(v)
        else:
            v = '[ ' + ' '.join(v) + ' ]'
        cmd('alter {} {} = {}'.format(device.lower(), k, v))


def alterparams(**kwargs):
    for k, v in kwargs.items():
        cmd('alterparam {} = {}'.format(k, v))
    cmd('reset')


def linear_sweep(start, stop, step):
    """
    Numbers from start to stop (inclusive), separated by step.

    These match the values used in a dc linear sweep

    Returns
    -------
    ndarray

    Examples
    --------

    >>> ns.linear_sweep(0, 100, 20)
    array([  0,  20,  40,  60,  80, 100])

    """
    if (start > stop and step > 0) or (start < stop and step < 0):
        raise ValueError("Can't sweep from", start, 'to', stop, 'with step',
                         step)
    ret = []
    nextval = start
    while True:
        if np.sign(step) * nextval - np.sign(step) * stop >= (
                np.finfo(float).eps * 1e3):
            return np.array(ret)
        ret.append(nextval)
        nextval = nextval + step


def source(filename):
    """
    Evaluate a ngspice input file

    This function is the same as the ngspice source command, so the first line
    of the file is considered a title line, lines beginning with the character
    ``*`` are considered comments and are ignored, etc.

    Parameters
    ----------
    filename : str
        A file containing a circuit netlist.
    """
    cmd("source '{}'".format(filename))


def xspice_enabled():
    """
    Was libngspice compiled with XSpice support?

    Returns
    -------
    bool
    """
    return '** XSPICE extensions included' in cmd('version -f')


initialize()
