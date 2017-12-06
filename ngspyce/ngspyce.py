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
    'save',
    'destroy',
    'decibel',
    'alter',
    'linear_sweep',
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
    success : list of str
        A list of lines of the captured output

    Examples
    --------
    Print all default variables

    >>> cmd('print all')
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
    captured_output.clear()
    spice.ngSpice_Command(command.encode('ascii'))
    logger.debug('Command %s returned %s', command, captured_output)
    return captured_output


def circ(netlist_lines):
    """
    Specify a netlist

    Parameters
    ----------
    netlist_lines : str or sequence of str
        Lines of the netlist, either as a sequence, or a single multi-line
        string.  Indentation and white space doesn't matter.  (Unlike a
        netlist file, the first line doesn't need to be a comment, and you
        don't need to provide the `.end`.)

    Returns
    -------
    success : int
        Returns a `1` upon error, otherwise `0`.

    Examples
    --------
    Using a sequence of lines:

    >>> circ(['va a 0 dc 1',
    ...       'r a 0 2'])
    0

    Using a single string:

    >>> circ('''va a 0 dc 1
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
    plots : list of str
        List of the plot names available

    Examples
    --------
    Get list of plots:

    >>> plots()
    ['ac1', 'dc1', 'const']

    Get lists of vectors available in different plots:

    >>> vectors(plot='const').keys()
    dict_keys(['echarge', 'e', 'TRUE', 'FALSE', 'no', 'i', ... 'c', 'boltz'])
    >>> vectors(plot='ac1').keys()
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
    List the vectors present in the specified plot

    List the voltages, currents, etc present in the specified plot.
    Defaults to the last plot.
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
    Return a dictionary with the specified vectors

    Parameters
    ----------
    names : iterable of strings
        Names of vectors to retrieve.  If `names` is None, return all
        available vectors

    Returns
    -------
    vectors : dict
        Dictionary of vectors.  Keys are vector names and values are Numpy
        arrays containing the data.

    Examples
    --------
    Do an AC sweep and then retrieve the frequency axis and output voltage

    >>> ac('dec', 3, 1e3, 10e6)
    >>> ac_results = vectors(['frequency', 'vout'])

    """
    if names is None:
        plot = spice.ngSpice_CurPlot()
        names = vector_names(plot)
    return dict(zip(names, map(vector, names)))


def vector(name, plot=None):
    """
    Return a numpy.ndarray with the specified vector

    Uses the current plot by default.
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
    Return dict with model parameters for device or model
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
    Return dict with device state
    """
    lines = cmd('show ' + device.lower())

    ret = dict(description=lines.pop(0))
    ret.update({parts[0]: try_float(parts[1])
                for parts in map(str.split, lines)})
    return ret


def alter_model(device, **params):
    for k, v in params.items():
        cmd('altermod {} {} = {:.6e}'.format(device, k, v))


def ac(mode, npoints, fstart, fstop):
    """
    Perform small-signal AC analysis

    Parameters
    ----------
    mode : {'lin', 'dec', oct'}
        Frequency axis spacing: linear, decade, or octave.
    npoints : int
        If mode is 'lin', this is the total number of points for the sweep.
        Otherwise, this is the number of points per decade or per octave.
    fstart : float or str
        Starting frequency.
    fstop : float or str
        Final frequency.

    Returns
    -------
    results : dict
        Dictionary of test results.  Frequency points are in
        ``results['frequency']``, with corresponding voltages and currents
        under their own key names, such as ``results['vout']``

    Examples
    --------
    Sweep from 1 kHz to 10 MHz with 3 points per decade

    >>> results = ac('dec', 3, 1e3, 10e6)
    >>> len(results['frequency'])
    13

    Sweep from 20 to 20 kHz in 21 linearly spaced points

    >>> results = ac('lin', 21, 20, 20e3)
    >>> len(results['frequency'])
    21
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

    sweeps is one or more sequences of (src, start, stop, increment)
    src can be an independent voltage or current source, a resistor, or "TEMP".

    Returned vectors are reshaped so each axis corresponds to one sweep.

    Examples
    --------
    >>> dc('va', 0, 1, 1, 'vb', 0, 2, 2)
    {'a': array([[ 0.,  0.], [ 1.,  1.]]),
     'b': array([[ 0.,  2.], [ 0.,  2.]]), ...}
    """
    # TODO: support more than two sweeps
    cmd('dc ' + ' '.join(map(str, sweeps)))
    sweepvalues = [linear_sweep(*sweep[1:])
                   for sweep in group(sweeps, 4)]
    sweeplengths = tuple(map(len, sweepvalues))
    ret = {k: v.reshape(sweeplengths, order='F')
           for k, v in vectors().items()}
    for ii, (name, values) in enumerate(zip(sweeps[::4], sweepvalues)):
        shape = [length if ii == jj else 1
                 for jj, length in enumerate(sweeplengths)]
        ret[name] = values.reshape(shape, order='F')
    return ret


def operating_point():
    """
    Analyze DC operating point
    """
    cmd('op')
    return vectors()


def save(vector_name):
    cmd('save ' + vector_name)


def destroy(plotname='all'):
    """
    Erase plot from memory
    """
    cmd('destroy ' + plotname)


def decibel(x):
    return 10. * np.log10(np.abs(x))


def alter(device, **parameters):
    """
    Alter device parameters

    Examples
    --------
    >>> alter('R1', resistance=200)
    >>> alter('vin', ac=2, dc=3)
    """
    for k, v in parameters.items():
        if not isinstance(v, (list, tuple)):
            v = str(v)
        else:
            v = '[ ' + ' '.join(v) + ' ]'
        cmd('alter {} {} = {}'.format(device.lower(), k, v))


def linear_sweep(start, stop, step):
    """
    Voltages used in a dc transfer curve analysis linear sweep
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
    Read a ngspice input file

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
    Return True if libngspice was compiled with XSpice support
    """
    return '** XSPICE extensions included' in cmd('version -f')


initialize()
