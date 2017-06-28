from ctypes import (CDLL, CFUNCTYPE, Structure, c_int, c_char_p, c_void_p,
                    c_bool, c_double, POINTER, pointer, cast, c_short,
                    py_object)

from ctypes.util import find_library
import numpy as np
import logging
import os
import platform

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

# libngspice source code is listed before the relevant ctype structs
if os.name == 'nt':  # Windows
    # http://stackoverflow.com/a/13277363
    curr_dir_before = os.getcwd()

    drive = os.getenv("SystemDrive") or 'C:'

    # Python and DLL must both be same number of bits
    if platform.architecture()[0] == '64bit':
        spice_path = os.path.join(drive, os.sep, 'Spice64')
    elif platform.architecture()[0] == '32bit':
        spice_path = os.path.join(drive, os.sep, 'Spice')
    else:
        raise RuntimeError("Couldn't determine if Python is 32-bit or 64-bit")

    """
    https://sourceforge.net/p/ngspice/discussion/133842/thread/1cece652/#4e32/5ab8/9027
    On Windows, when environment variable SPICE_LIB_DIR is empty, ngspice
    looks in `C:\Spice64\share\ngspice\scripts`.  If the variable is not empty
    it tries `%SPICE_LIB_DIR%\scripts\spinit`
    """

    if 'SPICE_LIB_DIR' not in os.environ:
        os.environ['SPICE_LIB_DIR'] = os.path.join(spice_path, 'share',
                                                   'ngspice')
    os.chdir(os.path.join(spice_path, 'bin_dll'))
    spice = CDLL('ngspice')
    os.chdir(curr_dir_before)
else:  # Linux, etc.
    spice = CDLL(find_library('ngspice'))

captured_output = []


@CFUNCTYPE(c_int, c_char_p, c_int, c_void_p)
def printfcn(output, id, ret):
    """Callback for libngspice to print a message"""
    global captured_output
    prefix, _, content = output.decode('ascii').partition(' ')
    if prefix == 'stderr':
        logger.error(content)
    else:
        captured_output.append(content)
    return 0


@CFUNCTYPE(c_int, c_char_p, c_int, c_void_p)
def statfcn(status, id, ret):
    """
    Callback for libngspice to report simulation status like 'tran 5%'
    """
    logger.debug(status.decode('ascii'))
    return 0


@CFUNCTYPE(c_int, c_int, c_bool, c_bool, c_int, c_void_p)
def controlled_exit(exit_status, immediate_unloading, requested_exit,
                    libngspice_id, ret):
    logger.debug('ControlledExit',
                 dict(exit_status=exit_status,
                      immediate_unloading=immediate_unloading,
                      requested_exit=requested_exit,
                      libngspice_id=libngspice_id, ret=ret))

# typedef struct vecvalues {
    # char* name; /* name of a specific vector */
    # double creal; /* actual data value */
    # double cimag; /* actual data value */
    # bool is_scale;/* if 'name' is the scale vector */
    # bool is_complex;/* if the data are complex numbers */
# } vecvalues, *pvecvalues;


class vecvalues(Structure):
    _fields_ = [
        ('name', c_char_p),
        ('creal', c_double),
        ('cimag', c_double),
        ('is_scale', c_bool),
        ('is_complex', c_bool)]

# typedef struct vecvaluesall {
    # int veccount; /* number of vectors in plot */
    # int vecindex; /* index of actual set of vectors. i.e. the number of accepted data points */
    # pvecvalues *vecsa; /* values of actual set of vectors, indexed from 0 to veccount - 1 */
# } vecvaluesall, *pvecvaluesall;


class vecvaluesall(Structure):
    _fields_ = [
        ('veccount', c_int),
        ('vecindex', c_int),
        ('vecsa', POINTER(POINTER(vecvalues)))]


@CFUNCTYPE(c_int, POINTER(vecvaluesall), c_int, c_int, c_void_p)
def send_data(vecvaluesall, num_structs, libngspice_id, ret):
    logger.debug('SendData', dict(vecvaluesall=vecvaluesall,
                                  num_structs=num_structs,
                                  libngspice_id=libngspice_id,
                                  ret=ret))

def initialize():
    spice.ngSpice_Init(printfcn, statfcn, controlled_exit, send_data, None, None,
                       None)
    # Prevent paging output of commands (hangs)
    cmd('set nomoremode')

# int  ngSpice_Command(char* command);
spice.ngSpice_Command.argtypes = [c_char_p]


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

# int ngSpice_Circ(char**)
spice.ngSpice_Circ.argtypes = [POINTER(c_char_p)]
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

#struct ngcomplex {
#    double cx_real;
#    double cx_imag;
#} ;

spice.ngSpice_AllPlots.restype = POINTER(c_char_p)
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


spice.ngSpice_AllVecs.argtypes = [c_char_p]
spice.ngSpice_AllVecs.restype = POINTER(c_char_p)
spice.ngSpice_CurPlot.restype = c_char_p
def vector_names(plot=None):
    """
    List the vectors present in the specified plot

    List the voltages, currents, etc present in the specified plot.
    Defaults to the last plot.
    """
    names = []
    if plot is None:
        plot = spice.ngSpice_CurPlot()
    veclist = spice.ngSpice_AllVecs(plot)
    ii = 0
    while True:
        if not veclist[ii]:
            return names
        names.append(veclist[ii].decode('ascii'))
        ii += 1


class ngcomplex(Structure):
    _fields_ = [
        ('cx_real', c_double),
        ('cx_imag', c_double)]
# /* Dvec flags. */
# enum dvec_flags {
#   VF_REAL = (1 << 0),       /* The data is real. */
#   VF_COMPLEX = (1 << 1),    /* The data is complex. */
#   VF_ACCUM = (1 << 2),      /* writedata should save this vector. */
#   VF_PLOT = (1 << 3),       /* writedata should incrementally plot it. */
#   VF_PRINT = (1 << 4),      /* writedata should print this vector. */
#   VF_MINGIVEN = (1 << 5),   /* The v_minsignal value is valid. */
#   VF_MAXGIVEN = (1 << 6),   /* The v_maxsignal value is valid. */
#   VF_PERMANENT = (1 << 7)   /* Don't garbage collect this vector. */
# };


class dvec_flags(object):
    vf_real = (1 << 0)       # The data is real.
    vf_complex = (1 << 1)    # The data is complex.
    vf_accum = (1 << 2)      # writedata should save this vector.
    vf_plot = (1 << 3)       # writedata should incrementally plot it.
    vf_print = (1 << 4)      # writedata should print this vector.
    vf_mingiven = (1 << 5)   # The v_minsignal value is valid.
    vf_maxgiven = (1 << 6)   # The v_maxsignal value is valid.
    vf_permanent = (1 << 7)  # Don't garbage collect this vector.

# /* vector info obtained from any vector in ngspice.dll.
   # Allows direct access to the ngspice internal vector structure,
   # as defined in include/ngspice/devc.h .*/
# typedef struct vector_info {
#    char *v_name;		/* Same as so_vname. */
#    int v_type;			/* Same as so_vtype. */
#    short v_flags;		/* Flags (a combination of VF_*). */
#    double *v_realdata;		/* Real data. */
#    ngcomplex_t *v_compdata;	/* Complex data. */
#    int v_length;		/* Length of the vector. */
#} vector_info, *pvector_info;


class vector_info(Structure):
    _fields_ = [
        ('v_name', c_char_p),
        ('v_type', c_int),
        ('v_flags', c_short),
        ('v_realdata', POINTER(c_double)),
        ('v_compdata', POINTER(ngcomplex)),
        ('v_length', c_int)]

# /* get info about a vector */
# pvector_info ngGet_Vec_Info(char* vecname);
spice.ngGet_Vec_Info.restype = POINTER(vector_info)
spice.ngGet_Vec_Info.argtypes = [c_char_p]

# Unit names for use with pint or other unit libraries
vector_type = [
    'dimensionless',      # notype = 0
    'second',             # time = 1
    'hertz',              # frequency = 2
    'volt',               # voltage = 3
    'ampere',             # current = 4
    'NotImplemented',     # output_n_dens = 5
    'NotImplemented',     # output_noise = 6
    'NotImplemented',     # input_n_dens = 7
    'NotImplemented',     # input_noise = 8
    'NotImplemented',     # pole = 9
    'NotImplemented',     # zero = 10
    'NotImplemented',     # sparam = 11
    'NotImplemented',     # temp = 12
    'ohm',                # res = 13
    'ohm',                # impedance = 14
    'siemens',            # admittance = 15
    'watt',               # power = 16
    'dimensionless'       # phase = 17
    'NotImplemented',     # db = 18
    'farad'               # capacitance = 19
    'coulomb'             # charge = 21
]


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

#
# enum simulation_types {
#   ...
# };
class simulation_type(object):
    notype = 0
    time = 1
    frequency = 2
    voltage = 3
    current = 4
    output_n_dens = 5
    output_noise = 6
    input_n_dens = 7
    input_noise = 8
    pole = 9
    zero = 10
    sparam = 11
    temp = 12
    res = 13
    impedance = 14
    admittance = 15
    power = 16
    phase = 17
    db = 18
    capacitance = 19
    charge = 20


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


def save(vector):
    cmd('save ' + vector)


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

from sys import float_info
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
                float_info.epsilon * 1e3):
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
