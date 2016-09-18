
from ctypes import CDLL, CFUNCTYPE, Structure, c_int, c_char_p, c_void_p, \
        c_bool, c_double, POINTER, pointer, cast, c_short, py_object
import numpy as np
import logging
import itertools
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

# libngspice source code is listed before the relevant ctype structs
spice = CDLL('libngspice.so')
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
    """Callback for libngspice to report simulation status like 'tran 5%'"""
    logger.debug(status.decode('ascii'))
    return 0

@CFUNCTYPE(c_int, c_int, c_bool, c_bool, c_int, c_void_p)
def controlled_exit(exit_status, immediate_unloading, requested_exit,
                    libngspice_id, ret):
    logger.debug('ControlledExit', dict(exit_status=exit_status,
                                      immediate_unloading=immediate_unloading,
                                      requested_exit=requested_exit,
                                      libngspice_id=libngspice_id,
                                      ret=ret))

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

spice.ngSpice_Init(printfcn, statfcn, controlled_exit, send_data, None, None, 
                   None)

# int  ngSpice_Command(char* command);
spice.ngSpice_Command.argtypes = [c_char_p]

def cmd(command):
    """Send a commang to the ngspice engine"""
    max_length = 1023
    if len(command) > max_length:
        raise Exception('Command length', len(command), 'greater than',
                        max_length)
    captured_output.clear()
    spice.ngSpice_Command(command.encode('ascii'))
    logger.debug('Command %s returned %s', command, captured_output)
    return captured_output

# int ngSpice_Circ(char**)
spice.ngSpice_Circ.argtypes = [POINTER(c_char_p)]

def circ(netlist_lines):
    """Specify a netlist
    
    Accepts an array of lines, or a multi-line string
    """
    if issubclass(type(netlist_lines), str):
        netlist_lines = netlist_lines.split('\n')
    netlist_lines = [line.encode('ascii') for line in netlist_lines]
    # First line is ignored by the engine
    netlist_lines.insert(0, b'* First line')
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
    """List available plots (result sets)"""
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
def vectorNames(plot=None):
    """List the vectors present in the specified plot 
    
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
   vf_real = (1 << 0)       # The data is real. */
   vf_complex = (1 << 1)    # The data is complex. */
   vf_accum = (1 << 2)      # writedata should save this vector. */
   vf_plot = (1 << 3)       # writedata should incrementally plot it. */
   vf_print = (1 << 4)      # writedata should print this vector. */
   vf_mingiven = (1 << 5)   # The v_minsignal value is valid. */
   vf_maxgiven = (1 << 6)   # The v_maxsignal value is valid. */
   vf_permanent = (1 << 7)  # Don't garbage collect this vector. */

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
    """Return a dictionary with the specified vectors
    
    If names is None, return all available vectors"""
    if names is None:
        plot = spice.ngSpice_CurPlot()
        names = vectorNames(plot)
    return dict(zip(names, map(vector, names)))

def vector(name, plot=None):
    """Return a numpy.ndarray with the specified vector
    
    Uses the current plot by default."""
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
                                           shape=(vec.v_length,2))
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
    '''Parse s as float if possible, otherwise return s'''
    try:
        return float(s)
    except ValueError:
        try:
            return float(s.replace(',', '.'))
        except ValueError:
            return s

def model_parameters(device):
    '''Return dict with model parameters for device'''
    lines = cmd('showmod ' + device)
    ret = dict(description=lines.pop(0))
    ret.update({parts[0]: try_float(parts[1])
                for parts in map(str.split, lines)})
    return ret

def alter_model(device, **params):
    for k,v in params.items():
        cmd('altermod {} {} = {:.6e}'.format(device, k, v))

def ac(mode, npoints, fstart, fstop):
    '''Perform small-signal AC analysis

    Examples
    ========
    Sweep from 1kHz to 10MHz with 3 points per decade

    >>> ac('dec', 3, 1e3, 10e6)

    Sweep from 0 to 20kHz in 21 linearly spaced points

    >>> ac('lin', 21, 0, 20e3)
    '''
    modes = ('dec', 'lin', 'oct')
    if mode.lower() not in modes:
        raise Exception(mode, 'not a valid AC sweep mode', modes)
    if fstop < fstart:
        raise Exception('Start frequency', fstart,
                        'greater than stop frequency', fstop)
    cmd('ac {} {} {} {}'.format(mode, npoints, fstart, fstop))
    return vectors()

def dc(*sweeps):
    '''Analyze DC transfer function

    sweeps is one or more sequences of (src, start, stop, increment)
    src can be an independent voltage or current source, a resistor, or "TEMP".

    Example
    =======
    Sweep Vgs from 0 to 5V in 0.1V steps while stepping temperature

    >>> dc('vgs', 0, 5, .1, 'temp', -20, 80, 10)
    '''
    cmd('dc ' + ' '.join(map(str, sweeps)))
    return vectors()

def operating_point():
    '''Analyze DC operating point'''
    cmd('op')
    return vectors()

def save(vector):
    cmd('save ' + vector)

def destroy(plotname='all'):
    '''Erase plot from memory'''
    cmd('destroy ' + plotname)

def decibel(x):
    return 10. * np.log10(np.abs(x))

def alter(device, **parameters):
    '''Alter device parameters

    Examples
    ========
    >>> alter('R1', resistance=200)
    >>> alter('vin', ac=2, dc=3)
    '''
    for k,v in parameters.items():
        if not isinstance(v, (list, tuple)):
            v = str(v)
        else:
            v = '[ ' + ' '.join(v) + ' ]'
        cmd('alter {} {} = {}'.format(device.lower(), k, v))

