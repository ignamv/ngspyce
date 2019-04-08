import os
import platform
import logging
from ctypes import (CDLL, CFUNCTYPE, Structure, c_int, c_char_p, c_void_p,
                    c_bool, c_double, POINTER, c_short)
from ctypes.util import find_library

logger = logging.getLogger(__name__)

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
    try:
        lib_location = os.environ['LIBNGSPICE']
    except KeyError:
        lib_location = find_library('ngspice')
    spice = CDLL(lib_location)

captured_output = []


@CFUNCTYPE(c_int, c_char_p, c_int, c_void_p)
def printfcn(output, _id, _ret):
    """Callback for libngspice to print a message"""
    global captured_output
    prefix, _, content = output.decode('ascii').partition(' ')
    if prefix == 'stderr':
        logger.error(content)
    else:
        captured_output.append(content)
    return 0


@CFUNCTYPE(c_int, c_char_p, c_int, c_void_p)
def statfcn(status, _id, _ret):
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
def send_data(vecvaluesall_, num_structs, libngspice_id, ret):
    logger.debug('SendData', dict(vecvaluesall=vecvaluesall_,
                                  num_structs=num_structs,
                                  libngspice_id=libngspice_id,
                                  ret=ret))


# int  ngSpice_Command(char* command);
spice.ngSpice_Command.argtypes = [c_char_p]

# int ngSpice_Circ(char**)
spice.ngSpice_Circ.argtypes = [POINTER(c_char_p)]
spice.ngSpice_AllPlots.restype = POINTER(c_char_p)

spice.ngSpice_AllVecs.argtypes = [c_char_p]
spice.ngSpice_AllVecs.restype = POINTER(c_char_p)
spice.ngSpice_CurPlot.restype = c_char_p


# struct ngcomplex {
#    double cx_real;
#    double cx_imag;
# } ;

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
    vf_real = (1 << 0)  # The data is real.
    vf_complex = (1 << 1)  # The data is complex.
    vf_accum = (1 << 2)  # writedata should save this vector.
    vf_plot = (1 << 3)  # writedata should incrementally plot it.
    vf_print = (1 << 4)  # writedata should print this vector.
    vf_mingiven = (1 << 5)  # The v_minsignal value is valid.
    vf_maxgiven = (1 << 6)  # The v_maxsignal value is valid.
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
# } vector_info, *pvector_info;


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
    'dimensionless',  # notype = 0
    'second',  # time = 1
    'hertz',  # frequency = 2
    'volt',  # voltage = 3
    'ampere',  # current = 4
    'NotImplemented',  # output_n_dens = 5
    'NotImplemented',  # output_noise = 6
    'NotImplemented',  # input_n_dens = 7
    'NotImplemented',  # input_noise = 8
    'NotImplemented',  # pole = 9
    'NotImplemented',  # zero = 10
    'NotImplemented',  # sparam = 11
    'NotImplemented',  # temp = 12
    'ohm',  # res = 13
    'ohm',  # impedance = 14
    'siemens',  # admittance = 15
    'watt',  # power = 16
    'dimensionless'  # phase = 17
    'NotImplemented',  # db = 18
    'farad'  # capacitance = 19
    'coulomb'  # charge = 21
]


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
