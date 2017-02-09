import unittest

import ngspyce as ns
from numpy import pi


class NgspiceTest(unittest.TestCase):
    def setUp(self):
        ns.destroy()

    def assertEqualDictOfArray(self, dict1, dict2):
        self.assertEqual(sorted(dict1.keys()),
                         sorted(dict2.keys()))
        for k, X in dict1.items():
            Y = dict2[k]
            self.assertEqual(len(X), len(Y))
            for x, y in zip(X, Y):
                self.assertAlmostEqual(x, y)

    def assertVectors(self, vectors):
        self.assertEqualDictOfArray(ns.vectors(), vectors)


class TestPlatform(NgspiceTest):
    # Test that spinit can be found and is loading default codemodels
    # TODO: Should these just print warnings or actually fail the test?
    def test_spice2poly_cm(self):
        self.assertTrue(any([line.startswith('spice2poly') for line in
                             ns.cmd('devhelp')[1:]]))

    def test_analog_cm(self):
        self.assertTrue(any([line.startswith('climit') for line in
                             ns.cmd('devhelp')[1:]]))

    def test_digital_cm(self):
        self.assertTrue(any([line.startswith('adc_bridge') for line in
                             ns.cmd('devhelp')[1:]]))

    def test_table_cm(self):
        self.assertTrue(any([line.startswith('table2d') for line in
                             ns.cmd('devhelp')[1:]]))

    def test_xtradev_cm(self):
        self.assertTrue(any([line.startswith('aswitch') for line in
                             ns.cmd('devhelp')[1:]]))

    def test_xtraevt_cm(self):
        self.assertTrue(any([line.startswith('d_to_real') for line in
                             ns.cmd('devhelp')[1:]]))


class TestBasicCircuits(NgspiceTest):
    def test_vsource(self):
        ns.circ('va a 0 dc 1')
        ns.operating_point()
        self.assertEqual(ns.vectors(),
                         {'a'        : [1],
                          'va#branch': [0]})

    def test_resistor(self):
        ns.circ(['va a 0 dc 1', 'r a 0 2'])
        ns.operating_point()
        self.assertEqual(ns.vectors(),
                         {'a'        : [1],
                          'va#branch': [-0.5]})

    def test_capacitor(self):
        ns.circ(['va a 0 ac 1 dc 0', 'c a 0 1'])
        ns.ac('lin', 1, 1, 1)
        self.assertVectors({'frequency': [1],
                            'a'        : [1],
                            'va#branch': [-2j * pi],
                            })

    def test_inductor(self):
        # operating point does not converge if I use a voltage source here
        ns.circ(['ia a 0 ac 1 dc 0', 'l1 a 0 1'])
        ns.ac('lin', 1, 1, 1)
        self.assertVectors({'frequency': [1],
                            'a'        : [-2j * pi],
                            'l1#branch': [-1],
                            })


class TestCommands(NgspiceTest):
    def test_cmd(self):
        self.assertEqual(ns.cmd('print planck'),
                         ['planck = 6.626200e-34'])
        self.assertEqual(ns.cmd('print' + ' '*200 + 'kelvin'),
                         ['kelvin = -2.73150e+02'])

        # Command too long
        self.assertRaises(ValueError, ns.cmd, 'print' + ' '*2000 + 'kelvin')

    def test_source(self):
        ns.source('../examples/npn/npn.net')
        self.assertEqual(ns.model_parameters(model='QBC337AP')['bf'], 175)

    def test_plots(self):
        ns.circ('va a 0 dc 1')
        for ii in range(3):
            ns.operating_point()
        self.assertEqual(ns.plots(),
                         ['op3', 'op2', 'op1', 'const'])

    def test_vector_names(self):
        ns.cmd('set curplot = new')
        names = list('abdf')
        for name in names:
            ns.cmd('let {} = 0'.format(name))
        self.assertEqual(sorted(ns.vector_names()), names)

    def test_vector(self):
        ns.cmd('let myvector = unitvec(4)')
        self.assertEqual(list(ns.vector('myvector')), [1, 1, 1, 1])

    def test_alter(self):
        ns.circ('r n 0 1')
        ns.alter('r', resistance=2, temp=3)
        ns.operating_point()  # Necessary for resistance to be calculated
        self.assertEqual(ns.vector('@r[resistance]'), 2)
        self.assertEqual(ns.vector('@r[temp]'), 3)

    def test_altermod(self):
        ns.circ(['r n 0 rmodel', '.model rmodel R res = 3'])
        ns.alter_model('r', res=4)
        ns.operating_point()
        self.assertEqual(ns.vector('@r[resistance]'), 4)

    def test_model_parameters(self):
        ns.circ(['r n 0 rmodel', '.model rmodel R res = 3'])
        self.assertEqual(ns.model_parameters(model='rmodel')['r'], 3)
        self.assertEqual(ns.model_parameters(device='r')['r'], 3)

        # Must specify device or model, and not both
        self.assertRaises(ValueError, ns.model_parameters)
        self.assertRaises(ValueError, ns.model_parameters, model='rmodel',
                          device='r')

    def test_ac(self):
        ns.circ(['va a 0 ac 1 dc 0', 'c a 0 1'])
        results = ns.ac('lin', 1, 1, 1)
        self.assertEqual(results.keys(),  {'a', 'va#branch', 'frequency'})

        # Invalid mode
        self.assertRaises(ValueError, ns.ac, 'foo', 1, 2, 3)

        # fstart > fstop
        self.assertRaises(ValueError, ns.ac, 'lin', 2, 3, 2)


class TestHelpers(unittest.TestCase):
    def test_decibel(self):
        self.assertEqual(list(ns.decibel([1, 10, 100])), [0, 10, 20])


class TestLinearSweep(unittest.TestCase):
    def setUp(self):
        ns.circ('va a 0 dc 0')

    def assertEqualNdarray(self, xs1, xs2):
        self.assertEqual(xs1.shape, xs2.shape)
        for x1, x2 in zip(xs1.flat, xs2.flat):
            self.assertEqual(x1, x2)

    def _test_sweep(self, *args):
        ns.dc('va', *args)
        self.assertEqualNdarray(ns.vector('a'),
                                ns.linear_sweep(*args))

    def test_linearsweep(self):
        testcases = [
                     (0, 10, 1),
                     (0, -10, -1),
                     # (.3, .3, 0),
                     (0, 10, .1),
                     (1.23, 4.56, 0.789),
                     ]
        for sweep in testcases:
            with self.subTest(sweep=sweep):
                self._test_sweep(*sweep)

        # Invalid sweeps
        self.assertRaises(ValueError, ns.linear_sweep, 9, 1, 1)
        self.assertRaises(ValueError, ns.linear_sweep, 1, 9, -1)


if __name__ == '__main__':
    unittest.main()
