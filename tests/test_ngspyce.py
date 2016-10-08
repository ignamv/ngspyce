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

class TestBasicCircuits(NgspiceTest):
    def test_vsource(self):
        ns.circ('va a 0 dc 1')
        ns.operating_point()
        self.assertEqual(ns.vectors(), 
                         {'a'        : [1]
                         ,'va#branch': [0]})

    def test_resistor(self):
        ns.circ(['va a 0 dc 1', 'r a 0 2'])
        ns.operating_point()
        self.assertEqual(ns.vectors(), 
                         {'a'        : [1]
                         ,'va#branch': [-0.5]})

    def test_capacitor(self):
        ns.circ(['va a 0 ac 1 dc 0', 'c a 0 1'])
        ns.ac('lin', 1, 1, 1)
        self.assertVectors({'frequency': [1]
                           ,'a'        : [1]
                           ,'va#branch': [-2j * pi]
                           })

    def test_inductor(self):
        # operating point does not converge if I use a voltage source here
        ns.circ(['ia a 0 ac 1 dc 0', 'l1 a 0 1'])
        ns.ac('lin', 1, 1, 1)
        self.assertVectors({'frequency': [1]
                           ,'a'        : [-2j * pi]
                           ,'l1#branch': [-1]
                           })

class TestCommands(NgspiceTest):
    def test_print(self):
        self.assertEqual(ns.cmd('print planck'),
                         ['planck = 6.626200e-34'])

    def test_plots(self):
        ns.circ('va a 0 dc 1')
        for ii in range(3):
            ns.operating_point()
        self.assertEqual(ns.plots(),
                         ['op3', 'op2', 'op1', 'const'])

    def test_vectorNames(self):
        ns.cmd('set curplot = new')
        names = list('abdf')
        for name in names:
            ns.cmd('let {} = 0'.format(name))
        self.assertEqual(sorted(ns.vectorNames()), names)

    def test_vector(self):
        ns.cmd('let myvector = unitvec(4)')
        self.assertEqual(list(ns.vector('myvector')), [1, 1, 1, 1])

    def test_alter(self):
        ns.circ('r n 0 1')
        ns.alter('r', resistance=2, temp=3)
        ns.operating_point() # Necessary for resistance to be calculated
        self.assertEqual(ns.vector('@r[resistance]'), 2)
        self.assertEqual(ns.vector('@r[temp]'), 3)

    def test_altermod(self):
        ns.circ(['r n 0 rmodel', '.model rmodel R res = 3'])
        ns.alter_model('r', res=4)
        ns.operating_point()
        self.assertEqual(ns.vector('@r[resistance]'), 4)

class TestHelpers(unittest.TestCase):
    def test_decibel(self):
        self.assertEqual(list(ns.decibel([1, 10, 100])), [0, 10, 20])

class TestLinearSweep(unittest.TestCase):
    def setUp(self):
        ns.circ('va a 0 dc 0')

    def assertEqualNdarray(self, xs1, xs2):
        for x1, x2 in zip(xs1.flat, xs2.flat):
            avg = .5 * (x1 + x2)
            if avg == 0:
                self.assertEqual(x1, x2)
            else:
                self.assertAlmostEqual(x1 / avg, x2 / avg, places=10)

    def _test_sweep(self, *args):
        ns.dc('va', *args)
        self.assertEqualNdarray(ns.vector('a')
                               ,ns.linear_sweep(*args))

    def test_linearsweep(self):
        testcases = [
             (0, 10, 1)
            ,(0, -10, -1)
            ,(0, 10, .1)
            ,(1.23, 4.56, 0.789)
            ,(1.23, -4.56, 0.789)
            ,(1.23, 4.56, -0.789)
        ]
        for sweep in testcases:
            with self.subTest(sweep=sweep):
                self._test_sweep(*sweep)

