import nose
from unittest import TestCase
from numpy import allclose

class testOutput(TestCase)
    def setup_func(self):
        times_numeric, uptake_numeric = self.model.evaluate(times,params)
        pass

    def teardown_func(self):
        del self.result_numeric
        del self.result_analytic

    def testTimes(self):
        assert self.times_numeric == self.times

    def testSizeCProfile(self):
        assert self.cprofile_numeric.shape == self.cprofile_analytic.shape

    def testLengthUptake(self):
        assert len(self.uptake_numeric) == len(self.uptake_analytic)

    def testValueUptake(self):
        assert allclose(self.uptake_numeric,self.uptake_analytic)
