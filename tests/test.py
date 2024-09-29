import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class TestFeatures(unittest.TestCase):

    # Test if MinMaxScaler is initialized correctly
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        self.assertIsInstance(scaler, MinMaxScaler, "Scaler is not a MinMaxScaler object")
        
    # Test if MinMaxScaler correctly fits the data and returns expected min and max values
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        np.testing.assert_array_equal(scaler.data_max_, np.array([1., 18.]),
                                      err_msg="Scaler fit does not return maximum values [1., 18.]")
        np.testing.assert_array_equal(scaler.data_min_, np.array([-1., 2.]),
                                      err_msg="Scaler fit does not return minimum values [-1., 2.]")
        
    # Test if MinMaxScaler transforms data correctly
    def test_min_max_scaler_transform(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        np.testing.assert_array_almost_equal(result, expected,
                                             err_msg=f"Scaler transform does not return expected values. Got: {result.reshape(1,-1)}")
        
    # Test if MinMaxScaler transforms a single value correctly
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]])
        np.testing.assert_array_almost_equal(result, expected,
                                             err_msg=f"Scaler transform does not return expected values. Expect [[1.5, 0.]]. Got: {result}")
        
    # Test if StandardScaler is initialized correctly
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        self.assertIsInstance(scaler, StandardScaler, "Scaler is not a StandardScaler object")
        
    # Test if StandardScaler correctly computes the mean after fitting
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        np.testing.assert_array_almost_equal(scaler.mean_, expected,
                                             err_msg=f"Scaler fit does not return expected mean {expected}. Got {scaler.mean_}")
        
    # Test if StandardScaler transforms data correctly
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)
        np.testing.assert_array_almost_equal(result, expected,
                                             err_msg=f"Scaler transform does not return expected values. Expect {expected.reshape(1,-1)}. Got: {result.reshape(1,-1)}")
        
    # Test if StandardScaler transforms a single value correctly
    def test_standard_scaler_single_value(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler.fit(data)
        result = scaler.transform([[2., 2.]])
        np.testing.assert_array_almost_equal(result, expected,
                                             err_msg=f"Scaler transform does not return expected values. Expect {expected.reshape(1,-1)}. Got: {result}")

if __name__ == '__main__':
    unittest.main()
