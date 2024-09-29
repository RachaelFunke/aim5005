import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # TODO: There is a bug here... Look carefully! 
        return (x-self.minimum) / diff_max_min
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
        ### Implementation
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Ensure x is a numpy array.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert isinstance(x, np.ndarray), "Expected the input to be a numpy array"
        return x
        
    def fit(self, x: np.ndarray) -> None:
        x = self._check_is_array(x)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Standardize the given vector.
        """
        x = self._check_is_array(x)
        return (x - self.mean) / self.std
    
    def fit_transform(self, x: list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)    
     ### Label enconder
class LabelEncoder:
    def __init__(self):
        self.classes_ = None
        self.class_to_index = None
    
    def fit(self, y: List) -> None:
        """
        Fit label encoder by finding the unique classes and sorting them.
        """
        self.classes_ = np.unique(y)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        
    def transform(self, y: List) -> np.ndarray:
        """
        Transform labels to normalized encoding.
        """
        if self.classes_ is None:
            raise ValueError("LabelEncoder not fitted yet. Call `fit` before `transform`.")
        return np.array([self.class_to_index[cls] for cls in y])
    
    def fit_transform(self, y: List) -> np.ndarray:
        """
        Fit label encoder and transform labels in one step.
        """
        self.fit(y)
        return self.transform(y)
    # Test for StandardScaler
scaler = StandardScaler()
data = np.array([[1, 2], [3, 4], [5, 6]])
scaler.fit(data)
transformed = scaler.transform(data)

print("Mean:", scaler.mean)          # Expected: [3. 4.]
print("Std:", scaler.std)            # Expected: [1.63299316 1.63299316]
print("Transformed:", transformed)   # Should be standardized values

