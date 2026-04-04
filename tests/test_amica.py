
import unittest
import numpy as np
from amica import Amica, AmicaConfig

class TestAmicaNative(unittest.TestCase):
    def test_run_simple(self):
        """Test basic fitting on random data."""
        rng = np.random.RandomState(42)
        n_channels = 4
        n_samples = 200
        data = rng.randn(n_channels, n_samples)
        
        # Use simple config for speed
        config = AmicaConfig(
            max_iter=10,
            num_mix_comps=2,
            do_sphere=True,
            do_mean=True,
            mineig=1e-20 # Accommodate scaled variance (1e-12)
        )
        
        amica = Amica(config=config, random_state=42)
        res = amica.fit(data)
        
        self.assertIsNotNone(res)
        self.assertEqual(res.unmixing_matrix.shape, (n_channels, n_channels))
        self.assertEqual(res.mixing_matrix.shape, (n_channels, n_channels))
        self.assertEqual(len(res.log_likelihood), 10)
        
        # Test transform
        sources = amica.transform(data)
        self.assertEqual(sources.shape, (n_channels, n_samples))
        
        # Test inverse
        recon = amica.inverse_transform(sources)
        self.assertEqual(recon.shape, (n_channels, n_samples))
        
        # Check reconstruction error ( should be small if n_components=n_channels)
        err = np.mean((data - recon)**2)
        self.assertLess(err, 1e-10)

if __name__ == '__main__':
    unittest.main()
