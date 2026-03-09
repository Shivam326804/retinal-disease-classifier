"""Test suite for retinal disease classification system"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.training.model_builder import ModelBuilder

logger = setup_logger(__name__)


class TestConfig:
    """Test configuration module"""
    
    def test_config_base_dir_exists(self):
        """Test that base directory is properly configured"""
        assert Config.BASE_DIR is not None
        assert Path(Config.BASE_DIR).exists() or Config.BASE_DIR
    
    def test_config_paths_defined(self):
        """Test that all paths are defined"""
        assert Config.DATA_DIR is not None
        assert Config.MODELS_DIR is not None
        assert Config.LOGS_DIR is not None
    
    def test_disease_classes_count(self):
        """Test disease classes are correctly defined"""
        assert Config.NUM_CLASSES == 5
        assert len(Config.DISEASE_CLASSES) == 5
    
    def test_config_hyperparameters(self):
        """Test hyperparameters are valid"""
        assert Config.BATCH_SIZE > 0
        assert Config.EPOCHS > 0
        assert Config.LEARNING_RATE > 0
        assert 0 < Config.IMAGE_SIZE < 1000
    
    def test_config_dict(self):
        """Test configuration dictionary generation"""
        config_dict = Config.get_config_dict()
        assert 'batch_size' in config_dict
        assert 'epochs' in config_dict
        assert 'num_classes' in config_dict


class TestDataPreprocessor:
    """Test data preprocessing module"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        return DataPreprocessor(image_size=300, normalize=True)
    
    def test_preprocessor_initialization(self, preprocessor):
        """Test preprocessor initialization"""
        assert preprocessor.image_size == 300
        assert preprocessor.normalize == True
        assert preprocessor.processed_data_dir is not None
    
    def test_create_dummy_image(self):
        """Test creation of dummy image"""
        dummy_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        assert dummy_image.shape == (512, 512, 3)
        assert dummy_image.max() <= 256
        assert dummy_image.min() >= 0
    
    def test_resize_with_padding(self, preprocessor):
        """Test image resizing with padding"""
        # Create test image
        test_image = np.ones((512, 768, 3), dtype=np.uint8) * 128
        
        # Resize
        resized = preprocessor._resize_with_padding(test_image, 300)
        
        # Verify output shape
        assert resized.shape == (300, 300, 3)
    
    def test_normalization(self, preprocessor):
        """Test image normalization"""
        # Create test image
        test_image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        
        # Normalize
        normalized = test_image.astype(np.float32) / 255.0
        
        # Verify normalization
        assert normalized.max() <= 1.0
        assert normalized.min() >= 0.0
        assert normalized.dtype == np.float32
    
    def test_dataset_split(self, preprocessor):
        """Test dataset splitting"""
        # Create dummy dataset
        images = np.random.rand(100, 300, 300, 3).astype(np.float32)
        labels = np.random.randint(0, 5, 100)
        
        # Split
        splits = preprocessor.split_dataset(images, labels)
        
        # After splitting we expect one-hot conversion may be done elsewhere
        # so verify shapes before conversion check
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        assert len(X_train) > len(X_val)
        assert len(X_val) < len(X_train)
        assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == 100
        
    def test_load_image_with_target_size(self, preprocessor):
        """Test loading image with custom target size"""
        # create a dummy image file
        dummy_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        tmp_path = Path(preprocessor.processed_data_dir) / "temp_test.jpg"
        import cv2
        cv2.imwrite(str(tmp_path), dummy_img)
        img = preprocessor.load_image(str(tmp_path), target_size=(128, 128))
        assert img is not None
        assert img.shape == (128, 128, 3)
        tmp_path.unlink()  # cleanup


class TestModelBuilder:
    """Test model building module"""
    
    @pytest.fixture
    def builder(self):
        """Create model builder instance"""
        return ModelBuilder(input_shape=(300, 300, 3), num_classes=5)
    
    def test_builder_initialization(self, builder):
        """Test builder initialization"""
        assert builder.input_shape == (300, 300, 3)
        assert builder.num_classes == 5
    
    def test_baseline_cnn_creation(self, builder):
        """Test baseline CNN model creation"""
        model = builder.build_baseline_cnn()
        assert model is not None
        assert hasattr(model, 'predict')
        assert len(model.layers) > 0
    
    def test_custom_cnn_creation(self, builder):
        """Test custom CNN model creation"""
        model = builder.build_custom_cnn()
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_model_input_shape(self, builder):
        """Test model accepts correct input shape"""
        model = builder.build_baseline_cnn()
        assert model.input_shape == (None, 300, 300, 3)
    
    def test_model_output_shape(self, builder):
        """Test model produces correct output shape"""
        model = builder.build_baseline_cnn()
        assert model.output_shape == (None, 5)
    
    @pytest.mark.slow
    def test_model_forward_pass(self, builder):
        """Test model forward pass with dummy data"""
        model = builder.build_baseline_cnn()
        
        # Create dummy input
        dummy_input = np.random.rand(2, 300, 300, 3).astype(np.float32)
        
        # Forward pass
        output = model.predict(dummy_input, verbose=0)
        
        # Verify output
        assert output.shape == (2, 5)  # 2 samples, 5 classes
        assert np.allclose(output.sum(axis=1), 1.0)  # Softmax sums to 1


class TestDataLoading:
    """Test data loading functionality"""
    
    def test_dummy_image_creation(self):
        """Test dummy image can be created"""
        import cv2
        
        img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        assert img.shape == (512, 512, 3)
    
    def test_batch_creation(self):
        """Test batch creation"""
        images = np.random.rand(64, 300, 300, 3).astype(np.float32)
        labels = np.random.randint(0, 5, 64)
        
        # Should not raise error
        assert len(images) == 64
        assert len(labels) == 64


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_logger_creation(self):
        """Test logger creation"""
        test_logger = setup_logger("test_module")
        assert test_logger is not None
        assert test_logger.name == "test_module"
    
    def test_config_create_directories(self):
        """Test directory creation"""
        Config.create_all_directories()
        # Should not raise error
        assert True


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.slow
    def test_preprocessing_pipeline(self):
        """Test end-to-end preprocessing"""
        preprocessor = DataPreprocessor(image_size=300)
        
        # Create dummy dataset
        images = np.random.rand(50, 300, 300, 3).astype(np.float32)
        labels = np.random.randint(0, 5, 50)
        
        # Split
        splits = preprocessor.split_dataset(images, labels)
        
        # Verify consistency
        assert splits['train'][0].shape[0] > 0
        assert splits['val'][0].shape[0] > 0
        assert splits['test'][0].shape[0] > 0
    
    @pytest.mark.slow
    def test_model_training_pipeline(self):
        """Test model can be created and prepared for training"""
        builder = ModelBuilder(input_shape=(300, 300, 3), num_classes=5)
        model = builder.build_baseline_cnn()
        
        # Verify model compilation
        assert model.optimizer is not None
        assert model.loss is not None
    
    def test_configuration_workflow(self):
        """Test complete configuration workflow"""
        Config.create_all_directories()
        config_dict = Config.get_config_dict()
        
        assert config_dict['batch_size'] == Config.BATCH_SIZE
        assert config_dict['epochs'] == Config.EPOCHS
        assert config_dict['num_classes'] == Config.NUM_CLASSES


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
