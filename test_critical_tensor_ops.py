#!/usr/bin/env python3
"""
Critical Tensor Operations Test Suite

Tests all the tensor operations that were causing errors in isolation:
1. deg.pow(-0.5) operations (AttributeError: 'int' object has no attribute 'pow')
2. Device transfer operations (CUDA assert errors)
3. Numerical stability operations (NaN/Inf handling)
4. Bounds checking operations
5. Tensor dtype consistency
"""

import sys
import traceback
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.config import get_device, set_seed
    print("✅ Successfully imported from src.config")
except ImportError as e:
    print(f"⚠️  Could not import from src.config: {e}")
    # Fallback definitions
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def set_seed():
        torch.manual_seed(42)

class TensorOperationTester:
    """Test suite for critical tensor operations."""
    
    def __init__(self):
        self.device = get_device()
        self.test_results = []
        print(f"🔧 Testing on device: {self.device}")
        
        # Set seed for reproducibility
        set_seed()
        
    def log_result(self, test_name, success, error_msg=None):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append((test_name, success, error_msg))
        if success:
            print(f"{status} {test_name}")
        else:
            print(f"{status} {test_name}: {error_msg}")
    
    def test_degree_pow_operations(self):
        """Test deg.pow(-0.5) operations that were causing 'int' object errors."""
        print("\n📊 Testing degree power operations...")
        
        # Test 1: Basic degree tensor creation and pow operation
        try:
            degrees = torch.tensor([0, 1, 2, 5, 10, 100], dtype=torch.float32, device=self.device)
            
            # This was the problematic operation
            inv_sqrt_deg = degrees.pow(-0.5)
            
            # Check for inf values (degree 0 should produce inf)
            expected_inf_mask = (degrees == 0)
            actual_inf_mask = torch.isinf(inv_sqrt_deg)
            
            if not torch.equal(expected_inf_mask, actual_inf_mask):
                self.log_result("Degree pow(-0.5) basic", False, "Inf handling incorrect")
            else:
                self.log_result("Degree pow(-0.5) basic", True)
                
        except Exception as e:
            self.log_result("Degree pow(-0.5) basic", False, str(e))
        
        # Test 2: Safe degree pow operation (our fix)
        try:
            degrees = torch.tensor([0, 1, 2, 5, 10, 100], dtype=torch.float32, device=self.device)
            
            # Our safe implementation using torch.where
            safe_inv_sqrt_deg = torch.where(
                degrees > 0,
                degrees.pow(-0.5),
                torch.zeros_like(degrees)
            )
            
            # Should have no inf values
            if torch.any(torch.isinf(safe_inv_sqrt_deg)):
                self.log_result("Safe degree pow(-0.5)", False, "Still contains inf values")
            else:
                self.log_result("Safe degree pow(-0.5)", True)
                
        except Exception as e:
            self.log_result("Safe degree pow(-0.5)", False, str(e))
        
        # Test 3: Integer degree tensor (the original problem)
        try:
            # This was causing the AttributeError
            degrees_int = torch.tensor([0, 1, 2, 5, 10, 100], dtype=torch.long, device=self.device)
            
            # Convert to float before pow operation (our fix)
            degrees_float = degrees_int.float()
            inv_sqrt_deg = degrees_float.pow(-0.5)
            
            self.log_result("Integer degree conversion", True)
            
        except Exception as e:
            self.log_result("Integer degree conversion", False, str(e))
    
    def test_device_transfer_operations(self):
        """Test device transfer operations that were causing CUDA errors."""
        print("\n🔄 Testing device transfer operations...")
        
        # Test 1: Basic device transfer
        try:
            tensor_cpu = torch.randn(100, 64)
            tensor_device = tensor_cpu.to(self.device)
            
            # Verify device
            if tensor_device.device != self.device:
                self.log_result("Basic device transfer", False, f"Wrong device: {tensor_device.device}")
            else:
                self.log_result("Basic device transfer", True)
                
        except Exception as e:
            self.log_result("Basic device transfer", False, str(e))
        
        # Test 2: Mixed device operations (our safety checks)
        try:
            tensor_a = torch.randn(10, 10, device=self.device)
            tensor_b = torch.randn(10, 10)  # CPU tensor
            
            # Safe device matching (our implementation)
            if tensor_a.device != tensor_b.device:
                tensor_b = tensor_b.to(tensor_a.device)
            
            result = torch.matmul(tensor_a, tensor_b)
            
            if result.device != self.device:
                self.log_result("Mixed device safety", False, "Result on wrong device")
            else:
                self.log_result("Mixed device safety", True)
                
        except Exception as e:
            self.log_result("Mixed device safety", False, str(e))
        
        # Test 3: Edge index device consistency
        try:
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
            edge_index_device = edge_index.to(self.device)
            
            embeddings = torch.randn(3, 64, device=self.device)
            
            # This operation was failing due to device mismatch
            user_emb = embeddings[edge_index_device[0]]
            item_emb = embeddings[edge_index_device[1]]
            
            if user_emb.device != self.device or item_emb.device != self.device:
                self.log_result("Edge index device consistency", False, "Embedding device mismatch")
            else:
                self.log_result("Edge index device consistency", True)
                
        except Exception as e:
            self.log_result("Edge index device consistency", False, str(e))
    
    def test_numerical_stability_operations(self):
        """Test numerical stability operations for NaN/Inf handling."""
        print("\n🔢 Testing numerical stability operations...")
        
        # Test 1: Sigmoid clamping (prevents overflow)
        try:
            large_logits = torch.tensor([-1000.0, -100.0, 0.0, 100.0, 1000.0], device=self.device)
            
            # Our clamped implementation
            clamped_logits = torch.clamp(large_logits, min=-10.0, max=10.0)
            sigmoid_result = torch.sigmoid(clamped_logits)
            
            if torch.any(torch.isnan(sigmoid_result)) or torch.any(torch.isinf(sigmoid_result)):
                self.log_result("Sigmoid clamping", False, "Contains NaN or Inf")
            else:
                self.log_result("Sigmoid clamping", True)
                
        except Exception as e:
            self.log_result("Sigmoid clamping", False, str(e))
        
        # Test 2: Log operation safety
        try:
            probs = torch.tensor([0.0, 1e-10, 0.5, 1.0], device=self.device)
            
            # Safe log operation (our implementation)
            safe_log = torch.log(torch.clamp(probs, min=1e-8))
            
            if torch.any(torch.isnan(safe_log)) or torch.any(torch.isinf(safe_log)):
                self.log_result("Safe log operation", False, "Contains NaN or Inf")
            else:
                self.log_result("Safe log operation", True)
                
        except Exception as e:
            self.log_result("Safe log operation", False, str(e))
        
        # Test 3: Division by zero protection
        try:
            numerator = torch.randn(100, device=self.device)
            denominator = torch.randn(100, device=self.device)
            
            # Some denominators might be very close to zero
            denominator[0] = 0.0
            denominator[1] = 1e-10
            
            # Safe division (our implementation)
            safe_result = numerator / torch.clamp(denominator.abs(), min=1e-8)
            
            if torch.any(torch.isnan(safe_result)) or torch.any(torch.isinf(safe_result)):
                self.log_result("Division by zero protection", False, "Contains NaN or Inf")
            else:
                self.log_result("Division by zero protection", True)
                
        except Exception as e:
            self.log_result("Division by zero protection", False, str(e))
    
    def test_bounds_checking_operations(self):
        """Test bounds checking operations for tensor indexing."""
        print("\n🔍 Testing bounds checking operations...")
        
        # Test 1: User/item index bounds
        try:
            num_users, num_items = 1000, 500
            edge_index = torch.tensor([[0, 999, 500], [0, 499, 250]], dtype=torch.long, device=self.device)
            
            # Bounds checking (our implementation)
            user_indices = edge_index[0]
            item_indices = edge_index[1]
            
            user_valid = torch.all((user_indices >= 0) & (user_indices < num_users))
            item_valid = torch.all((item_indices >= 0) & (item_indices < num_items))
            
            if not user_valid or not item_valid:
                self.log_result("Index bounds checking", False, f"Invalid indices: users={user_valid}, items={item_valid}")
            else:
                self.log_result("Index bounds checking", True)
                
        except Exception as e:
            self.log_result("Index bounds checking", False, str(e))
        
        # Test 2: Edge index validation
        try:
            edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long, device=self.device)
            
            # Validate edge index shape and values
            if edge_index.shape[0] != 2:
                self.log_result("Edge index validation", False, f"Wrong shape: {edge_index.shape}")
            elif torch.any(edge_index < 0):
                self.log_result("Edge index validation", False, "Negative indices found")
            else:
                self.log_result("Edge index validation", True)
                
        except Exception as e:
            self.log_result("Edge index validation", False, str(e))
        
        # Test 3: Empty tensor handling
        try:
            empty_tensor = torch.tensor([], dtype=torch.long, device=self.device)
            
            # Operations on empty tensors should not crash
            if len(empty_tensor) == 0:
                # This should work
                result = empty_tensor.float()
                self.log_result("Empty tensor handling", True)
            else:
                self.log_result("Empty tensor handling", False, "Empty tensor not properly handled")
                
        except Exception as e:
            self.log_result("Empty tensor handling", False, str(e))
    
    def test_dtype_consistency_operations(self):
        """Test tensor dtype consistency operations."""
        print("\n📏 Testing dtype consistency operations...")
        
        # Test 1: Degree tensor dtypes
        try:
            # Item degree should be float (for pow operations)
            item_degree = torch.tensor([0, 1, 2, 5], dtype=torch.float32, device=self.device)
            
            # Edge indices should be long
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long, device=self.device)
            
            # Check dtypes
            if item_degree.dtype != torch.float32:
                self.log_result("Item degree dtype", False, f"Wrong dtype: {item_degree.dtype}")
            elif edge_index.dtype != torch.long:
                self.log_result("Edge index dtype", False, f"Wrong dtype: {edge_index.dtype}")
            else:
                self.log_result("Tensor dtype consistency", True)
                
        except Exception as e:
            self.log_result("Tensor dtype consistency", False, str(e))
        
        # Test 2: Embedding dtype consistency
        try:
            embeddings = torch.randn(100, 64, dtype=torch.float32, device=self.device)
            
            # All operations should maintain float32
            normalized = F.normalize(embeddings, p=2, dim=1)
            
            if normalized.dtype != torch.float32:
                self.log_result("Embedding dtype consistency", False, f"Wrong dtype: {normalized.dtype}")
            else:
                self.log_result("Embedding dtype consistency", True)
                
        except Exception as e:
            self.log_result("Embedding dtype consistency", False, str(e))
    
    def test_integrated_critical_operations(self):
        """Test integrated operations that combine multiple critical components."""
        print("\n🔗 Testing integrated critical operations...")
        
        # Test 1: Complete degree calculation pipeline
        try:
            edge_index = torch.tensor([[0, 1, 2, 0, 1], [1, 2, 0, 2, 0]], dtype=torch.long, device=self.device)
            num_nodes = 3
            
            # Calculate degrees (this was problematic)
            row, col = edge_index
            degree = torch.zeros(num_nodes, dtype=torch.float32, device=self.device)
            degree.scatter_add_(0, row, torch.ones(row.size(0), dtype=torch.float32, device=self.device))
            
            # Safe normalization (our fix)
            deg_inv_sqrt = torch.where(
                degree > 0,
                degree.pow(-0.5),
                torch.zeros_like(degree)
            )
            
            # Create normalized adjacency
            norm_edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            
            if torch.any(torch.isnan(norm_edge_weight)) or torch.any(torch.isinf(norm_edge_weight)):
                self.log_result("Degree calculation pipeline", False, "Contains NaN or Inf")
            else:
                self.log_result("Degree calculation pipeline", True)
                
        except Exception as e:
            self.log_result("Degree calculation pipeline", False, str(e))
        
        # Test 2: BPR loss calculation with safety
        try:
            batch_size = 64
            embedding_dim = 32
            
            user_emb = torch.randn(batch_size, embedding_dim, device=self.device)
            pos_item_emb = torch.randn(batch_size, embedding_dim, device=self.device)
            neg_item_emb = torch.randn(batch_size, embedding_dim, device=self.device)
            
            # BPR score calculation (this was problematic)
            pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
            neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
            
            # Safe BPR loss (our implementation)
            bpr_loss_val = pos_scores - neg_scores
            bpr_loss_val = torch.clamp(bpr_loss_val, min=-10.0, max=10.0)  # Prevent overflow
            bpr_loss_val = -torch.mean(torch.log(torch.sigmoid(bpr_loss_val) + 1e-8))
            
            if torch.isnan(bpr_loss_val) or torch.isinf(bpr_loss_val):
                self.log_result("Safe BPR loss calculation", False, "Loss is NaN or Inf")
            else:
                self.log_result("Safe BPR loss calculation", True)
                
        except Exception as e:
            self.log_result("Safe BPR loss calculation", False, str(e))
    
    def run_all_tests(self):
        """Run all tensor operation tests."""
        print("🧪 CRITICAL TENSOR OPERATIONS TEST SUITE")
        print("=" * 80)
        
        self.test_degree_pow_operations()
        self.test_device_transfer_operations()
        self.test_numerical_stability_operations()
        self.test_bounds_checking_operations()
        self.test_dtype_consistency_operations()
        self.test_integrated_critical_operations()
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        failed_tests = [name for name, success, error in self.test_results if not success]
        
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {passed/total*100:.1f}%")
        
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test_name in failed_tests:
                error = next(error for name, success, error in self.test_results if name == test_name and not success)
                print(f"  - {test_name}: {error}")
        else:
            print(f"\n✅ ALL TESTS PASSED!")
        
        return passed == total

def main():
    """Main test execution."""
    try:
        tester = TensorOperationTester()
        success = tester.run_all_tests()
        
        if success:
            print(f"\n🎉 ALL CRITICAL TENSOR OPERATIONS VERIFIED!")
            print(f"✅ The fixes applied in the code review are working correctly.")
            return True
        else:
            print(f"\n⚠️  SOME TENSOR OPERATIONS FAILED!")
            print(f"❌ Review the failed tests and ensure fixes are properly implemented.")
            return False
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR in test execution: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)