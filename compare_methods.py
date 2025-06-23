import time
import numpy as np
import pandas as pd
from module import PredictionIntervalModel

def compare_prediction_methods():
    print("=== Comparing Prediction Methods ===\n")
    
    predictor = PredictionIntervalModel(confidence_level=0.95)
    
    print("Preparing data...")
    X_train, y_train, X_test = predictor.prepare_data()
    print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}\n")
    
    methods = {
        'Ultra Fast': predictor.ultra_fast_approach,
        'Improved': predictor.improved_approach,
        'Adaptive': predictor.adaptive_uncertainty_approach
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"=== Testing {method_name} Method ===")
        start_time = time.time()
        
        try:
            pi_lower, pi_upper = method_func(X_train, y_train, X_test)
            execution_time = time.time() - start_time
            
            if pi_lower is not None:
                quality = predictor.evaluate_prediction_quality(pi_lower, pi_upper)
                results[method_name] = {
                    'time': execution_time,
                    'mean_width': quality['mean_width'],
                    'mean_center': quality['mean_center'],
                    'success': True
                }
                print(f"{method_name} completed in {execution_time:.2f} seconds\n")
            else:
                results[method_name] = {'success': False}
                print(f"{method_name} failed\n")
                
        except Exception as e:
            results[method_name] = {'success': False, 'error': str(e)}
            print(f"{method_name} failed with error: {str(e)}\n")
    
    print("=== Method Comparison Summary ===")
    for method_name, result in results.items():
        if result['success']:
            print(f"{method_name}:")
            print(f"  Time: {result['time']:.2f}s")
            print(f"  Mean width: ${result['mean_width']:,.0f}")
            print(f"  Mean center: ${result['mean_center']:,.0f}")
        else:
            print(f"{method_name}: FAILED")
        print()
    
    return results

if __name__ == "__main__":
    results = compare_prediction_methods() 